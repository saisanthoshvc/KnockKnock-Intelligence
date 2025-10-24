# app_simple.py
# -------------------------------------------------------------------------
# KnockKnock Intelligence — Flask frontend for the RFP Question Extraction API
# - Uses Posit Connect-style API key auth (supports custom header names)
# - Works with both async (/extract/) and sync (/extract/sync) endpoints
# - Uploads DOCX/PDF/XLSX, polls job status, shows Review/Edit page, exports
# - Heavily instrumented logging to diagnose deployment issues on Connect
# -------------------------------------------------------------------------

import io
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

import requests
from flask import (
    Flask, render_template, request, jsonify, redirect,
    url_for, send_file
)
from werkzeug.utils import secure_filename

# ========= Flask setup =========
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")  # replace in prod

# ========= Upload config =========
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
ALLOWED_EXTENSIONS = {"docx", "pdf", "xlsx"}
MAX_MB = float(os.getenv("MAX_UPLOAD_MB", "16"))
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = int(MAX_MB * 1024 * 1024)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ========= Env-driven API config (required on Posit Connect) =========
def _get_env_float(name: str, default: float) -> float:
    try:
        v = os.getenv(name, "").strip()
        return float(v) if v else default
    except Exception:
        return default

API_BASE_URL = (os.getenv(
    "RFP_API_URL",
    "https://connect.affiniusaiplatform.com/content/000d3572-937a-4d5c-ab7e-7ec80d80c4ce"
) or "").rstrip("/")

API_KEY_HEADER_NAME = os.getenv("RFP_API_KEY_HEADER_NAME", "Authorization").strip()
API_AUTH_SCHEME = os.getenv("RFP_API_AUTH_SCHEME", "Raw").strip()  # Raw | Bearer | Key
API_KEY = os.getenv("RFP_API_KEY", "").strip()

POLL_INTERVAL_SECONDS = _get_env_float("POLL_INTERVAL_SECONDS", 2.0)
POLL_TIMEOUT_SECONDS = _get_env_float("POLL_TIMEOUT_SECONDS", 240.0)
REQUEST_TIMEOUT_SECONDS = _get_env_float("REQUEST_TIMEOUT_SECONDS", 60.0)

# Stats file (local JSON)
STATS_FILE = os.getenv("STATS_FILE", "stats.json")

# In-memory stores for this process
job_data: Dict[str, Dict[str, Any]] = {}
job_edits: Dict[str, Any] = {}

# ========= Helpers =========
def _log(s: str):
    print(f"[{datetime.now().isoformat()}] {s}", flush=True)

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def _ep(path: str) -> str:
    """Join base URL and path safely. Accepts path with or without leading slash."""
    return f"{API_BASE_URL}/{path.lstrip('/')}"

def _auth_header() -> Dict[str, str]:
    """Builds the authentication header from env configuration."""
    if not API_KEY:
        return {}
    if API_KEY_HEADER_NAME.lower() == "authorization":
        scheme = API_AUTH_SCHEME.lower()
        if scheme == "bearer":
            return {"Authorization": f"Bearer {API_KEY}"}
        elif scheme == "key":
            return {"Authorization": f"Key {API_KEY}"}
        else:
            return {"Authorization": API_KEY}
    return {API_KEY_HEADER_NAME: API_KEY}

def _requests_post(url: str, **kwargs):
    """POST wrapper with consistent timeout & logging."""
    try:
        _log(f"POST {url}")
        return requests.post(url, timeout=REQUEST_TIMEOUT_SECONDS, **kwargs)
    except requests.exceptions.RequestException as e:
        _log(f"POST ERROR {url}: {e}")
        return None

def _requests_get(url: str, **kwargs):
    try:
        _log(f"GET {url}")
        return requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS, **kwargs)
    except requests.exceptions.RequestException as e:
        _log(f"GET ERROR {url}: {e}")
        return None

def _requests_delete(url: str, **kwargs):
    try:
        _log(f"DELETE {url}")
        return requests.delete(url, timeout=REQUEST_TIMEOUT_SECONDS, **kwargs)
    except requests.exceptions.RequestException as e:
        _log(f"DELETE ERROR {url}: {e}")
        return None

# ========= Stats =========
def load_stats() -> Dict[str, Any]:
    try:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE, "r") as f:
                stats = json.load(f)
        else:
            stats = {}
        stats.setdefault("documents_processed", 0)
        stats.setdefault("questions_extracted", 0)
        stats.setdefault("total_processing_time", 0.0)
        docs = stats["documents_processed"] or 1
        stats["avg_processing_time"] = round(stats["total_processing_time"] / docs, 1)
        stats.setdefault("accuracy_rate", 0.0)
        stats["last_updated"] = datetime.now().isoformat()
        return stats
    except Exception as e:
        _log(f"load_stats error: {e}")
        return {
            "documents_processed": 0,
            "questions_extracted": 0,
            "total_processing_time": 0.0,
            "avg_processing_time": 0.0,
            "accuracy_rate": 0.0,
            "last_updated": datetime.now().isoformat(),
        }

def save_stats(stats: Dict[str, Any]):
    try:
        stats["last_updated"] = datetime.now().isoformat()
        with open(STATS_FILE, "w") as f:
            json.dump(stats, f, indent=2)
    except Exception as e:
        _log(f"save_stats error: {e}")

def update_stats(questions_count: int, processing_time_sec: float, questions_data: Optional[list] = None):
    stats = load_stats()
    stats["documents_processed"] += 1
    stats["questions_extracted"] += int(questions_count or 0)
    stats["total_processing_time"] += float(processing_time_sec or 0)

    # Compute "accuracy" as avg confidence (if present)
    if questions_data and isinstance(questions_data, list):
        total_conf = 0.0
        n = 0
        for q in questions_data:
            c = q.get("confidence")
            if isinstance(c, (int, float)):
                total_conf += float(c)
                n += 1
        if n > 0:
            stats["accuracy_rate"] = round((total_conf / n) * 100.0, 1)
        else:
            stats.setdefault("accuracy_rate", 85.0)
    else:
        stats.setdefault("accuracy_rate", 85.0)
    save_stats(stats)
    return stats

# ========= Routes =========
@app.route("/")
def index():
    # Quick config sanity to help debugging on Connect
    _log(f"Serving index with API_BASE_URL={API_BASE_URL}, "
         f"HEADER={API_KEY_HEADER_NAME}, SCHEME={API_AUTH_SCHEME}, "
         f"KEY_PRESENT={'yes' if bool(API_KEY) else 'no'}")
    return render_template("index.html", api_config={
        "base_url": API_BASE_URL,
        "header_name": API_KEY_HEADER_NAME,
        "auth_scheme": API_AUTH_SCHEME,
        "poll_interval": POLL_INTERVAL_SECONDS,
        "poll_timeout": POLL_TIMEOUT_SECONDS,
        "request_timeout": REQUEST_TIMEOUT_SECONDS,
    }, stats=load_stats())

@app.route("/health")
def health():
    return jsonify({"ok": True, "time": datetime.now().isoformat()})

@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Accepts file upload + options, calls Extract API (sync or async)."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": "Invalid file type. Allowed: docx, pdf, xlsx"}), 400

    # Parse options
    use_llm = str(request.form.get("use_llm", "false")).lower() == "true"
    use_sync = str(request.form.get("use_sync", "false")).lower() == "true"
    mode = request.form.get("mode", "balanced") or "balanced"

    # Save to disk (temp)
    filename = secure_filename(f.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(path)

    try:
        # Determine mime
        ext = filename.lower().rsplit(".", 1)[-1]
        mime = "application/octet-stream"
        if ext == "docx":
            mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif ext == "pdf":
            mime = "application/pdf"
        elif ext == "xlsx":
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        # Prepare multipart form; send booleans as strings
        data = {"use_llm": "true" if use_llm else "false", "mode": mode}
        files = {"file": (filename, open(path, "rb"), mime)}

        # Correct endpoints:
        #   async → POST /extract/
        #   sync  → POST /extract/sync
        endpoint = "extract/sync" if use_sync else "extract/"
        url = _ep(endpoint)
        headers = _auth_header()

        _log(f"UPLOAD: use_sync={use_sync}, use_llm={use_llm}, mode={mode}")
        _log(f"UPLOAD: URL={url}, headers={list(headers.keys())}, file={filename}, mime={mime}")

        resp = _requests_post(url, headers=headers, files=files, data=data)

        # Close file handle
        try:
            files["file"][1].close()
        except Exception:
            pass

        if resp is None:
            return jsonify({"error": "Network error calling extractor API"}), 502

        _log(f"UPLOAD: status={resp.status_code}, body_snip={resp.text[:500]!r}")

        if resp.status_code != 200:
            return jsonify({"error": f"HTTP {resp.status_code}: {resp.text}"}), 502

        payload = {}
        try:
            payload = resp.json()
        except Exception:
            return jsonify({"error": f"Non-JSON response: {resp.text[:500]}"}), 502

        # Sync flow → questions returned immediately
        if use_sync and isinstance(payload, dict) and "questions" in payload:
            q = payload["questions"] or []
            job_id = f"sync_{int(time.time())}"
            job_data[job_id] = {"status": "completed", "questions": q, "timestamp": datetime.now().isoformat()}
            update_stats(len(q), processing_time_sec=5, questions_data=q)
            return jsonify({"success": True, "job_id": job_id, "questions": q})

        # Async flow → expect job_id
        job_id = payload.get("job_id")
        if not job_id:
            return jsonify({"error": f"Extractor did not return a job_id. Body: {payload}"}), 502

        # Track in memory; UI will poll
        job_data[job_id] = {"status": "processing", "timestamp": datetime.now().isoformat()}
        return jsonify({"success": True, "job_id": job_id, "async": True})

    finally:
        # Cleanup temp file
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            _log(f"UPLOAD: failed to remove temp file {path}: {e}")

@app.route("/api/poll/<job_id>")
def api_poll(job_id: str):
    """Poll job status from extractor; if completed, return questions."""
    if job_id not in job_data:
        return jsonify({"error": "Job not found"}), 404

    # Already completed in memory
    if job_data[job_id].get("status") == "completed":
        return jsonify(job_data[job_id])

    headers = _auth_header()

    # Poll /jobs/{job_id}
    url = _ep(f"jobs/{job_id}")
    resp = _requests_get(url, headers=headers)
    if resp is None:
        return jsonify({"error": "Network error polling job"}), 502

    _log(f"POLL: status={resp.status_code}, body_snip={resp.text[:500]!r}")

    if resp.status_code == 404:
        # Some backends remove job status; try /jobs/{id}/questions
        qurl = _ep(f"jobs/{job_id}/questions")
        qresp = _requests_get(qurl, headers=headers)
        if qresp and qresp.status_code == 200:
            try:
                questions = qresp.json()
                job_data[job_id] = {"status": "completed", "questions": questions}
                update_stats(len(questions), processing_time_sec=30, questions_data=questions)
                return jsonify({"status": "completed", "questions": questions})
            except Exception:
                pass
        return jsonify({"status": "completed", "questions": []})

    if resp.status_code != 200:
        return jsonify({"error": f"HTTP {resp.status_code}: {resp.text}"}), 502

    try:
        state = resp.json()
    except Exception:
        return jsonify({"error": f"Non-JSON response: {resp.text[:500]}"}), 502

    # Merge state
    job_data[job_id].update(state)

    status = str(state.get("status", "")).lower()
    if status == "completed":
        # Best case: questions embedded
        if "questions" in state and state["questions"] is not None:
            q = state["questions"]
            update_stats(len(q), processing_time_sec=30, questions_data=q)
            return jsonify(state)
        # Else fetch explicitly
        qurl = _ep(f"jobs/{job_id}/questions")
        qresp = _requests_get(qurl, headers=headers)
        if qresp and qresp.status_code == 200:
            try:
                questions = qresp.json()
                state["questions"] = questions
                job_data[job_id]["questions"] = questions
                update_stats(len(questions), processing_time_sec=30, questions_data=questions)
                return jsonify(state)
            except Exception:
                pass
        return jsonify({"status": "completed", "questions": []})

    if status == "failed":
        return jsonify(state)

    # Time-out guard for very long jobs (e.g., 10 minutes)
    started = job_data[job_id].get("timestamp")
    if started:
        try:
            dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
            if datetime.now() - dt > timedelta(minutes=10):
                job_data[job_id]["status"] = "timeout"
                return jsonify({"status": "timeout", "error": "Processing timeout"})
        except Exception:
            pass

    return jsonify({"status": "processing"})

@app.route("/api/job/<job_id>/questions")
def api_job_questions(job_id: str):
    """Return questions from memory or fetch from extractor."""
    if job_id not in job_data:
        return jsonify({"error": "Job not found"}), 404

    job = job_data[job_id]
    if job.get("status") == "completed" and "questions" in job:
        return jsonify(job["questions"])

    headers = _auth_header()
    qurl = _ep(f"jobs/{job_id}/questions")
    qresp = _requests_get(qurl, headers=headers)
    if qresp and qresp.status_code == 200:
        try:
            questions = qresp.json()
            job["questions"] = questions
            return jsonify(questions)
        except Exception:
            pass
    return jsonify({"error": "Failed to fetch questions"}), 502

@app.route("/review/<job_id>")
def review(job_id: str):
    """Render the Review & Edit page."""
    if job_id not in job_data:
        return redirect(url_for("index"))

    job = job_data[job_id]
    questions = job.get("questions") or []

    # If we have no questions yet, try to fetch
    if not questions:
        headers = _auth_header()
        qurl = _ep(f"jobs/{job_id}/questions")
        qresp = _requests_get(qurl, headers=headers)
        if qresp and qresp.status_code == 200:
            try:
                questions = qresp.json()
                job["questions"] = questions
            except Exception:
                pass

    # Overlay user edits if present
    if job_id in job_edits and job_edits[job_id]:
        # Merge by qid if available
        ed = {q.get("qid"): q for q in job_edits[job_id] if isinstance(q, dict)}
        merged = []
        for q in (questions or []):
            qid = q.get("qid")
            merged.append(ed.get(qid, q))
        questions = merged

    _log(f"REVIEW: job_id={job_id}, questions={len(questions)}")
    return render_template("review.html", job_id=job_id, questions=questions, job=job)

@app.route("/api/save_edits/<job_id>", methods=["POST"])
def api_save_edits(job_id: str):
    """Save edit payload (list of question dicts)."""
    try:
        data = request.get_json(force=True, silent=False) or {}
        edits = data.get("questions", [])
        if not isinstance(edits, list):
            return jsonify({"error": "Invalid payload: questions must be a list."}), 400

        job_edits.setdefault(job_id, [])
        # Upsert by qid if present
        index_by_qid = {q.get("qid"): i for i, q in enumerate(job_edits[job_id]) if isinstance(q, dict)}
        for q in edits:
            if not isinstance(q, dict):
                continue
            qid = q.get("qid")
            if qid in index_by_qid:
                job_edits[job_id][index_by_qid[qid]] = q
            else:
                job_edits[job_id].append(q)

        _log(f"SAVE_EDITS: job_id={job_id}, saved={len(edits)}")
        return jsonify({"success": True})
    except Exception as e:
        _log(f"SAVE_EDITS ERROR: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/export/<job_id>/<fmt>")
def api_export(job_id: str, fmt: str):
    """Export merged (original + edits) questions as DOCX or XLSX (fallback CSV)."""
    if job_id not in job_data:
        return jsonify({"error": "Job not found"}), 404

    original = job_data[job_id].get("questions") or []
    edits = job_edits.get(job_id, [])
    by_qid = {q.get("qid"): q for q in edits if isinstance(q, dict)}

    merged = []
    for q in original:
        qid = q.get("qid")
        merged.append(by_qid.get(qid, q))

    _log(f"EXPORT: job_id={job_id}, fmt={fmt}, rows={len(merged)}")

    # Optional filtering (approved/high_confidence)
    approved_only = request.args.get("approved", "false").lower() == "true"
    high_conf = request.args.get("high_confidence", "false").lower() == "true"

    def pass_filters(q: Dict[str, Any]) -> bool:
        if approved_only and str(q.get("status", "")).lower() != "approved":
            return False
        if high_conf:
            try:
                return float(q.get("confidence", 0.0)) >= 0.8
            except Exception:
                return False
        return True

    filtered = [q for q in merged if pass_filters(q)]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if fmt.lower() == "docx":
        try:
            from docx import Document

            doc = Document()
            doc.add_heading("RFP Questions Report", 0)
            doc.add_paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            doc.add_paragraph(f"Job: {job_id}")
            doc.add_paragraph(f"Total Questions: {len(filtered)}")
            doc.add_paragraph("")

            for i, q in enumerate(filtered, start=1):
                doc.add_heading(f"Question {i}", level=2)
                doc.add_paragraph(f"Text: {q.get('text') or q.get('question') or ''}")
                doc.add_paragraph(f"Confidence: {q.get('confidence', '—')}")
                doc.add_paragraph(f"Type: {q.get('type', '—')}")
                section = None
                sec_path = q.get("section_path")
                if isinstance(sec_path, list) and sec_path:
                    section = sec_path[0]
                elif isinstance(sec_path, str):
                    section = sec_path
                doc.add_paragraph(f"Section: {section or '—'}")
                doc.add_paragraph("")

            bio = io.BytesIO()
            doc.save(bio)
            bio.seek(0)
            return send_file(
                bio,
                as_attachment=True,
                download_name=f"RFP_Questions_{ts}.docx",
                mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        except Exception as e:
            _log(f"DOCX export failed, fallback to CSV: {e}")
            fmt = "csv"  # continue to CSV fallback

    if fmt.lower() == "xlsx":
        try:
            import pandas as pd
            from pandas import DataFrame

            df: DataFrame = pd.json_normalize(filtered)
            bio = io.BytesIO()
            with pd.ExcelWriter(bio, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Questions")
            bio.seek(0)
            return send_file(
                bio,
                as_attachment=True,
                download_name=f"RFP_Questions_{ts}.xlsx",
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as e:
            _log(f"XLSX export failed, fallback to CSV: {e}")
            fmt = "csv"  # continue to CSV fallback

    # CSV fallback
    try:
        import csv
        sio = io.StringIO()
        if filtered:
            # Flatten a useful subset
            keys = ["qid", "text", "question", "confidence", "type", "status", "section_path", "numbering", "category"]
            w = csv.DictWriter(sio, fieldnames=keys)
            w.writeheader()
            for q in filtered:
                row = {k: q.get(k) for k in keys}
                w.writerow(row)
        return send_file(
            io.BytesIO(sio.getvalue().encode("utf-8")),
            as_attachment=True,
            download_name=f"RFP_Questions_{ts}.csv",
            mimetype="text/csv",
        )
    except Exception as e:
        _log(f"CSV export failed: {e}")
        return jsonify({"error": "Export failed"}), 500

# ========= Entrypoint =========
if __name__ == "__main__":
    _log("Starting KnockKnock Intelligence (Flask)")
    _log(f"Config → BASE={API_BASE_URL}, HEADER={API_KEY_HEADER_NAME}, SCHEME={API_AUTH_SCHEME}, "
         f"KEY_PRESENT={'yes' if bool(API_KEY) else 'no'}, POLL={POLL_INTERVAL_SECONDS}s, "
         f"TIMEOUT={REQUEST_TIMEOUT_SECONDS}s")
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5002")), debug=True)
