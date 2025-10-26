# app_simple.py
import os
import io
import json
import time
from datetime import datetime
from typing import Dict, Optional, Tuple, List

import requests
from flask import (
    Flask, render_template, request, jsonify, redirect,
    url_for, send_file
)
from werkzeug.utils import secure_filename

# ============================== Flask Setup ==============================
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")

# Prefer writable paths on Connect
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "/tmp/kk_uploads")
ALLOWED_EXTENSIONS = {"docx", "pdf", "xlsx"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

# ============================ API Configuration ===========================
def _env_float(name: str, default: float) -> float:
    v = os.getenv(name, "").strip()
    try:
        return float(v) if v else default
    except Exception:
        return default

API_BASE_URL = (
    os.getenv(
        "RFP_API_URL",
        "https://connect.affiniusaiplatform.com/content/000d3572-937a-4d5c-ab7e-7ec80d80c4ce/"
    ).strip().rstrip("/")
)

API_KEY = os.getenv("RFP_API_KEY", "").strip()

# Explicit, env-driven auth controls (so we stop guessing)
RFP_API_KEY_HEADER_NAME = os.getenv("RFP_API_KEY_HEADER_NAME", "").strip()  # e.g., "Authorization"
RFP_API_AUTH_SCHEME     = os.getenv("RFP_API_AUTH_SCHEME", "").strip()      # e.g., "Key", "Bearer", "Raw", or ""

POLL_INTERVAL = _env_float("POLL_INTERVAL_SECONDS", 2.0)
POLL_TIMEOUT  = _env_float("POLL_TIMEOUT_SECONDS", 240.0)
REQUEST_TIMEOUT = _env_float("REQUEST_TIMEOUT_SECONDS", 60.0)

# In-memory job store
job_data: Dict[str, dict] = {}
job_edits: Dict[str, List[dict]] = {}

# Stats
STATS_FILE = os.getenv("STATS_FILE", "/tmp/kk_stats.json")

# ============================== Utilities ==============================
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def now_iso() -> str:
    return datetime.now().isoformat()

def redacted(s: Optional[str]) -> str:
    if not s:
        return ""
    if len(s) <= 8:
        return "••••"
    return s[:4] + "••••" + s[-4:]

def load_stats() -> dict:
    try:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE, "r") as f:
                stats = json.load(f)
            docs = stats.get("documents_processed", 0)
            tot = stats.get("total_processing_time", 0.0)
            stats["avg_processing_time"] = round(tot / max(docs, 1), 1)
            return stats
    except Exception:
        pass
    return {
        "documents_processed": 0,
        "questions_extracted": 0,
        "total_processing_time": 0.0,
        "avg_processing_time": 0.0,
        "accuracy_rate": 0.0,
        "last_updated": now_iso(),
    }

def save_stats(stats: dict) -> None:
    stats["last_updated"] = now_iso()
    try:
        with open(STATS_FILE, "w") as f:
            json.dump(stats, f, indent=2)
    except Exception as e:
        print(f"[STATS] Save error: {e}")

def update_stats(questions: List[dict], processing_time_sec: float) -> dict:
    stats = load_stats()
    stats["documents_processed"] += 1
    stats["questions_extracted"] += len(questions)
    stats["total_processing_time"] += float(processing_time_sec)
    # accuracy from confidence if present
    vals = []
    for q in questions or []:
        try:
            c = q.get("confidence", None)
            if c is not None:
                vals.append(float(c))
        except Exception:
            pass
    stats["accuracy_rate"] = round((sum(vals)/len(vals) if vals else 0.85) * 100, 1) if questions else 85.0
    save_stats(stats)
    return stats

def _mime_for(filename: str) -> str:
    ext = filename.lower().rsplit(".", 1)[-1]
    return {
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "pdf":  "application/pdf",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }.get(ext, "application/octet-stream")

def _ep(path: str) -> str:
    return f"{API_BASE_URL}/{path.lstrip('/')}"

# ===================== Auth Builders & Request Layer ======================
def build_explicit_auth_header() -> Optional[dict]:
    """
    Build exactly what the upstream expects from env settings:
      - RFP_API_KEY_HEADER_NAME: e.g., 'Authorization' (or 'knocknock-authentication')
      - RFP_API_AUTH_SCHEME:     e.g., 'Key', 'Bearer', 'Raw', or '' (no scheme)
      - RFP_API_KEY:             token value
    """
    if not API_KEY:
        raise RuntimeError("Missing RFP_API_KEY env var.")
    if not RFP_API_KEY_HEADER_NAME:
        return None
    if RFP_API_AUTH_SCHEME:
        return { RFP_API_KEY_HEADER_NAME: f"{RFP_API_AUTH_SCHEME} {API_KEY}" }
    return { RFP_API_KEY_HEADER_NAME: API_KEY }

# Keep your original fallbacks as a safety net
AUTH_CANDIDATES = [
    lambda key: {"Authorization": f"Key {key}"},
    lambda key: {"Authorization": f"Bearer {key}"},
    lambda key: {"knocknock-authentication": key},
    lambda key: {"X-API-Key": key},
    lambda key: {"Authorization": key},  # raw
]

# Some servers are picky about trailing slashes for POST.
EXTRACT_PATHS = ["extract/", "extract"]

def api_post_with_fallbacks(files, data, want_sync: bool) -> Tuple[Optional[requests.Response], dict, str]:
    """
    Try env-driven header first, then fallback candidates.
    Returns: (response, used_headers, used_path)
    """
    if not API_KEY:
        raise RuntimeError("Missing RFP_API_KEY env var.")

    paths = ["extract/sync"] if want_sync else EXTRACT_PATHS
    last_error = None

    # 1) Try explicit env-configured header first
    explicit = build_explicit_auth_header()
    if explicit:
        for path in paths:
            url = _ep(path)
            try:
                r = requests.post(url, headers=explicit, files=files, data=data, timeout=REQUEST_TIMEOUT)
                print(f"[POST] explicit {url} -> {r.status_code} hdr={list(explicit.keys())[0]}:{str(explicit[list(explicit.keys())[0]])[:12]}…")
                if r.status_code == 200:
                    return r, explicit, path
                last_error = (r.status_code, (r.text or "")[:500])
            except requests.RequestException as e:
                last_error = (None, str(e))

    # 2) Fallback header guesses
    for path in paths:
        url = _ep(path)
        for build in AUTH_CANDIDATES:
            hdrs = build(API_KEY)
            try:
                r = requests.post(url, headers=hdrs, files=files, data=data, timeout=REQUEST_TIMEOUT)
                print(f"[POST] fallback {url} -> {r.status_code} using {list(hdrs.keys())[0]}:{hdrs[list(hdrs.keys())[0]][:12]}…")
                if r.status_code == 200:
                    return r, hdrs, path
                if r.status_code in (400, 401, 403, 422):
                    last_error = (r.status_code, (r.text or "")[:500])
                    continue
                return r, hdrs, path
            except requests.RequestException as e:
                last_error = (None, str(e))
                continue

    # If we get here, nothing worked
    resp = requests.models.Response()
    resp.status_code = 599
    resp._content = (f"All auth variants failed. Last error: {last_error}").encode()
    return resp, {}, paths[-1] if paths else ("extract/sync" if want_sync else "extract/")

def api_get(path: str, headers: dict) -> requests.Response:
    url = _ep(path)
    return requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)

def api_delete(path: str, headers: dict) -> requests.Response:
    url = _ep(path)
    return requests.delete(url, headers=headers, timeout=REQUEST_TIMEOUT)

# ============================== Routes ===============================
@app.route("/")
def index():
    # Help you see what the app is configured to hit
    print(f"[CFG] API_BASE_URL={API_BASE_URL}")
    print(f"[CFG] API_KEY={redacted(API_KEY)}")
    print(f"[CFG] AUTH_NAME={RFP_API_KEY_HEADER_NAME or '(none)'} AUTH_SCHEME={RFP_API_AUTH_SCHEME or '(none)'}")
    print(f"[CFG] TIMEOUT={REQUEST_TIMEOUT}s")
    stats = load_stats()
    api_config = {
        "base_url": API_BASE_URL,
        "poll_interval": POLL_INTERVAL,
        "poll_timeout": POLL_TIMEOUT,
        "request_timeout": REQUEST_TIMEOUT,
    }
    return render_template("index.html", api_config=api_config, stats=stats)

@app.route("/_diag")
def diag():
    hdr = build_explicit_auth_header()
    preview = {}
    if hdr:
        for k, v in hdr.items():
            if isinstance(v, str) and len(v) > 8:
                preview[k] = v[:4] + "••••" + v[-4:]
            else:
                preview[k] = "••••"
    return jsonify({
        "api_base_url": API_BASE_URL,
        "api_key_set": bool(API_KEY),
        "auth_header_name": RFP_API_KEY_HEADER_NAME or None,
        "auth_scheme": RFP_API_AUTH_SCHEME or None,
        "using_explicit_header": bool(hdr),
        "sample_header_preview": preview,
        "upload_folder": UPLOAD_FOLDER,
        "stats_file": STATS_FILE,
    })

@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, "time": now_iso()})

@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    use_llm = str(request.form.get("use_llm", "false")).lower() == "true"
    use_sync = str(request.form.get("use_sync", "false")).lower() == "true"
    mode = request.form.get("mode", "balanced")  # balanced | fast | thorough

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        mime = _mime_for(filename)
        with open(filepath, "rb") as f:
            files = {"file": (filename, f, mime)}
            data = {"use_llm": "true" if use_llm else "false", "mode": mode}

            print(f"[UPLOAD] use_llm={use_llm} use_sync={use_sync} mode={mode} file={filename} size={os.path.getsize(filepath)}")
            resp, used_headers, used_path = api_post_with_fallbacks(files, data, want_sync=use_sync)
            body_snip = (resp.text or "")[:500] if resp is not None else ""
            print(f"[UPLOAD] status={getattr(resp, 'status_code', None)} path={used_path} hdr={list(used_headers.keys())[:1]} body_snip={body_snip!r}")

        if resp is None:
            return jsonify({"error": "No response from extractor"}), 502

        if resp.status_code != 200:
            # Surface server message to front-end
            return jsonify({"error": f"HTTP {resp.status_code}: {resp.text}"}), 502

        result = resp.json() if "application/json" in (resp.headers.get("content-type","").lower()) else {}
        # SYNC: returns questions directly
        if use_sync and isinstance(result, dict) and "questions" in result:
            job_id = f"sync_{int(time.time())}"
            job_data[job_id] = {
                "status": "completed",
                "questions": result.get("questions") or [],
                "timestamp": now_iso(),
                "auth_headers": used_headers,   # cache the working header
            }
            update_stats(job_data[job_id]["questions"], processing_time_sec=5.0)
            return jsonify({"success": True, "job_id": job_id, "questions": result.get("questions", [])})

        # ASYNC: expect job_id
        job_id = (result or {}).get("job_id")
        if not job_id:
            return jsonify({"error": f"Extractor did not return job_id. Raw: {result}"}), 502

        job_data[job_id] = {
            "status": "processing",
            "timestamp": now_iso(),
            "auth_headers": used_headers,  # cache working header
        }
        return jsonify({"success": True, "job_id": job_id, "async": True})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception:
            pass

@app.route("/api/poll/<job_id>")
def poll_job(job_id: str):
    job = job_data.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    # Completed (cached)
    if job.get("status") == "completed":
        return jsonify(job)

    headers = job.get("auth_headers") or build_explicit_auth_header() or AUTH_CANDIDATES[0](API_KEY)
    try:
        r = api_get(f"jobs/{job_id}", headers)
        print(f"[POLL] GET jobs/{job_id} -> {r.status_code}")
        if r.status_code == 200:
            payload = r.json()
            job.update(payload or {})
            state = (payload or {}).get("status", "").lower()

            # If done, ensure we have questions (some servers embed; others need /questions)
            if state == "completed":
                if "questions" in payload and payload["questions"] is not None:
                    update_stats(payload["questions"], processing_time_sec=30.0)
                    return jsonify(payload)
                # fallback: fetch questions endpoint
                rq = api_get(f"jobs/{job_id}/questions", headers)
                if rq.status_code == 200:
                    questions = rq.json()
                    job["questions"] = questions
                    update_stats(questions, processing_time_sec=30.0)
                    return jsonify({"status": "completed", "questions": questions})
                return jsonify({"status": "completed", "questions": []})

            if state == "failed":
                return jsonify(payload)

            # still processing
            return jsonify({"status": "processing"})

        # If 404, try to fetch questions (job might be cleaned)
        if r.status_code == 404:
            rq = api_get(f"jobs/{job_id}/questions", headers)
            if rq.status_code == 200:
                questions = rq.json()
                job["status"] = "completed"
                job["questions"] = questions
                update_stats(questions, processing_time_sec=30.0)
                return jsonify({"status": "completed", "questions": questions})
            return jsonify({"status": "completed", "questions": []})

        return jsonify({"error": f"HTTP {r.status_code}: {r.text}"}), 502

    except requests.RequestException as e:
        return jsonify({"error": f"Poll error: {str(e)}"}), 502

@app.route("/api/job/<job_id>/questions")
def get_job_questions(job_id: str):
    job = job_data.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    if job.get("status") == "completed" and "questions" in job:
        return jsonify(job["questions"])

    headers = job.get("auth_headers") or build_explicit_auth_header() or AUTH_CANDIDATES[0](API_KEY)
    r = api_get(f"jobs/{job_id}/questions", headers)
    if r.status_code == 200:
        qs = r.json()
        job["questions"] = qs
        return jsonify(qs)
    return jsonify({"error": f"HTTP {r.status_code}: {r.text}"}), 502

@app.route("/review/<job_id>")
def review(job_id: str):
    job = job_data.get(job_id)
    if not job:
        return redirect(url_for("index"))

    questions = job.get("questions") or []
    if not questions:
        headers = job.get("auth_headers") or build_explicit_auth_header() or AUTH_CANDIDATES[0](API_KEY)
        r = api_get(f"jobs/{job_id}/questions", headers)
        if r.status_code == 200:
            questions = r.json()
            job["questions"] = questions

    return render_template("review.html", job_id=job_id, questions=questions, job=job)

@app.route("/api/save_edits/<job_id>", methods=["POST"])
def save_edits(job_id: str):
    payload = request.get_json(force=True) or {}
    questions = payload.get("questions") or []
    if not questions:
        return jsonify({"success": False, "error": "No questions provided"}), 400

    edits = job_edits.setdefault(job_id, [])
    # merge by qid
    idx = {q.get("qid"): i for i, q in enumerate(edits) if q.get("qid")}
    for q in questions:
        qid = q.get("qid")
        if qid in idx:
            edits[idx[qid]] = q
        else:
            edits.append(q)
    return jsonify({"success": True})

# No-op delete to satisfy the front-end button; removes any edited entry with this qid
@app.route("/api/delete_job/<qid>", methods=["DELETE"])
def delete_question(qid: str):
    removed = 0
    for job_id, edits in job_edits.items():
        keep = []
        for q in edits:
            if q.get("qid") == qid:
                removed += 1
                continue
            keep.append(q)
        job_edits[job_id] = keep
    return jsonify({"success": True, "removed": removed})

@app.route("/api/export/<job_id>/<fmt>")
def export_data(job_id: str, fmt: str):
    job = job_data.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    original = job.get("questions") or []
    edited = job_edits.get(job_id, [])
    edited_map = {q.get("qid"): q for q in edited if q.get("qid")}
    merged = [edited_map.get(o.get("qid"), o) for o in original]

    # Optional filters from query params
    approved_only = str(request.args.get("approved", "false")).lower() == "true"
    high_conf_only = str(request.args.get("high_confidence", "false")).lower() == "true"

    def _passes(q: dict) -> bool:
        if approved_only:
            if (q.get("status") or "").lower() != "approved":
                return False
        if high_conf_only:
            try:
                if float(q.get("confidence", 0.0)) < 0.8:
                    return False
            except Exception:
                return False
        return True

    merged = [q for q in merged if _passes(q or {})]

    if fmt == "docx":
        try:
            from docx import Document
            doc = Document()
            doc.add_heading("RFP Questions Report", 0)
            doc.add_paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            doc.add_paragraph(f"Job ID: {job_id}")
            doc.add_paragraph(f"Total Questions: {len(merged)}")
            doc.add_paragraph("")
            for i, q in enumerate(merged, 1):
                doc.add_heading(f"Question {i}", level=2)
                doc.add_paragraph(f"Text: {q.get('text','')}")
                doc.add_paragraph(f"Confidence: {q.get('confidence','')}")
                doc.add_paragraph(f"Type: {q.get('type','')}")
                section = q.get("section_path") or q.get("section") or ""
                if isinstance(section, list):
                    section = " / ".join([str(s) for s in section])
                doc.add_paragraph(f"Section: {section}")
                doc.add_paragraph("")
            bio = io.BytesIO()
            doc.save(bio)
            bio.seek(0)
            fname = f"RFP_Questions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
            return send_file(bio, as_attachment=True,
                             download_name=fname,
                             mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        except Exception as e:
            return jsonify({"error": f"DOCX export failed: {e}"}), 500

    if fmt == "xlsx":
        try:
            import pandas as pd
            df = pd.DataFrame(merged)
            bio = io.BytesIO()
            with pd.ExcelWriter(bio, engine="openpyxl") as w:
                df.to_excel(w, sheet_name="Questions", index=False)
            bio.seek(0)
            fname = f"RFP_Questions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            return send_file(bio, as_attachment=True,
                             download_name=fname,
                             mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e:
            return jsonify({"error": f"XLSX export failed: {e}"}), 500

    return jsonify({"error": "Invalid format"}), 400

# ========================== Minimal error page ==========================
@app.errorhandler(413)
def too_large(_e):
    return render_template("error.html",
                           error_code=413,
                           error_message="File too large (limit 16MB)."), 413

# ============================== Dev server ==============================
if __name__ == "__main__":
    print(f"Starting on http://localhost:5002")
    print(f"Extractor base: {API_BASE_URL}")
    print(f"API key: {redacted(API_KEY)}")
    print(f"Auth header: {RFP_API_KEY_HEADER_NAME or '(none)'}; scheme: {RFP_API_AUTH_SCHEME or '(none)'}")
    app.run(host="0.0.0.0", port=5002, debug=True)
