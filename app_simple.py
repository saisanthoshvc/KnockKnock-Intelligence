# app_simple.py
# --------------------------------------------------------------------------------------
# ENV VARS you must set in Posit Connect (Settings -> Environment):
#   RFP_API_URL=https://connect.affiniusaiplatform.com/content/000d3572-937a-4d5c-ab7e-7ec80d80c4ce
#   RFP_API_KEY_HEADER_NAME=knocknock-authentication
#   RFP_API_AUTH_SCHEME=Raw
#   RFP_API_KEY=<YOUR extractor API key value for "knocknock-authentication">
#
# Optional (have sensible defaults):
#   POLL_INTERVAL_SECONDS=2
#   POLL_TIMEOUT_SECONDS=240
#   REQUEST_TIMEOUT_SECONDS=60
#   FLASK_SECRET_KEY=<random string>   # if you want to override default
#
# IMPORTANT:
# - Do NOT use "Authorization: Bearer ..." for this extractor.
# - The extractor expects the header EXACTLY as:
#       knocknock-authentication: <token>
#   (no "Bearer", no "Key" prefix).
# --------------------------------------------------------------------------------------

import io
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any

import requests
from flask import (
    Flask, render_template, request, jsonify, redirect,
    url_for, send_file
)
from werkzeug.utils import secure_filename

# --------------------------------------------------------------------------------------
# Flask app & config
# --------------------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-me-in-prod")

UPLOAD_FOLDER = os.path.abspath(os.getenv("UPLOAD_FOLDER", "uploads"))
ALLOWED_EXTENSIONS = {"docx", "pdf", "xlsx"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Effective API config (driven by env on Posit Connect)
api_config: Dict[str, Any] = {
    "base_url": os.getenv(
        "RFP_API_URL",
        "https://connect.affiniusaiplatform.com/content/000d3572-937a-4d5c-ab7e-7ec80d80c4ce",
    ).rstrip("/"),
    "api_key": os.getenv("RFP_API_KEY", ""),  # set in Connect Environment
    "header_name": os.getenv("RFP_API_KEY_HEADER_NAME", "knocknock-authentication"),
    "auth_scheme": os.getenv("RFP_API_AUTH_SCHEME", "Raw"),  # Raw | Bearer | Key
    "poll_interval": float(os.getenv("POLL_INTERVAL_SECONDS", "2")),
    "poll_timeout": float(os.getenv("POLL_TIMEOUT_SECONDS", "240")),
    "request_timeout": float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60")),
}

# In-memory state
job_data: Dict[str, Dict[str, Any]] = {}
job_edits: Dict[str, Any] = {}

# Stats (persist to file in app dir)
STATS_FILE = os.path.abspath("stats.json")


# --------------------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_stats() -> Dict[str, Any]:
    try:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE, "r") as f:
                stats = json.load(f)
            dp = stats.get("documents_processed", 0)
            total = float(stats.get("total_processing_time", 0))
            stats["avg_processing_time"] = round(total / dp, 1) if dp else 0.0
            return stats
    except Exception as e:
        print(f"[STATS] read error: {e}")

    return {
        "documents_processed": 0,
        "questions_extracted": 0,
        "total_processing_time": 0.0,
        "avg_processing_time": 0.0,
        "accuracy_rate": 0.0,
        "last_updated": datetime.now().isoformat(),
    }


def save_stats(stats: Dict[str, Any]) -> None:
    try:
        stats["last_updated"] = datetime.now().isoformat()
        with open(STATS_FILE, "w") as f:
            json.dump(stats, f, indent=2)
    except Exception as e:
        print(f"[STATS] write error: {e}")


def update_stats(questions_count: int, processing_time: float, questions_data=None):
    stats = load_stats()
    stats["documents_processed"] += 1
    stats["questions_extracted"] += int(questions_count)
    stats["total_processing_time"] += float(processing_time)

    # derive accuracy from confidence if available
    if questions_data:
        total_conf, n = 0.0, 0
        for q in questions_data:
            try:
                c = q.get("confidence", None)
                if c is not None:
                    total_conf += float(c)
                    n += 1
            except Exception:
                pass
        if n > 0:
            stats["accuracy_rate"] = round((total_conf / n) * 100, 1)
        elif stats["accuracy_rate"] == 0.0:
            stats["accuracy_rate"] = 85.0
    elif stats["accuracy_rate"] == 0.0:
        stats["accuracy_rate"] = 85.0

    save_stats(stats)
    return stats


def get_auth_header() -> Dict[str, str]:
    """
    Build the correct header for the extractor:
      - If header_name == "Authorization": honor scheme (Bearer/Key/Raw)
      - Else: send the header as-is with the token (RAW)
    """
    name = api_config["header_name"].strip()
    token = (api_config["api_key"] or "").strip()
    scheme = (api_config["auth_scheme"] or "Raw").strip().lower()

    if not name or not token:
        print("[AUTH] Missing header name or token")
        return {}

    if name.lower() == "authorization":
        if scheme == "bearer":
            return {"Authorization": f"Bearer {token}"}
        elif scheme == "key":
            return {"Authorization": f"Key {token}"}
        return {"Authorization": token}

    # RAW, non-Authorization header (what your extractor expects)
    return {name: token}


def _ep(path: str) -> str:
    return f"{api_config['base_url']}/{path.lstrip('/')}"


def make_api_request(method: str, endpoint: str, **kwargs):
    """
    Centralized API call. Important details:
      - Use exact paths with trailing slash where required by FastAPI app.
      - Let 'requests' set Content-Type for multipart (do NOT override).
    """
    url = _ep(endpoint)
    headers = get_auth_header()
    # Allow caller to add extra headers if needed (but don't force Content-Type)
    user_headers = kwargs.pop("headers", {})
    headers.update(user_headers or {})

    # Debug (redact token)
    redacted = {k: ("••••••••" if k.lower() == api_config["header_name"].lower() else v) for k, v in headers.items()}
    print(f"[HTTP] {method.upper()} {url}")
    print(f"[HTTP] headers: {redacted}")

    try:
        if method.upper() == "GET":
            return requests.get(url, headers=headers, timeout=api_config["request_timeout"])
        if method.upper() == "POST":
            return requests.post(url, headers=headers, timeout=api_config["request_timeout"], **kwargs)
        if method.upper() == "DELETE":
            return requests.delete(url, headers=headers, timeout=api_config["request_timeout"])
    except requests.exceptions.RequestException as e:
        print(f"[HTTP] Request error: {e}")
        return None


# --------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------
@app.route("/")
def index():
    stats = load_stats()
    return render_template("index.html", api_config=api_config, stats=stats)


@app.route("/api/config")
def api_config_echo():
    """Debug: confirm effective config on the server (token redacted)."""
    cfg = dict(api_config)
    if cfg.get("api_key"):
        cfg["api_key"] = "••••••••"
    return jsonify(cfg)


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """
    1) Save file locally (temp).
    2) POST to /extract/ or /extract/sync with:
         - multipart 'file'
         - form fields: use_llm, mode
    3) Return job_id for async OR questions for sync.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if not f or f.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(f.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(f.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    f.save(filepath)

    # Client options
    use_llm = str(request.form.get("use_llm", "false")).lower() in ("true", "1", "yes", "y")
    use_sync = str(request.form.get("use_sync", "false")).lower() in ("true", "1", "yes", "y")
    mode = request.form.get("mode", "balanced").strip().lower()
    if mode not in {"balanced", "fast", "thorough"}:
        mode = "balanced"

    print(f"[UPLOAD] name={filename}, size={os.path.getsize(filepath)} bytes, use_llm={use_llm}, use_sync={use_sync}, mode={mode}")

    # Determine mime type from extension
    ext = filename.rsplit(".", 1)[-1].lower()
    mime = "application/octet-stream"
    if ext == "docx":
        mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif ext == "pdf":
        mime = "application/pdf"
    elif ext == "xlsx":
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    files = {"file": (filename, open(filepath, "rb"), mime)}
    data = {
        # Send booleans as strings for form-data to avoid Pydantic "pattern" issues on some setups
        "use_llm": "true" if use_llm else "false",
        "mode": mode,
    }

    try:
        endpoint = "extract/sync" if use_sync else "extract/"
        # IMPORTANT: exact trailing slash like the docs
        if use_sync:
            # docs show "/extract/sync" (no trailing slash here), FastAPI handles it
            endpoint = "extract/sync"
        else:
            endpoint = "extract/"

        print(f"[UPLOAD] POST -> {endpoint} data={data} (multipart)")

        resp = make_api_request("POST", endpoint, files=files, data=data)
        # Close file handle ASAP
        files["file"][1].close()

        if not resp:
            return jsonify({"error": "No response from extractor"}), 502

        print(f"[UPLOAD] HTTP {resp.status_code}")
        txt_preview = (resp.text or "")[:600]
        print(f"[UPLOAD] body: {txt_preview}")

        if resp.status_code != 200:
            return jsonify({"error": f"HTTP {resp.status_code}: {resp.text}"}), 500

        payload = resp.json() if resp.headers.get("content-type", "").lower().startswith("application/json") else {}
        print(f"[UPLOAD] JSON keys: {list(payload.keys())}")

        # SYNC path returns questions directly
        if use_sync and isinstance(payload, dict) and "questions" in payload:
            questions = payload.get("questions") or []
            job_id = f"sync_{int(time.time())}"
            job_data[job_id] = {
                "status": "completed",
                "questions": questions,
                "timestamp": datetime.now().isoformat(),
            }
            # Stats: assume quick sync
            update_stats(len(questions), 5.0, questions)
            return jsonify({"success": True, "job_id": job_id, "questions": questions})

        # ASYNC path returns a job_id
        job_id = payload.get("job_id")
        if not job_id:
            return jsonify({"error": "No job_id returned by extractor"}), 502

        job_data[job_id] = {"status": "processing", "timestamp": datetime.now().isoformat()}
        return jsonify({"success": True, "job_id": job_id, "async": True})

    except Exception as e:
        print(f"[UPLOAD] exception: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        # local cleanup
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception:
            pass


@app.route("/api/poll/<job_id>")
def poll_job(job_id):
    if job_id not in job_data:
        return jsonify({"error": "Job not found"}), 404

    # If we already have completion
    if job_data[job_id].get("status") == "completed":
        return jsonify(job_data[job_id])

    resp = make_api_request("GET", f"jobs/{job_id}")
    if not resp:
        return jsonify({"error": "No response while polling"}), 502

    print(f"[POLL] HTTP {resp.status_code}")
    if resp.status_code == 404:
        # Sometimes jobs are GC'd—try direct questions endpoint
        q = make_api_request("GET", f"jobs/{job_id}/questions")
        if q and q.status_code == 200:
            questions = q.json()
            job_data[job_id] = {"status": "completed", "questions": questions}
            update_stats(len(questions), 30.0, questions)
            return jsonify(job_data[job_id])
        return jsonify({"status": "completed", "questions": []})

    if resp.status_code != 200:
        return jsonify({"error": f"HTTP {resp.status_code}: {resp.text}"}), 500

    data = resp.json()
    job_data[job_id].update(data)
    status = (data.get("status") or "").lower()

    print(f"[POLL] status={status}")

    if status == "completed":
        if "questions" in data and data["questions"] is not None:
            qs = data["questions"]
            update_stats(len(qs), 30.0, qs)
            return jsonify(data)
        # Fallback: fetch questions endpoint
        q = make_api_request("GET", f"jobs/{job_id}/questions")
        if q and q.status_code == 200:
            questions = q.json()
            job_data[job_id]["questions"] = questions
            update_stats(len(questions), 30.0, questions)
            return jsonify({"status": "completed", "questions": questions})
        return jsonify({"status": "completed", "questions": []})

    if status == "failed":
        return jsonify(data)

    # Handle long-running timeouts (10 minutes guard)
    try:
        started = job_data[job_id].get("timestamp")
        if started:
            dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
            if datetime.now() - dt > timedelta(minutes=10):
                job_data[job_id]["status"] = "timeout"
                return jsonify({"status": "timeout", "error": "Processing timeout"})
    except Exception:
        pass

    return jsonify({"status": "processing"})


@app.route("/api/job/<job_id>/questions")
def get_job_questions(job_id):
    if job_id not in job_data:
        return jsonify({"error": "Job not found"}), 404

    if job_data[job_id].get("status") == "completed" and "questions" in job_data[job_id]:
        return jsonify(job_data[job_id]["questions"])

    q = make_api_request("GET", f"jobs/{job_id}/questions")
    if q and q.status_code == 200:
        questions = q.json()
        job_data[job_id]["questions"] = questions
        return jsonify(questions)

    return jsonify({"error": "Failed to fetch questions"}), 500


@app.route("/review/<job_id>")
def review(job_id):
    if job_id not in job_data:
        return redirect(url_for("index"))

    job = job_data[job_id]
    questions = job.get("questions", [])

    # If missing, try fetch
    if not questions:
        q = make_api_request("GET", f"jobs/{job_id}/questions")
        if q and q.status_code == 200:
            questions = q.json()
            job["questions"] = questions

    # Apply edits overlay if present
    if job_id in job_edits and job_edits[job_id]:
        # build dict by qid for override
        ed = {q.get("qid"): q for q in job_edits[job_id]}
        merged = []
        for orig in questions:
            qid = orig.get("qid")
            merged.append(ed.get(qid, orig))
        questions = merged

    return render_template("review.html", job_id=job_id, questions=questions, job=job)


@app.route("/api/save_edits/<job_id>", methods=["POST"])
def save_edits(job_id):
    payload = request.get_json(silent=True) or {}
    items = payload.get("questions", [])
    if not items:
        return jsonify({"success": True})  # nothing to save, but not an error

    if job_id not in job_edits:
        job_edits[job_id] = []

    # index current edits by qid
    idx = {q.get("qid"): i for i, q in enumerate(job_edits[job_id])}
    for q in items:
        qid = q.get("qid")
        if qid in idx:
            job_edits[job_id][idx[qid]] = q
        else:
            job_edits[job_id].append(q)

    return jsonify({"success": True})


@app.route("/api/export/<job_id>/<fmt>")
def export_data(job_id, fmt):
    if job_id not in job_data:
        return jsonify({"error": "Job not found"}), 404

    original = job_data[job_id].get("questions", [])
    edits = job_edits.get(job_id, [])
    ed = {q.get("qid"): q for q in edits}
    questions = [ed.get(q.get("qid"), q) for q in original]

    # Optional filters via query params
    approved_only = str(request.args.get("approved", "false")).lower() in ("true", "1", "yes", "y")
    high_conf_only = str(request.args.get("high_confidence", "false")).lower() in ("true", "1", "yes", "y")

    if approved_only:
        questions = [q for q in questions if (q.get("status") or "").lower() == "approved"]
    if high_conf_only:
        def _ok(v):
            try:
                return float(v) >= 0.8
            except Exception:
                return False
        questions = [q for q in questions if _ok(q.get("confidence"))]

    # DOCX
    if fmt.lower() == "docx":
        try:
            from docx import Document
            doc = Document()
            doc.add_heading("RFP Questions Report", 0)
            doc.add_paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            doc.add_paragraph(f"Job ID: {job_id}")
            doc.add_paragraph(f"Total Questions: {len(questions)}")
            doc.add_paragraph("")

            for i, q in enumerate(questions, 1):
                doc.add_heading(f"Question {i}", level=2)
                doc.add_paragraph(f"Text: {q.get('text', 'N/A')}")
                doc.add_paragraph(f"Confidence: {q.get('confidence', 'N/A')}")
                doc.add_paragraph(f"Type: {q.get('type', 'N/A')}")
                section = (q.get("section_path") or ["N/A"])
                doc.add_paragraph(f"Section: {section[0] if section else 'N/A'}")
                doc.add_paragraph("")

            bio = io.BytesIO()
            doc.save(bio)
            bio.seek(0)
            fname = f"RFP_Questions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
            return send_file(
                bio,
                as_attachment=True,
                download_name=fname,
                mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        except Exception as e:
            return jsonify({"error": f"DOCX export failed: {e}"}), 500

    # XLSX
    if fmt.lower() == "xlsx":
        try:
            import pandas as pd
            df = pd.DataFrame(questions)
            bio = io.BytesIO()
            with pd.ExcelWriter(bio, engine="openpyxl") as w:
                df.to_excel(w, index=False, sheet_name="RFP Questions")
            bio.seek(0)
            fname = f"RFP_Questions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            return send_file(
                bio,
                as_attachment=True,
                download_name=fname,
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as e:
            return jsonify({"error": f"XLSX export failed: {e}"}), 500

    return jsonify({"error": "Invalid format"}), 400


# Simple health route (handy for Connect)
@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, "time": datetime.now().isoformat()})


# --------------------------------------------------------------------------------------
# WSGI entrypoint
# --------------------------------------------------------------------------------------
# In Posit Connect, set Entrypoint to:  app_simple:app
if __name__ == "__main__":
    print("[BOOT] Effective API config:", {**api_config, "api_key": "••••••••" if api_config.get("api_key") else ""})
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5002")), debug=True)
