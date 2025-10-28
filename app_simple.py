# app_simple.py
import os
import io
import json
import time
import uuid
import logging
import traceback
from datetime import datetime, timedelta

import requests
from flask import (
    Flask, render_template, request, jsonify, redirect,
    url_for, send_file, g
)
from werkzeug.utils import secure_filename

# --------------------------
# Logging setup (Posit logs)
# --------------------------
class JsonFormatter(logging.Formatter):
    def format(self, record):
        base = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Include extra if present
        for k, v in getattr(record, "__dict__", {}).items():
            if k not in ("args", "msg", "levelname", "levelno", "name",
                         "pathname", "filename", "module", "exc_info",
                         "exc_text", "stack_info", "lineno", "funcName",
                         "created", "msecs", "relativeCreated", "thread",
                         "threadName", "processName", "process"):
                base[k] = v
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)
        return json.dumps(base)

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.environ.get("LOG_FORMAT", "text").lower()  # 'text' | 'json'
handler = logging.StreamHandler()
handler.setLevel(LOG_LEVEL)
if LOG_FORMAT == "json":
    handler.setFormatter(JsonFormatter())
else:
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    ))

root_logger = logging.getLogger()
root_logger.handlers = []  # clear default to avoid dupes
root_logger.addHandler(handler)
root_logger.setLevel(LOG_LEVEL)

log = logging.getLogger("knockknock.app")

# --------------------------
# Flask app
# --------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "your-secret-key-here")

APP_VERSION = os.environ.get("APP_VERSION", "1.0.0")

# ---------- Paths (use /tmp on Posit Connect) ----------
UPLOAD_FOLDER = os.environ.get("UPLOAD_DIR", "/tmp/knockknock_uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

STATS_FILE = os.environ.get("STATS_FILE", "/tmp/knockknock_stats.json")

# ---------- Allowed uploads ----------
ALLOWED_EXTENSIONS = {"docx", "pdf", "xlsx"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

# ---------- API Config (env-first) ----------
def env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default

DEFAULT_API_CONFIG = {
    "base_url": os.environ.get(
        "RFP_API_URL",
        "https://connect.affiniusaiplatform.com/content/000d3572-937a-4d5c-ab7e-7ec80d80c4ce/",
    ),
    "api_key": os.environ.get("RFP_API_KEY", ""),  # set in Connect, may be blank if public
    "header_name": os.environ.get("RFP_API_KEY_HEADER_NAME", "Authorization"),
    "auth_scheme": os.environ.get("RFP_API_AUTH_SCHEME", "Key"),  # Key | Bearer | Raw
    "poll_interval": env_int("POLL_INTERVAL_SECONDS", 2),
    "poll_timeout": env_int("POLL_TIMEOUT_SECONDS", 240),
    "request_timeout": env_int("REQUEST_TIMEOUT_SECONDS", 60),
}

api_config = DEFAULT_API_CONFIG.copy()

# ---------- In-memory state ----------
job_data = {}
job_edits = {}

# ---------- Helpers ----------
def mask_token(tok: str) -> str:
    if not tok:
        return ""
    if len(tok) <= 8:
        return "***"
    return f"{tok[:4]}...{tok[-4:]}"

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def get_auth_header():
    """
    Build auth header according to env:
      - Authorization: Key <token>     (Posit Connect programmatic)
      - Authorization: Bearer <token>  (if service expects bearer)
      - <custom-name>: <token>         (Raw custom header)
    """
    hdr = api_config["header_name"]
    scheme = (api_config["auth_scheme"] or "").lower()
    token = api_config["api_key"]

    # Logging (masked)
    log.debug("AUTH_CONFIG",
              extra={"event": "AUTH_CONFIG",
                     "header_name": hdr,
                     "auth_scheme": api_config["auth_scheme"],
                     "token_present": bool(token)})

    if not token:
        return {}  # allow unauthenticated if target is public

    if hdr.lower() == "authorization":
        if scheme == "key":
            return {"Authorization": f"Key {token}"}
        elif scheme == "bearer":
            return {"Authorization": f"Bearer {token}"}
        elif scheme == "raw":
            return {"Authorization": token}
        else:
            return {"Authorization": token}
    else:
        # custom header name
        if scheme == "raw":
            return {hdr: token}
        elif scheme in ("key", "bearer"):
            return {hdr: f"{scheme.title()} {token}"}
        else:
            return {hdr: token}

def make_api_request(method, endpoint, **kwargs):
    url = f"{api_config['base_url'].rstrip('/')}/{endpoint.lstrip('/')}"
    headers = get_auth_header()
    headers.update(kwargs.pop("headers", {}))

    # Safe log for headers (show names and masked values only)
    safe_headers = {}
    for k, v in headers.items():
        if isinstance(v, str):
            if k.lower() == "authorization":
                # keep scheme only
                parts = v.split(" ", 1)
                safe_headers[k] = parts[0] if len(parts) > 1 else "***"
            else:
                safe_headers[k] = mask_token(v)
        else:
            safe_headers[k] = "***"

    t0 = time.monotonic()
    try:
        log.info("API_REQUEST_START",
                 extra={"event": "API_REQUEST_START",
                        "method": method.upper(),
                        "url": url,
                        "headers": safe_headers})
        if method.upper() == "GET":
            resp = requests.get(url, headers=headers,
                                timeout=api_config["request_timeout"])
        elif method.upper() == "POST":
            resp = requests.post(url, headers=headers,
                                 timeout=api_config["request_timeout"], **kwargs)
        elif method.upper() == "DELETE":
            resp = requests.delete(url, headers=headers,
                                   timeout=api_config["request_timeout"])
        else:
            log.error("API_REQUEST_UNSUPPORTED_METHOD",
                      extra={"event": "API_REQUEST_UNSUPPORTED_METHOD",
                             "method": method})
            return None

        elapsed = round((time.monotonic() - t0) * 1000, 1)
        ct = resp.headers.get("Content-Type", "")
        # Detect accidental redirect to login page (auth failure)
        login_redirect = False
        if ("text/html" in ct) and (
            "__login__" in resp.text.lower() or "<title>sign in" in resp.text.lower()
        ):
            login_redirect = True

        log.info("API_REQUEST_DONE",
                 extra={"event": "API_REQUEST_DONE",
                        "method": method.upper(),
                        "url": url,
                        "status": resp.status_code,
                        "content_type": ct,
                        "login_redirect": login_redirect,
                        "elapsed_ms": elapsed})

        if login_redirect:
            class Dummy:
                status_code = 401
                text = "Not authorized: redirected to login page. Check RFP_API_KEY_* env vars and extractor auth."
                def json(self): return {"error": self.text}
            return Dummy()

        return resp

    except requests.exceptions.RequestException as e:
        elapsed = round((time.monotonic() - t0) * 1000, 1)
        log.error("API_REQUEST_ERROR",
                  extra={"event": "API_REQUEST_ERROR",
                         "url": url,
                         "error": str(e),
                         "elapsed_ms": elapsed})
        return None

# ---------- Stats ----------
def load_stats():
    try:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE, "r") as f:
                stats = json.load(f)
            docs = stats.get("documents_processed", 0)
            stats["avg_processing_time"] = round(
                stats.get("total_processing_time", 0) / docs, 1
            ) if docs > 0 else 0.0
            return stats
    except Exception as e:
        log.warning("STATS_LOAD_ERROR", extra={"event": "STATS_LOAD_ERROR", "error": str(e)})
    return {
        "documents_processed": 0,
        "questions_extracted": 0,
        "total_processing_time": 0,
        "avg_processing_time": 0.0,
        "accuracy_rate": 0.0,
        "last_updated": datetime.now().isoformat(),
    }

def save_stats(stats):
    try:
        stats["last_updated"] = datetime.now().isoformat()
        with open(STATS_FILE, "w") as f:
            json.dump(stats, f, indent=2)
        log.info("STATS_SAVED", extra={"event": "STATS_SAVED", "stats": stats})
    except Exception as e:
        log.error("STATS_SAVE_ERROR", extra={"event": "STATS_SAVE_ERROR", "error": str(e)})

def update_stats(questions_count, processing_time, questions_data=None):
    stats = load_stats()
    stats["documents_processed"] += 1
    stats["questions_extracted"] += questions_count
    stats["total_processing_time"] += processing_time

    if questions_count > 0 and questions_data:
        total_conf = 0.0
        n = 0
        for q in questions_data:
            c = q.get("confidence")
            if isinstance(c, (int, float)):
                total_conf += float(c)
                n += 1
        stats["accuracy_rate"] = round((total_conf / n) * 100, 1) if n else stats.get("accuracy_rate", 85.0)
    else:
        stats["accuracy_rate"] = max(stats.get("accuracy_rate", 0.0), 85.0)

    save_stats(stats)
    return stats

# ---------- Request lifecycle logging ----------
@app.before_request
def add_request_context():
    g.req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    g.t0 = time.monotonic()
    log.info("REQ_START",
             extra={"event": "REQ_START",
                    "req_id": g.req_id,
                    "method": request.method,
                    "path": request.path,
                    "remote_addr": request.remote_addr})

@app.after_request
def after(resp):
    elapsed = round((time.monotonic() - getattr(g, "t0", time.monotonic())) * 1000, 1)
    log.info("REQ_DONE",
             extra={"event": "REQ_DONE",
                    "req_id": getattr(g, "req_id", "-"),
                    "status": resp.status_code,
                    "elapsed_ms": elapsed})
    return resp

@app.teardown_request
def teardown(exc):
    if exc:
        log.error("REQ_EXCEPTION",
                  extra={"event": "REQ_EXCEPTION",
                         "req_id": getattr(g, "req_id", "-"),
                         "error": str(exc),
                         "trace": traceback.format_exc()})

# ---------- Startup connectivity check ----------
@app.before_first_request
def startup_check():
    log.info("APP_START",
             extra={"event": "APP_START",
                    "version": APP_VERSION,
                    "upload_dir": UPLOAD_FOLDER,
                    "stats_file": STATS_FILE})
    # Try health endpoints (non-fatal if unavailable)
    for ep in ("health/ready", "health/", "info"):
        resp = make_api_request("GET", ep)
        if resp and resp.status_code == 200:
            log.info("UPSTREAM_HEALTH_OK",
                     extra={"event": "UPSTREAM_HEALTH_OK", "endpoint": ep})
            return
        else:
            code = resp.status_code if resp else "NO_RESPONSE"
            log.warning("UPSTREAM_HEALTH_FAIL",
                        extra={"event": "UPSTREAM_HEALTH_FAIL",
                               "endpoint": ep, "status": code})

# ---------- Routes ----------
@app.route("/")
def index():
    stats = load_stats()
    log.debug("INDEX_VIEW", extra={"event": "INDEX_VIEW", "stats": stats})
    return render_template("index.html", api_config=api_config, stats=stats)

@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        log.warning("UPLOAD_NO_FILE", extra={"event": "UPLOAD_NO_FILE"})
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if f.filename == "":
        log.warning("UPLOAD_EMPTY_FILENAME", extra={"event": "UPLOAD_EMPTY_FILENAME"})
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(f.filename):
        log.warning("UPLOAD_INVALID_TYPE", extra={"event": "UPLOAD_INVALID_TYPE", "filename": f.filename})
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(f.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(filepath)
    size = os.path.getsize(filepath)
    log.info("UPLOAD_SAVED",
             extra={"event": "UPLOAD_SAVED",
                    "filename": filename, "path": filepath, "size_bytes": size})

    # Booleans (string for FastAPI robustness)
    use_llm_flag = request.form.get("use_llm", "false").lower() in ("true", "1", "yes", "on")
    use_sync_flag = request.form.get("use_sync", "false").lower() in ("true", "1", "yes", "on")
    mode = request.form.get("mode", "balanced")
    log.info("UPLOAD_PARAMS",
             extra={"event": "UPLOAD_PARAMS",
                    "use_llm": use_llm_flag, "use_sync": use_sync_flag, "mode": mode})

    try:
        with open(filepath, "rb") as doc:
            files = {"file": doc}
            data = {
                "use_llm": "true" if use_llm_flag else "false",
                "mode": mode,
            }
            endpoint = "extract/sync" if use_sync_flag else "extract/"
            log.info("EXTRACT_CALL",
                     extra={"event": "EXTRACT_CALL", "endpoint": endpoint})
            resp = make_api_request("POST", endpoint, files=files, data=data)

        if not resp:
            log.error("EXTRACT_NO_RESPONSE", extra={"event": "EXTRACT_NO_RESPONSE"})
            return jsonify({"error": "API request failed (no response)"}), 502

        if resp.status_code == 200:
            payload = resp.json()
            log.info("EXTRACT_OK",
                     extra={"event": "EXTRACT_OK", "sync": use_sync_flag,
                            "keys": list(payload.keys())})
            if use_sync_flag and isinstance(payload, dict) and "questions" in payload:
                job_id = f"sync_{int(time.time())}"
                job_data[job_id] = {
                    "status": "completed",
                    "questions": payload["questions"],
                    "timestamp": datetime.now().isoformat(),
                }
                update_stats(len(payload["questions"]), 5, payload["questions"])
                return jsonify({"success": True, "job_id": job_id, "questions": payload["questions"]})
            else:
                job_id = payload.get("job_id")
                if job_id:
                    job_data[job_id] = {"status": "processing", "timestamp": datetime.now().isoformat()}
                    log.info("JOB_STARTED", extra={"event": "JOB_STARTED", "job_id": job_id})
                    return jsonify({"success": True, "job_id": job_id, "async": True})
                else:
                    log.error("EXTRACT_NO_JOB_ID", extra={"event": "EXTRACT_NO_JOB_ID"})
                    return jsonify({"error": "No job ID returned from API"}), 502
        else:
            try:
                msg = resp.json()
            except Exception:
                msg = resp.text[:500]
            log.error("EXTRACT_ERROR",
                      extra={"event": "EXTRACT_ERROR",
                             "status": resp.status_code, "message": msg})
            return jsonify({"error": f"HTTP {resp.status_code}: {msg}"}), resp.status_code

    except Exception as e:
        log.exception("UPLOAD_HANDLER_EXCEPTION",
                      extra={"event": "UPLOAD_HANDLER_EXCEPTION",
                             "error": str(e)})
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                log.debug("UPLOAD_CLEANED",
                          extra={"event": "UPLOAD_CLEANED", "path": filepath})
        except Exception as e:
            log.warning("UPLOAD_CLEANUP_FAIL",
                        extra={"event": "UPLOAD_CLEANUP_FAIL", "error": str(e)})

@app.route("/api/poll/<job_id>")
def poll_job(job_id):
    if job_id not in job_data:
        log.warning("POLL_UNKNOWN_JOB", extra={"event": "POLL_UNKNOWN_JOB", "job_id": job_id})
        return jsonify({"error": "Job not found"}), 404

    if job_data[job_id].get("status") == "completed":
        log.debug("POLL_ALREADY_DONE", extra={"event": "POLL_ALREADY_DONE", "job_id": job_id})
        return jsonify(job_data[job_id])

    log.info("POLL_CALL", extra={"event": "POLL_CALL", "job_id": job_id})
    resp = make_api_request("GET", f"jobs/{job_id}")
    if not resp:
        log.error("POLL_NO_RESPONSE", extra={"event": "POLL_NO_RESPONSE", "job_id": job_id})
        return jsonify({"error": "Failed to poll job status"}), 502

    if resp.status_code == 404:
        log.info("POLL_404_TRY_QUESTIONS", extra={"event": "POLL_404_TRY_QUESTIONS", "job_id": job_id})
        q = make_api_request("GET", f"jobs/{job_id}/questions")
        if q and q.status_code == 200:
            questions = q.json()
            job_data[job_id]["status"] = "completed"
            job_data[job_id]["questions"] = questions
            update_stats(len(questions), 30, questions)
            log.info("POLL_RECOVERED_FROM_404",
                     extra={"event": "POLL_RECOVERED_FROM_404", "job_id": job_id, "count": len(questions)})
            return jsonify({"status": "completed", "questions": questions})
        log.warning("POLL_404_NO_QUESTIONS", extra={"event": "POLL_404_NO_QUESTIONS", "job_id": job_id})
        return jsonify({"status": "completed", "questions": []})

    if resp.status_code != 200:
        try:
            msg = resp.json()
        except Exception:
            msg = resp.text[:500]
        log.error("POLL_ERROR",
                  extra={"event": "POLL_ERROR", "job_id": job_id,
                         "status": resp.status_code, "message": msg})
        return jsonify({"error": f"HTTP {resp.status_code}: {msg}"}), resp.status_code

    data = resp.json()
    job_data[job_id].update(data)
    status = data.get("status", "processing")
    log.info("POLL_STATUS",
             extra={"event": "POLL_STATUS", "job_id": job_id, "status": status})

    if status == "completed":
        if data.get("questions") is not None:
            qs = data["questions"]
            update_stats(len(qs), 30, qs)
            log.info("POLL_COMPLETED_WITH_INLINE",
                     extra={"event": "POLL_COMPLETED_WITH_INLINE",
                            "job_id": job_id, "count": len(qs)})
            return jsonify(data)
        q = make_api_request("GET", f"jobs/{job_id}/questions")
        if q and q.status_code == 200:
            qs = q.json()
            job_data[job_id]["questions"] = qs
            update_stats(len(qs), 30, qs)
            data["questions"] = qs
            log.info("POLL_COMPLETED_FETCHED",
                     extra={"event": "POLL_COMPLETED_FETCHED",
                            "job_id": job_id, "count": len(qs)})
            return jsonify(data)
        log.warning("POLL_COMPLETED_NO_QUESTIONS",
                    extra={"event": "POLL_COMPLETED_NO_QUESTIONS", "job_id": job_id})
        return jsonify({"status": "completed", "questions": []})

    if status == "processing":
        start = job_data[job_id].get("timestamp")
        if start:
            try:
                started = datetime.fromisoformat(start)
                if datetime.now() - started > timedelta(minutes=10):
                    job_data[job_id]["status"] = "timeout"
                    log.warning("POLL_TIMEOUT",
                                extra={"event": "POLL_TIMEOUT", "job_id": job_id})
                    return jsonify({"status": "timeout", "error": "Processing timeout; try sync or no-LLM"})
            except Exception as e:
                log.debug("POLL_TIME_PARSE_FAIL",
                          extra={"event": "POLL_TIME_PARSE_FAIL", "error": str(e)})
        return jsonify({"status": "processing", "progress": data.get("progress")})

    if status == "failed":
        log.error("POLL_FAILED",
                  extra={"event": "POLL_FAILED", "job_id": job_id, "data": data})
        return jsonify(data)

    return jsonify({"status": "processing"})

@app.route("/api/job/<job_id>/questions")
def get_job_questions(job_id):
    if job_id not in job_data:
        log.warning("QUESTIONS_UNKNOWN_JOB",
                    extra={"event": "QUESTIONS_UNKNOWN_JOB", "job_id": job_id})
        return jsonify({"error": "Job not found"}), 404

    job = job_data[job_id]
    if job.get("status") == "completed" and "questions" in job:
        log.debug("QUESTIONS_CACHED",
                  extra={"event": "QUESTIONS_CACHED", "job_id": job_id,
                         "count": len(job["questions"])})
        return jsonify(job["questions"])

    resp = make_api_request("GET", f"jobs/{job_id}/questions")
    if resp and resp.status_code == 200:
        questions = resp.json()
        job["questions"] = questions
        log.info("QUESTIONS_FETCHED",
                 extra={"event": "QUESTIONS_FETCHED", "job_id": job_id, "count": len(questions)})
        return jsonify(questions)
    log.error("QUESTIONS_FETCH_ERROR",
              extra={"event": "QUESTIONS_FETCH_ERROR", "job_id": job_id,
                     "status": getattr(resp, 'status_code', 'NO_RESP')})
    return jsonify({"error": "Failed to fetch questions"}), 500

@app.route("/review/<job_id>")
def review(job_id):
    if job_id not in job_data:
        log.warning("REVIEW_UNKNOWN_JOB",
                    extra={"event": "REVIEW_UNKNOWN_JOB", "job_id": job_id})
        return redirect(url_for("index"))

    job = job_data[job_id]
    questions = job.get("questions") or []
    if not questions:
        resp = make_api_request("GET", f"jobs/{job_id}/questions")
        if resp and resp.status_code == 200:
            questions = resp.json()
            job["questions"] = questions
            log.info("REVIEW_FETCHED_QUESTIONS",
                     extra={"event": "REVIEW_FETCHED_QUESTIONS",
                            "job_id": job_id, "count": len(questions)})

    log.info("REVIEW_VIEW",
             extra={"event": "REVIEW_VIEW", "job_id": job_id, "count": len(questions)})
    return render_template("review.html", job_id=job_id, questions=questions, job=job)

@app.route("/api/save_edits/<job_id>", methods=["POST"])
def save_edits(job_id):
    data = request.get_json() or {}
    incoming = data.get("questions", [])
    if job_id not in job_edits:
        job_edits[job_id] = []

    index = {q.get("qid"): i for i, q in enumerate(job_edits[job_id])}
    for q in incoming:
        qid = q.get("qid")
        if qid in index:
            job_edits[job_id][index[qid]] = q
        else:
            job_edits[job_id].append(q)

    log.info("SAVE_EDITS",
             extra={"event": "SAVE_EDITS", "job_id": job_id, "count": len(incoming)})
    return jsonify({"success": True})

@app.route("/api/export/<job_id>/<format>")
def export_data(job_id, format):
    if job_id not in job_data:
        log.warning("EXPORT_UNKNOWN_JOB",
                    extra={"event": "EXPORT_UNKNOWN_JOB", "job_id": job_id})
        return jsonify({"error": "Job not found"}), 404

    original = job_data[job_id].get("questions", [])
    edits = job_edits.get(job_id, [])
    edits_by_id = {q.get("qid"): q for q in edits}
    questions = [edits_by_id.get(q.get("qid"), q) for q in original]

    approved_only = request.args.get("approved") in ("true", "1", "yes", "on")
    high_conf = request.args.get("high_confidence") in ("true", "1", "yes", "on")

    def keep(q):
        if approved_only and q.get("status") != "approved":
            return False
        if high_conf and (float(q.get("confidence", 0)) < 0.8):
            return False
        return True

    questions = [q for q in questions if keep(q)]
    log.info("EXPORT_START",
             extra={"event": "EXPORT_START", "job_id": job_id, "format": format,
                    "count": len(questions),
                    "approved_only": approved_only, "high_conf_only": high_conf})

    if format == "docx":
        try:
            from docx import Document
            doc = Document()
            doc.add_heading("RFP Questions Report", 0)
            doc.add_paragraph(f"Generated on: {datetime.now():%Y-%m-%d %H:%M:%S}")
            doc.add_paragraph(f"Job ID: {job_id}")
            doc.add_paragraph(f"Total Questions: {len(questions)}")
            doc.add_paragraph("")

            for i, q in enumerate(questions, 1):
                doc.add_heading(f"Question {i}", level=2)
                doc.add_paragraph(f'Text: {q.get("text","N/A")}')
                doc.add_paragraph(f'Confidence: {q.get("confidence","N/A")}')
                doc.add_paragraph(f'Type: {q.get("type","N/A")}')
                section = (q.get("section_path") or ["N/A"])[0]
                doc.add_paragraph(f"Section: {section}")
                doc.add_paragraph("")

            bio = io.BytesIO()
            doc.save(bio)
            bio.seek(0)
            fname = f"RFP_Questions_Report_{datetime.now():%Y%m%d_%H%M%S}.docx"
            log.info("EXPORT_DONE", extra={"event": "EXPORT_DONE", "format": "docx", "job_id": job_id})
            return send_file(
                bio, as_attachment=True, download_name=fname,
                mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        except ImportError:
            content = [
                "RFP Questions Export",
                f"Generated on: {datetime.now():%Y-%m-%d %H:%M:%S}",
                f"Job ID: {job_id}",
                f"Total Questions: {len(questions)}", ""
            ]
            for i, q in enumerate(questions, 1):
                section = (q.get("section_path") or [""])[0] if q.get("section_path") else ""
                content.append(f"{i}. {q.get('text','N/A')}")
                content.append(f"   Confidence: {q.get('confidence','N/A')}")
                content.append(f"   Type: {q.get('type','N/A')}")
                content.append(f"   Section: {section}")
                content.append("")
            body = "\n".join(content)
            fname = f"RFP_Questions_Report_{datetime.now():%Y%m%d_%H%M%S}.txt"
            log.info("EXPORT_DONE_FALLBACK",
                     extra={"event": "EXPORT_DONE_FALLBACK", "format": "txt", "job_id": job_id})
            return body, 200, {
                "Content-Type": "text/plain",
                "Content-Disposition": f'attachment; filename="{fname}"'
            }

    elif format == "xlsx":
        try:
            import pandas as pd
            import openpyxl  # ensure engine available
            df = pd.DataFrame(questions)
            bio = io.BytesIO()
            with pd.ExcelWriter(bio, engine="openpyxl") as w:
                df.to_excel(w, sheet_name="RFP Questions", index=False)
            bio.seek(0)
            fname = f"RFP_Questions_Report_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
            log.info("EXPORT_DONE", extra={"event": "EXPORT_DONE", "format": "xlsx", "job_id": job_id})
            return send_file(
                bio, as_attachment=True, download_name=fname,
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except ImportError:
            import csv
            out = io.StringIO()
            rows = []
            for q in questions:
                rows.append({
                    "qid": q.get("qid", ""),
                    "text": q.get("text", ""),
                    "confidence": q.get("confidence", ""),
                    "type": q.get("type", ""),
                    "status": q.get("status", ""),
                    "section": (q.get("section_path") or [""])[0] if q.get("section_path") else "",
                    "numbering": q.get("numbering", ""),
                    "category": q.get("category", "")
                })
            if rows:
                writer = csv.DictWriter(out, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            fname = f"RFP_Questions_Report_{datetime.now():%Y%m%d_%H%M%S}.csv"
            log.info("EXPORT_DONE_FALLBACK",
                     extra={"event": "EXPORT_DONE_FALLBACK", "format": "csv", "job_id": job_id})
            return out.getvalue(), 200, {
                "Content-Type": "text/csv",
                "Content-Disposition": f'attachment; filename="{fname}"'
            }

    log.error("EXPORT_INVALID_FORMAT",
              extra={"event": "EXPORT_INVALID_FORMAT", "format": format, "job_id": job_id})
    return jsonify({"error": "Invalid format"}), 400

# Optional tiny health for UI app
@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok", "version": APP_VERSION}), 200

if __name__ == "__main__":
    log.info("DEV_START",
             extra={"event": "DEV_START",
                    "message": "Starting RFP Extraction App on port 5002..."})
    app.run(debug=True, port=5002, host="0.0.0.0")
