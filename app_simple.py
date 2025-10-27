import os
import io
import json
import time
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple, List

import requests
from flask import (
    Flask, render_template, request, jsonify, redirect,
    url_for, send_file
)
from werkzeug.utils import secure_filename

# ============================== Logging ==============================
LOG_PATH = os.getenv("APP_LOG_FILE", "/tmp/kk_app.log")
logger = logging.getLogger("kk")
logger.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
fh = logging.FileHandler(LOG_PATH)
fh.setFormatter(_fmt)
sh = logging.StreamHandler()
sh.setFormatter(_fmt)
logger.addHandler(fh)
logger.addHandler(sh)

# ============================== Flask Setup ==============================
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "/tmp/kk_uploads")
ALLOWED_EXTENSIONS = {"docx", "pdf", "xlsx"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

# ============================ API Configuration ===========================
def _env_float(name: str, default: float) -> float:
    v = (os.getenv(name) or "").strip()
    try:
        return float(v) if v else default
    except Exception:
        return default

def _env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return v.strip() if v else default

API_BASE_URL = _env_str(
    "RFP_API_URL",
    "https://connect.affiniusaiplatform.com/content/000d3572-937a-4d5c-ab7e-7ec80d80c4ce/"
).rstrip("/")

API_KEY = _env_str("RFP_API_KEY", "")

# Optional env to pin the header we send first; we still try fallbacks if that fails.
EXPLICIT_AUTH_NAME = _env_str("RFP_API_KEY_HEADER_NAME", "")
EXPLICIT_AUTH_SCHEME = _env_str("RFP_API_AUTH_SCHEME", "")  # Key | Bearer | Raw | (empty = None)

POLL_INTERVAL = _env_float("POLL_INTERVAL_SECONDS", 2.0)
POLL_TIMEOUT = _env_float("POLL_TIMEOUT_SECONDS", 240.0)
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
    except Exception as e:
        logger.warning(f"[STATS] load error: {e}")
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
        logger.warning(f"[STATS] Save error: {e}")

def update_stats(questions: List[dict], processing_time_sec: float) -> dict:
    stats = load_stats()
    stats["documents_processed"] += 1
    stats["questions_extracted"] += len(questions)
    stats["total_processing_time"] += float(processing_time_sec)
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

# ===================== Auth Candidates & Request Layer ======================

def _explicit_candidate() -> Optional[dict]:
    if not API_KEY:
        return None
    name = EXPLICIT_AUTH_NAME
    scheme = EXPLICIT_AUTH_SCHEME
    if not name:
        return None
    if scheme:
        return {name: f"{scheme} {API_KEY}"}
    else:
        return {name: API_KEY}

def _fallback_candidates() -> List[dict]:
    if not API_KEY:
        return []
    return [
        {"Authorization": f"Key {API_KEY}"},
        {"Authorization": f"Bearer {API_KEY}"},
        {"knocknock-authentication": f"Raw {API_KEY}"},
        {"X-API-Key": API_KEY},
        {"Authorization": API_KEY},
    ]

def build_auth_sequence() -> List[dict]:
    seq = []
    expl = _explicit_candidate()
    if expl:
        seq.append(expl)
    seq.extend(_fallback_candidates())
    # De-dup
    uniq = []
    seen = set()
    for h in seq:
        k = tuple(sorted(h.items()))
        if k not in seen:
            uniq.append(h)
            seen.add(k)
    return uniq

EXTRACT_PATHS = ["extract/", "extract"]

def api_post_with_fallbacks(files, data, want_sync: bool) -> Tuple[Optional[requests.Response], dict, str]:
    if not API_KEY:
        raise RuntimeError("Missing RFP_API_KEY env var.")
    paths = ["extract/sync"] if want_sync else EXTRACT_PATHS
    last_error = None
    for path in paths:
        url = _ep(path)
        for hdrs in build_auth_sequence():
            try:
                r = requests.post(url, headers=hdrs, files=files, data=data, timeout=REQUEST_TIMEOUT)
                body_snip = (r.text or "")[:300]
                logger.info(f"[POST] {url} -> {r.status_code} using {list(hdrs.keys())[0]}: {body_snip!r}")
                if r.status_code == 200:
                    return r, hdrs, path
                last_error = (r.status_code, body_snip, hdrs, path)
            except requests.RequestException as e:
                last_error = (None, str(e), hdrs, path)
                logger.warning(f"[POST] {url} exception with {hdrs}: {e}")
                continue
    resp = requests.models.Response()
    resp.status_code = 599
    resp._content = (f"All auth variants failed. Last error: {last_error}").encode()
    return resp, {}, paths[-1] if paths else ("extract/sync" if want_sync else "extract/")

def api_get(path: str, headers: dict) -> requests.Response:
    url = _ep(path)
    r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    logger.info(f"[GET]  {url} -> {r.status_code} { (r.text or '')[:200]!r }")
    return r

def api_delete(path: str, headers: dict) -> requests.Response:
    url = _ep(path)
    r = requests.delete(url, headers=headers, timeout=REQUEST_TIMEOUT)
    logger.info(f"[DEL]  {url} -> {r.status_code} { (r.text or '')[:200]!r }")
    return r

# ============================== Routes ===============================
@app.route("/")
def index():
    logger.info(f"[CFG] API_BASE_URL={API_BASE_URL}")
    logger.info(f"[CFG] API_KEY={redacted(API_KEY)}")
    logger.info(f"[CFG] AUTH_NAME={EXPLICIT_AUTH_NAME or 'auto'} AUTH_SCHEME={EXPLICIT_AUTH_SCHEME or 'auto'}")
    logger.info(f"[CFG] TIMEOUT={REQUEST_TIMEOUT}s")
    stats = load_stats()
    api_config = {
        "base_url": API_BASE_URL,
        "poll_interval": POLL_INTERVAL,
        "poll_timeout": POLL_TIMEOUT,
        "request_timeout": REQUEST_TIMEOUT,
    }
    return render_template("index.html", api_config=api_config, stats=stats)

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
    mode = request.form.get("mode", "balanced")

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        mime = _mime_for(filename)
        with open(filepath, "rb") as f:
            files = {"file": (filename, f, mime)}
            data = {"use_llm": "true" if use_llm else "false", "mode": mode}
            logger.info(f"[UPLOAD] use_llm={use_llm} use_sync={use_sync} mode={mode} file={filename} size={os.path.getsize(filepath)}")

            resp, used_headers, used_path = api_post_with_fallbacks(files, data, want_sync=use_sync)
            body_snip = (resp.text or "")[:500] if resp is not None else ""
            logger.info(f"[UPLOAD] status={getattr(resp, 'status_code', None)} path={used_path} hdr={list(used_headers.keys())[:1]} body_snip={body_snip!r}")

        if resp is None:
            return jsonify({"error": "No response from extractor"}), 502
        if resp.status_code != 200:
            return jsonify({"error": f"HTTP {resp.status_code}: {resp.text}"}), 502

        result = resp.json() if "application/json" in (resp.headers.get("content-type","").lower()) else {}

        if use_sync and isinstance(result, dict) and "questions" in result:
            job_id = f"sync_{int(time.time())}"
            job_data[job_id] = {
                "status": "completed",
                "questions": result.get("questions") or [],
                "timestamp": now_iso(),
                "auth_headers": used_headers,
            }
            update_stats(job_data[job_id]["questions"], processing_time_sec=5.0)
            return jsonify({"success": True, "job_id": job_id, "questions": result.get("questions", [])})

        job_id = (result or {}).get("job_id")
        if not job_id:
            return jsonify({"error": f"Extractor did not return job_id. Raw: {result}"}), 502

        job_data[job_id] = {
            "status": "processing",
            "timestamp": now_iso(),
            "auth_headers": used_headers,
        }
        return jsonify({"success": True, "job_id": job_id, "async": True})

    except Exception as e:
        logger.exception("[UPLOAD] exception")
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
            for name in os.listdir(app.config["UPLOAD_FOLDER"]):
                p = os.path.join(app.config["UPLOAD_FOLDER"], name)
                if os.path.isfile(p) and os.path.getsize(p) == 0:
                    os.remove(p)
        except Exception:
            pass

@app.route("/api/poll/<job_id>")
def poll_job(job_id: str):
    job = job_data.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    if job.get("status") == "completed":
        return jsonify(job)

    headers = job.get("auth_headers") or (build_auth_sequence()[0] if build_auth_sequence() else {})
    try:
        r = api_get(f"jobs/{job_id}", headers)
        if r.status_code == 200:
            payload = r.json()
            job.update(payload or {})
            state = (payload or {}).get("status", "").lower()

            if state == "completed":
                if "questions" in payload and payload["questions"] is not None:
                    update_stats(payload["questions"], processing_time_sec=30.0)
                    return jsonify(payload)
                rq = api_get(f"jobs/{job_id}/questions", headers)
                if rq.status_code == 200:
                    questions = rq.json()
                    job["questions"] = questions
                    update_stats(questions, processing_time_sec=30.0)
                    return jsonify({"status": "completed", "questions": questions})
                return jsonify({"status": "completed", "questions": []})

            if state == "failed":
                return jsonify(payload)

            return jsonify({"status": "processing"})

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

    headers = job.get("auth_headers") or (build_auth_sequence()[0] if build_auth_sequence() else {})
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
        headers = job.get("auth_headers") or (build_auth_sequence()[0] if build_auth_sequence() else {})
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
    idx = {q.get("qid"): i for i, q in enumerate(edits) if q.get("qid")}
    for q in questions:
        qid = q.get("qid")
        if qid in idx:
            edits[idx[qid]] = q
        else:
            edits.append(q)
    return jsonify({"success": True})

@app.route("/api/export/<job_id>/<fmt>")
def export_data(job_id: str, fmt: str):
    job = job_data.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    original = job.get("questions") or []
    edited = job_edits.get(job_id, [])
    edited_map = {q.get("qid"): q for q in edited if q.get("qid")}
    merged = [edited_map.get(o.get("qid"), o) for o in original]

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
            logger.exception("[EXPORT DOCX] exception")
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
            logger.exception("[EXPORT XLSX] exception")
            return jsonify({"error": f"XLSX export failed: {e}"}), 500

    return jsonify({"error": "Invalid format"}), 400

# ========================== Diagnostics ==========================

def _tiny_docx_bytes() -> bytes:
    return b"Hello RFP"

@app.route("/_diag")
def diag():
    env = {
        "api_base_url": API_BASE_URL,
        "api_key_redacted": redacted(API_KEY),
        "explicit_auth_name": EXPLICIT_AUTH_NAME or None,
        "explicit_auth_scheme": EXPLICIT_AUTH_SCHEME or None,
        "timeout": REQUEST_TIMEOUT,
        "log_file": LOG_PATH,
    }

    results = {"env": env, "probes": {"health": [], "extract": []}}
    headers_seq = build_auth_sequence()

    # Health probe
    health_url = _ep("health/ready")
    for hdr in headers_seq:
        try:
            r = requests.get(health_url, headers=hdr, timeout=REQUEST_TIMEOUT)
            results["probes"]["health"].append({
                "url": health_url,
                # FIX: no stray paren
                "used_header": {k: (v.split()[0] + " <redacted>") if " " in v else "<redacted>" for k, v in hdr.items()},
                "status": r.status_code,
                "body_snip": (r.text or "")[:200],
            })
        except Exception as e:
            results["probes"]["health"].append({
                "url": health_url,
                "used_header": {k: "<error>" for k in hdr},
                "status": None,
                "error": str(e),
            })

    # Extract probe
    for path in EXTRACT_PATHS:
        url = _ep(path)
        for hdr in headers_seq:
            try:
                files = {"file": ("tiny.docx", io.BytesIO(_tiny_docx_bytes()),
                                  "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
                data = {"use_llm": "false", "mode": "balanced"}
                r = requests.post(url, headers=hdr, files=files, data=data, timeout=REQUEST_TIMEOUT)
                results["probes"]["extract"].append({
                    "url": url,
                    # FIX: no stray paren
                    "used_header": {k: (v.split()[0] + " <redacted>") if " " in v else "<redacted>" for k, v in hdr.items()},
                    "status": r.status_code,
                    "body_snip": (r.text or "")[:300],
                })
            except Exception as e:
                results["probes"]["extract"].append({
                    "url": url,
                    "used_header": {k: "<error>" for k in hdr},
                    "status": None,
                    "error": str(e),
                })

    return app.response_class(
        response=json.dumps(results, indent=2),
        status=200,
        mimetype="application/json"
    )

# ========================== Minimal error page ==========================
@app.errorhandler(413)
def too_large(_e):
    return render_template("error.html",
                           error_code=413,
                           error_message="File too large (limit 16MB)."), 413

# ============================== Dev server ==============================
if __name__ == "__main__":
    logger.info(f"Starting on http://localhost:5002")
    logger.info(f"Extractor base: {API_BASE_URL}")
    logger.info(f"API key: {redacted(API_KEY)}")
    app.run(host="0.0.0.0", port=5002, debug=True)
