import os
import io
import json
import time
import typing as t
from datetime import datetime, timedelta

import requests
from flask import (
    Flask, render_template, request, jsonify, redirect,
    url_for, send_file
)
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
load_dotenv()

# ------------------------------------------------------------------------------
# Flask App
# ------------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "dev-secret-change-me")

# ------------------------------------------------------------------------------
# Config (ENV-driven; no code edits required when deploying to Connect)
# ------------------------------------------------------------------------------
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
ALLOWED_EXTENSIONS = {"docx", "pdf", "xlsx"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DEFAULT_API_CONFIG = {
    # Your Posit Connect extractor deployment. Either with or without trailing slash is fine.
    "base_url": os.getenv(
        "RFP_API_URL",
        "https://connect.affiniusaiplatform.com/content/000d3572-937a-4d5c-ab7e-7ec80d80c4ce",
    ).rstrip("/"),
    # API key (NEVER hardcode in production; this is pulled from Connect ENV)
    "api_key": os.getenv("RFP_API_KEY", "CHANGEME"),
    # If your extractor expects a custom header, set it here, e.g. "knocknock-authentication"
    "header_name": os.getenv("RFP_API_KEY_HEADER_NAME", "knocknock-authentication"),
    # One of: Bearer | Key | Raw. Use Raw when header_name is custom.
    "auth_scheme": os.getenv("RFP_API_AUTH_SCHEME", "Raw"),
    "poll_interval": float(os.getenv("POLL_INTERVAL_SECONDS", "2")),
    "poll_timeout": float(os.getenv("POLL_TIMEOUT_SECONDS", "240")),
    "request_timeout": int(os.getenv("REQUEST_TIMEOUT_SECONDS", "60")),
}

api_config = DEFAULT_API_CONFIG.copy()

# ------------------------------------------------------------------------------
# App Stats (simple JSON file)
# ------------------------------------------------------------------------------
STATS_FILE = os.getenv("STATS_FILE", "stats.json")


def load_stats() -> dict:
    try:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE, "r") as f:
                stats = json.load(f)
        else:
            stats = {}
    except Exception:
        stats = {}
    # Defaults
    stats.setdefault("documents_processed", 0)
    stats.setdefault("questions_extracted", 0)
    stats.setdefault("total_processing_time", 0.0)
    # Derived
    dp = max(stats.get("documents_processed", 0), 1)
    stats["avg_processing_time"] = round(
        float(stats.get("total_processing_time", 0.0)) / dp, 1
    )
    stats.setdefault("accuracy_rate", 0.0)
    stats["last_updated"] = datetime.now().isoformat()
    return stats


def save_stats(stats: dict):
    try:
        stats["last_updated"] = datetime.now().isoformat()
        with open(STATS_FILE, "w") as f:
            json.dump(stats, f, indent=2)
    except Exception as e:
        print(f"DEBUG: Error saving stats: {e}")


def update_stats(questions_count: int, processing_time_seconds: float, questions: t.Optional[t.List[dict]] = None):
    stats = load_stats()
    stats["documents_processed"] += 1
    stats["questions_extracted"] += int(questions_count)
    stats["total_processing_time"] += float(processing_time_seconds)

    # compute mean confidence if present
    if questions:
        vals = []
        for q in questions:
            c = q.get("confidence")
            try:
                if c is not None:
                    vals.append(float(c))
            except Exception:
                pass
        if vals:
            stats["accuracy_rate"] = round(sum(vals) / len(vals) * 100.0, 1)
        else:
            # fallback if extractor doesn't supply confidence
            stats["accuracy_rate"] = max(stats.get("accuracy_rate", 85.0), 85.0)
    else:
        stats["accuracy_rate"] = max(stats.get("accuracy_rate", 85.0), 85.0)

    save_stats(stats)
    return stats


# ------------------------------------------------------------------------------
# In-Memory Job Cache (best effort; we still rely on remote polling)
# ------------------------------------------------------------------------------
job_data: dict[str, dict] = {}
job_edits: dict[str, list] = {}

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_auth_header() -> dict:
    """Build the authentication header for the extractor call."""
    name = api_config["header_name"].strip()
    key = api_config["api_key"].strip()
    scheme = (api_config.get("auth_scheme") or "Raw").strip()

    if name.lower() == "authorization":
        if scheme.lower() == "bearer":
            return {"Authorization": f"Bearer {key}"}
        elif scheme.lower() == "key":
            return {"Authorization": f"Key {key}"}
        else:
            return {"Authorization": key}
    # Custom header (e.g. knocknock-authentication)
    return {name: key}


def _join(base: str, path: str) -> str:
    return f"{base.rstrip('/')}/{path.lstrip('/')}"


def _mime_for(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "docx":
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if ext == "pdf":
        return "application/pdf"
    if ext == "xlsx":
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return "application/octet-stream"


def make_api_request(method: str, endpoint: str, **kwargs) -> t.Optional[requests.Response]:
    """
    Single point for calling the extractor. Handles:
      - trailing slash normalization
      - **never** setting Content-Type when files=... (requests sets boundary)
    """
    # Normalize common paths. We'll try exact path first, then a fallback with/without slash.
    candidates = [endpoint]
    if endpoint.endswith("/"):
        candidates.append(endpoint.rstrip("/"))
    else:
        candidates.append(endpoint.rstrip("/") + "/")

    headers = get_auth_header()
    # Never override multipart headers
    if "files" not in kwargs:
        headers.update(kwargs.get("headers", {}))

    for ep in candidates:
        url = _join(api_config["base_url"], ep)
        try:
            if method.upper() == "GET":
                resp = requests.get(url, headers=headers, timeout=api_config["request_timeout"])
            elif method.upper() == "POST":
                resp = requests.post(url, headers=headers, timeout=api_config["request_timeout"], **kwargs)
            elif method.upper() == "DELETE":
                resp = requests.delete(url, headers=headers, timeout=api_config["request_timeout"])
            else:
                continue
            # Prefer a successful response immediately
            if resp is not None and (resp.status_code < 400 or resp.status_code == 422):
                # We still return 422 to the caller to decide on fallback
                return resp
            # If 404, try next variant
        except requests.exceptions.RequestException as e:
            print(f"DEBUG: extractor request error to {url}: {e}")
            # continue to next candidate
            continue
    return None


def _looks_like_validation_error(resp: t.Optional[requests.Response]) -> bool:
    if resp is None:
        return False
    if resp.status_code in (400, 422):
        body = (resp.text or "").lower()
        return ("expected pattern" in body) or ("validation" in body) or ("did not match" in body)
    return False


def _redact_headers(h: dict) -> dict:
    safe = {}
    for k, v in (h or {}).items():
        if isinstance(v, str) and len(v) > 8:
            safe[k] = v[:4] + "••••••••"
        else:
            safe[k] = v
    return safe


# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------
@app.route("/")
def index():
    stats = load_stats()
    return render_template("index.html", api_config=api_config, stats=stats)


@app.route("/debug/extractor")
def debug_extractor():
    """Live probe from Connect to verify base_url, auth header, and common docs endpoints."""
    base = api_config["base_url"]
    hdr = _redact_headers(get_auth_header())
    probes = {}
    for p in ("/", "/health", "/info", "/openapi.json", "/docs"):
        url = _join(base, p)
        try:
            r = requests.get(url, headers=get_auth_header(), timeout=10)
            probes[p] = {
                "url": url,
                "status": r.status_code,
                "ok": r.status_code in (200, 307, 308),
                "preview": (r.text or "")[:200],
            }
        except Exception as e:
            probes[p] = {"url": url, "error": str(e)}
    return jsonify({"base_url": base, "headers": hdr, "probes": probes})


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """
    Upload and extract with robust fallbacks:
      1) Try sync if requested.
      2) On 400/422 validation, fall back to async.
      3) For async, progressively simplify the form (mode→none, use_llm→none) if needed.
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

    # UI flags
    use_llm = request.form.get("use_llm", "false").lower() == "true"
    use_sync = request.form.get("use_sync", "false").lower() == "true"
    # The backend may be strict about allowed mode strings; start with 'balanced'
    mode = (request.form.get("mode") or "balanced").strip()

    print(f"DEBUG upload: file={filename}, use_llm={use_llm}, mode={mode}, use_sync={use_sync}")

    def form_payload(m: t.Optional[str] = None, llm: t.Optional[bool] = None) -> dict:
        payload = {}
        if llm is not None:
            payload["use_llm"] = "true" if llm else "false"
        if m:
            payload["mode"] = m
        return payload

    try:
        # Prepare reusable bytes
        with open(filepath, "rb") as fh:
            file_bytes = fh.read()
        mime = _mime_for(filename)

        # ------------------------------
        # 1) Try SYNC (if requested)
        # ------------------------------
        if use_sync:
            files = {"file": (filename, io.BytesIO(file_bytes), mime)}
            data = form_payload(m=mode, llm=use_llm)
            print(f"DEBUG sync attempt: endpoint=extract/sync data={data}")
            r = make_api_request("POST", "extract/sync", files=files, data=data)

            if r is not None and r.status_code == 200:
                try:
                    payload = r.json() or {}
                except Exception:
                    payload = {}
                if "questions" in payload:
                    job_id = f"sync_{int(time.time())}"
                    job_data[job_id] = {
                        "status": "completed",
                        "questions": payload["questions"],
                        "timestamp": datetime.now().isoformat(),
                    }
                    update_stats(len(payload["questions"]), 5, payload["questions"])
                    return jsonify({"success": True, "job_id": job_id, "questions": payload["questions"]})
                else:
                    print("DEBUG sync: 200 but no questions → falling back to async")

            if _looks_like_validation_error(r):
                print(f"DEBUG sync: validation error {r.status_code}, body={r.text[:300]}")
                # fall through to async
            else:
                if r is not None and r.status_code >= 500:
                    return jsonify({"error": f"Sync failed (HTTP {r.status_code}): {r.text}"}), 502
                # continue to async in all other cases

        # ------------------------------
        # 2) ASYNC primary
        # ------------------------------
        def async_attempts() -> t.Optional[dict]:
            """Try async with (mode+use_llm) → (use_llm only) → (no form), with endpoint variants."""
            attempts = [
                ("extract", form_payload(m=mode, llm=use_llm)),
                ("extract", form_payload(m=None, llm=use_llm)),
                ("extract", {}),  # final bare minimum
            ]
            for ep, data in attempts:
                files = {"file": (filename, io.BytesIO(file_bytes), mime)}
                print(f"DEBUG async attempt: endpoint={ep} data={data}")
                r = make_api_request("POST", ep, files=files, data=data)
                if r is None:
                    print("DEBUG async attempt: no response (network or endpoint mismatch)")
                    continue
                if r.status_code == 200:
                    try:
                        payload = r.json() or {}
                    except Exception:
                        payload = {}
                    job_id = payload.get("job_id")
                    if job_id:
                        job_data[job_id] = {"status": "processing", "timestamp": datetime.now().isoformat()}
                        return {"success": True, "job_id": job_id, "async": True}
                    else:
                        print("DEBUG async: 200 but no job_id in payload → continuing")
                        continue
                if _looks_like_validation_error(r):
                    print(f"DEBUG async validation {r.status_code}, body={r.text[:300]} → trying simpler payload")
                    continue
                print(f"DEBUG async error {r.status_code}: {r.text[:300]}")
            return None

        ok = async_attempts()
        if ok:
            return jsonify(ok)

        return jsonify({"error": "Extractor API failed to accept the upload"}), 502

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
    """
    Poll the extractor for job status. If our in-memory cache doesn't have it
    (multi-process on Connect), we still poll the remote job endpoint.
    """
    # If we already have completion cached:
    cached = job_data.get(job_id)
    if cached and cached.get("status") == "completed":
        return jsonify(cached)

    # Always hit remote to be robust across multiple processes on Connect
    r = make_api_request("GET", f"jobs/{job_id}")
    if r is not None and r.status_code == 200:
        data = r.json() if r.headers.get("content-type", "").lower().startswith("application/json") else {}
        status = (data or {}).get("status", "").lower()

        # Cache a minimal view
        job_data[job_id] = {**(job_data.get(job_id) or {}), **(data or {})}

        if status == "completed":
            # Some deployments embed questions directly; others separate them
            if "questions" in data and data["questions"] is not None:
                update_stats(len(data["questions"]), 30, data["questions"])
                return jsonify(data)

            # Try fetching questions separately
            qr = make_api_request("GET", f"jobs/{job_id}/questions")
            if qr is not None and qr.status_code == 200:
                try:
                    questions = qr.json() or []
                except Exception:
                    questions = []
                job_data[job_id]["questions"] = questions
                job_data[job_id]["status"] = "completed"
                update_stats(len(questions), 30, questions)
                return jsonify({"status": "completed", "questions": questions})

            # Completed but we couldn't fetch questions
            job_data[job_id]["status"] = "completed"
            job_data[job_id].setdefault("questions", [])
            return jsonify({"status": "completed", "questions": []})

        if status == "failed":
            return jsonify(data)

        # Else still processing
        # Guardrail: mark timeout after 10 minutes
        try:
            started = job_data[job_id].get("timestamp")
            if started:
                start_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
                if datetime.now() - start_dt > timedelta(minutes=10):
                    job_data[job_id]["status"] = "timeout"
                    return jsonify({"status": "timeout", "error": "AI processing timeout"})
        except Exception:
            pass

        return jsonify({"status": "processing"})

    # 404: some backends delete completed jobs; try to fetch questions anyway
    if r is not None and r.status_code == 404:
        qr = make_api_request("GET", f"jobs/{job_id}/questions")
        if qr is not None and qr.status_code == 200:
            try:
                questions = qr.json() or []
            except Exception:
                questions = []
            job_data[job_id] = {"status": "completed", "questions": questions, "timestamp": datetime.now().isoformat()}
            update_stats(len(questions), 30, questions)
            return jsonify({"status": "completed", "questions": questions})
        return jsonify({"status": "completed", "questions": []})

    err = f"Failed to poll job {job_id}"
    if r is not None:
        err = f"HTTP {r.status_code}: {r.text}"
    return jsonify({"error": err}), 502


@app.route("/api/job/<job_id>/questions")
def get_job_questions(job_id: str):
    cached = job_data.get(job_id, {})
    if cached.get("status") == "completed" and "questions" in cached:
        return jsonify(cached["questions"])

    r = make_api_request("GET", f"jobs/{job_id}/questions")
    if r is not None and r.status_code == 200:
        try:
            questions = r.json() or []
        except Exception:
            questions = []
        job_data.setdefault(job_id, {})["questions"] = questions
        job_data[job_id]["status"] = "completed"
        return jsonify(questions)
    return jsonify({"error": "Failed to fetch questions"}), 502


@app.route("/review/<job_id>")
def review(job_id: str):
    job = job_data.get(job_id, {"status": "processing", "questions": []})

    # If no cached questions, attempt remote fetch so page isn't empty on refresh
    if not job.get("questions"):
        r = make_api_request("GET", f"jobs/{job_id}/questions")
        if r is not None and r.status_code == 200:
            try:
                job["questions"] = r.json() or []
            except Exception:
                job["questions"] = []
            job["status"] = "completed"
            job_data[job_id] = job

    # Apply local edits overlay if any
    if job_id in job_edits and job.get("questions"):
        edits_by_qid = {q.get("qid"): q for q in job_edits[job_id]}
        merged = []
        for q in job["questions"]:
            qid = q.get("qid")
            merged.append(edits_by_qid.get(qid, q))
        job["questions"] = merged

    return render_template("review.html", job_id=job_id, questions=job.get("questions", []), job=job)


@app.route("/api/save_edits/<job_id>", methods=["POST"])
def save_edits(job_id: str):
    data = request.get_json(silent=True) or {}
    incoming = data.get("questions", [])

    job_edits.setdefault(job_id, [])
    index_by_qid = {q.get("qid"): i for i, q in enumerate(job_edits[job_id])}

    for q in incoming:
        qid = q.get("qid")
        if qid in index_by_qid:
            job_edits[job_id][index_by_qid[qid]] = q
        else:
            job_edits[job_id].append(q)

    print(f"DEBUG edits: job={job_id} total_saved={len(job_edits[job_id])}")
    return jsonify({"success": True})


@app.route("/api/export/<job_id>/<fmt>")
def export_data(job_id: str, fmt: str):
    """
    Export merged (original + edits) questions to DOCX or XLSX.
    Supports optional filters via query params:
      - approved=true
      - high_confidence=true (>= 0.8)
    """
    job = job_data.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    # Merge originals with edits
    originals = job.get("questions", [])
    edited = job_edits.get(job_id, [])
    edited_by_qid = {q.get("qid"): q for q in edited}
    merged: list[dict] = [edited_by_qid.get(q.get("qid"), q) for q in originals]

    # Filters
    approved_only = request.args.get("approved", "false").lower() == "true"
    high_conf_only = request.args.get("high_confidence", "false").lower() == "true"

    def keep(q: dict) -> bool:
        if approved_only and q.get("status") != "approved":
            return False
        if high_conf_only:
            try:
                if float(q.get("confidence", 0)) < 0.8:
                    return False
            except Exception:
                return False
        return True

    filtered = [q for q in merged if keep(q)]

    if fmt.lower() == "docx":
        try:
            from docx import Document

            doc = Document()
            doc.add_heading("RFP Questions Report", 0)
            doc.add_paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            doc.add_paragraph(f"Job ID: {job_id}")
            doc.add_paragraph(f"Total Questions: {len(filtered)}")
            doc.add_paragraph("")

            for i, q in enumerate(filtered, 1):
                doc.add_heading(f"Question {i}", level=2)
                doc.add_paragraph(f"Text: {q.get('text', 'N/A')}")
                doc.add_paragraph(f"Confidence: {q.get('confidence', 'N/A')}")
                doc.add_paragraph(f"Type: {q.get('type', 'N/A')}")
                # section can be string or list
                section = q.get("section_path")
                if isinstance(section, list):
                    section = section[0] if section else None
                doc.add_paragraph(f"Section: {section or 'N/A'}")
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

    if fmt.lower() == "xlsx":
        try:
            import pandas as pd

            df = pd.DataFrame(filtered or [])
            bio = io.BytesIO()
            with pd.ExcelWriter(bio, engine="openpyxl") as w:
                df.to_excel(w, sheet_name="RFP Questions", index=False)
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


# ------------------------------------------------------------------------------
# Small Health Endpoint (useful on Connect)
# ------------------------------------------------------------------------------
@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, "ts": datetime.now().isoformat()}), 200


# ------------------------------------------------------------------------------
# Dev server
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting KnockKnock Intelligence (Flask) on http://localhost:5002")
    app.run(host="0.0.0.0", port=5002, debug=True)
