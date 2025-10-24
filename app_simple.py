# app_simple.py — KnockKnock Intelligence (Flask)
# Hotfix build: proper root route, extractor auth header (knocknock-authentication),
# sync→async fallback, debug endpoints, MIME detection, review/edit/export preserved.

import os
import io
import csv
import json
import time
import mimetypes
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import requests
from flask import (
    Flask, render_template, request, jsonify, redirect, url_for, send_file
)
from werkzeug.utils import secure_filename

# ------------------------------------------------------------------------------
# Flask
# ------------------------------------------------------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.getenv("FLASK_SECRET", "dev-secret-change-me")

# ------------------------------------------------------------------------------
# Config (ENV first; defaults match your working Streamlit)
# ------------------------------------------------------------------------------
RFP_API_URL = os.getenv(
    "RFP_API_URL",
    "https://connect.affiniusaiplatform.com/content/000d3572-937a-4d5c-ab7e-7ec80d80c4ce",
).rstrip("/")

# *** IMPORTANT: extractor key + header ***
# Use *one* of these header styles:
#   1) Authorization + Bearer  => set HEADER_NAME=Authorization, SCHEME=Bearer
#   2) Custom header (RECOMMENDED here): knocknock-authentication: <key>
RFP_API_KEY = os.getenv("RFP_API_KEY", "hOSWPr4qZpvlzYOB0pKv6DXkf9HB8emB")
RFP_API_KEY_HEADER_NAME = os.getenv("RFP_API_KEY_HEADER_NAME", "knocknock-authentication")
RFP_API_AUTH_SCHEME = os.getenv("RFP_API_AUTH_SCHEME", "Raw")  # Raw means no scheme prefix

POLL_INTERVAL_SECONDS = float(os.getenv("POLL_INTERVAL_SECONDS", "2"))
POLL_TIMEOUT_SECONDS = float(os.getenv("POLL_TIMEOUT_SECONDS", "240"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))

ALLOWED_EXTENSIONS = {"docx", "pdf", "xlsx"}
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH_MB", "16")) * 1024 * 1024
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/knockknock_uploads")
CACHE_DIR = os.getenv("JOB_CACHE_DIR", "/tmp/knockknock_cache")
STATS_FILE = os.getenv("STATS_FILE", "/tmp/knockknock_stats.json")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# Tiny file cache for jobs & edits
# ------------------------------------------------------------------------------
def _cache(job_id: str) -> str:
    return os.path.join(CACHE_DIR, f"{job_id}.json")

def _cache_edits(job_id: str) -> str:
    return os.path.join(CACHE_DIR, f"{job_id}.edits.json")

def load_job(job_id: str) -> Optional[Dict[str, Any]]:
    p = _cache(job_id)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None

def save_job(job_id: str, data: Dict[str, Any]) -> None:
    try:
        with open(_cache(job_id), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[CACHE] save_job error: {e}")

def load_edits(job_id: str) -> List[Dict[str, Any]]:
    p = _cache_edits(job_id)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def save_edits(job_id: str, rows: List[Dict[str, Any]]) -> None:
    try:
        with open(_cache_edits(job_id), "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[CACHE] save_edits error: {e}")

# ------------------------------------------------------------------------------
# Stats
# ------------------------------------------------------------------------------
def load_stats() -> Dict[str, Any]:
    try:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE, "r", encoding="utf-8") as f:
                s = json.load(f)
            docs = s.get("documents_processed", 0)
            s["avg_processing_time"] = round(s.get("total_processing_time", 0) / max(docs, 1), 1) if docs else 0.0
            return s
    except Exception:
        pass
    return {
        "documents_processed": 0,
        "questions_extracted": 0,
        "total_processing_time": 0,
        "avg_processing_time": 0.0,
        "accuracy_rate": 0.0,
        "last_updated": datetime.now().isoformat(),
    }

def save_stats(stats: Dict[str, Any]) -> None:
    stats["last_updated"] = datetime.now().isoformat()
    try:
        with open(STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
    except Exception as e:
        print(f"[STATS] save error: {e}")

def update_stats_from_questions(questions: List[Dict[str, Any]], processing_time: float) -> Dict[str, Any]:
    s = load_stats()
    s["documents_processed"] += 1
    s["questions_extracted"] += len(questions)
    s["total_processing_time"] += processing_time
    confs = [float(q.get("confidence")) for q in questions if isinstance(q.get("confidence"), (int, float, str)) and str(q.get("confidence")).strip() != ""]
    try:
        confs = [float(c) for c in confs]
    except Exception:
        confs = []
    if confs:
        s["accuracy_rate"] = round(sum(confs) / len(confs) * 100.0, 1)
    save_stats(s)
    return s

# ------------------------------------------------------------------------------
# Extractor API glue
# ------------------------------------------------------------------------------
def ep(path: str) -> str:
    return f"{RFP_API_URL}/{path.lstrip('/')}"

def build_auth_headers() -> Dict[str, str]:
    """
    Build EXACT header for the extractor. For knocknock-authentication, send Raw:
      knocknock-authentication: <key>
    For Authorization/Bearer:
      Authorization: Bearer <key>
    """
    if not RFP_API_KEY:
        return {}
    if RFP_API_KEY_HEADER_NAME.lower() == "authorization":
        scheme = (RFP_API_AUTH_SCHEME or "Raw").lower()
        if scheme == "bearer":
            return {"Authorization": f"Bearer {RFP_API_KEY}"}
        elif scheme == "key":
            return {"Authorization": f"Key {RFP_API_KEY}"}
        else:
            return {"Authorization": RFP_API_KEY}
    # custom header (recommended for you)
    return {RFP_API_KEY_HEADER_NAME: RFP_API_KEY}

def rq_get(url: str) -> Optional[requests.Response]:
    try:
        return requests.get(url, headers=build_auth_headers(), timeout=REQUEST_TIMEOUT_SECONDS)
    except Exception as e:
        print("[HTTP][GET][ERROR]", e)
        return None

def rq_post(url: str, **kwargs) -> Optional[requests.Response]:
    try:
        headers = build_auth_headers()
        headers.update(kwargs.pop("headers", {}))
        return requests.post(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS, **kwargs)
    except Exception as e:
        print("[HTTP][POST][ERROR]", e)
        return None

# ------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def guess_mime(filename: str) -> str:
    ext = filename.rsplit(".", 1)[1].lower()
    if ext == "docx":
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if ext == "pdf":
        return "application/pdf"
    if ext == "xlsx":
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return mimetypes.guess_type(filename)[0] or "application/octet-stream"

def redacted_headers_for_debug() -> Dict[str, str]:
    h = build_auth_headers()
    red = {}
    for k, v in h.items():
        red[k] = v[:4] + "••••••••" if isinstance(v, str) else "•••"
    return red

# ------------------------------------------------------------------------------
# UI routes
# ------------------------------------------------------------------------------
@app.get("/")
def home():
    cfg_public = {
        "base_url": RFP_API_URL,
        "header_name": RFP_API_KEY_HEADER_NAME,
        "auth_scheme": RFP_API_AUTH_SCHEME,
        "poll_interval": POLL_INTERVAL_SECONDS,
        "poll_timeout": POLL_TIMEOUT_SECONDS,
        "max_upload_mb": MAX_CONTENT_LENGTH // (1024 * 1024),
    }
    return render_template("index.html", api_config=cfg_public, stats=load_stats())

@app.get("/review/<job_id>")
def review(job_id: str):
    job = load_job(job_id)
    if not job:
        # hydrate from extractor once
        r = rq_get(ep(f"/jobs/{job_id}"))
        if r and r.status_code == 200:
            job = r.json() or {}
            job["job_id"] = job_id
            save_job(job_id, job)
        else:
            return redirect(url_for("home"))
    orig = job.get("questions") or []
    edits = load_edits(job_id)
    questions = merge_questions(orig, edits)
    return render_template("review.html", job_id=job_id, questions=questions, job=job)

# ------------------------------------------------------------------------------
# API routes for frontend JS
# ------------------------------------------------------------------------------
@app.post("/api/upload")
def api_upload():
    """
    Upload & extract. If /extract/sync fails with a validation/pattern error,
    automatically fall back to /extract/ (async) and tell the UI.
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        f = request.files["file"]
        if f.filename == "":
            return jsonify({"error": "No file selected"}), 400
        if not allowed_file(f.filename):
            return jsonify({"error": f"Invalid file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"}), 400

        use_llm = str(request.form.get("use_llm", "false")).lower() == "true"
        use_sync = str(request.form.get("use_sync", "false")).lower() == "true"
        mode = request.form.get("mode") or "balanced"

        filename = secure_filename(f.filename)
        local_path = os.path.join(UPLOAD_DIR, filename)
        f.save(local_path)

        # Re-open for posting; keep bytes around for fallback
        mime = guess_mime(filename)
        with open(local_path, "rb") as fh:
            file_bytes = fh.read()

        files = {"file": (filename, io.BytesIO(file_bytes), mime)}
        data = {"use_llm": "true" if use_llm else "false", "mode": mode}

        print(f"[UPLOAD] header={redacted_headers_for_debug()} url={RFP_API_URL} sync={use_sync} mode={mode} mime={mime}")

        # 1) Try sync if requested
        if use_sync:
            r = rq_post(ep("/extract/sync"), files=files, data=data)
            if r is None:
                return jsonify({"error": "Extractor API unreachable"}), 502

            # If sync success: deliver questions immediately
            if r.status_code == 200:
                try:
                    resp = r.json()
                except Exception:
                    return jsonify({"error": f"Invalid JSON from extractor (sync): {r.text[:300]}"}), 502

                if "questions" in resp:
                    job_id = f"sync_{int(time.time())}"
                    job = {"status": "completed", "questions": resp.get("questions") or [], "job_id": job_id}
                    save_job(job_id, job)
                    update_stats_from_questions(job["questions"], processing_time=5.0)
                    # Clean up file
                    try: os.remove(local_path)
                    except: pass
                    return jsonify({"success": True, "job_id": job_id, "questions": job["questions"], "async": False})

                # Sync didn’t return questions → fall back
                print(f"[UPLOAD][SYNC] 200 but no questions; fallback to async. Body: {resp}")
            else:
                body_preview = (r.text or "")[:400]
                print(f"[UPLOAD][SYNC][HTTP {r.status_code}] {body_preview}")

                # Heuristic: pattern/validation issues → auto-fallback to async
                if "did not match the expected pattern" in body_preview or "ValidationError" in body_preview:
                    print("[UPLOAD][SYNC] Pattern/validation error → falling back to async")
                else:
                    # Return explicit error for non-validation failures
                    try: os.remove(local_path)
                    except: pass
                    return jsonify({"error": f"Sync failed (HTTP {r.status_code}): {body_preview}"}), 502

        # 2) Async (primary path or sync fallback)
        files = {"file": (filename, io.BytesIO(file_bytes), mime)}  # rebuild stream
        r = rq_post(ep("/extract/"), files=files, data=data)
        try: os.remove(local_path)
        except: pass

        if r is None:
            return jsonify({"error": "Extractor API unreachable"}), 502
        if r.status_code != 200:
            return jsonify({"error": f"Async failed (HTTP {r.status_code}): {r.text[:400]}"}), 502

        resp = r.json() or {}
        job_id = resp.get("job_id")
        if not job_id:
            return jsonify({"error": "No job_id returned by extractor (async)"}), 502

        job = {"status": "processing", "job_id": job_id, "timestamp": datetime.now().isoformat()}
        save_job(job_id, job)
        return jsonify({"success": True, "job_id": job_id, "async": True})

    except Exception as e:
        print("[UPLOAD][ERROR]", e)
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.get("/api/poll/<job_id>")
def api_poll(job_id: str):
    try:
        job = load_job(job_id) or {"job_id": job_id, "status": "processing"}
        r = rq_get(ep(f"/jobs/{job_id}"))

        if r is None:
            return jsonify({"error": "Extractor API unreachable"}), 502

        if r.status_code == 404:
            # GC on server; try direct questions
            q = rq_get(ep(f"/jobs/{job_id}/questions"))
            if q and q.status_code == 200:
                qs = q.json() or []
                job.update({"status": "completed", "questions": qs})
                save_job(job_id, job)
                update_stats_from_questions(qs, processing_time=30.0)
                return jsonify(job)
            job.update({"status": "completed", "questions": []})
            save_job(job_id, job)
            return jsonify(job)

        if r.status_code != 200:
            return jsonify({"error": f"HTTP {r.status_code}: {r.text[:400]}"}), 502

        status = r.json() or {}
        job.update(status)
        job["job_id"] = job_id

        if job.get("status") == "completed":
            if "questions" in job and job["questions"] is not None:
                save_job(job_id, job)
                update_stats_from_questions(job["questions"], processing_time=30.0)
                return jsonify(job)
            q = rq_get(ep(f"/jobs/{job_id}/questions"))
            if q and q.status_code == 200:
                job["questions"] = q.json() or []
                save_job(job_id, job)
                update_stats_from_questions(job["questions"], processing_time=30.0)
                return jsonify(job)
            job["questions"] = []
            save_job(job_id, job)
            return jsonify(job)

        save_job(job_id, job)
        return jsonify({"status": "processing"})

    except Exception as e:
        print("[POLL][ERROR]", e)
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.get("/api/job/<job_id>/questions")
def api_get_questions(job_id: str):
    job = load_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    orig = job.get("questions") or []
    edits = load_edits(job_id)
    return jsonify(merge_questions(orig, edits))

@app.post("/api/save_edits/<job_id>")
def api_save_edits(job_id: str):
    try:
        payload = request.get_json(force=True) or {}
        rows = payload.get("questions") or []
        if not isinstance(rows, list):
            return jsonify({"error": "questions must be a list"}), 400
        rows = [r for r in rows if isinstance(r, dict)]
        save_edits(job_id, rows)
        # merge immediately for UX
        job = load_job(job_id) or {"job_id": job_id, "status": "completed", "questions": []}
        job["questions"] = merge_questions(job.get("questions") or [], rows)
        save_job(job_id, job)
        return jsonify({"success": True})
    except Exception as e:
        print("[SAVE_EDITS][ERROR]", e)
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# ------------------------------------------------------------------------------
# Export
# ------------------------------------------------------------------------------
def merge_questions(orig: List[Dict[str, Any]], edits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not edits:
        return orig or []
    by_qid = {}
    for e in edits:
        qid = e.get("qid")
        if qid is not None:
            by_qid[str(qid)] = e
    out = []
    for q in orig or []:
        qid = q.get("qid")
        if qid is not None and str(qid) in by_qid:
            out.append(by_qid[str(qid)])
        else:
            out.append(q)
    # include any new manual edits without qid
    for e in edits:
        if e.get("qid") is None:
            out.append(e)
    return out

def _questions_for_export(job_id: str) -> List[Dict[str, Any]]:
    job = load_job(job_id)
    if not job:
        return []
    return merge_questions(job.get("questions") or [], load_edits(job_id))

@app.get("/api/export/<job_id>/json")
def export_json(job_id: str):
    rows = _questions_for_export(job_id)
    data = json.dumps(rows, indent=2, ensure_ascii=False).encode("utf-8")
    fname = f"RFP_Questions_{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    return send_file(io.BytesIO(data), as_attachment=True, download_name=fname, mimetype="application/json")

@app.get("/api/export/<job_id>/csv")
def export_csv(job_id: str):
    rows = _questions_for_export(job_id)
    keys = set()
    for r in rows:
        keys.update(r.keys())
    keys = list(sorted(keys))
    s = io.StringIO()
    w = csv.DictWriter(s, fieldnames=keys)
    w.writeheader()
    for r in rows:
        w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v) for k, v in r.items()})
    data = s.getvalue().encode("utf-8")
    fname = f"RFP_Questions_{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return send_file(io.BytesIO(data), as_attachment=True, download_name=fname, mimetype="text/csv")

@app.get("/api/export/<job_id>/xlsx")
def export_xlsx(job_id: str):
    try:
        import pandas as pd
    except Exception:
        return jsonify({"error": "pandas is required for xlsx export"}), 500
    rows = _questions_for_export(job_id)
    df = pd.json_normalize(rows) if rows else pd.DataFrame()
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="RFP Questions")
    buf.seek(0)
    fname = f"RFP_Questions_{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return send_file(buf, as_attachment=True, download_name=fname, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

@app.get("/api/export/<job_id>/docx")
def export_docx(job_id: str):
    try:
        from docx import Document
    except Exception:
        return jsonify({"error": "python-docx is required for docx export"}), 500
    rows = _questions_for_export(job_id)
    doc = Document()
    doc.add_heading("RFP Questions Report", 0)
    doc.add_paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    doc.add_paragraph(f"Job ID: {job_id}")
    doc.add_paragraph(f"Total Questions: {len(rows)}")
    doc.add_paragraph("")
    for i, q in enumerate(rows, 1):
        doc.add_heading(f"Question {i}", level=2)
        doc.add_paragraph(f"Text: {q.get('text','')}")
        conf = q.get("confidence")
        try: conf = f"{float(conf):.3f}"
        except Exception: conf = conf if conf is not None else ""
        doc.add_paragraph(f"Confidence: {conf}")
        doc.add_paragraph(f"Type: {q.get('type','')}")
        section = None
        if isinstance(q.get("section_path"), list) and q.get("section_path"):
            section = q["section_path"][0]
        elif "section" in q: section = q.get("section")
        elif "section_name" in q: section = q.get("section_name")
        doc.add_paragraph(f"Section: {section or ''}")
        doc.add_paragraph("")
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    fname = f"RFP_Questions_{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
    return send_file(buf, as_attachment=True, download_name=fname, mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

# ------------------------------------------------------------------------------
# Debug & health
# ------------------------------------------------------------------------------
@app.get("/debug/extractor")
def debug_extractor():
    """Check headers + hit /health and /info to confirm auth & base URL."""
    hdr = redacted_headers_for_debug()
    out = {"base_url": RFP_API_URL, "headers": hdr, "probes": {}}
    for p in ["/health", "/health/ready", "/info", "/openapi.json"]:
        r = rq_get(ep(p))
        out["probes"][p] = {
            "ok": bool(r and r.status_code == 200),
            "status": (r.status_code if r else None),
            "preview": (r.text[:400] if r and r.text else None),
        }
    return jsonify(out)

@app.get("/health")
def health():
    return jsonify({"ok": True, "ts": datetime.now().isoformat()})

@app.get("/health/ready")
def ready():
    return jsonify({"ready": True, "ts": datetime.now().isoformat()})

@app.get("/info")
def info():
    return jsonify({
        "app": "KnockKnock Intelligence (Flask)",
        "extractor_base": RFP_API_URL,
        "auth_header": RFP_API_KEY_HEADER_NAME,
        "auth_scheme": RFP_API_AUTH_SCHEME,
        "poll_interval": POLL_INTERVAL_SECONDS,
        "poll_timeout": POLL_TIMEOUT_SECONDS,
        "max_upload_mb": MAX_CONTENT_LENGTH // (1024 * 1024),
    })

# ------------------------------------------------------------------------------
# Local dev
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting KnockKnock Intelligence (Flask) on http://127.0.0.1:5002")
    app.run(host="0.0.0.0", port=5002, debug=True)
