# app_simple.py
# Flask web app for RFP Question Extraction on Posit Connect
# - Upload DOCX/PDF/XLSX
# - Call extractor API (sync or async)
# - Poll job until completed
# - Review & inline edit
# - Export (DOCX/XLSX/CSV/JSON; PDF optional if reportlab installed)
# - Persistent light-weight caching across workers (files under /tmp/knockknock_cache)
# - Friendly health endpoints for Connect

import os
import io
import csv
import json
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import requests
from flask import (
    Flask, render_template, request, jsonify, redirect, url_for,
    send_file
)
from werkzeug.utils import secure_filename

# ------------------------------------------------------------------------------
# Flask app
# ------------------------------------------------------------------------------
# Ensure your HTML is under ./templates and static files (if any) under ./static
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.getenv("FLASK_SECRET", "dev-secret-change-me")  # set securely in Connect

# ------------------------------------------------------------------------------
# Configuration (read from environment; sensible defaults)
# ------------------------------------------------------------------------------
RFP_API_URL = os.getenv(
    "RFP_API_URL",
    "https://connect.affiniusaiplatform.com/content/000d3572-937a-4d5c-ab7e-7ec80d80c4ce",
).rstrip("/")

RFP_API_KEY = os.getenv("RFP_API_KEY", "")  # <-- set this in Connect (extractor key)
RFP_API_KEY_HEADER_NAME = os.getenv("RFP_API_KEY_HEADER_NAME", "Authorization")
RFP_API_AUTH_SCHEME = os.getenv("RFP_API_AUTH_SCHEME", "Bearer")  # Bearer | Key | Raw

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
# Helpers: persistence (tiny file-based cache so edits/jobs survive worker restarts)
# ------------------------------------------------------------------------------
def _cache_path(job_id: str) -> str:
    return os.path.join(CACHE_DIR, f"{job_id}.json")

def _cache_edits_path(job_id: str) -> str:
    return os.path.join(CACHE_DIR, f"{job_id}.edits.json")

def load_job(job_id: str) -> Optional[Dict[str, Any]]:
    p = _cache_path(job_id)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None

def save_job(job_id: str, data: Dict[str, Any]) -> None:
    try:
        with open(_cache_path(job_id), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[CACHE] save_job error: {e}")

def load_edits(job_id: str) -> List[Dict[str, Any]]:
    p = _cache_edits_path(job_id)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def save_edits(job_id: str, questions: List[Dict[str, Any]]) -> None:
    try:
        with open(_cache_edits_path(job_id), "w", encoding="utf-8") as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[CACHE] save_edits error: {e}")

# ------------------------------------------------------------------------------
# Helpers: stats (file)
# ------------------------------------------------------------------------------
def load_stats() -> Dict[str, Any]:
    try:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE, "r", encoding="utf-8") as f:
                stats = json.load(f)
            if stats.get("documents_processed", 0) > 0:
                stats["avg_processing_time"] = round(
                    stats.get("total_processing_time", 0) / max(stats.get("documents_processed", 1), 1), 1
                )
            else:
                stats["avg_processing_time"] = 0.0
            return stats
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
    stats = load_stats()
    stats["documents_processed"] += 1
    stats["questions_extracted"] += len(questions)
    stats["total_processing_time"] += processing_time

    # derive accuracy from confidence if present
    confidences = []
    for q in questions:
        c = q.get("confidence")
        if isinstance(c, (int, float)):
            confidences.append(float(c))
    if confidences:
        stats["accuracy_rate"] = round(sum(confidences) / len(confidences) * 100.0, 1)
    save_stats(stats)
    return stats

# ------------------------------------------------------------------------------
# Helpers: API calls
# ------------------------------------------------------------------------------
def ep(path: str) -> str:
    return f"{RFP_API_URL}/{path.lstrip('/')}"

def build_auth_headers() -> Dict[str, str]:
    if not RFP_API_KEY:
        return {}
    if RFP_API_KEY_HEADER_NAME.lower() == "authorization":
        scheme = RFP_API_AUTH_SCHEME.lower()
        if scheme == "bearer":
            return {"Authorization": f"Bearer {RFP_API_KEY}"}
        elif scheme == "key":
            return {"Authorization": f"Key {RFP_API_KEY}"}
        else:
            return {"Authorization": RFP_API_KEY}
    return {RFP_API_KEY_HEADER_NAME: RFP_API_KEY}

def rq_get(url: str) -> requests.Response:
    return requests.get(url, headers=build_auth_headers(), timeout=REQUEST_TIMEOUT_SECONDS)

def rq_post(url: str, **kwargs) -> requests.Response:
    headers = build_auth_headers()
    headers.update(kwargs.pop("headers", {}))
    return requests.post(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS, **kwargs)

# ------------------------------------------------------------------------------
# Helpers: questions merge (original + edits by qid if present)
# ------------------------------------------------------------------------------
def merge_questions(original: List[Dict[str, Any]], edits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not edits:
        return original or []
    by_qid = {str(q.get("qid")): q for q in edits if q.get("qid") is not None}
    merged = []
    for q in original or []:
        qid = str(q.get("qid")) if q.get("qid") is not None else None
        if qid and qid in by_qid:
            merged.append(by_qid[qid])
        else:
            merged.append(q)
    # include any newly added edited questions without qid matches
    for e in edits:
        if e.get("qid") is None:
            merged.append(e)
    return merged

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ------------------------------------------------------------------------------
# Routes: UI
# ------------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    stats = load_stats()
    cfg_public = {
        "base_url": RFP_API_URL,
        "header_name": RFP_API_KEY_HEADER_NAME,
        "auth_scheme": RFP_API_AUTH_SCHEME,
        "poll_interval": POLL_INTERVAL_SECONDS,
        "poll_timeout": POLL_TIMEOUT_SECONDS,
        "max_upload_mb": MAX_CONTENT_LENGTH // (1024 * 1024),
        # hide key in UI
    }
    return render_template("index.html", api_config=cfg_public, stats=stats)

@app.route("/review/<job_id>", methods=["GET"])
def review(job_id: str):
    job = load_job(job_id)
    if not job:
        # Try fetching status once to hydrate cache
        try:
            r = rq_get(ep(f"/jobs/{job_id}"))
            if r.status_code == 200:
                job = r.json()
                job["job_id"] = job_id
                save_job(job_id, job)
        except Exception:
            pass
        if not job:
            return redirect(url_for("home"))

    orig = job.get("questions") or []
    edits = load_edits(job_id)
    questions = merge_questions(orig, edits)
    # For convenience, expose a few columns the templates might use
    return render_template("review.html", job_id=job_id, questions=questions, job=job)

# ------------------------------------------------------------------------------
# Routes: API glue for frontend JS (upload, poll, save, export, etc.)
# ------------------------------------------------------------------------------
@app.post("/api/upload")
def api_upload():
    """
    Upload a file and start extraction (sync or async).
    Form fields:
      - file: DOCX/PDF/XLSX
      - use_llm: "true" | "false"
      - use_sync: "true" | "false"
      - mode: "balanced"|"fast"|"thorough" (defaults to balanced)
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        f = request.files["file"]
        if f.filename == "":
            return jsonify({"error": "No file selected"}), 400
        if not allowed_file(f.filename):
            return jsonify({"error": f"Invalid file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"}), 400

        filename = secure_filename(f.filename)
        local_path = os.path.join(UPLOAD_DIR, filename)
        f.save(local_path)

        use_llm = str(request.form.get("use_llm", "false")).lower() == "true"
        use_sync = str(request.form.get("use_sync", "false")).lower() == "true"
        mode = request.form.get("mode") or "balanced"

        files = {"file": (filename, open(local_path, "rb"), "application/octet-stream")}
        data = {"use_llm": str(use_llm).lower(), "mode": mode}

        endpoint = "/extract/sync" if use_sync else "/extract/"
        print(f"[UPLOAD] → {endpoint} use_llm={use_llm} mode={mode} file={filename}")

        r = rq_post(ep(endpoint), files=files, data=data)
        try:
            files["file"][1].close()
        except Exception:
            pass
        try:
            os.remove(local_path)
        except Exception:
            pass

        if r is None:
            return jsonify({"error": "Extractor API unreachable"}), 502

        if r.status_code != 200:
            print(f"[UPLOAD] HTTP {r.status_code} → {r.text[:400]}")
            return jsonify({"error": f"HTTP {r.status_code}: {r.text}"}), 502

        resp = r.json()

        # Sync path returns questions directly
        if use_sync and isinstance(resp, dict) and "questions" in resp:
            job_id = f"sync_{int(time.time())}"
            job = {"status": "completed", "questions": resp.get("questions") or [], "job_id": job_id}
            save_job(job_id, job)
            update_stats_from_questions(job["questions"], processing_time=5.0)
            return jsonify({"success": True, "job_id": job_id, "questions": job["questions"], "async": False})

        # Async path returns job_id
        job_id = resp.get("job_id")
        if not job_id:
            return jsonify({"error": "No job_id returned by extractor"}), 502

        job = {"status": "processing", "job_id": job_id, "timestamp": datetime.now().isoformat()}
        save_job(job_id, job)
        return jsonify({"success": True, "job_id": job_id, "async": True})

    except Exception as e:
        print("[UPLOAD][ERROR]", e)
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.get("/api/poll/<job_id>")
def api_poll(job_id: str):
    """
    Poll job status. If completed, return questions (inline or fetched from /jobs/{id}/questions).
    """
    try:
        job = load_job(job_id) or {"job_id": job_id, "status": "processing"}
        r = rq_get(ep(f"/jobs/{job_id}"))

        if r is None:
            return jsonify({"error": "Extractor API unreachable"}), 502

        # If extractor has already GC'd the job, try direct questions fetch
        if r.status_code == 404:
            q = rq_get(ep(f"/jobs/{job_id}/questions"))
            if q is not None and q.status_code == 200:
                questions = q.json() or []
                job.update({"status": "completed", "questions": questions})
                save_job(job_id, job)
                update_stats_from_questions(questions, processing_time=30.0)
                return jsonify(job)
            # else fallback to soft-completed with empty questions
            job.update({"status": "completed", "questions": []})
            save_job(job_id, job)
            return jsonify(job)

        if r.status_code != 200:
            return jsonify({"error": f"HTTP {r.status_code}: {r.text}"}), 502

        status = r.json() or {}
        job.update(status)
        job["job_id"] = job_id

        # Completed with inline questions
        if job.get("status") == "completed":
            if "questions" in job and job["questions"] is not None:
                save_job(job_id, job)
                update_stats_from_questions(job["questions"], processing_time=30.0)
                return jsonify(job)
            # Completed but no inline questions → fetch separately
            q = rq_get(ep(f"/jobs/{job_id}/questions"))
            if q is not None and q.status_code == 200:
                job["questions"] = q.json() or []
                save_job(job_id, job)
                update_stats_from_questions(job["questions"], processing_time=30.0)
                return jsonify(job)
            # Completed, no questions found
            job["questions"] = []
            save_job(job_id, job)
            return jsonify(job)

        # Processing
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
    merged = merge_questions(orig, edits)
    return jsonify(merged)

@app.post("/api/save_edits/<job_id>")
def api_save_edits(job_id: str):
    try:
        payload = request.get_json(force=True, silent=False) or {}
        questions = payload.get("questions") or []
        # Basic normalization: ensure list of dicts
        if not isinstance(questions, list):
            return jsonify({"error": "questions must be a list"}), 400
        questions = [q for q in questions if isinstance(q, dict)]
        save_edits(job_id, questions)
        # also update the merged view for immediate consistency
        job = load_job(job_id) or {"job_id": job_id, "status": "completed", "questions": []}
        job["questions"] = merge_questions(job.get("questions") or [], questions)
        save_job(job_id, job)
        return jsonify({"success": True})
    except Exception as e:
        print("[SAVE_EDITS][ERROR]", e)
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# ------------------------------------------------------------------------------
# Export endpoints
# ------------------------------------------------------------------------------
def _questions_for_export(job_id: str) -> List[Dict[str, Any]]:
    job = load_job(job_id)
    if not job:
        return []
    orig = job.get("questions") or []
    edits = load_edits(job_id)
    return merge_questions(orig, edits)

@app.get("/api/export/<job_id>/json")
def export_json(job_id: str):
    rows = _questions_for_export(job_id)
    data = json.dumps(rows, indent=2, ensure_ascii=False)
    fname = f"RFP_Questions_{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    return send_file(io.BytesIO(data.encode("utf-8")), as_attachment=True,
                     download_name=fname, mimetype="application/json")

@app.get("/api/export/<job_id>/csv")
def export_csv(job_id: str):
    rows = _questions_for_export(job_id)
    if not rows:
        rows = []
    # pick union of keys
    keys = set()
    for r in rows:
        keys.update(r.keys())
    keys = list(sorted(keys))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=keys)
    w.writeheader()
    for r in rows:
        w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v) for k, v in r.items()})
    data = buf.getvalue().encode("utf-8")
    fname = f"RFP_Questions_{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return send_file(io.BytesIO(data), as_attachment=True,
                     download_name=fname, mimetype="text/csv")

@app.get("/api/export/<job_id>/xlsx")
def export_xlsx(job_id: str):
    try:
        import pandas as pd
    except Exception:
        return jsonify({"error": "pandas is required for xlsx export"}), 500

    rows = _questions_for_export(job_id)
    df = pd.json_normalize(rows) if rows else pd.DataFrame(columns=["qid","text","confidence","type","section_path"])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="RFP Questions", index=False)
    buf.seek(0)
    fname = f"RFP_Questions_{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return send_file(buf, as_attachment=True,
                     download_name=fname,
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

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

    for i, q in enumerate(rows, start=1):
        doc.add_heading(f"Question {i}", level=2)
        doc.add_paragraph(f"Text: {q.get('text','')}")
        # handle confidence as float
        conf = q.get("confidence")
        try:
            conf = f"{float(conf):.3f}"
        except Exception:
            conf = conf if conf is not None else ""
        doc.add_paragraph(f"Confidence: {conf}")
        doc.add_paragraph(f"Type: {q.get('type','')}")
        # section: try various fields commonly returned
        section = None
        if isinstance(q.get("section_path"), list) and q.get("section_path"):
            section = q["section_path"][0]
        elif "section" in q:
            section = q.get("section")
        elif "section_name" in q:
            section = q.get("section_name")
        doc.add_paragraph(f"Section: {section or ''}")
        doc.add_paragraph("")

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    fname = f"RFP_Questions_{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
    return send_file(buf, as_attachment=True,
                     download_name=fname,
                     mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

@app.get("/api/export/<job_id>/pdf")
def export_pdf(job_id: str):
    """
    Optional PDF export (requires reportlab). If not available, returns 501.
    """
    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import simpleSplit
    except Exception:
        return jsonify({"error": "PDF export requires 'reportlab'. Install it or use DOCX/XLSX/CSV/JSON."}), 501

    rows = _questions_for_export(job_id)
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    width, height = LETTER
    x, y = 50, height - 50

    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, "RFP Questions Report")
    y -= 20
    c.setFont("Helvetica", 10)
    c.drawString(x, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  Job ID: {job_id}  |  Total: {len(rows)}")
    y -= 20

    for i, q in enumerate(rows, start=1):
        if y < 80:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 10)

        c.setFont("Helvetica-Bold", 11)
        c.drawString(x, y, f"Question {i}")
        y -= 14
        c.setFont("Helvetica", 10)

        def draw_multiline(label: str, val: str):
            nonlocal y
            text = f"{label}: {val or ''}"
            lines = simpleSplit(text, "Helvetica", 10, width - 2 * x)
            for line in lines:
                if y < 60:
                    c.showPage()
                    y = height - 50
                    c.setFont("Helvetica", 10)
                c.drawString(x, y, line)
                y -= 12

        draw_multiline("Text", str(q.get("text", "")))
        conf = q.get("confidence")
        try:
            conf = f"{float(conf):.3f}"
        except Exception:
            conf = conf if conf is not None else ""
        draw_multiline("Confidence", str(conf))
        draw_multiline("Type", str(q.get("type", "")))

        section = None
        if isinstance(q.get("section_path"), list) and q.get("section_path"):
            section = q["section_path"][0]
        elif "section" in q:
            section = q.get("section")
        elif "section_name" in q:
            section = q.get("section_name")
        draw_multiline("Section", str(section or ""))

        y -= 6

    c.showPage()
    c.save()
    buf.seek(0)
    fname = f"RFP_Questions_{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(buf, as_attachment=True, download_name=fname, mimetype="application/pdf")

# ------------------------------------------------------------------------------
# Health & info (handy for Connect)
# ------------------------------------------------------------------------------
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
