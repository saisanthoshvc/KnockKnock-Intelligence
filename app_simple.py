
# app_simple.py
import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Tuple

from flask import Flask, request, render_template, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import requests

################################
# Logging to stdout for Connect
################################
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("kk-ui")

################################
# Config helpers
################################
def _get_env(name: str, default: str = "") -> str:
    v = os.environ.get(name, default)
    if isinstance(v, str):
        return v.strip()
    return v

API_BASE_URL = _get_env("RFP_API_URL", _get_env("API_BASE_URL", "")).rstrip("/")
AUTH_NAME = _get_env("RFP_API_KEY_HEADER_NAME", _get_env("AUTH_NAME", "Authorization"))
AUTH_SCHEME = _get_env("RFP_API_AUTH_SCHEME", _get_env("AUTH_SCHEME", "Raw"))
API_KEY = _get_env("RFP_API_KEY", _get_env("API_KEY", ""))
TIMEOUT = float(_get_env("REQUEST_TIMEOUT_SECONDS", "60") or "60")

UPLOAD_DIR = _get_env("UPLOAD_FOLDER", "uploads")
GENERATED_DIR = _get_env("GENERATED_FOLDER", "generated")

ALLOWED_EXTENSIONS = {".docx", ".pdf", ".xlsx"}

################################
# Safe filesystem setup
################################
def _safe_mkdir(p: str):
    try:
        os.makedirs(p, exist_ok=True)
        testfile = os.path.join(p, ".write_test")
        with open(testfile, "w") as f:
            f.write(datetime.utcnow().isoformat())
        os.remove(testfile)
        return True, ""
    except Exception as e:
        return False, str(e)

################################
# Auth header builder
################################
def build_auth_header(key: str, name: str, scheme: str) -> Dict[str, str]:
    """Build the outbound auth header according to scheme."""
    if not key:
        return {}
    s = (scheme or "").strip().lower()
    if s == "raw":
        return {name: key}
    if s == "key":
        return {name: f"Key {key}"}
    if s == "bearer":
        return {name: f"Bearer {key}"}
    if s == "token":
        return {name: f"Token {key}"}
    return {name: key}

def _mask(v: str, show: int = 4) -> str:
    if not v:
        return ""
    if len(v) <= show:
        return "*" * len(v)
    return f"{v[:show]}{'•'*(max(0,len(v)-2*show))}{v[-show:]}"

################################
# App factory
################################
def create_app() -> Flask:
    ok_u, err_u = _safe_mkdir(UPLOAD_DIR)
    ok_g, err_g = _safe_mkdir(GENERATED_DIR)

    app = Flask(__name__, template_folder="templates", static_folder=None)

    # Boot diagnostics
    log.info("[BOOT] API_BASE_URL=%s", API_BASE_URL or "<empty>")
    log.info("[BOOT] AUTH_NAME=%s AUTH_SCHEME=%s", AUTH_NAME, AUTH_SCHEME)
    log.info("[BOOT] API_KEY=%s", _mask(API_KEY))
    log.info("[BOOT] TIMEOUT=%ss", TIMEOUT)
    log.info("[BOOT] UPLOAD_DIR=%s (ok=%s err=%s)", UPLOAD_DIR, ok_u, err_u)
    log.info("[BOOT] GENERATED_DIR=%s (ok=%s err=%s)", GENERATED_DIR, ok_g, err_g)

    # Health & debug
    @app.get("/healthz")
    def healthz():
        return jsonify(status="ok", time=datetime.utcnow().isoformat())

    @app.get("/_debug")
    def debug_info():
        d = {
            "api_base_url": API_BASE_URL,
            "auth_name": AUTH_NAME,
            "auth_scheme": AUTH_SCHEME,
            "key_preview": _mask(API_KEY),
            "timeout": TIMEOUT,
            "cwd": os.getcwd(),
            "files": os.listdir("."),
            "templates_exists": os.path.isdir("templates"),
            "upload_dir": UPLOAD_DIR,
            "generated_dir": GENERATED_DIR,
        }
        return jsonify(d)

    # Safe render that falls back if templates are missing or error
    def _render(template, **ctx):
        try:
            if os.path.isdir("templates") and os.path.isfile(os.path.join("templates", template)):
                return render_template(template, **ctx)
        except Exception as e:
            log.exception("Template render failed; falling back: %s", e)
        # Fallback minimal HTML
        if template == "index.html":
            return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>KnockKnock Intelligence</title></head>
<body>
  <h1>KnockKnock Intelligence</h1>
  <form method="post" action="/extract" enctype="multipart/form-data">
    <p><input type="file" name="file" required /></p>
    <p>
      <label>Use LLM? <input type="checkbox" name="use_llm" value="true" checked></label>
      <label>Mode:
        <select name="mode">
          <option value="balanced" selected>balanced</option>
          <option value="fast">fast</option>
          <option value="deep">deep</option>
        </select>
      </label>
    </p>
    <button type="submit">Upload & Extract</button>
  </form>
</body></html>"""
        if template == "error.html":
            message = ctx.get("message","Unknown error")
            return f"""<!doctype html><html><body>
<h3>Error</h3><pre>{message}</pre>
<p><a href="/">Back</a></p>
</body></html>"""
        if template == "review.html":
            payload_json = json.dumps(ctx.get("payload", {}))[:5000]
            return f"""<!doctype html><html><body>
<h3>Result</h3><pre>{payload_json}</pre>
<p><a href="/">Back</a></p>
</body></html>"""
        return f"<pre>Missing template: {template}</pre>", 200, {"Content-Type": "text/html"}

    @app.get("/")
    def index():
        return _render("index.html")

    # Helpers
    def _allowed(filename: str) -> bool:
        _, ext = os.path.splitext(filename or "")
        return ext.lower() in ALLOWED_EXTENSIONS

    def _ensure_api_ready() -> Tuple[bool, str]:
        if not API_BASE_URL:
            return False, "RFP_API_URL (or API_BASE_URL) is not set"
        try:
            r = requests.get(f"{API_BASE_URL}/health/ready", timeout=TIMEOUT)
            if r.status_code >= 500:
                return False, f"Extractor unhealthy: {r.status_code} {r.text[:200]}"
        except Exception as e:
            log.warning("Extractor health check failed: %s", e)
        return True, ""

    # Routes
    @app.post("/extract")
    def extract():
        if "file" not in request.files:
            return _render("error.html", message="No file part in the request")
        f = request.files["file"]
        if not f or not f.filename:
            return _render("error.html", message="No selected file")
        if not _allowed(f.filename):
            return _render("error.html", message="Unsupported file type; allowed: .docx, .pdf, .xlsx")

        ok, why = _ensure_api_ready()
        if not ok:
            return _render("error.html", message=f"Config/Extractor not ready: {why}")

        filename = secure_filename(f.filename)
        local_path = os.path.join(UPLOAD_DIR, filename)
        try:
            f.save(local_path)
        except Exception as e:
            log.exception("Failed to save upload: %s", e)
            return _render("error.html", message=f"Failed to save upload: {e}")

        use_llm = request.form.get("use_llm", "true")
        use_llm_str = "true" if str(use_llm).lower() in ("1", "true", "on", "yes") else "false"
        mode = request.form.get("mode", "balanced") or "balanced"

        files = {"file": (filename, open(local_path, "rb"))}
        data = {"use_llm": use_llm_str, "mode": mode}
        headers = build_auth_header(API_KEY, AUTH_NAME, AUTH_SCHEME)

        try:
            url = f"{API_BASE_URL}/extract/"
            log.info("[HTTP] POST %s data=%s headers=%s", url, data, {k: ("***" if k.lower() == AUTH_NAME.lower() else v) for k, v in headers.items()})
            resp = requests.post(url, files=files, data=data, headers=headers, timeout=TIMEOUT)
            text_preview = (resp.text or "")[:500]
            log.info("[HTTP] -> %s %s", resp.status_code, text_preview.replace(API_KEY, "***"))
        finally:
            try:
                files["file"][1].close()
            except Exception:
                pass

        if resp.status_code >= 400:
            try:
                payload = resp.json()
            except Exception:
                payload = {"error": "HTTPError", "message": (resp.text or "")[:500]}
            return _render("error.html", message=f"Extractor error ({resp.status_code}): {json.dumps(payload)[:800]}")

        try:
            payload = resp.json()
        except Exception as e:
            return _render("error.html", message=f"Invalid JSON from extractor: {e} :: {text_preview}")

        job_id = payload.get("job_id")
        if not job_id:
            return _render("error.html", message=f"Extractor response missing job_id: {json.dumps(payload)[:800]}")

        return redirect(url_for("job_status", job_id=job_id))

    @app.get("/jobs/<job_id>")
    def job_status(job_id: str):
        headers = build_auth_header(API_KEY, AUTH_NAME, AUTH_SCHEME)
        url = f"{API_BASE_URL}/jobs/{job_id}"
        try:
            r = requests.get(url, headers=headers, timeout=TIMEOUT)
            log.info("[HTTP] GET %s -> %s %s", url, r.status_code, (r.text or "")[:200])
        except Exception as e:
            return _render("error.html", message=f"Failed to reach extractor: {e}")

        if r.status_code >= 400:
            return _render("error.html", message=f"Extractor returned {r.status_code}: {(r.text or '')[:500]}")

        try:
            payload = r.json()
        except Exception as e:
            return _render("error.html", message=f"Invalid JSON from extractor: {e} :: {(r.text or '')[:500]}")

        status = (payload or {}).get("status", "unknown")
        if status == "completed":
            return redirect(url_for("job_questions", job_id=job_id))
        if status in ("pending", "processing"):
            # Auto refresh simple page
            html = f"""<!doctype html><html><body>
<h3>Job {job_id}</h3>
<p>Status: {status}</p>
<meta http-equiv="refresh" content="2">
<p><a href="{url_for('job_status', job_id=job_id)}">Refresh</a></p>
</body></html>"""
            return html
        # failed or unknown
        return _render("error.html", message=f"Job status: {status} :: {json.dumps(payload)[:500]}")

    @app.get("/jobs/<job_id>/questions")
    def job_questions(job_id: str):
        headers = build_auth_header(API_KEY, AUTH_NAME, AUTH_SCHEME)
        url = f"{API_BASE_URL}/jobs/{job_id}/questions"
        try:
            r = requests.get(url, headers=headers, timeout=TIMEOUT)
            log.info("[HTTP] GET %s -> %s %s", url, r.status_code, (r.text or "")[:200])
        except Exception as e:
            return _render("error.html", message=f"Failed to reach extractor: {e}")

        if r.status_code >= 400:
            return _render("error.html", message=f"Extractor returned {r.status_code}: {(r.text or '')[:500]}")

        try:
            payload = r.json()
        except Exception as e:
            return _render("error.html", message=f"Invalid JSON from extractor: {e} :: {(r.text or '')[:500]}")

        return _render("review.html", payload=payload)

    return app

# WSGI entrypoint for Posit Connect (and useful for local dev)
app = create_app()

if __name__ == "__main__":
    # For local testing only; Connect runs gunicorn for you
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=True)
