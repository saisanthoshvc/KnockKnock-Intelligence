# app_simple.py
import os, json, time, logging, uuid
from pathlib import Path
from typing import Dict, Any, Tuple
import requests
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("knockknock-ui")

# ---------- Flask ----------
app = Flask(__name__, template_folder="templates")

# ---------- Config ----------
def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return v if v is not None and str(v).strip() != "" else default

API_BASE_URL  = _env("RFP_API_URL", _env("API_BASE_URL"))
API_KEY       = _env("RFP_API_KEY", _env("API_TOKEN"))
AUTH_NAME     = _env("RFP_API_KEY_HEADER_NAME", _env("API_AUTH_NAME", "Authorization"))
AUTH_SCHEME   = _env("RFP_API_AUTH_SCHEME", _env("API_AUTH_SCHEME", "Key"))  # Key|Bearer|Token|Raw
TIMEOUT_SEC   = float(_env("REQUEST_TIMEOUT_SECONDS", "60"))
UPLOAD_DIR    = Path(_env("UPLOAD_FOLDER", "uploads"))
GEN_DIR       = Path(_env("GENERATED_FOLDER", "generated"))

# Make dirs writable on Posit
for d in (UPLOAD_DIR, GEN_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Masked config for logs
def _mask(s: str, keep: int = 4) -> str:
    if not s: return ""
    if len(s) <= keep: return "*" * len(s)
    return (s[:keep] + "••••" + s[-keep:])

logger.info("[CFG] API_BASE_URL=%s", API_BASE_URL or "<unset>")
logger.info("[CFG] API_KEY=%s", _mask(API_KEY))
logger.info("[CFG] AUTH_NAME=%s AUTH_SCHEME=%s", AUTH_NAME, AUTH_SCHEME)
logger.info("[CFG] TIMEOUT=%ss", TIMEOUT_SEC)

# ---------- Helpers ----------
def auth_headers() -> Dict[str, str]:
    """
    Compose the auth header. Examples:
      - Key    -> Authorization: Key <token>
      - Bearer -> Authorization: Bearer <token>
      - Token  -> Authorization: Token <token>
      - Raw    -> <AUTH_NAME>: <token>   (no scheme)
    """
    if not API_KEY:
        return {}
    scheme = AUTH_SCHEME.strip().lower()
    if scheme == "raw":
        return {AUTH_NAME: API_KEY}
    elif scheme in ("key", "bearer", "token"):
        proper = {"key":"Key", "bearer":"Bearer", "token":"Token"}[scheme]
        return {AUTH_NAME: f"{proper} {API_KEY}"}
    else:
        # default to Key
        return {AUTH_NAME: f"Key {API_KEY}"}

def guess_mime(filename: str) -> str:
    fn = filename.lower()
    if fn.endswith(".docx"):
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if fn.endswith(".pdf"):
        return "application/pdf"
    if fn.endswith(".xlsx"):
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return "application/octet-stream"

def load_local_stats() -> Dict[str, Any]:
    """
    Load your pretty-dashboard counters so index.html can render without blowing up.
    Looks for templates/stats.json then ./stats.json. Falls back to zeros.
    """
    default_stats = {
        "documents_processed": 0,
        "questions_extracted": 0,
        "avg_confidence": 0.0,
        "uptime_hours": 0
    }
    for p in (Path("templates")/"stats.json", Path("stats.json")):
        try:
            if p.exists():
                with p.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return {**default_stats, **data}
        except Exception as e:
            logger.warning("Failed to load %s: %s", p, e)
    return default_stats

def check_health() -> Tuple[bool, Dict[str, Any]]:
    """
    Try /health/ or /health/ready or /info. Non-fatal if unreachable.
    """
    if not API_BASE_URL:
        return False, {"reason": "no_api_url"}
    for path in ("/health/", "/health/ready", "/info"):
        url = API_BASE_URL.rstrip("/") + path
        try:
            r = requests.get(url, headers=auth_headers(), timeout=TIMEOUT_SEC)
            logger.info("[HTTP] GET %s -> %s", url, r.status_code)
            if r.ok:
                payload = {}
                try:
                    payload = r.json()
                except Exception:
                    payload = {"text": (r.text or "")[:500]}
                return True, payload
        except Exception as e:
            logger.warning("Health check failed %s: %s", url, e)
    return False, {"reason": "unreachable"}

def safe_render(template: str, **ctx):
    """
    Render Jinja, but never crash the app. If the template raises, return a minimal page.
    With this build we also pre-populate ctx so your fancy template should render.
    """
    # Ensure the variables your template expects are present
    ctx.setdefault("stats", load_local_stats())
    online, info = check_health()
    ctx.setdefault("system_online", online)
    ctx.setdefault("api_info", info)
    ctx.setdefault("api_base_url", API_BASE_URL)
    ctx.setdefault("auth_name", AUTH_NAME)
    ctx.setdefault("auth_scheme", AUTH_SCHEME)

    try:
        return render_template(template, **ctx)
    except Exception as e:
        logger.error("Template render failed; falling back: %s", e, exc_info=True)
        # Minimal fallback so the app still works if template has strict refs
        html = f"""
        <!doctype html>
        <html><head><meta charset="utf-8"><title>KnockKnock Intelligence</title></head>
        <body style="font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; padding:24px">
          <h1>KnockKnock Intelligence</h1>
          <p><b>System online:</b> {'✅' if online else '❌'}</p>
          <form action="{url_for('upload')}" method="post" enctype="multipart/form-data">
            <p><input type="file" name="file" required></p>
            <label><input type="checkbox" name="use_llm" checked> Use LLM?</label>
            <label> Mode:
              <select name="mode">
                <option value="balanced" selected>balanced</option>
                <option value="fast">fast</option>
                <option value="max">max</option>
              </select>
            </label>
            <p><button type="submit">Upload & Extract</button></p>
          </form>
        </body></html>
        """
        return html

# ---------- Routes ----------
@app.get("/healthz")
def healthz():
    online, payload = check_health()
    return jsonify({"online": online, "details": payload})

@app.get("/_debug")
def debug():
    return jsonify({
        "api_base_url": API_BASE_URL,
        "auth_name": AUTH_NAME,
        "auth_scheme": AUTH_SCHEME,
        "has_api_key": bool(API_KEY),
        "upload_dir": str(UPLOAD_DIR.resolve()),
        "generated_dir": str(GEN_DIR.resolve()),
        "stats": load_local_stats()
    })

@app.get("/")
def index():
    # Render your nice template with guaranteed defaults
    return safe_render("index.html")

@app.post("/upload")
def upload():
    file = request.files.get("file")
    if not file or file.filename == "":
        return safe_render("error.html", message="No file provided.")

    # form fields
    use_llm = "true" if request.form.get("use_llm") in ("on", "true", "1") else "false"
    mode = request.form.get("mode", "balanced")
    if mode not in ("fast", "balanced", "max"):
        mode = "balanced"

    # Save locally (helps debugging)
    ext = Path(file.filename).suffix
    safe_name = f"{uuid.uuid4().hex}{ext}"
    save_path = UPLOAD_DIR / safe_name
    file.stream.seek(0)
    file.save(save_path)

    if not API_BASE_URL:
        return safe_render("error.html", message="API base URL is not configured.")

    url = API_BASE_URL.rstrip("/") + "/extract/"
    headers = auth_headers()
    files = {"file": (file.filename, open(save_path, "rb"), guess_mime(file.filename))}
    data = {"use_llm": use_llm, "mode": mode}

    try:
        r = requests.post(url, headers=headers, files=files, data=data, timeout=TIMEOUT_SEC)
        logger.info("[HTTP] POST %s -> %s", url, r.status_code)
    finally:
        try:
            files["file"][1].close()
        except Exception:
            pass

    if not r.ok:
        # Show a meaningful message from extractor (common cause: auth header mismatch)
        preview = (r.text or "")[:400]
        return safe_render("error.html",
                           message=f"Extractor responded {r.status_code}",
                           detail=preview)

    resp = {}
    try:
        resp = r.json()
    except Exception:
        return safe_render("error.html", message="Extractor returned non-JSON response.", detail=(r.text or "")[:400])

    job_id = resp.get("job_id")
    if not job_id:
        return safe_render("error.html", message="No job_id returned by extractor.", detail=json.dumps(resp))

    # Poll for completion
    job_url = API_BASE_URL.rstrip("/") + f"/jobs/{job_id}"
    deadline = time.time() + max(TIMEOUT_SEC, 60)
    status = "pending"
    last_payload = {}

    while time.time() < deadline:
        try:
            jr = requests.get(job_url, headers=headers, timeout=TIMEOUT_SEC)
            logger.info("[HTTP] GET %s -> %s", job_url, jr.status_code)
            if not jr.ok:
                break
            last_payload = jr.json()
            status = str(last_payload.get("status", "")).lower()
            if status in ("completed", "failed"):
                break
        except Exception as e:
            logger.warning("Polling error: %s", e)
            break
        time.sleep(2)

    if status != "completed":
        return safe_render("error.html",
                           message=f"Job did not complete (status={status}).",
                           detail=json.dumps(last_payload)[:800])

    # Fetch questions
    q_url = job_url + "/questions"
    try:
        qr = requests.get(q_url, headers=headers, timeout=TIMEOUT_SEC)
        logger.info("[HTTP] GET %s -> %s", q_url, qr.status_code)
        if not qr.ok:
            return safe_render("error.html",
                               message=f"Failed to fetch questions ({qr.status_code}).",
                               detail=(qr.text or "")[:400])
        questions = qr.json() or []
    except Exception as e:
        return safe_render("error.html", message=f"Error fetching questions: {e}")

    summary = last_payload.get("summary") or {}
    return safe_render("review.html",
                       questions=questions,
                       summary=summary,
                       job_id=job_id,
                       stats=load_local_stats(),
                       system_online=True)

# -------- Gunicorn entrypoint --------
# Posit uses `app_simple:app` as entrypoint
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
