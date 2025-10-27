# app_simple.py
import os
import json
import time
import logging
from typing import Tuple, Optional, Dict, Any

from flask import (
    Flask, render_template, request, jsonify, redirect, url_for
)
import requests
from dotenv import load_dotenv

# -------------------- Bootstrap --------------------
load_dotenv()

app = Flask(__name__, template_folder="templates")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("knockknock")

# -------------------- Config --------------------
API_BASE_URL = os.getenv("RFP_API_URL") or os.getenv("API_BASE_URL") or "http://localhost:8000"
API_KEY = os.getenv("RFP_API_KEY") or os.getenv("API_KEY") or ""
AUTH_NAME = os.getenv("RFP_API_KEY_HEADER_NAME") or os.getenv("AUTH_NAME") or "Authorization"
AUTH_SCHEME = (os.getenv("RFP_API_AUTH_SCHEME") or os.getenv("AUTH_SCHEME") or "Key").strip()
TIMEOUT = float(os.getenv("REQUEST_TIMEOUT_SECONDS") or os.getenv("TIMEOUT_SECONDS") or 60)

# Basic telemetry so you can see config at startup (token not printed)
logger.info("[CFG] API_BASE_URL=%s", API_BASE_URL)
if API_KEY:
    logger.info("[CFG] API_KEY=*** (loaded)")
else:
    logger.warning("[CFG] API_KEY is empty")
logger.info("[CFG] AUTH_NAME=%s AUTH_SCHEME=%s", AUTH_NAME, AUTH_SCHEME)
logger.info("[CFG] TIMEOUT=%.1fs", TIMEOUT)

# In-memory cache for sync responses that don't return a job_id
SYNC_RESULTS: Dict[str, Dict[str, Any]] = {}


def _auth_header() -> Dict[str, str]:
    """
    Compose the auth header value based on AUTH_SCHEME and AUTH_NAME, without leaking the token to logs.
    Supports: Raw (just the token), Key (default), Bearer, Api-Key.
    """
    token = (API_KEY or "").strip()
    if not token:
        return {}
    scheme = AUTH_SCHEME.lower()
    if scheme == "raw":
        value = token
    elif scheme == "bearer":
        value = f"Bearer {token}"
    elif scheme in ("key", "api-key", "api_key"):
        prefix = "Key" if scheme == "key" else "Api-Key"
        value = f"{prefix} {token}"
    else:
        # Fallback: "<scheme> <token>"
        value = f"{AUTH_SCHEME} {token}".strip()
    return {AUTH_NAME: value}


def _api(method: str, path: str, **kwargs) -> Tuple[Optional[requests.Response], Optional[str]]:
    """
    Tiny wrapper for requests to your extractor, adding auth + timeout and returning (response, error_msg).
    """
    url = API_BASE_URL.rstrip("/") + path
    headers = kwargs.pop("headers", {}) or {}
    headers.update(_auth_header())
    try:
        r = requests.request(method, url, headers=headers, timeout=TIMEOUT, **kwargs)
        logger.info("[API] %s %s -> %s", method, url, r.status_code)
        return r, None
    except requests.RequestException as e:
        logger.error("[API] %s %s failed: %s", method, url, e)
        return None, str(e)


def _safe_stats() -> Dict[str, Any]:
    """
    Provide stats to the template without breaking if a file is missing.
    If you keep a real stats.json, it will load it; otherwise default to zeros.
    """
    defaults = {
        "documents_processed": 0,
        "questions_extracted": 0,
        "accuracy_rate": 0,
        "avg_processing_time": 0
    }
    try:
        # Try in /templates first (repo layout), then project root.
        for p in ("templates/stats.json", "stats.json"):
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    defaults.update({k: data.get(k, defaults[k]) for k in defaults.keys()})
                    break
    except Exception as e:
        logger.warning("Failed to read stats.json: %s", e)
    return defaults


# -------------------- Routes --------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", stats=_safe_stats())


@app.route("/api/ping", methods=["GET"])
def ping():
    return jsonify(ok=True, now=int(time.time()))


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """
    Accepts: multipart form with fields:
      - file (required)
      - use_llm: "true"/"false"
      - use_sync: "true"/"false"
      - mode: e.g., "balanced"
    Proxies to:
      - POST /extract/ (async)
      - POST /extract/sync (sync)
    """
    f = request.files.get("file")
    if not f:
        return jsonify(success=False, error="No file provided"), 400

    use_llm = (request.form.get("use_llm", "true").lower() in ("1", "true", "yes"))
    use_sync = (request.form.get("use_sync", "false").lower() in ("1", "true", "yes"))
    mode = request.form.get("mode", "balanced")

    files = {"file": (f.filename, f.stream, f.mimetype or "application/octet-stream")}
    data = {
        "use_llm": "true" if use_llm else "false",
        "mode": mode
    }

    endpoint = "/extract/sync" if use_sync else "/extract/"
    resp, err = _api("POST", endpoint, files=files, data=data)
    if err or resp is None:
        return jsonify(success=False, error=f"Network error: {err or 'unknown'}"), 502

    # Try JSON first; if not JSON, surface raw text for debugging.
    payload: Dict[str, Any] = {}
    raw_text = ""
    content_type = resp.headers.get("content-type", "")
    try:
        if "application/json" in content_type:
            payload = resp.json()  # type: ignore[assignment]
        else:
            raw_text = (resp.text or "")[:1000]
    except Exception:
        raw_text = (resp.text or "")[:1000]

    if resp.status_code >= 300:
        # Bubble up extractor error text so the browser alert shows the real reason
        return jsonify(success=False,
                       error=f"Extractor responded {resp.status_code}",
                       detail=payload or raw_text), resp.status_code

    # If the extractor returns a job_id, use it
    job_id = payload.get("job_id")
    if job_id:
        return jsonify(success=True, async=(not use_sync), job_id=job_id)

    # Some sync implementations return the full questions instead of a job_id
    if payload.get("questions"):
        local_id = f"local_{int(time.time()*1000)}"
        SYNC_RESULTS[local_id] = payload
        return jsonify(success=True, async=False, job_id=local_id)

    # Fallback — treat as error with visible details
    return jsonify(success=False,
                   error="Unexpected extractor response",
                   detail=payload or raw_text), 502


@app.route("/api/poll/<job_id>", methods=["GET"])
def api_poll(job_id: str):
    # Local sync “job” — always completed
    if job_id.startswith("local_") and job_id in SYNC_RESULTS:
        return jsonify({
            "status": "completed",
            "progress": {"message": "completed (local)"},
            "job_id": job_id
        })

    resp, err = _api("GET", f"/jobs/{job_id}")
    if err or resp is None:
        return jsonify(status="failed", error=f"Network error: {err or 'unknown'}"), 502

    try:
        data = resp.json()
    except Exception:
        return jsonify(status="failed", error=f"Bad response: {resp.text[:500]}"), 502

    # Normalize minimal fields the front-end uses
    return jsonify({
        "status": data.get("status", "processing"),
        "progress": data.get("progress") or {"message": f"status={data.get('status','?')}"},
        "error": data.get("error"),
        "job_id": data.get("job_id", job_id)
    })


@app.route("/review/<job_id>", methods=["GET"])
def review(job_id: str):
    questions: Any = []
    summary: Any = None

    if job_id.startswith("local_") and job_id in SYNC_RESULTS:
        payload = SYNC_RESULTS.get(job_id) or {}
        questions = payload.get("questions") or []
        summary = payload.get("summary")
    else:
        # Pull questions from extractor
        resp, err = _api("GET", f"/jobs/{job_id}/questions")
        if err or resp is None:
            return render_template("error.html", message=f"Network error: {err or 'unknown'}"), 502
        try:
            questions = resp.json()
        except Exception:
            return render_template("error.html", message=f"Bad response: {resp.text[:500]}"), 502

    return render_template("review.html", job_id=job_id, questions=questions, summary=summary)


# Allow running locally
if __name__ == "__main__":
    # When testing locally, run on 0.0.0.0 so Posit tunnels (or you) can reach it
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=True)
