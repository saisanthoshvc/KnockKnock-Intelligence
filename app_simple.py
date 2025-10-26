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
from werkzeug.middleware.proxy_fix import ProxyFix

# ============================== Flask Setup ==============================
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")

# Make the app path-aware behind proxies (Posit Connect)
# This lets Flask pick up X-Forwarded-Prefix so request.script_root is correct
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
ALLOWED_EXTENSIONS = {"docx", "pdf", "xlsx"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024 # 16MB

# ============================ API Configuration ===========================
def _env_float(name: str, default: float) -> float:
v = os.getenv(name, "").strip()
try:
return float(v) if v else default
except Exception:
return default

API_BASE_URL = (os.getenv(
"RFP_API_URL",
"https://connect.affiniusaiplatform.com/content/000d3572-937a-4d5c-ab7e-7ec80d80c4ce/"
).strip().rstrip("/"))

API_KEY = os.getenv("RFP_API_KEY", "").strip()

POLL_INTERVAL = _env_float("POLL_INTERVAL_SECONDS", 2.0)
POLL_TIMEOUT = _env_float("POLL_TIMEOUT_SECONDS", 240.0)
REQUEST_TIMEOUT = _env_float("REQUEST_TIMEOUT_SECONDS", 60.0)

# In-memory job store
job_data: Dict[str, dict] = {}
job_edits: Dict[str, List[dict]] = {}

# Stats
STATS_FILE = os.getenv("STATS_FILE", "stats.json")

app.run(host="0.0.0.0", port=5002, debug=True)
