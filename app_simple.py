# app_simple.py
import os
import io
import json
import time
import threading
import logging
from datetime import datetime, timedelta

import requests
from flask import (
    Flask, render_template, request, jsonify,
    redirect, url_for, send_file
)
from werkzeug.utils import secure_filename

# ------------------------------------------------------------------------------
# App & logging
# ------------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key-here')

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
log = logging.getLogger("knockknock.app")
if not log.handlers:
    _h = logging.StreamHandler()
    _f = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    _h.setFormatter(_f)
    log.addHandler(_h)
log.setLevel(LOG_LEVEL)

def log_event(event: str, **fields):
    """Emit a single-line JSON log (easy to read in Posit Connect logs)."""
    safe_fields = {}
    for k, v in fields.items():
        # avoid dumping very long blobs
        if isinstance(v, (str, bytes)) and len(str(v)) > 500:
            safe_fields[k] = str(v)[:500] + "...[truncated]"
        else:
            safe_fields[k] = v
    log.info(json.dumps({"event": event, **safe_fields}))

APP_VERSION = os.environ.get("APP_VERSION", "1.0.0")

# ------------------------------------------------------------------------------
# Paths (use /tmp on Posit Connect)
# ------------------------------------------------------------------------------
UPLOAD_FOLDER = os.environ.get('UPLOAD_DIR', '/tmp/knockknock_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

STATS_FILE = os.environ.get('STATS_FILE', '/tmp/knockknock_stats.json')

# ------------------------------------------------------------------------------
# Allowed uploads
# ------------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {'docx', 'pdf', 'xlsx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get("MAX_UPLOAD_MB", "16")) * 1024 * 1024

# ------------------------------------------------------------------------------
# API Config (env-first)
# ------------------------------------------------------------------------------
def env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default

DEFAULT_API_CONFIG = {
    'base_url': os.environ.get(
        'RFP_API_URL',
        'https://connect.affiniusaiplatform.com/content/000d3572-937a-4d5c-ab7e-7ec80d80c4ce/'
    ),
    'api_key': os.environ.get('RFP_API_KEY', ''),                        # your Connect API key if auth required
    'header_name': os.environ.get('RFP_API_KEY_HEADER_NAME', 'Authorization'),
    'auth_scheme': os.environ.get('RFP_API_AUTH_SCHEME', 'Key'),         # Key | Bearer | Raw
    'poll_interval': env_int('POLL_INTERVAL_SECONDS', 2),
    'poll_timeout': env_int('POLL_TIMEOUT_SECONDS', 240),
    'request_timeout': env_int('REQUEST_TIMEOUT_SECONDS', 60),
}
api_config = DEFAULT_API_CONFIG.copy()

# ------------------------------------------------------------------------------
# In-memory state
# ------------------------------------------------------------------------------
job_data = {}
job_edits = {}

# ------------------------------------------------------------------------------
# Stats helpers
# ------------------------------------------------------------------------------
def load_stats():
    try:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE, 'r') as f:
                stats = json.load(f)
            docs = stats.get("documents_processed", 0)
            stats["avg_processing_time"] = round(
                stats.get("total_processing_time", 0) / docs, 1
            ) if docs else 0.0
            return stats
    except Exception as e:
        log_event("STATS_LOAD_ERROR", error=str(e))
    return {
        "documents_processed": 0,
        "questions_extracted": 0,
        "total_processing_time": 0,
        "avg_processing_time": 0.0,
        "accuracy_rate": 0.0,
        "last_updated": datetime.now().isoformat()
    }

def save_stats(stats):
    try:
        stats["last_updated"] = datetime.now().isoformat()
        with open(STATS_FILE, 'w') as f:
            json.dump(stats, f, indent=2)
    except Exception as e:
        log_event("STATS_SAVE_ERROR", error=str(e))

def update_stats(questions_count, processing_time, questions_data=None):
    stats = load_stats()
    stats["documents_processed"] += 1
    stats["questions_extracted"] += questions_count
    stats["total_processing_time"] += processing_time

    # compute avg confidence when available
    if questions_count > 0 and questions_data:
        total_conf = 0.0
        n = 0
        for q in questions_data:
            c = q.get('confidence')
            if isinstance(c, (int, float)):
                total_conf += float(c)
                n += 1
        if n:
            stats["accuracy_rate"] = round((total_conf / n) * 100, 1)
        else:
            stats["accuracy_rate"] = max(stats.get("accuracy_rate", 0.0), 85.0)
    else:
        stats["accuracy_rate"] = max(stats.get("accuracy_rate", 0.0), 85.0)

    save_stats(stats)
    log_event("STATS_UPDATED", documents_processed=stats["documents_processed"],
              questions_extracted=stats["questions_extracted"],
              avg_processing_time=stats["avg_processing_time"],
              accuracy_rate=stats["accuracy_rate"])
    return stats

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_auth_header():
    """
    Build auth header according to env:
      - Authorization: Key <token>     (Posit Connect programmatic access)
      - Authorization: Bearer <token>  (if your service expects bearer)
      - <custom-name>: <token>         (Raw custom header)
    """
    hdr = api_config['header_name']
    scheme = (api_config['auth_scheme'] or '').lower()
    token = api_config['api_key']

    if not token:
        # allow unauthenticated if target is public (login not required)
        return {}

    if hdr.lower() == 'authorization':
        if scheme == 'key':
            return {'Authorization': f'Key {token}'}
        if scheme == 'bearer':
            return {'Authorization': f'Bearer {token}'}
        if scheme == 'raw':
            return {'Authorization': token}
        # default
        return {'Authorization': token}
    else:
        if scheme == 'raw':
            return {hdr: token}
        if scheme in ('key', 'bearer'):
            return {hdr: f'{scheme.title()} {token}'}
        return {hdr: token}

def make_api_request(method, endpoint, **kwargs):
    url = f"{api_config['base_url'].rstrip('/')}/{endpoint.lstrip('/')}"
    headers = get_auth_header()
    headers.update(kwargs.pop('headers', {}))

    try:
        log_event("API_REQUEST_OUT", method=method.upper(), url=url,
                  header_name=list(headers.keys())[0] if headers else None)
        if method.upper() == 'GET':
            resp = requests.get(url, headers=headers, timeout=api_config['request_timeout'])
        elif method.upper() == 'POST':
            resp = requests.post(url, headers=headers, timeout=api_config['request_timeout'], **kwargs)
        elif method.upper() == 'DELETE':
            resp = requests.delete(url, headers=headers, timeout=api_config['request_timeout'])
        else:
            return None

        # Detect accidental redirect to login page (auth failure)
        ct = resp.headers.get('Content-Type', '')
        if 'text/html' in ct:
            txt_low = resp.text.lower()
            if '__login__' in txt_low or '<title>sign in' in txt_low or '<title>login' in txt_low:
                log_event("API_AUTH_HTML_LOGIN", url=url, code=resp.status_code)
                class Dummy:
                    status_code = 401
                    text = "Not authorized: request appears redirected to a login page. Check RFP_API_* env vars."
                    def json(self): return {"error": self.text}
                return Dummy()

        log_event("API_REQUEST_DONE", url=url, status_code=resp.status_code)
        return resp

    except requests.exceptions.RequestException as e:
        log_event("API_REQUEST_ERROR", url=url, error=str(e))
        return None

# ------------------------------------------------------------------------------
# Flask 3-safe: run-once startup check
# ------------------------------------------------------------------------------
_initialized = False
_init_lock = threading.Lock()

def _startup_check():
    # Redact token presence only
    log_event("APP_START",
              version=APP_VERSION,
              upload_dir=UPLOAD_FOLDER,
              stats_file=STATS_FILE,
              api_base=api_config['base_url'],
              header_name=api_config['header_name'],
              auth_scheme=api_config['auth_scheme'],
              have_api_key=bool(api_config['api_key']))

    # Try a couple of health/info endpoints; non-fatal if unavailable
    for ep in ("health/ready", "health/", "info"):
        try:
            resp = make_api_request("GET", ep)
            if resp and resp.status_code == 200:
                log_event("UPSTREAM_HEALTH_OK", endpoint=ep)
                break
            else:
                code = resp.status_code if resp else "NO_RESPONSE"
                log_event("UPSTREAM_HEALTH_FAIL", endpoint=ep, status=code)
        except Exception as e:
            log_event("UPSTREAM_HEALTH_EXC", endpoint=ep, error=str(e))

@app.before_request
def _ensure_startup_once():
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if not _initialized:
            _startup_check()
            _initialized = True

# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------
@app.route('/')
def index():
    stats = load_stats()
    log_event("ROUTE_INDEX")
    return render_template('index.html', api_config=api_config, stats=stats)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    log_event("UPLOAD_START", form_keys=list(request.form.keys()), files_present='file' in request.files)

    if 'file' not in request.files:
        log_event("UPLOAD_NO_FILE")
        return jsonify({'error': 'No file provided'}), 400

    f = request.files['file']
    if f.filename == '':
        log_event("UPLOAD_EMPTY_FILENAME")
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(f.filename):
        log_event("UPLOAD_INVALID_TYPE", filename=f.filename)
        return jsonify({'error': 'Invalid file type'}), 400

    filename = secure_filename(f.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(filepath)
    log_event("UPLOAD_SAVED", filename=filename, path=filepath, size=os.path.getsize(filepath))

    # Coerce booleans to strings 'true'/'false' for FastAPI form parsing
    use_llm = request.form.get('use_llm', 'false').lower() in ('true', '1', 'yes', 'on')
    use_sync = request.form.get('use_sync', 'false').lower() in ('true', '1', 'yes', 'on')
    mode = request.form.get('mode', 'balanced')

    try:
        with open(filepath, 'rb') as doc:
            files = {'file': doc}
            data = {'use_llm': 'true' if use_llm else 'false', 'mode': mode}
            endpoint = 'extract/sync' if use_sync else 'extract/'
            log_event("EXTRACT_CALL", endpoint=endpoint, use_llm=data['use_llm'], mode=mode, use_sync=use_sync)
            resp = make_api_request('POST', endpoint, files=files, data=data)

        if not resp:
            log_event("EXTRACT_NO_RESPONSE")
            return jsonify({'error': 'API request failed (no response)'}), 502

        if resp.status_code == 200:
            payload = resp.json()
            log_event("EXTRACT_OK", keys=list(payload.keys()))
            if use_sync and isinstance(payload, dict) and 'questions' in payload:
                job_id = f"sync_{int(time.time())}"
                job_data[job_id] = {
                    'status': 'completed',
                    'questions': payload['questions'],
                    'timestamp': datetime.now().isoformat()
                }
                update_stats(len(payload['questions']), 5, payload['questions'])
                return jsonify({'success': True, 'job_id': job_id, 'questions': payload['questions']})
            else:
                job_id = payload.get('job_id')
                if job_id:
                    job_data[job_id] = {'status': 'processing', 'timestamp': datetime.now().isoformat()}
                    log_event("JOB_STARTED", job_id=job_id)
                    return jsonify({'success': True, 'job_id': job_id, 'async': True})
                log_event("EXTRACT_MISSING_JOB_ID")
                return jsonify({'error': 'No job ID returned from API'}), 502

        # Surface upstream error body
        try:
            msg = resp.json()
        except Exception:
            msg = resp.text
        log_event("EXTRACT_HTTP_ERROR", status=resp.status_code, body=msg)
        return jsonify({'error': f"HTTP {resp.status_code}: {msg}"}), resp.status_code

    except Exception as e:
        log_event("UPLOAD_EXC", error=str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                log_event("UPLOAD_CLEANED", filename=filename)
        except Exception as e:
            log_event("UPLOAD_CLEAN_ERROR", error=str(e))

@app.route('/api/poll/<job_id>')
def poll_job(job_id):
    log_event("POLL_START", job_id=job_id)

    if job_id not in job_data:
        log_event("POLL_NO_JOB", job_id=job_id)
        return jsonify({'error': 'Job not found'}), 404

    if job_data[job_id].get('status') == 'completed':
        log_event("POLL_ALREADY_COMPLETED", job_id=job_id)
        return jsonify(job_data[job_id])

    resp = make_api_request('GET', f'jobs/{job_id}')
    if not resp:
        log_event("POLL_NO_RESPONSE", job_id=job_id)
        return jsonify({'error': 'Failed to poll job status'}), 502

    if resp.status_code == 404:
        log_event("POLL_404_TRY_QUESTIONS", job_id=job_id)
        q = make_api_request('GET', f'jobs/{job_id}/questions')
        if q and q.status_code == 200:
            questions = q.json()
            job_data[job_id]['status'] = 'completed'
            job_data[job_id]['questions'] = questions
            update_stats(len(questions), 30, questions)
            log_event("POLL_COMPLETED_VIA_QUESTIONS", job_id=job_id, count=len(questions))
            return jsonify({'status': 'completed', 'questions': questions})
        return jsonify({'status': 'completed', 'questions': []})

    if resp.status_code != 200:
        try:
            msg = resp.json()
        except Exception:
            msg = resp.text
        log_event("POLL_HTTP_ERROR", job_id=job_id, status=resp.status_code, body=msg)
        return jsonify({'error': f"HTTP {resp.status_code}: {msg}"}), resp.status_code

    data = resp.json()
    job_data[job_id].update(data)
    status = data.get('status', 'processing')
    log_event("POLL_STATUS", job_id=job_id, status=status)

    if status == 'completed':
        if data.get('questions') is not None:
            qs = data['questions']
            update_stats(len(qs), 30, qs)
            log_event("POLL_COMPLETED_INLINE", job_id=job_id, count=len(qs))
            return jsonify(data)
        # completed but no inline questions
        q = make_api_request('GET', f'jobs/{job_id}/questions')
        if q and q.status_code == 200:
            qs = q.json()
            job_data[job_id]['questions'] = qs
            update_stats(len(qs), 30, qs)
            data['questions'] = qs
            log_event("POLL_COMPLETED_FETCHED", job_id=job_id, count=len(qs))
            return jsonify(data)
        log_event("POLL_COMPLETED_NO_QUESTIONS", job_id=job_id)
        return jsonify({'status': 'completed', 'questions': []})

    if status == 'processing':
        start = job_data[job_id].get('timestamp')
        if start:
            try:
                started = datetime.fromisoformat(start)
                if datetime.now() - started > timedelta(minutes=10):
                    job_data[job_id]['status'] = 'timeout'
                    log_event("POLL_TIMEOUT", job_id=job_id)
                    return jsonify({'status': 'timeout', 'error': 'Processing timeout; try sync or no-LLM'})
            except Exception as e:
                log_event("POLL_TS_PARSE_ERROR", error=str(e))
        return jsonify({'status': 'processing', 'progress': data.get('progress')})

    if status == 'failed':
        log_event("POLL_FAILED", job_id=job_id)
        return jsonify(data)

    return jsonify({'status': 'processing'})

@app.route('/api/job/<job_id>/questions')
def get_job_questions(job_id):
    log_event("GET_QUESTIONS_START", job_id=job_id)

    if job_id not in job_data:
        log_event("GET_QUESTIONS_NO_JOB", job_id=job_id)
        return jsonify({'error': 'Job not found'}), 404

    job = job_data[job_id]
    if job.get('status') == 'completed' and 'questions' in job:
        log_event("GET_QUESTIONS_FROM_CACHE", job_id=job_id, count=len(job['questions']))
        return jsonify(job['questions'])

    resp = make_api_request('GET', f'jobs/{job_id}/questions')
    if resp and resp.status_code == 200:
        questions = resp.json()
        job['questions'] = questions
        log_event("GET_QUESTIONS_OK", job_id=job_id, count=len(questions))
        return jsonify(questions)

    log_event("GET_QUESTIONS_ERROR", job_id=job_id, status=resp.status_code if resp else "NO_RESPONSE")
    return jsonify({'error': 'Failed to fetch questions'}), 500

@app.route('/review/<job_id>')
def review(job_id):
    log_event("ROUTE_REVIEW", job_id=job_id)
    if job_id not in job_data:
        return redirect(url_for('index'))

    job = job_data[job_id]
    questions = job.get('questions') or []
    if not questions:
        resp = make_api_request('GET', f'jobs/{job_id}/questions')
        if resp and resp.status_code == 200:
            questions = resp.json()
            job['questions'] = questions
            log_event("REVIEW_FETCHED_QUESTIONS", job_id=job_id, count=len(questions))
    return render_template('review.html', job_id=job_id, questions=questions, job=job)

@app.route('/api/save_edits/<job_id>', methods=['POST'])
def save_edits(job_id):
    data = request.get_json() or {}
    incoming = data.get('questions', [])
    if job_id not in job_edits:
        job_edits[job_id] = []

    index = {q.get('qid'): i for i, q in enumerate(job_edits[job_id])}
    updated = 0
    created = 0
    for q in incoming:
        qid = q.get('qid')
        if qid in index:
            job_edits[job_id][index[qid]] = q
            updated += 1
        else:
            job_edits[job_id].append(q)
            created += 1

    log_event("SAVE_EDITS", job_id=job_id, updated=updated, created=created, total=len(job_edits[job_id]))
    return jsonify({'success': True})

# The UI JavaScript calls /api/delete_job/<qid> (qid only). We'll remove from any edits cache.
@app.route('/api/delete_job/<qid>', methods=['DELETE'])
def delete_qid(qid):
    removed = 0
    for jid, lst in job_edits.items():
        before = len(lst)
        job_edits[jid] = [q for q in lst if q.get('qid') != qid]
        removed += before - len(job_edits[jid])
    log_event("DELETE_QID", qid=qid, removed=removed)
    return jsonify({'success': True, 'removed': removed})

@app.route('/api/export/<job_id>/<format>')
def export_data(job_id, format):
    log_event("EXPORT_START", job_id=job_id, format=format)

    if job_id not in job_data:
        log_event("EXPORT_NO_JOB", job_id=job_id)
        return jsonify({'error': 'Job not found'}), 404

    original = job_data[job_id].get('questions', [])
    edits = job_edits.get(job_id, [])
    edits_by_id = {q.get('qid'): q for q in edits}
    questions = [edits_by_id.get(q.get('qid'), q) for q in original]

    approved_only = request.args.get('approved') in ('true', '1', 'yes', 'on')
    high_conf = request.args.get('high_confidence') in ('true', '1', 'yes', 'on')

    def keep(q):
        if approved_only and q.get('status') != 'approved':
            return False
        try:
            conf = float(q.get('confidence', 0))
        except Exception:
            conf = 0.0
        if high_conf and conf < 0.8:
            return False
        return True

    questions = [q for q in questions if keep(q)]
    log_event("EXPORT_FILTERED", job_id=job_id, count=len(questions),
              approved_only=approved_only, high_conf=high_conf)

    if format == 'docx':
        try:
            from docx import Document
            doc = Document()
            doc.add_heading('RFP Questions Report', 0)
            doc.add_paragraph(f'Generated on: {datetime.now():%Y-%m-%d %H:%M:%S}')
            doc.add_paragraph(f'Job ID: {job_id}')
            doc.add_paragraph(f'Total Questions: {len(questions)}')
            doc.add_paragraph('')

            for i, q in enumerate(questions, 1):
                doc.add_heading(f'Question {i}', level=2)
                doc.add_paragraph(f'Text: {q.get("text","N/A")}')
                doc.add_paragraph(f'Confidence: {q.get("confidence","N/A")}')
                doc.add_paragraph(f'Type: {q.get("type","N/A")}')
                section = (q.get("section_path") or ["N/A"])[0]
                doc.add_paragraph(f'Section: {section}')
                doc.add_paragraph('')

            bio = io.BytesIO()
            doc.save(bio)
            bio.seek(0)
            fname = f"RFP_Questions_Report_{datetime.now():%Y%m%d_%H%M%S}.docx"
            log_event("EXPORT_DONE", job_id=job_id, format=format, filename=fname)
            return send_file(
                bio, as_attachment=True, download_name=fname,
                mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            )
        except ImportError:
            # fall back to plain text
            content = [
                "RFP Questions Export",
                f"Generated on: {datetime.now():%Y-%m-%d %H:%M:%S}",
                f"Job ID: {job_id}",
                f"Total Questions: {len(questions)}",
                ""
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
            log_event("EXPORT_DONE_FALLBACK_TEXT", job_id=job_id, filename=fname)
            return body, 200, {
                'Content-Type': 'text/plain',
                'Content-Disposition': f'attachment; filename="{fname}"'
            }

    elif format == 'xlsx':
        try:
            import pandas as pd
            import openpyxl  # ensure engine available
            df = pd.DataFrame(questions)
            bio = io.BytesIO()
            with pd.ExcelWriter(bio, engine='openpyxl') as w:
                df.to_excel(w, sheet_name='RFP Questions', index=False)
            bio.seek(0)
            fname = f"RFP_Questions_Report_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
            log_event("EXPORT_DONE", job_id=job_id, format=format, filename=fname)
            return send_file(
                bio, as_attachment=True, download_name=fname,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        except ImportError:
            # CSV fallback (no pandas)
            import csv
            out = io.StringIO()
            rows = []
            for q in questions:
                rows.append({
                    'qid': q.get('qid',''),
                    'text': q.get('text',''),
                    'confidence': q.get('confidence',''),
                    'type': q.get('type',''),
                    'status': q.get('status',''),
                    'section': (q.get('section_path') or [''])[0] if q.get('section_path') else '',
                    'numbering': q.get('numbering',''),
                    'category': q.get('category','')
                })
            if rows:
                writer = csv.DictWriter(out, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            fname = f"RFP_Questions_Report_{datetime.now():%Y%m%d_%H%M%S}.csv"
            log_event("EXPORT_DONE_FALLBACK_CSV", job_id=job_id, filename=fname)
            return out.getvalue(), 200, {
                'Content-Type': 'text/csv',
                'Content-Disposition': f'attachment; filename="{fname}"'
            }

    log_event("EXPORT_INVALID_FORMAT", job_id=job_id, format=format)
    return jsonify({'error': 'Invalid format'}), 400

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    log_event("DEV_SERVER_START", port=5002, host="0.0.0.0")
    print("Open your browser at http://localhost:5002")
    app.run(debug=True, port=5002, host='0.0.0.0')
