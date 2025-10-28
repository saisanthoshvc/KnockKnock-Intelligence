import os, json, time, io, csv
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import requests
import logging

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = Flask(__name__, template_folder="templates")
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key-here')

# -----------------------------------------------------------------------------
# Logging (JSON lines so they show nicely in Posit Connect logs)
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def jlog(**kwargs):
    try:
        logging.info(json.dumps(kwargs))
    except Exception:
        logging.info(str(kwargs))

# -----------------------------------------------------------------------------
# Paths (use /tmp on Posit Connect)
# -----------------------------------------------------------------------------
UPLOAD_FOLDER = os.environ.get('UPLOAD_DIR', '/tmp/knockknock_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

STATS_FILE = os.environ.get('STATS_FILE', '/tmp/knockknock_stats.json')

# -----------------------------------------------------------------------------
# Allowed uploads
# -----------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {'docx', 'pdf', 'xlsx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# -----------------------------------------------------------------------------
# Env helpers
# -----------------------------------------------------------------------------
def env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default

# -----------------------------------------------------------------------------
# API Config (env-first)
# -----------------------------------------------------------------------------
api_config = {
    'base_url': os.environ.get(
        'RFP_API_URL',
        'https://connect.affiniusaiplatform.com/content/000d3572-937a-4d5c-ab7e-7ec80d80c4ce/'
    ),
    'api_key': os.environ.get('RFP_API_KEY', ''),                         # <- set in Connect (or blank if public)
    'header_name': os.environ.get('RFP_API_KEY_HEADER_NAME', 'Authorization'),
    'auth_scheme': os.environ.get('RFP_API_AUTH_SCHEME', 'Key'),          # Key | Bearer | Raw
    'poll_interval': env_int('POLL_INTERVAL_SECONDS', 2),
    'poll_timeout': env_int('POLL_TIMEOUT_SECONDS', 240),
    'request_timeout': env_int('REQUEST_TIMEOUT_SECONDS', 60),
}

# -----------------------------------------------------------------------------
# In-memory state
# -----------------------------------------------------------------------------
job_data = {}
job_edits = {}

# -----------------------------------------------------------------------------
# Stats helpers
# -----------------------------------------------------------------------------
def load_stats():
    try:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE, 'r') as f:
                stats = json.load(f)
            docs = stats.get("documents_processed", 0)
            stats["avg_processing_time"] = round(stats.get("total_processing_time", 0) / docs, 1) if docs else 0.0
            return stats
    except Exception as e:
        jlog(event="STATS_LOAD_ERROR", error=str(e))
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
        jlog(event="STATS_SAVE_ERROR", error=str(e))

def update_stats(questions_count, processing_time, questions_data=None):
    stats = load_stats()
    stats["documents_processed"] += 1
    stats["questions_extracted"] += questions_count
    stats["total_processing_time"] += processing_time

    if questions_count > 0 and questions_data:
        total_conf, n = 0.0, 0
        for q in questions_data:
            c = q.get('confidence')
            if isinstance(c, (int, float)):
                total_conf += float(c); n += 1
        if n:
            stats["accuracy_rate"] = round((total_conf / n) * 100, 1)
        elif stats.get("accuracy_rate", 0) == 0:
            stats["accuracy_rate"] = 85.0
    else:
        if stats.get("accuracy_rate", 0) == 0:
            stats["accuracy_rate"] = 85.0

    save_stats(stats)
    return stats

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_auth_header():
    """Build auth header according to env."""
    hdr = api_config['header_name']
    scheme = api_config['auth_scheme'].lower()
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

    jlog(event="API_REQUEST_OUT", method=method, url=url, header_name=list(headers.keys())[0] if headers else None)
    try:
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
        if ('text/html' in ct) and ('__login__' in resp.text.lower() or '<title>sign in' in resp.text.lower()):
            class Dummy:
                status_code = 401
                text = "Not authorized: redirected to login page. Check RFP_API_* env vars."
                def json(self): return {"error": self.text}
            jlog(event="API_REQUEST_DONE", url=url, status_code=401, redirected_to_login=True)
            return Dummy()

        jlog(event="API_REQUEST_DONE", url=url, status_code=resp.status_code)
        return resp
    except requests.exceptions.RequestException as e:
        jlog(event="API_REQUEST_ERROR", url=url, error=str(e))
        return None

# -----------------------------------------------------------------------------
# One-time startup health ping (safe for Flask 3)
# -----------------------------------------------------------------------------
def startup_ping():
    jlog(event="APP_START",
         version="1.0.0",
         upload_dir=UPLOAD_FOLDER,
         stats_file=STATS_FILE,
         api_base=api_config['base_url'],
         header_name=api_config['header_name'],
         auth_scheme=api_config['auth_scheme'],
         have_api_key=bool(api_config['api_key']))
    r = make_api_request('GET', 'health/ready')
    if r and r.status_code == 200:
        jlog(event="UPSTREAM_HEALTH_OK", endpoint="health/ready")
    else:
        jlog(event="UPSTREAM_HEALTH_FAIL", status=(r.status_code if r else None))

startup_ping()

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route('/health/ready')
def health_ready():
    return jsonify({"status": "ok"}), 200

@app.route('/')
def index():
    jlog(event="ROUTE_INDEX")
    stats = load_stats()
    return render_template('index.html', api_config=api_config, stats=stats)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        jlog(event="UPLOAD_BEGIN", ok=False, reason="no_file_in_form")
        return jsonify({'error': 'No file provided'}), 400

    f = request.files['file']
    if f.filename == '':
        jlog(event="UPLOAD_BEGIN", ok=False, reason="empty_filename")
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(f.filename):
        jlog(event="UPLOAD_BEGIN", ok=False, reason="invalid_extension", filename=f.filename)
        return jsonify({'error': 'Invalid file type'}), 400

    filename = secure_filename(f.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(filepath)

    use_llm = request.form.get('use_llm', 'false').lower() in ('true', '1', 'yes', 'on')
    use_sync = request.form.get('use_sync', 'false').lower() in ('true', '1', 'yes', 'on')
    mode = request.form.get('mode', 'balanced')

    jlog(event="UPLOAD_SAVED", filename=filename, size=os.path.getsize(filepath),
         use_llm=use_llm, use_sync=use_sync, mode=mode)

    try:
        with open(filepath, 'rb') as doc:
            files = {'file': doc}
            data = {'use_llm': 'true' if use_llm else 'false', 'mode': mode}
            endpoint = 'extract/sync' if use_sync else 'extract/'
            resp = make_api_request('POST', endpoint, files=files, data=data)

        if not resp:
            jlog(event="UPLOAD_FAIL", reason="no_response")
            return jsonify({'error': 'API request failed (no response)'}), 502

        if resp.status_code == 200:
            payload = resp.json()
            if use_sync and isinstance(payload, dict) and 'questions' in payload:
                job_id = f"sync_{int(time.time())}"
                job_data[job_id] = {
                    'status': 'completed',
                    'questions': payload['questions'],
                    'timestamp': datetime.now().isoformat()
                }
                update_stats(len(payload['questions']), 5, payload['questions'])
                jlog(event="UPLOAD_OK_SYNC", job_id=job_id, questions=len(payload['questions']))
                return jsonify({'success': True, 'job_id': job_id, 'questions': payload['questions']})
            else:
                job_id = payload.get('job_id')
                if job_id:
                    job_data[job_id] = {'status': 'processing', 'timestamp': datetime.now().isoformat()}
                    jlog(event="UPLOAD_OK_ASYNC", job_id=job_id)
                    return jsonify({'success': True, 'job_id': job_id, 'async': True})
                else:
                    jlog(event="UPLOAD_FAIL", reason="missing_job_id", body=payload)
                    return jsonify({'error': 'No job ID returned from API'}), 502
        else:
            try:
                msg = resp.json()
            except Exception:
                msg = resp.text
            jlog(event="UPLOAD_FAIL", status=resp.status_code, message=msg)
            return jsonify({'error': f"HTTP {resp.status_code}: {msg}"}), resp.status_code

    except Exception as e:
        jlog(event="UPLOAD_EXCEPTION", error=str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                jlog(event="UPLOAD_CLEANUP_OK", filename=filename)
        except Exception as e:
            jlog(event="UPLOAD_CLEANUP_FAIL", filename=filename, error=str(e))

@app.route('/api/poll/<job_id>')
def poll_job(job_id):
    jlog(event="POLL_BEGIN", job_id=job_id)
    if job_id not in job_data:
        jlog(event="POLL_NOT_FOUND", job_id=job_id)
        return jsonify({'error': 'Job not found'}), 404

    if job_data[job_id].get('status') == 'completed':
        jlog(event="POLL_ALREADY_COMPLETE", job_id=job_id)
        return jsonify(job_data[job_id])

    resp = make_api_request('GET', f'jobs/{job_id}')
    if not resp:
        jlog(event="POLL_FAIL", job_id=job_id, reason="no_response")
        return jsonify({'error': 'Failed to poll job status'}), 502

    if resp.status_code == 404:
        q = make_api_request('GET', f'jobs/{job_id}/questions')
        if q and q.status_code == 200:
            questions = q.json()
            job_data[job_id]['status'] = 'completed'
            job_data[job_id]['questions'] = questions
            update_stats(len(questions), 30, questions)
            jlog(event="POLL_COMPLETE_RECOVERED", job_id=job_id, questions=len(questions))
            return jsonify({'status': 'completed', 'questions': questions})
        jlog(event="POLL_COMPLETE_EMPTY", job_id=job_id)
        return jsonify({'status': 'completed', 'questions': []})

    if resp.status_code != 200:
        try:
            msg = resp.json()
        except Exception:
            msg = resp.text
        jlog(event="POLL_FAIL", job_id=job_id, status=resp.status_code, message=msg)
        return jsonify({'error': f"HTTP {resp.status_code}: {msg}"}), resp.status_code

    data = resp.json()
    job_data[job_id].update(data)
    status = data.get('status', 'processing')

    if status == 'completed':
        if data.get('questions') is not None:
            qs = data['questions']
            update_stats(len(qs), 30, qs)
            jlog(event="POLL_COMPLETE_INLINE", job_id=job_id, questions=len(qs))
            return jsonify(data)
        q = make_api_request('GET', f'jobs/{job_id}/questions')
        if q and q.status_code == 200:
            qs = q.json()
            job_data[job_id]['questions'] = qs
            update_stats(len(qs), 30, qs)
            data['questions'] = qs
            jlog(event="POLL_COMPLETE_FETCHED", job_id=job_id, questions=len(qs))
            return jsonify(data)
        jlog(event="POLL_COMPLETE_NO_QUESTIONS", job_id=job_id)
        return jsonify({'status': 'completed', 'questions': []})

    if status == 'processing':
        start = job_data[job_id].get('timestamp')
        if start:
            try:
                started = datetime.fromisoformat(start)
                if datetime.now() - started > timedelta(minutes=10):
                    job_data[job_id]['status'] = 'timeout'
                    jlog(event="POLL_TIMEOUT", job_id=job_id)
                    return jsonify({'status': 'timeout', 'error': 'Processing timeout; try sync or no-LLM'})
            except Exception:
                pass
        jlog(event="POLL_PROCESSING", job_id=job_id, progress=data.get('progress'))
        return jsonify({'status': 'processing', 'progress': data.get('progress')})

    if status == 'failed':
        jlog(event="POLL_FAILED_STATUS", job_id=job_id, data=data)
        return jsonify(data)

    jlog(event="POLL_UNKNOWN_STATUS", job_id=job_id, status=status)
    return jsonify({'status': 'processing'})

@app.route('/api/job/<job_id>/questions')
def get_job_questions(job_id):
    if job_id not in job_data:
        return jsonify({'error': 'Job not found'}), 404

    job = job_data[job_id]
    if job.get('status') == 'completed' and 'questions' in job:
        return jsonify(job['questions'])

    resp = make_api_request('GET', f'jobs/{job_id}/questions')
    if resp and resp.status_code == 200:
        questions = resp.json()
        job['questions'] = questions
        return jsonify(questions)
    return jsonify({'error': 'Failed to fetch questions'}), 500

@app.route('/review/<job_id>')
def review(job_id):
    if job_id not in job_data:
        return redirect(url_for('index'))
    job = job_data[job_id]
    questions = job.get('questions') or []
    if not questions:
        resp = make_api_request('GET', f'jobs/{job_id}/questions')
        if resp and resp.status_code == 200:
            questions = resp.json()
            job['questions'] = questions
    return render_template('review.html', job_id=job_id, questions=questions, job=job)

@app.route('/api/save_edits/<job_id>', methods=['POST'])
def save_edits(job_id):
    data = request.get_json() or {}
    incoming = data.get('questions', [])
    if job_id not in job_edits:
        job_edits[job_id] = []
    index = {q.get('qid'): i for i, q in enumerate(job_edits[job_id])}
    for q in incoming:
        qid = q.get('qid')
        if qid in index:
            job_edits[job_id][index[qid]] = q
        else:
            job_edits[job_id].append(q)
    jlog(event="EDITS_SAVED", job_id=job_id, count=len(incoming))
    return jsonify({'success': True})

@app.route('/api/delete_job/<qid>', methods=['DELETE'])
def delete_row(qid):
    # purely client-side removal helper; also drop from any cached job if present
    removed = False
    for jid, jd in job_data.items():
        if 'questions' in jd and isinstance(jd['questions'], list):
            before = len(jd['questions'])
            jd['questions'] = [q for q in jd['questions'] if str(q.get('qid')) != str(qid)]
            after = len(jd['questions'])
            if after < before:
                removed = True
    jlog(event="ROW_DELETED", qid=qid, removed=removed)
    return jsonify({'success': True})

@app.route('/api/export/<job_id>/<format>')
def export_data(job_id, format):
    if job_id not in job_data:
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
        if high_conf and (float(q.get('confidence', 0)) < 0.8):
            return False
        return True

    questions = [q for q in questions if keep(q)]
    jlog(event="EXPORT_BEGIN", job_id=job_id, fmt=format, rows=len(questions),
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
            bio = io.BytesIO(); doc.save(bio); bio.seek(0)
            fname = f"RFP_Questions_Report_{datetime.now():%Y%m%d_%H%M%S}.docx"
            jlog(event="EXPORT_DONE", job_id=job_id, fmt="docx", bytes=len(bio.getvalue()))
            return send_file(bio, as_attachment=True,
                             download_name=fname,
                             mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document')
        except ImportError:
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
            jlog(event="EXPORT_DONE", job_id=job_id, fmt="txt", bytes=len(body.encode("utf-8")))
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
            jlog(event="EXPORT_DONE", job_id=job_id, fmt="xlsx", bytes=len(bio.getvalue()))
            return send_file(bio, as_attachment=True,
                             download_name=fname,
                             mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        except ImportError:
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
                writer.writeheader(); writer.writerows(rows)
            fname = f"RFP_Questions_Report_{datetime.now():%Y%m%d_%H%M%S}.csv"
            body = out.getvalue()
            jlog(event="EXPORT_DONE", job_id=job_id, fmt="csv", bytes=len(body.encode("utf-8")))
            return body, 200, {
                'Content-Type': 'text/csv',
                'Content-Disposition': f'attachment; filename="{fname}"'
            }

    return jsonify({'error': 'Invalid format'}), 400

# NOTE: No app.run() for Posit Connect gunicorn loader
if __name__ == '__main__':
    print("Starting RFP Extraction App on port 5002...")
    app.run(debug=True, port=5002, host='0.0.0.0')
