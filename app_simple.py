import os
import json
import time
import requests
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import io

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key-here')

# ---------- Paths (use /tmp on Posit Connect) ----------
UPLOAD_FOLDER = os.environ.get('UPLOAD_DIR', '/tmp/knockknock_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

STATS_FILE = os.environ.get('STATS_FILE', '/tmp/knockknock_stats.json')

# ---------- Allowed uploads ----------
ALLOWED_EXTENSIONS = {'docx', 'pdf', 'xlsx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# ---------- API Config (env-first) ----------
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
    'api_key': os.environ.get('RFP_API_KEY', ''),
    'header_name': os.environ.get('RFP_API_KEY_HEADER_NAME', 'Authorization'),
    'auth_scheme': os.environ.get('RFP_API_AUTH_SCHEME', 'Key'),  # Key | Bearer | Raw
    'poll_interval': env_int('POLL_INTERVAL_SECONDS', 2),
    'poll_timeout': env_int('POLL_TIMEOUT_SECONDS', 240),
    'request_timeout': env_int('REQUEST_TIMEOUT_SECONDS', 60),
}

api_config = DEFAULT_API_CONFIG.copy()

# ---------- Simple logging helper ----------
def log_event(event, **kv):
    try:
        print(json.dumps({"event": event, **kv}))
    except Exception:
        print(f"[{event}] {kv}")

# ---------- In-memory state ----------
job_data = {}   # job_id -> {'status', 'questions', 'timestamp', ...}
job_edits = {}  # job_id -> [edited question dicts]

# NEW: stores for KB responses (edited/saved by user)
job_responses = {}        # job_id -> list of response dicts (optional cache)
job_responses_edits = {}  # job_id -> [ {qid, text, response, status, sources} ]

# ---------- Stats helpers ----------
def load_stats():
    try:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE, 'r') as f:
                stats = json.load(f)
            docs = stats.get("documents_processed", 0)
            if docs > 0:
                stats["avg_processing_time"] = round(
                    stats.get("total_processing_time", 0) / docs, 1
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

    if questions_count > 0 and questions_data:
        total_conf = 0.0
        n = 0
        for q in questions_data:
            c = q.get('confidence')
            if isinstance(c, (int, float)):
                total_conf += float(c)
                n += 1
        stats["accuracy_rate"] = round((total_conf / n) * 100, 1) if n else stats.get("accuracy_rate", 85.0)
    else:
        stats["accuracy_rate"] = max(stats.get("accuracy_rate", 0.0), 85.0)

    save_stats(stats)
    return stats

# ---------- Utilities ----------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_auth_header():
    """
    Build auth header according to env:
      - Authorization: Key <token>
      - Authorization: Bearer <token>
      - <custom-name>: <token>
    """
    hdr = api_config['header_name']
    scheme = api_config['auth_scheme'].lower()
    token = api_config['api_key']

    if not token:
        return {}

    if hdr.lower() == 'authorization':
        if scheme == 'key':
            return {'Authorization': f'Key {token}'}
        elif scheme == 'bearer':
            return {'Authorization': f'Bearer {token}'}
        elif scheme == 'raw':
            return {'Authorization': token}
        else:
            return {'Authorization': token}
    else:
        if scheme == 'raw':
            return {hdr: token}
        elif scheme in ('key', 'bearer'):
            scheme_title = scheme.title()
            return {hdr: f'{scheme_title} {token}'}
        else:
            return {hdr: token}

def make_api_request(method, endpoint, **kwargs):
    url = f"{api_config['base_url'].rstrip('/')}/{endpoint.strip('/')}"
    headers = get_auth_header()
    headers.update(kwargs.pop('headers', {}))

    log_event("API_REQUEST_OUT", method=method.upper(), url=url, header_name=api_config['header_name'])

    try:
        if method.upper() == 'GET':
            resp = requests.get(url, headers=headers, timeout=api_config['request_timeout'])
        elif method.upper() == 'POST':
            resp = requests.post(url, headers=headers, timeout=api_config['request_timeout'], **kwargs)
        elif method.upper() == 'DELETE':
            resp = requests.delete(url, headers=headers, timeout=api_config['request_timeout'])
        else:
            return None

        # Detect redirect to login page (auth failure)
        ct = resp.headers.get('Content-Type', '')
        if ('text/html' in ct) and ('__login__' in resp.text.lower() or '<title>sign in' in resp.text.lower()):
            class Dummy:
                status_code = 401
                text = "Not authorized: redirected to login page. Check RFP_API_* env vars."
                def json(self): return {"error": self.text}
            log_event("API_REQUEST_DONE", url=url, status_code=401)
            return Dummy()

        log_event("API_REQUEST_DONE", url=url, status_code=resp.status_code)
        return resp
    except requests.exceptions.RequestException as e:
        log_event("API_REQUEST_ERROR", url=url, error=str(e))
        return None

# ---------- Request logging ----------
@app.before_request
def _log_in():
    log_event("HTTP_IN", method=request.method, path=request.path)

@app.after_request
def _log_out(response):
    log_event("HTTP_OUT", status=response.status_code, path=request.path)
    return response

# ---------- Helpers ----------
def merge_questions_with_edits(job_id):
    """Merge original job questions with local edits (status/text/confidence/type/section...)."""
    original = job_data.get(job_id, {}).get('questions', []) or []
    edits = job_edits.get(job_id, []) or []
    edits_by_id = {q.get('qid') or q.get('id') or q.get('question_id'): q for q in edits}

    def norm(q):
        q = dict(q or {})
        q['qid'] = q.get('qid') or q.get('id') or q.get('question_id')
        # normalize status
        status = (q.get('status') or '').strip().lower()
        if status not in ('approved', 'pending', 'rejected'):
            status = 'pending'
        q['status'] = status
        return q

    merged = []
    for oq in original:
        base = norm(oq)
        ov = edits_by_id.get(base['qid'])
        if ov:
            ov = norm(ov)
            for k in ('text','confidence','status','type','section_path','numbering','category'):
                if ov.get(k) not in (None, '', []):
                    base[k] = ov.get(k)
        merged.append(base)
    return merged

# ---------- Routes ----------
@app.route('/')
def index():
    # Probe upstream readiness (optional)
    health = make_api_request('GET', 'health/ready')
    if health and health.status_code == 200:
        log_event("UPSTREAM_HEALTH_OK", endpoint="health/ready")
    stats = load_stats()
    log_event("ROUTE_INDEX")
    return render_template('index.html', api_config=api_config, stats=stats)

# ---- Upload & Extract ----
@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        log_event("UPLOAD_MISSING_FILE")
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
    log_event("UPLOAD_SAVED", filename=filename, path=filepath)

    use_llm = request.form.get('use_llm', 'false').lower() in ('true', '1', 'yes', 'on')
    use_sync = request.form.get('use_sync', 'false').lower() in ('true', '1', 'yes', 'on')
    mode = request.form.get('mode', 'balanced')

    try:
        with open(filepath, 'rb') as doc:
            files = {'file': doc}
            data = {'use_llm': 'true' if use_llm else 'false', 'mode': mode}
            endpoint = 'extract/sync' if use_sync else 'extract/'
            log_event("EXTRACT_CALL", endpoint=endpoint, mode=mode, use_llm=use_llm, use_sync=use_sync)
            resp = make_api_request('POST', endpoint, files=files, data=data)

        if not resp:
            log_event("EXTRACT_NO_RESPONSE")
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
                log_event("EXTRACT_SYNC_OK", job_id=job_id, q=len(payload['questions']))
                return jsonify({'success': True, 'job_id': job_id, 'questions': payload['questions']})
            else:
                job_id = payload.get('job_id')
                if job_id:
                    job_data[job_id] = {'status': 'processing', 'timestamp': datetime.now().isoformat()}
                    log_event("EXTRACT_ASYNC_OK", job_id=job_id)
                    return jsonify({'success': True, 'job_id': job_id, 'async': True})
                else:
                    log_event("EXTRACT_MISSING_JOB_ID")
                    return jsonify({'error': 'No job ID returned from API'}), 502
        else:
            try:
                msg = resp.json()
            except Exception:
                msg = resp.text
            log_event("EXTRACT_HTTP_ERROR", status=resp.status_code, message=str(msg))
            return jsonify({'error': f"HTTP {resp.status_code}: {msg}"}), resp.status_code

    except Exception as e:
        log_event("UPLOAD_EXCEPTION", error=str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                log_event("UPLOAD_CLEANED", path=filepath)
        except Exception as e:
            log_event("UPLOAD_CLEANUP_ERROR", error=str(e))

# ---- Polling / Jobs ----
@app.route('/api/poll/<job_id>')
def poll_job(job_id):
    if job_id not in job_data:
        log_event("POLL_UNKNOWN_JOB", job_id=job_id)
        return jsonify({'error': 'Job not found'}), 404

    if job_data[job_id].get('status') == 'completed':
        log_event("POLL_ALREADY_DONE", job_id=job_id)
        return jsonify(job_data[job_id])

    resp = make_api_request('GET', f'jobs/{job_id}')
    if not resp:
        log_event("POLL_API_FAIL", job_id=job_id)
        return jsonify({'error': 'Failed to poll job status'}), 502

    if resp.status_code == 404:
        q = make_api_request('GET', f'jobs/{job_id}/questions')
        if q and q.status_code == 200:
            questions = q.json()
            job_data[job_id]['status'] = 'completed'
            job_data[job_id]['questions'] = questions
            update_stats(len(questions), 30, questions)
            log_event("POLL_RECOVERED_VIA_QUESTIONS", job_id=job_id, q=len(questions))
            return jsonify({'status': 'completed', 'questions': questions})
        log_event("POLL_NOT_FOUND_COMPLETING_EMPTY", job_id=job_id)
        return jsonify({'status': 'completed', 'questions': []})

    if resp.status_code != 200:
        try:
            msg = resp.json()
        except Exception:
            msg = resp.text
        log_event("POLL_HTTP_ERROR", job_id=job_id, status=resp.status_code, message=str(msg))
        return jsonify({'error': f"HTTP {resp.status_code}: {msg}"}), resp.status_code

    data = resp.json()
    job_data[job_id].update(data)

    status = data.get('status', 'processing')
    if status == 'completed':
        if data.get('questions') is not None:
            qs = data['questions']
            update_stats(len(qs), 30, qs)
            log_event("POLL_COMPLETED_INLINE", job_id=job_id, q=len(qs))
            return jsonify(data)
        q = make_api_request('GET', f'jobs/{job_id}/questions')
        if q and q.status_code == 200:
            qs = q.json()
            job_data[job_id]['questions'] = qs
            update_stats(len(qs), 30, qs)
            data['questions'] = qs
            log_event("POLL_COMPLETED_FETCHED", job_id=job_id, q=len(qs))
            return jsonify(data)
        log_event("POLL_COMPLETED_NO_QS", job_id=job_id)
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
            except Exception:
                pass
        return jsonify({'status': 'processing', 'progress': data.get('progress')})

    if status == 'failed':
        log_event("POLL_FAILED", job_id=job_id, data=data)
        return jsonify(data)

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

# ---- Review page ----
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

    log_event("SAVE_EDITS", job_id=job_id, count=len(incoming))
    return jsonify({'success': True})

# ---- Export questions (existing) ----
@app.route('/api/export/<job_id>/<format>')
def export_data(job_id, format):
    if job_id not in job_data:
        return jsonify({'error': 'Job not found'}), 404

    original = job_data[job_id].get('questions', []) or []
    edits = job_edits.get(job_id, []) or []
    edits_by_id = {q.get('qid'): q for q in edits}

    def normalize(q):
        q = dict(q or {})
        q['qid'] = q.get('qid') or q.get('id') or q.get('question_id')
        status = (q.get('status') or '').strip().lower()
        if status not in ('approved', 'pending', 'rejected'):
            status = 'pending'
        q['status'] = status
        try:
            c = float(q.get('confidence', 0))
            if c > 1:
                c = c / 100.0
        except Exception:
            c = 0.0
        q['confidence'] = round(c, 3)
        return q

    merged = []
    for oq in original:
        base = normalize(oq)
        override = normalize(edits_by_id.get(base.get('qid'), {}))
        for k in ('text', 'confidence', 'status', 'type', 'section_path', 'numbering', 'category'):
            v = override.get(k)
            if v not in (None, '', []):
                base[k] = v
        merged.append(base)

    approved_only = request.args.get('approved') in ('true', '1', 'yes', 'on')
    high_conf = request.args.get('high_confidence') in ('true', '1', 'yes', 'on')

    def keep(q):
        if approved_only and q.get('status') != 'approved':
            return False
        if high_conf and float(q.get('confidence', 0)) < 0.8:
            return False
        return True

    questions = [q for q in merged if keep(q)]

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
                doc.add_paragraph(f'Status: {q.get("status","pending").title()}')
                doc.add_paragraph(f'Confidence: {int(round(float(q.get("confidence",0))*100))}%')
                doc.add_paragraph(f'Type: {q.get("type","N/A")}')
                section = (q.get("section_path") or ["N/A"])[0] if q.get("section_path") else "N/A"
                doc.add_paragraph(f'Section: {section}')
                doc.add_paragraph('')

            bio = io.BytesIO()
            doc.save(bio)
            bio.seek(0)
            fname = f"RFP_Questions_Report_{datetime.now():%Y%m%d_%H%M%S}.docx"
            return send_file(
                bio, as_attachment=True, download_name=fname,
                mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            )
        except ImportError:
            content = [
                "RFP Questions Report",
                f"Generated on: {datetime.now():%Y-%m-%d %H:%M:%S}",
                f"Job ID: {job_id}",
                f"Total Questions: {len(questions)}",
                ""
            ]
            for i, q in enumerate(questions, 1):
                section = (q.get("section_path") or [""])[0] if q.get("section_path") else ""
                content.append(f"{i}. {q.get('text','N/A')}")
                content.append(f"   Status: {q.get('status','pending').title()}")
                content.append(f"   Confidence: {int(round(float(q.get('confidence',0))*100))}%")
                content.append(f"   Type: {q.get('type','N/A')}")
                content.append(f"   Section: {section}")
                content.append("")
            body = "\n".join(content)
            fname = f"RFP_Questions_Report_{datetime.now():%Y%m%d_%H%M%S}.txt"
            return body, 200, {
                'Content-Type': 'text/plain',
                'Content-Disposition': f'attachment; filename=\"{fname}\"'
            }

    elif format == 'xlsx':
        try:
            import pandas as pd
            import openpyxl
            df = pd.DataFrame(questions)
            cols = ['qid','text','status','confidence','type','numbering','category','section_path']
            existing = [c for c in cols if c in df.columns]
            df = df[existing + [c for c in df.columns if c not in existing]]
            bio = io.BytesIO()
            with pd.ExcelWriter(bio, engine='openpyxl') as w:
                df.to_excel(w, sheet_name='RFP Questions', index=False)
            bio.seek(0)
            fname = f"RFP_Questions_Report_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
            return send_file(
                bio, as_attachment=True, download_name=fname,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        except ImportError:
            import csv
            out = io.StringIO()
            rows = []
            for q in questions:
                rows.append({
                    'qid': q.get('qid',''),
                    'text': q.get('text',''),
                    'status': q.get('status',''),
                    'confidence': q.get('confidence',''),
                    'type': q.get('type',''),
                    'section': (q.get('section_path') or [''])[0] if q.get('section_path') else '',
                    'numbering': q.get('numbering',''),
                    'category': q.get('category','')
                })
            if rows:
                writer = csv.DictWriter(out, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            fname = f"RFP_Questions_Report_{datetime.now():%Y%m%d_%H%M%S}.csv"
            return out.getvalue(), 200, {
                'Content-Type': 'text/csv',
                'Content-Disposition': f'attachment; filename=\"{fname}\"'
            }

    return jsonify({'error': 'Invalid format'}), 400

# ===================== NEW: KB + RESPONSES =====================

# Approved questions for a job (used by responses page)
@app.route('/api/approved_questions/<job_id>')
def get_approved_questions(job_id):
    if job_id not in job_data:
        return jsonify({'error':'Job not found'}), 404
    merged = merge_questions_with_edits(job_id)
    approved = [q for q in merged if q.get('status') == 'approved']
    return jsonify(approved)

# Proxy a single KB query to upstream /kb/query
@app.route('/api/kb/query-one', methods=['POST'])
def kb_query_one():
    payload = request.get_json() or {}
    body = {
        "query": payload.get("query", ""),
        "max_results": int(payload.get("max_results", 5)),
        "temperature": float(payload.get("temperature", 0.1)),
        "max_tokens": int(payload.get("max_tokens", 1500)),
        "conversation_history": payload.get("conversation_history") or []
    }
    resp = make_api_request('POST', 'kb/query', json=body)
    if not resp:
        return jsonify({'error':'KB request failed (no response)'}), 502
    try:
        data = resp.json()
    except Exception:
        data = {"error": resp.text}
    return jsonify(data), resp.status_code

# Save edited KB responses
@app.route('/api/responses/save/<job_id>', methods=['POST'])
def save_responses(job_id):
    data = request.get_json() or {}
    incoming = data.get('responses', [])
    if job_id not in job_responses_edits:
        job_responses_edits[job_id] = []

    index = {r.get('qid'): i for i, r in enumerate(job_responses_edits[job_id])}
    for r in incoming:
        qid = r.get('qid')
        if qid in index:
            job_responses_edits[job_id][index[qid]] = r
        else:
            job_responses_edits[job_id].append(r)
    log_event("SAVE_RESPONSES", job_id=job_id, count=len(incoming))
    return jsonify({'success': True})

# Responses UI page
@app.route('/responses/<job_id>')
def responses(job_id):
    if job_id not in job_data:
        return redirect(url_for('index'))
    if job_id not in job_responses:
        job_responses[job_id] = []
    return render_template('llm_responses.html', job_id=job_id)

# Export KB responses (docx/xlsx)
@app.route('/api/export_responses/<job_id>/<format>')
def export_responses(job_id, format):
    if job_id not in job_data:
        return jsonify({'error': 'Job not found'}), 404

    # base from approved questions
    approved = merge_questions_with_edits(job_id)
    approved = [q for q in approved if q.get('status') == 'approved']

    # overlay saved responses
    edits = job_responses_edits.get(job_id, []) or []
    by_id = {e.get('qid'): e for e in edits}

    rows = []
    for q in approved:
        qid = q.get('qid')
        e = by_id.get(qid, {})
        rows.append({
            'qid': qid,
            'question': q.get('text', ''),
            'response': e.get('response',''),
            'status': e.get('status','ok'),
            'type': q.get('type',''),
            'section': (q.get('section_path') or [''])[0] if q.get('section_path') else '',
            'sources': e.get('sources', [])
        })

    only_success = request.args.get('success') in ('true','1','yes','on')
    if only_success:
        rows = [r for r in rows if r.get('status') == 'ok']

    id_str = request.args.get('qids')
    if id_str:
        wanted = set(id_str.split(','))
        rows = [r for r in rows if r.get('qid') in wanted]

    if format == 'docx':
        try:
            from docx import Document
            from datetime import datetime as _dt
            doc = Document()
            doc.add_heading('RFP KB Responses', 0)
            doc.add_paragraph(f'Generated on: {_dt.now():%Y-%m-%d %H:%M:%S}')
            doc.add_paragraph(f'Job ID: {job_id}')
            doc.add_paragraph(f'Total Responses: {len(rows)}')
            doc.add_paragraph('')

            for i, r in enumerate(rows, 1):
                doc.add_heading(f'Item {i}', level=2)
                doc.add_paragraph(f'Question: {r["question"]}')
                doc.add_paragraph(f'Response: {r["response"] or "(empty)"}')
                doc.add_paragraph(f'Status: {r["status"].title()}')
                doc.add_paragraph(f'Type: {r["type"] or "N/A"}')
                doc.add_paragraph(f'Section: {r["section"] or "N/A"}')
                if r['sources']:
                    doc.add_paragraph('Sources:')
                    for s in r['sources']:
                        name = s.get('name') or s.get('uri') or 'source'
                        uri = s.get('uri')
                        line = f"- {name}" + (f" ({uri})" if uri else "")
                        doc.add_paragraph(line)
                doc.add_paragraph('')

            bio = io.BytesIO()
            doc.save(bio); bio.seek(0)
            fname = f"RFP_KB_Responses_{_dt.now():%Y%m%d_%H%M%S}.docx"
            return send_file(bio, as_attachment=True, download_name=fname,
                mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document')
        except ImportError:
            return jsonify({'error':'python-docx not installed on server'}), 500

    elif format == 'xlsx':
        try:
            import pandas as pd
            import openpyxl
            import io as _io, json as _json
            flat = []
            for r in rows:
                flat.append({
                    'qid': r['qid'],
                    'question': r['question'],
                    'response': r['response'],
                    'status': r['status'],
                    'type': r['type'],
                    'section': r['section'],
                    'sources_json': _json.dumps(r['sources'])
                })
            df = pd.DataFrame(flat)
            bio = _io.BytesIO()
            with pd.ExcelWriter(bio, engine='openpyxl') as w:
                df.to_excel(w, sheet_name='KB Responses', index=False)
            bio.seek(0)
            from datetime import datetime as _dt
            fname = f"RFP_KB_Responses_{_dt.now():%Y%m%d_%H%M%S}.xlsx"
            return send_file(bio, as_attachment=True, download_name=fname,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        except ImportError:
            return jsonify({'error':'pandas/openpyxl not installed on server'}), 500

    return jsonify({'error': 'Invalid format'}), 400

# ===================== END KB =====================

if __name__ == '__main__':
    print("Starting RFP Extraction App on port 5002...")
    print("Open your browser at http://localhost:5002")
    app.run(debug=True, port=5002, host='0.0.0.0')
