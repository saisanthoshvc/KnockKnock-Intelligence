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

def env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default

DEFAULT_API_CONFIG = {
    'base_url': os.environ.get(
        'RFP_API_URL',
        'https://connect.affiniusaiplatform.com/content/000d3572-937a-4d5c-ab7e-7ec80d80c4ce/'
    ),
    'api_key': os.environ.get('RFP_API_KEY', ''),                         # <- set in Connect
    'header_name': os.environ.get('RFP_API_KEY_HEADER_NAME', 'Authorization'),
    'auth_scheme': os.environ.get('RFP_API_AUTH_SCHEME', 'Key'),          # Key | Bearer | Raw
    'poll_interval': env_int('POLL_INTERVAL_SECONDS', 2),
    'poll_timeout': env_int('POLL_TIMEOUT_SECONDS', 240),
    'request_timeout': env_int('REQUEST_TIMEOUT_SECONDS', 60),
}

api_config = DEFAULT_API_CONFIG.copy()

# ---------- In-memory state ----------
job_data = {}
job_edits = {}

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
        print(f"[STATS] Error saving stats: {e}")

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
        # default if API didn’t send confidences
        stats["accuracy_rate"] = max(stats.get("accuracy_rate", 0.0), 85.0)

    save_stats(stats)
    return stats

# ---------- Utilities ----------
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
    scheme = api_config['auth_scheme'].lower()
    token = api_config['api_key']

    if not token:
        return {}  # allow unauthenticated if target is public (login not required)

    if hdr.lower() == 'authorization':
        if scheme == 'key':
            return {'Authorization': f'Key {token}'}
        elif scheme == 'bearer':
            return {'Authorization': f'Bearer {token}'}
        elif scheme == 'raw':
            # rare, but supported
            return {'Authorization': token}
        else:
            return {'Authorization': token}
    else:
        # custom header name
        if scheme == 'raw':
            return {hdr: token}
        elif scheme in ('key', 'bearer'):
            scheme_title = scheme.title()
            return {hdr: f'{scheme_title} {token}'}
        else:
            return {hdr: token}

def make_api_request(method, endpoint, **kwargs):
    url = f"{api_config['base_url'].rstrip('/')}/{endpoint.lstrip('/')}"
    headers = get_auth_header()
    # merge any extra headers from caller
    headers.update(kwargs.pop('headers', {}))

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
            # Posit returned login HTML because Authorization header wasn’t accepted
            class Dummy:
                status_code = 401
                text = "Not authorized: your request was redirected to the login page. Check RFP_API_KEY_* env vars."
                def json(self): return {"error": self.text}
            return Dummy()

        return resp
    except requests.exceptions.RequestException as e:
        print(f"[API] Request error: {e}")
        return None

# ---------- Routes ----------
@app.route('/')
def index():
    stats = load_stats()
    return render_template('index.html', api_config=api_config, stats=stats)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(f.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filename = secure_filename(f.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(filepath)

    # Booleans must be lowercase strings for robust FastAPI parsing
    use_llm = request.form.get('use_llm', 'false').lower() in ('true', '1', 'yes', 'on')
    use_sync = request.form.get('use_sync', 'false').lower() in ('true', '1', 'yes', 'on')
    mode = request.form.get('mode', 'balanced')

    try:
        with open(filepath, 'rb') as doc:
            files = {'file': doc}
            data = {
                'use_llm': 'true' if use_llm else 'false',
                'mode': mode
            }
            endpoint = 'extract/sync' if use_sync else 'extract/'
            resp = make_api_request('POST', endpoint, files=files, data=data)

        if not resp:
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
                # assume sync is fast
                update_stats(len(payload['questions']), 5, payload['questions'])
                return jsonify({'success': True, 'job_id': job_id, 'questions': payload['questions']})
            else:
                job_id = payload.get('job_id')
                if job_id:
                    job_data[job_id] = {
                        'status': 'processing',
                        'timestamp': datetime.now().isoformat()
                    }
                    return jsonify({'success': True, 'job_id': job_id, 'async': True})
                else:
                    return jsonify({'error': 'No job ID returned from API'}), 502
        else:
            # surface exact upstream error
            try:
                msg = resp.json()
            except Exception:
                msg = resp.text
            return jsonify({'error': f"HTTP {resp.status_code}: {msg}"}), resp.status_code

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception:
            pass

@app.route('/api/poll/<job_id>')
def poll_job(job_id):
    if job_id not in job_data:
        return jsonify({'error': 'Job not found'}), 404

    # already done?
    if job_data[job_id].get('status') == 'completed':
        return jsonify(job_data[job_id])

    resp = make_api_request('GET', f'jobs/{job_id}')
    if not resp:
        return jsonify({'error': 'Failed to poll job status'}), 502

    # If upstream cleaned up, try direct questions fetch
    if resp.status_code == 404:
        q = make_api_request('GET', f'jobs/{job_id}/questions')
        if q and q.status_code == 200:
            questions = q.json()
            job_data[job_id]['status'] = 'completed'
            job_data[job_id]['questions'] = questions
            update_stats(len(questions), 30, questions)
            return jsonify({'status': 'completed', 'questions': questions})
        return jsonify({'status': 'completed', 'questions': []})

    if resp.status_code != 200:
        try:
            msg = resp.json()
        except Exception:
            msg = resp.text
        return jsonify({'error': f"HTTP {resp.status_code}: {msg}"}), resp.status_code

    data = resp.json()
    job_data[job_id].update(data)

    status = data.get('status', 'processing')
    if status == 'completed':
        if data.get('questions') is not None:
            qs = data['questions']
            update_stats(len(qs), 30, qs)
            return jsonify(data)
        # fetch questions endpoint if completed but missing inline
        q = make_api_request('GET', f'jobs/{job_id}/questions')
        if q and q.status_code == 200:
            qs = q.json()
            job_data[job_id]['questions'] = qs
            update_stats(len(qs), 30, qs)
            data['questions'] = qs
            return jsonify(data)
        return jsonify({'status': 'completed', 'questions': []})

    if status == 'processing':
        # 10-minute guard
        start = job_data[job_id].get('timestamp')
        if start:
            try:
                started = datetime.fromisoformat(start)
                if datetime.now() - started > timedelta(minutes=10):
                    job_data[job_id]['status'] = 'timeout'
                    return jsonify({'status': 'timeout', 'error': 'Processing timeout; try sync or no-LLM'})
            except Exception:
                pass
        return jsonify({'status': 'processing', 'progress': data.get('progress')})

    if status == 'failed':
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

    # index existing by qid
    index = {q.get('qid'): i for i, q in enumerate(job_edits[job_id])}
    for q in incoming:
        qid = q.get('qid')
        if qid in index:
            job_edits[job_id][index[qid]] = q
        else:
            job_edits[job_id].append(q)

    return jsonify({'success': True})

@app.route('/api/export/<job_id>/<format>')
def export_data(job_id, format):
    if job_id not in job_data:
        return jsonify({'error': 'Job not found'}), 404

    original = job_data[job_id].get('questions', [])
    edits = job_edits.get(job_id, [])
    edits_by_id = {q.get('qid'): q for q in edits}

    # merge edits
    questions = [edits_by_id.get(q.get('qid'), q) for q in original]

    # optional export filters
    approved_only = request.args.get('approved') in ('true', '1', 'yes', 'on')
    high_conf = request.args.get('high_confidence') in ('true', '1', 'yes', 'on')

    def keep(q):
        if approved_only and q.get('status') != 'approved':
            return False
        if high_conf and (float(q.get('confidence', 0)) < 0.8):
            return False
        return True

    questions = [q for q in questions if keep(q)]

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
            return send_file(bio, as_attachment=True,
                             download_name=fname,
                             mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document')
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
            return send_file(bio, as_attachment=True,
                             download_name=fname,
                             mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
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
            return out.getvalue(), 200, {
                'Content-Type': 'text/csv',
                'Content-Disposition': f'attachment; filename="{fname}"'
            }

    return jsonify({'error': 'Invalid format'}), 400


if __name__ == '__main__':
    print("Starting RFP Extraction App on port 5002...")
    print("Open your browser at http://localhost:5002")
    app.run(debug=True, port=5002, host='0.0.0.0')
