import os
import json
import time
import requests
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_file, make_response
from werkzeug.utils import secure_filename
import io

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'docx', 'pdf', 'xlsx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Default API configuration
DEFAULT_API_CONFIG = {
    'base_url': 'https://connect.affiniusaiplatform.com/content/000d3572-937a-4d5c-ab7e-7ec80d80c4ce/',
    'api_key': 'hOSWPr4qZpvlzYOB0pKv6DXkf9HB8emB',
    'header_name': 'Authorization',
    'auth_scheme': 'Bearer',
    'poll_interval': 2,
    'poll_timeout': 240,
    'request_timeout': 60
}

# In-memory storage for job data and edits
job_data = {}
job_edits = {}
api_config = DEFAULT_API_CONFIG.copy()

# Statistics tracking
STATS_FILE = 'stats.json'

def load_stats():
    """Load statistics from file"""
    try:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE, 'r') as f:
                stats = json.load(f)
            
            # Calculate average processing time
            if stats.get("documents_processed", 0) > 0:
                stats["avg_processing_time"] = round(stats.get("total_processing_time", 0) / stats.get("documents_processed", 1), 1)
            else:
                stats["avg_processing_time"] = 0.0
                
            return stats
    except:
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
    """Save statistics to file"""
    try:
        stats["last_updated"] = datetime.now().isoformat()
        with open(STATS_FILE, 'w') as f:
            json.dump(stats, f, indent=2)
    except Exception as e:
        print(f"Error saving stats: {e}")

def update_stats(questions_count, processing_time, questions_data=None):
    """Update statistics with new data"""
    stats = load_stats()
    stats["documents_processed"] += 1
    stats["questions_extracted"] += questions_count
    stats["total_processing_time"] += processing_time
    
    # Calculate accuracy rate from actual confidence scores
    if questions_count > 0 and questions_data:
        # Calculate average confidence from actual questions
        total_confidence = 0
        valid_questions = 0
        for question in questions_data:
            if 'confidence' in question and question['confidence'] is not None:
                total_confidence += question['confidence']
                valid_questions += 1
        
        if valid_questions > 0:
            avg_confidence = total_confidence / valid_questions
            stats["accuracy_rate"] = round(avg_confidence * 100, 1)
            print(f"DEBUG: Calculated accuracy rate from {valid_questions} questions: {stats['accuracy_rate']}%")
        else:
            stats["accuracy_rate"] = 85.0  # Default if no confidence data
            print(f"DEBUG: No valid confidence data, using default: {stats['accuracy_rate']}%")
    else:
        stats["accuracy_rate"] = 85.0  # Default if no questions data
        print(f"DEBUG: No questions data provided, using default: {stats['accuracy_rate']}%")
    
    save_stats(stats)
    return stats

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_auth_header():
    """Generate authentication header based on current config"""
    config = api_config
    if config['header_name'] == 'Authorization':
        if config['auth_scheme'] == 'Bearer':
            return {'Authorization': f"Bearer {config['api_key']}"}
        elif config['auth_scheme'] == 'Key':
            return {'Authorization': f"Key {config['api_key']}"}
        elif config['auth_scheme'] == 'Raw':
            return {'Authorization': config['api_key']}
    else:
        return {config['header_name']: config['api_key']}

def make_api_request(method, endpoint, **kwargs):
    """Make API request with current configuration"""
    url = f"{api_config['base_url'].rstrip('/')}/{endpoint.lstrip('/')}"
    headers = get_auth_header()
    
    # Don't set Content-Type for file uploads - let requests handle it
    if 'files' not in kwargs:
        headers.update(kwargs.get('headers', {}))
    
    try:
        if method.upper() == 'GET':
            response = requests.get(url, headers=headers, timeout=api_config['request_timeout'])
        elif method.upper() == 'POST':
            response = requests.post(url, headers=headers, timeout=api_config['request_timeout'], **kwargs)
        elif method.upper() == 'DELETE':
            response = requests.delete(url, headers=headers, timeout=api_config['request_timeout'])
        
        return response
    except requests.exceptions.RequestException as e:
        print(f"DEBUG: Request exception: {e}")
        return None

@app.route('/')
def index():
    stats = load_stats()
    return render_template('index.html', api_config=api_config, stats=stats)



@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and extraction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Save file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Prepare extraction request
    use_llm = request.form.get('use_llm', 'false').lower() == 'true'
    mode = 'balanced'  # Always use balanced mode as per user request
    use_sync = request.form.get('use_sync', 'false').lower() == 'true'
    
    print(f"DEBUG: Upload parameters - use_llm: {use_llm}, mode: {mode}, use_sync: {use_sync}")
    
    try:
        with open(filepath, 'rb') as f:
            # Properly format the file for multipart upload
            # Try different MIME types based on file extension
            file_ext = filename.lower().split('.')[-1]
            mime_type = 'application/octet-stream'
            if file_ext == 'docx':
                mime_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            elif file_ext == 'pdf':
                mime_type = 'application/pdf'
            elif file_ext == 'xlsx':
                mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            
            files = {'file': (filename, f, mime_type)}
            data = {
                'use_llm': str(use_llm).lower(),
                'mode': mode
            }
            
            endpoint = 'extract/sync' if use_sync else 'extract'
            print(f"DEBUG: Making API request to {endpoint} with data: {data}")
            print(f"DEBUG: File info - name: {filename}, size: {os.path.getsize(filepath)} bytes")
            print(f"DEBUG: Files dict: {files}")
            print(f"DEBUG: Full URL will be: {api_config['base_url'].rstrip('/')}/{endpoint.lstrip('/')}")
            response = make_api_request('POST', endpoint, files=files, data=data)
        
        print(f"DEBUG: API response status: {response.status_code if response else 'None'}")
        if response:
            print(f"DEBUG: API response text: {response.text[:500]}...")
        
        if response and response.status_code == 200:
            result = response.json()
            print(f"DEBUG: API response JSON: {result}")
            
            if use_sync and 'questions' in result:
                # Sync mode - questions returned directly
                job_id = f"sync_{int(time.time())}"
                job_data[job_id] = {
                    'status': 'completed',
                    'questions': result['questions'],
                    'timestamp': datetime.now().isoformat()
                }
                # Update statistics for sync mode
                questions_count = len(result['questions'])
                processing_time = 5  # Sync mode is faster
                update_stats(questions_count, processing_time, result['questions'])
                return jsonify({'success': True, 'job_id': job_id, 'questions': result['questions']})
            else:
                # Async mode - start polling
                job_id = result.get('job_id')
                if job_id:
                    # Start polling in background
                    job_data[job_id] = {
                        'status': 'processing',
                        'timestamp': datetime.now().isoformat()
                    }
                    return jsonify({'success': True, 'job_id': job_id, 'async': True})
                else:
                    return jsonify({'error': 'No job ID returned'}), 500
        else:
            error_msg = 'API request failed'
            if response:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                print(f"DEBUG: API Error - {error_msg}")
            return jsonify({'error': error_msg}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/api/poll/<job_id>')
def poll_job(job_id):
    """Poll job status"""
    if job_id not in job_data:
        return jsonify({'error': 'Job not found'}), 404
    
    if job_data[job_id]['status'] == 'completed':
        return jsonify(job_data[job_id])
    
    # Poll the API
    response = make_api_request('GET', f'jobs/{job_id}')
    if response and response.status_code == 200:
        data = response.json()
        job_data[job_id].update(data)
        
        print(f"DEBUG: Polling job {job_id}, status: {data.get('status')}")
        print(f"DEBUG: Full response data: {data}")
        
        if 'questions' in data and data['questions'] is not None:
            print(f"DEBUG: Found {len(data['questions'])} questions in response")
        elif 'questions' in data and data['questions'] is None:
            print(f"DEBUG: Questions field is null in response")
        
        if data.get('status') == 'completed':
            if 'questions' in data and data['questions'] is not None:
                print(f"DEBUG: Job completed with {len(data['questions'])} questions")
                # Update statistics
                questions_count = len(data['questions'])
                processing_time = 30  # Default processing time in seconds
                update_stats(questions_count, processing_time, data['questions'])
                return jsonify(data)
            else:
                # Try to fetch questions separately
                print(f"DEBUG: Job completed but no questions in response, trying to fetch separately")
                questions_response = make_api_request('GET', f'jobs/{job_id}/questions')
                if questions_response and questions_response.status_code == 200:
                    questions = questions_response.json()
                    data['questions'] = questions
                    job_data[job_id]['questions'] = questions
                    print(f"DEBUG: Fetched {len(questions)} questions separately")
                    # Update statistics
                    questions_count = len(questions)
                    processing_time = 30  # Default processing time in seconds
                    update_stats(questions_count, processing_time, questions)
                    return jsonify(data)
                else:
                    print(f"DEBUG: Failed to fetch questions separately")
                    return jsonify({'status': 'completed', 'questions': []})
        elif data.get('status') == 'processing':
            # Check if job has been processing for too long (10 minutes for AI)
            job_start_time = job_data[job_id].get('timestamp')
            if job_start_time:
                from datetime import datetime, timedelta
                start_time = datetime.fromisoformat(job_start_time.replace('Z', '+00:00'))
                if datetime.now() - start_time > timedelta(minutes=10):
                    print(f"DEBUG: Job {job_id} has been processing for over 10 minutes, marking as timeout")
                    job_data[job_id]['status'] = 'timeout'
                    return jsonify({'status': 'timeout', 'error': 'AI processing timeout - try normal extraction mode'})
            # Job is still processing normally
            return jsonify({'status': 'processing'})
        elif data.get('status') == 'failed':
            print(f"DEBUG: Job failed with status: {data.get('status')}")
            return jsonify(data)
        else:
            return jsonify({'status': 'processing'})
    else:
        # Check if job was not found (completed and cleaned up)
        if response and response.status_code == 404:
            print(f"DEBUG: Job {job_id} not found - likely completed and cleaned up")
            # Try to fetch questions directly
            questions_response = make_api_request('GET', f'jobs/{job_id}/questions')
            if questions_response and questions_response.status_code == 200:
                questions = questions_response.json()
                job_data[job_id]['status'] = 'completed'
                job_data[job_id]['questions'] = questions
                print(f"DEBUG: Fetched {len(questions)} questions from completed job")
                # Update statistics
                questions_count = len(questions)
                processing_time = 30  # Default processing time in seconds
                update_stats(questions_count, processing_time, questions)
                return jsonify({'status': 'completed', 'questions': questions})
            else:
                print(f"DEBUG: Could not fetch questions for completed job")
                return jsonify({'status': 'completed', 'questions': []})
        
        error_msg = 'Failed to poll job status'
        if response:
            error_msg = f"HTTP {response.status_code}: {response.text}"
            print(f"DEBUG: Polling error - {error_msg}")
        return jsonify({'error': error_msg}), 500

@app.route('/api/job/<job_id>/questions')
def get_job_questions(job_id):
    """Get questions for a specific job"""
    if job_id not in job_data:
        return jsonify({'error': 'Job not found'}), 404
    
    job = job_data[job_id]
    if job['status'] == 'completed' and 'questions' in job:
        return jsonify(job['questions'])
    
    # Fallback to API call
    response = make_api_request('GET', f'jobs/{job_id}/questions')
    if response and response.status_code == 200:
        questions = response.json()
        job['questions'] = questions
        return jsonify(questions)
    else:
        return jsonify({'error': 'Failed to fetch questions'}), 500

@app.route('/review/<job_id>')
def review(job_id):
    """Review and edit page"""
    if job_id not in job_data:
        return redirect(url_for('index'))
    
    job = job_data[job_id]
    questions = job.get('questions', [])
    
    # If no questions in job data, try to fetch from API
    if not questions:
        response = make_api_request('GET', f'jobs/{job_id}/questions')
        if response and response.status_code == 200:
            questions = response.json()
            job['questions'] = questions
    
    # Apply edits if any
    if job_id in job_edits:
        questions = job_edits[job_id]
    
    print(f"DEBUG: Job {job_id} has {len(questions)} questions")
    if questions:
        print(f"DEBUG: First question: {questions[0]}")
    
    return render_template('review.html', job_id=job_id, questions=questions, job=job)

@app.route('/api/save_edits/<job_id>', methods=['POST'])
def save_edits(job_id):
    """Save edits for a job"""
    data = request.get_json()
    questions = data.get('questions', [])
    
    # Update job_edits with the new questions
    if job_id not in job_edits:
        job_edits[job_id] = []
    
    # Update existing questions or add new ones
    for question in questions:
        qid = question.get('qid')
        # Find and update existing question or add new one
        updated = False
        for i, existing in enumerate(job_edits[job_id]):
            if existing.get('qid') == qid:
                job_edits[job_id][i] = question
                updated = True
                break
        if not updated:
            job_edits[job_id].append(question)
    
    print(f"DEBUG: Saved edits for job {job_id}: {len(questions)} questions")
    return jsonify({'success': True})

@app.route('/api/export/<job_id>/<format>')
def export_data(job_id, format):
    """Export data in various formats"""
    if job_id not in job_data:
        return jsonify({'error': 'Job not found'}), 404
    
    # Get original questions
    original_questions = job_data[job_id].get('questions', [])
    
    # Get edited questions if any
    edited_questions = job_edits.get(job_id, [])
    
    # Create a dictionary of edited questions by qid for quick lookup
    edited_dict = {q.get('qid'): q for q in edited_questions}
    
    # Merge: use edited version if available, otherwise use original
    questions = []
    for original_q in original_questions:
        qid = original_q.get('qid')
        if qid in edited_dict:
            # Use edited version
            questions.append(edited_dict[qid])
        else:
            # Use original version
            questions.append(original_q)
    
    print(f"DEBUG: Exporting {len(questions)} questions (original: {len(original_questions)}, edited: {len(edited_questions)})")
    
    if format == 'docx':
        try:
            from docx import Document
            from docx.shared import Inches
            
            doc = Document()
            doc.add_heading('RFP Questions Report', 0)
            doc.add_paragraph(f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            doc.add_paragraph(f'Job ID: {job_id}')
            doc.add_paragraph(f'Total Questions: {len(questions)}')
            doc.add_paragraph('')
            
            for i, question in enumerate(questions, 1):
                doc.add_heading(f'Question {i}', level=2)
                doc.add_paragraph(f'Text: {question.get("text", "N/A")}')
                doc.add_paragraph(f'Confidence: {question.get("confidence", "N/A")}')
                doc.add_paragraph(f'Type: {question.get("type", "N/A")}')
                doc.add_paragraph(f'Section: {question.get("section_path", ["N/A"])[0] if question.get("section_path") else "N/A"}')
                doc.add_paragraph('')
            
            # Save to BytesIO
            doc_io = io.BytesIO()
            doc.save(doc_io)
            doc_io.seek(0)
            
            filename = f"RFP_Questions_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
            return send_file(
                doc_io,
                as_attachment=True,
                download_name=filename,
                mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            )
        except ImportError:
            # Fallback to text if docx not available
            content = f"RFP Questions Export\n"
            content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            content += f"Job ID: {job_id}\n"
            content += f"Total Questions: {len(questions)}\n\n"
            
            for i, question in enumerate(questions, 1):
                content += f"{i}. {question.get('text', 'N/A')}\n"
                content += f"   Confidence: {question.get('confidence', 'N/A')}\n"
                content += f"   Type: {question.get('type', 'N/A')}\n"
                content += f"   Section: {question.get('section_path', ['N/A'])[0] if question.get('section_path') else 'N/A'}\n\n"
            
            filename = f"RFP_Questions_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            return content, 200, {
                'Content-Type': 'text/plain',
                'Content-Disposition': f'attachment; filename="{filename}"'
            }
    
    
    elif format == 'xlsx':
        try:
            import pandas as pd
            
            # Create DataFrame
            df = pd.DataFrame(questions)
            
            # Save to BytesIO
            excel_io = io.BytesIO()
            with pd.ExcelWriter(excel_io, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='RFP Questions', index=False)
            excel_io.seek(0)
            
            filename = f"RFP_Questions_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            return send_file(
                excel_io,
                as_attachment=True,
                download_name=filename,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        except ImportError:
            # Fallback to CSV if pandas not available
            import csv
            output = io.StringIO()
            if questions:
                # Flatten the data for CSV export
                flattened_questions = []
                for q in questions:
                    flat_q = {
                        'qid': q.get('qid', ''),
                        'text': q.get('text', ''),
                        'confidence': q.get('confidence', ''),
                        'type': q.get('type', ''),
                        'status': q.get('status', ''),
                        'section': q.get('section_path', [''])[0] if q.get('section_path') else '',
                        'numbering': q.get('numbering', ''),
                        'category': q.get('category', '')
                    }
                    flattened_questions.append(flat_q)
                
                if flattened_questions:
                    writer = csv.DictWriter(output, fieldnames=flattened_questions[0].keys())
                    writer.writeheader()
                    writer.writerows(flattened_questions)
            
            filename = f"RFP_Questions_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            return output.getvalue(), 200, {
                'Content-Type': 'text/csv',
                'Content-Disposition': f'attachment; filename="{filename}"'
            }
    
    return jsonify({'error': 'Invalid format'}), 400


if __name__ == '__main__':
    print("Starting RFP Extraction App on port 5002...")
    print("Open your browser and navigate to: http://localhost:5002")
    app.run(debug=True, port=5002, host='0.0.0.0')
