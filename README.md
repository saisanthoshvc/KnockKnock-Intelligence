# 🚀 KnockKnock Intelligence - AI-Powered RFP Question Extraction

<div align="center">

![KnockKnock Intelligence](https://img.shields.io/badge/KnockKnock-Intelligence-blue?style=for-the-badge&logo=robot)
![Python](https://img.shields.io/badge/Python-3.7+-green?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.3.3-red?style=for-the-badge&logo=flask)
![AI](https://img.shields.io/badge/AI-Powered-purple?style=for-the-badge&logo=brain)

**Professional AI-powered RFP question extraction with modern glassmorphism UI**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-http://localhost:5002-brightgreen?style=for-the-badge)](http://localhost:5002)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

</div>

---

## 🎯 **Overview**

**KnockKnock Intelligence** is a cutting-edge, production-grade web application that leverages AI to automatically extract and categorize questions from RFP (Request for Proposal) documents. Built with modern web technologies and featuring a stunning glassmorphism UI, it transforms complex document analysis into an intuitive, professional experience.

### ✨ **Key Features**

- 🤖 **AI-Powered Classification** - Advanced machine learning for intelligent question categorization
- ⚡ **Dual Processing Modes** - Choose between AI classification or fast sync processing
- 🎨 **Modern Glassmorphism UI** - Professional, responsive design with smooth animations
- 📊 **Real-time Analytics** - Live statistics and performance metrics
- ✏️ **Inline Editing** - Edit questions directly with auto-save functionality
- 📤 **Multi-format Export** - Download results as DOCX, Excel, or CSV
- 🔍 **Advanced Filtering** - Filter by confidence, type, section, and search terms
- 📱 **Fully Responsive** - Perfect experience across all devices
- ⌨️ **Keyboard Shortcuts** - Power user features for enhanced productivity
- 🎭 **Theme Support** - Beautiful dark mode with professional styling

---

## 🏗️ **Architecture & Technology Stack**

### **Backend**
- **Framework**: Flask 2.3.3 with Python 3.7+
- **API Integration**: Posit Connect API with Bearer token authentication
- **Data Processing**: pandas, openpyxl for Excel export
- **Document Processing**: python-docx for Word document generation
- **Session Management**: Flask sessions for job tracking
- **Statistics**: JSON-based persistent storage

### **Frontend**
- **Design System**: Custom CSS with glassmorphism effects
- **Animations**: CSS3 transitions, transforms, and keyframe animations
- **Interactivity**: Vanilla JavaScript with modern ES6+ features
- **Responsive Design**: Mobile-first approach with flexible layouts
- **User Experience**: Toast notifications, loading states, auto-save
- **Accessibility**: Keyboard navigation and screen reader support

### **API Integration**
- **Authentication**: Bearer token with Posit Connect
- **Endpoints**: `/extract/`, `/jobs/{id}`, `/jobs/{id}/questions`
- **Processing**: Asynchronous job polling with real-time status updates
- **Error Handling**: Comprehensive error management and fallback mechanisms

---

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.7 or higher
- pip package manager
- Modern web browser (Chrome, Firefox, Safari, Edge)

### **Installation**

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/knockknock-intelligence.git
   cd knockknock-intelligence
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv3
   source venv3/bin/activate  # On Windows: venv3\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app_simple.py
   ```

5. **Access the application**
   Open your browser and navigate to: `http://localhost:5002`

---

## 📖 **Usage Guide**

### **1. Upload & Parse Documents**

#### **Supported File Formats**
- **DOCX** - Microsoft Word documents
- **PDF** - Portable Document Format
- **XLSX** - Microsoft Excel spreadsheets
- **Maximum Size**: 16MB per file

#### **Processing Modes**
- **🤖 AI Classification**: Advanced AI-powered question categorization (recommended)
- **⚡ Sync Mode**: Fast processing for smaller documents

#### **Upload Process**
1. Click "Upload & Parse" in the header or scroll to the upload section
2. Drag and drop your file or click to browse
3. Select your processing mode (AI Classification or Sync Mode)
4. Click "Extract Questions" to begin processing
5. Monitor real-time progress with animated status indicators

### **2. Review & Edit Questions**

#### **Interactive Table Features**
- **Inline Editing**: Click on any question text to edit directly
- **Confidence Adjustment**: Modify confidence scores with sliders
- **Status Management**: Mark questions as approved, pending, or rejected
- **Real-time Validation**: Instant feedback on data quality

#### **Advanced Filtering**
- **Confidence Filter**: Set minimum confidence threshold
- **Type Filter**: Filter by question type (technical, business, etc.)
- **Status Filter**: Show only approved, pending, or rejected questions
- **Search**: Full-text search across all question content

#### **Bulk Operations**
- **Select All**: Choose all questions at once
- **Save All Changes**: Persist all modifications with one click
- **Export Filtered**: Download only filtered results

### **3. Export & Download**

#### **Supported Export Formats**
- **📄 DOCX**: Professional Word documents with formatting
- **📊 Excel**: Spreadsheet format with data analysis capabilities
- **📋 CSV**: Comma-separated values for data import

#### **Export Options**
- **All Questions**: Export complete dataset
- **Approved Only**: Export only approved questions
- **High Confidence**: Export questions above confidence threshold
- **Custom Filtering**: Export based on applied filters

---

## 🔧 **API Documentation**

### **Authentication**
```http
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json
```

### **Core Endpoints**

#### **Upload Document**
```http
POST /api/upload
Content-Type: multipart/form-data

Parameters:
- file: Document file (DOCX, PDF, XLSX)
- use_llm: Boolean (AI classification)
- use_sync: Boolean (Sync mode)
- mode: String (balanced, fast, thorough)
```

#### **Poll Job Status**
```http
GET /api/poll/{job_id}

Response:
{
  "status": "processing|completed|failed",
  "progress": {
    "step": "extracting|classifying",
    "message": "Processing document..."
  },
  "questions": [...] // Available when completed
}
```

#### **Get Questions**
```http
GET /api/job/{job_id}/questions

Response:
{
  "questions": [
    {
      "qid": "unique_question_id",
      "text": "Question content",
      "confidence": 0.85,
      "type": "technical",
      "section_path": ["Section Name"],
      "status": "pending"
    }
  ]
}
```

#### **Save Edits**
```http
POST /api/save_edits/{job_id}
Content-Type: application/json

Body:
{
  "questions": [
    {
      "qid": "question_id",
      "text": "Updated question text",
      "confidence": 0.90,
      "status": "approved"
    }
  ]
}
```

#### **Export Data**
```http
GET /api/export/{job_id}/{format}?approved=true&high_confidence=true

Formats: docx, xlsx, csv
```

---

## 🎨 **Frontend Architecture**

### **Design System**

#### **Color Palette**
```css
/* Dark Theme (Default) */
--primary-bg: #0f172a;
--secondary-bg: #1e293b;
--accent-blue: #3b82f6;
--accent-purple: #8b5cf6;
--text-primary: #f8fafc;
--text-secondary: #cbd5e1;

/* Light Theme */
--primary-bg: #ffffff;
--secondary-bg: #f8fafc;
--accent-blue: #2563eb;
--text-primary: #1e293b;
--text-secondary: #64748b;
```

#### **Glassmorphism Effects**
- **Backdrop Blur**: `backdrop-filter: blur(10px)`
- **Transparency**: `rgba(255, 255, 255, 0.1)`
- **Gradient Overlays**: Multi-layer gradient backgrounds
- **Shadow System**: Layered shadows for depth

#### **Animation System**
- **Micro-interactions**: Hover effects, focus states
- **Page Transitions**: Smooth scrolling, fade effects
- **Loading States**: Spinner animations, progress bars
- **Toast Notifications**: Slide-in notifications

### **Component Architecture**

#### **Base Template (`base.html`)**
- **Global Styles**: CSS variables, typography, layout
- **Navigation**: Header with smooth scrolling
- **Theme System**: Light/dark mode toggle
- **Toast System**: Global notification manager
- **Loading Overlay**: Full-screen loading states

#### **Homepage (`index.html`)**
- **Upload Interface**: Drag & drop with visual feedback
- **Processing Options**: Radio button selection
- **Real-time Stats**: Live metrics dashboard
- **Progress Tracking**: Animated status indicators

#### **Review Page (`review.html`)**
- **Data Table**: Sortable, filterable question list
- **Inline Editing**: Direct text editing capabilities
- **Filter System**: Advanced filtering controls
- **Export Modal**: Format selection and options

### **JavaScript Features**

#### **Core Functionality**
```javascript
// File Upload with Validation
function handleFile(file) {
  if (!validateFile(file)) return;
  // Process file with visual feedback
}

// Auto-save with Debouncing
function setupAutoSave() {
  // Auto-save after 2 seconds of inactivity
}

// Real-time Statistics
function refreshStats() {
  // Update metrics every 30 seconds
}
```

#### **User Experience Enhancements**
- **Keyboard Shortcuts**: Ctrl+S (save), Ctrl+F (search), Escape (close)
- **Drag & Drop**: Visual feedback for file uploads
- **Form Validation**: Real-time input validation
- **Error Handling**: Graceful error recovery
- **Performance**: Optimized rendering for large datasets

---

## 📊 **Performance & Analytics**

### **Real-time Metrics**
- **Documents Processed**: Total uploads tracked
- **Questions Extracted**: Cumulative question count
- **Accuracy Rate**: Average confidence score
- **Processing Time**: Average processing duration

### **Performance Optimizations**
- **Lazy Loading**: Load data on demand
- **Debounced Input**: Reduce API calls during typing
- **Caching**: Session-based data caching
- **Compression**: Optimized asset delivery

---

## 🛠️ **Development**

### **Project Structure**
```
knockknock-intelligence/
├── app_simple.py              # Main Flask application
├── requirements.txt           # Python dependencies
├── stats.json                # Application statistics
├── templates/                # HTML templates
│   ├── base.html            # Base template with global styles
│   ├── index.html           # Homepage with upload interface
│   ├── review.html          # Review & edit page
│   └── error.html           # Error page template
├── backup/                   # Version backups
│   ├── v1/ ... v9/          # Historical versions
└── final_deployment/         # Production-ready version
```

### **Key Dependencies**
```txt
Flask==2.3.3                 # Web framework
requests==2.31.0             # HTTP client
python-docx==0.8.11          # Word document generation
pandas                       # Data manipulation
openpyxl                     # Excel file support
```

### **Environment Setup**
```bash
# Development
export FLASK_ENV=development
export FLASK_DEBUG=1

# Production
export FLASK_ENV=production
export FLASK_DEBUG=0
```

---

## 🔒 **Security & Privacy**

### **Data Protection**
- **Server-side Processing**: All AI processing on secure servers
- **Session Management**: Secure session handling
- **Input Validation**: Comprehensive input sanitization
- **Error Handling**: No sensitive data in error messages

### **API Security**
- **Authentication**: Bearer token authentication
- **Rate Limiting**: Built-in request throttling
- **CORS**: Proper cross-origin resource sharing
- **HTTPS**: Secure communication protocols

---

## 🚀 **Deployment**

### **Production Deployment**
1. **Environment Setup**
   ```bash
   export FLASK_ENV=production
   pip install -r requirements.txt
   ```

2. **WSGI Server** (Recommended)
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5002 app_simple:app
   ```

3. **Docker Deployment**
   ```dockerfile
   FROM python:3.7-slim
   COPY . /app
   WORKDIR /app
   RUN pip install -r requirements.txt
   EXPOSE 5002
   CMD ["python", "app_simple.py"]
   ```

### **Environment Variables**
```bash
export API_KEY=your_posit_connect_key
export API_BASE_URL=https://connect.affiniusaiplatform.com/content/000d3572-937a-4d5c-ab7e-7ec80d80c4ce
export FLASK_SECRET_KEY=your_secret_key
```

---

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### **Development Workflow**
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- **Posit Connect** for AI processing capabilities
- **Flask Community** for the excellent web framework
- **Open Source Contributors** for the amazing libraries used

---

## 📞 **Support**

- **Documentation**: [Full Documentation]([docs/](https://connect.affiniusaiplatform.com/content/000d3572-937a-4d5c-ab7e-7ec80d80c4ce/docs))

---

<div align="center">

**Built with ❤️ by the KnockKnock Intelligence Team**

</div>
