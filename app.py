import os
import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import traceback

# Import our custom agents
from agents.controller_agent import ControllerAgent
from agents.pdf_rag_agent import PDFRAGAgent
from agents.web_search_agent import WebSearchAgent
from agents.arxiv_agent import ArxivAgent
from utils.logging_utils import setup_logging, log_decision, log_agent_result  # Added log_agent_result

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('sample_pdfs', exist_ok=True)

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize agents
controller_agent = ControllerAgent()
pdf_rag_agent = PDFRAGAgent()
web_search_agent = WebSearchAgent()
arxiv_agent = ArxivAgent()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
    
@app.route('/health')
def health():
    return jsonify({'status': 'healthy'}), 200
    
@app.route('/')
def index():
    """Serve the main interface"""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    """Main endpoint to process user queries"""
    try:
        data = request.json
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        logger.info(f"Received query: {query}")
        
        # Use controller to decide which agents to call
        decision = controller_agent.route_query(query)
        
        # Log the decision
        log_decision(query, decision)
        
        # Execute the decided agents
        results = {}
        final_answer = ""
        
        if 'pdf_rag' in decision['agents']:
            logger.info("Calling PDF RAG agent")
            pdf_result = pdf_rag_agent.process_query(query)
            results['pdf_rag'] = pdf_result
            log_agent_result('pdf_rag', query, pdf_result)  # Added: Log agent result
            
        if 'web_search' in decision['agents']:
            logger.info("Calling Web Search agent")
            web_result = web_search_agent.process_query(query)
            results['web_search'] = web_result
            log_agent_result('web_search', query, web_result)  # Added: Log agent result
            
        if 'arxiv' in decision['agents']:
            logger.info("Calling ArXiv agent")
            arxiv_result = arxiv_agent.process_query(query)
            results['arxiv'] = arxiv_result
            log_agent_result('arxiv', query, arxiv_result)  # Added: Log agent result
        
        # Synthesize final answer using controller
        final_answer = controller_agent.synthesize_answer(query, results)
        
        response = {
            'answer': final_answer,
            'agents_used': decision['agents'],
            'reasoning': decision['reasoning'],
            'sources': results,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    """Upload and ingest PDF into RAG system"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Ingest the PDF into the RAG system
            success = pdf_rag_agent.ingest_pdf(filepath)
            
            if success:
                # Retention Policy: Delete the uploaded PDF immediately after ingestion to minimize storage of potentially sensitive data (e.g., PII).
                # Files are not retained long-term; only extracted text chunks are stored in the FAISS index for RAG queries.
                os.remove(filepath)
                return jsonify({
                    'message': f'PDF {filename} uploaded, ingested, and deleted successfully (per retention policy)',
                    'filename': filename
                })
            else:
                os.remove(filepath)  # Clean up even on failure
                return jsonify({'error': 'Failed to ingest PDF'}), 500
                
        return jsonify({'error': 'Invalid file type. Only PDF files allowed.'}), 400
        
    except Exception as e:
        logger.error(f"Error uploading PDF: {str(e)}")
        return jsonify({'error': 'Failed to upload PDF'}), 500

@app.route('/logs')
def get_logs():
    """Retrieve system logs and decisions"""
    try:
        logs = []
        log_file = 'logs/decisions.log'
        
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        continue
        
        # Return last 50 logs
        return jsonify(logs[-50:])
        
    except Exception as e:
        logger.error(f"Error retrieving logs: {str(e)}")
        return jsonify({'error': 'Failed to retrieve logs'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.route('/test_logs')
def test_logs():
    """Create test logs for debugging"""
    try:
        import os
        import json
        from datetime import datetime
        
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        # Create sample log entries
        sample_logs = [
            {
                'timestamp': datetime.now().isoformat(),
                'query': 'What is artificial intelligence?',
                'decision': {
                    'agents': ['web_search'],
                    'reasoning': 'Query asks for general information about AI, using web search'
                },
                'type': 'routing_decision'
            },
            {
                'timestamp': datetime.now().isoformat(),
                'query': 'Find papers about neural networks',
                'decision': {
                    'agents': ['arxiv'],
                    'reasoning': 'Query asks for academic papers, routing to ArXiv agent'
                },
                'type': 'routing_decision'
            },
            {
                'timestamp': datetime.now().isoformat(),
                'query': 'Summarize uploaded document',
                'decision': {
                    'agents': ['pdf_rag'],
                    'reasoning': 'Query about document content, using PDF RAG agent'
                },
                'type': 'routing_decision'
            }
        ]
        
        # Write to log file
        with open('logs/decisions.log', 'w') as f:
            for log in sample_logs:
                f.write(json.dumps(log) + '\n')
        
        return jsonify({'message': f'Created {len(sample_logs)} test log entries'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting Flask app on port {port}")
    logger.info(f"Debug mode: {debug_mode}")
    
    app.run(host='0.0.0.0', port=port, debug=False)
