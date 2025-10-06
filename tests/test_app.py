import pytest
from app import app  # Assuming app.py is in the parent directory
from agents.controller_agent import ControllerAgent
from agents.pdf_rag_agent import PDFRAGAgent

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Multi-Agent AI System' in response.data

def test_ask_question(client):
    response = client.post('/ask', json={'query': 'Test query'})
    assert response.status_code == 200
    data = response.json
    assert 'answer' in data

def test_upload_pdf(client, tmp_path):
    # Create a temp PDF file
    pdf_path = tmp_path / 'test.pdf'
    pdf_path.write_bytes(b'%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj 3 0 obj<</Type/Page/MediaBox[0 0 3 3]>>endobj\nxref\n0 4\n0000000000 65535 f\n0000000010 00000 n\n0000000053 00000 n\n0000000102 00000 n\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n149\n%%EOF')
    
    response = client.post('/upload_pdf', data={'file': (open(pdf_path, 'rb'), 'test.pdf')})
    assert response.status_code == 200
    data = response.json
    assert 'message' in data

def test_controller_route_query():
    controller = ControllerAgent()
    decision = controller.route_query('What is AI?')
    assert 'agents' in decision
    assert 'reasoning' in decision

def test_pdf_rag_ingest(tmp_path):
    rag_agent = PDFRAGAgent()
    pdf_path = tmp_path / 'test.pdf'
    pdf_path.write_bytes(b'%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj 3 0 obj<</Type/Page/MediaBox[0 0 3 3]>>endobj\nxref\n0 4\n0000000000 65535 f\n0000000010 00000 n\n0000000053 00000 n\n0000000102 00000 n\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n149\n%%EOF')
    success = rag_agent.ingest_pdf(str(pdf_path))
    assert success is True