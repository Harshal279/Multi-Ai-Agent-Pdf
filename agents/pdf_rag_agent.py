import os
import json
import logging
import faiss
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import pickle
from dotenv import load_dotenv
from groq import Groq
import pdfplumber
from reportlab.lib.pagesizes import letter  # NEW: For PDF creation
from reportlab.pdfgen import canvas     # NEW: For PDF creation
import time  # NEW: For retries

logger = logging.getLogger(__name__)

class PDFRAGAgent:
    """Agent for PDF document processing and retrieval"""
    
    def __init__(self):
        # FIXED: Load env and validate API key
        load_dotenv()
        api_key =  Groq(api_key=os.getenv("GROQ_API_KEY"))
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is missing. Set it in your .env file or Render dashboard.")
        self.groq_client = Groq(api_key=api_key)
        self.model = "llama-3.1-8b-instant"
        
        # Initialize embedding model
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # FAISS index and metadata storage
        self.index_file = os.path.join('rag_data', 'faiss_index.bin')  # FIXED: Cross-platform
        self.metadata_file = os.path.join('rag_data', 'metadata.pkl')
        self.documents_file = os.path.join('rag_data', 'documents.pkl')
        
        # Create directories
        os.makedirs('rag_data', exist_ok=True)
        os.makedirs('sample_pdfs', exist_ok=True)  # FIXED: Ensure dir exists
        os.makedirs('uploads', exist_ok=True)      # FIXED: For uploaded PDFs
        
        # Load existing index or create new one
        self.index = None
        self.metadata = []
        self.documents = []
        self._load_or_create_index()
        
        # Ingest sample PDFs if index is empty
        self._ingest_sample_pdfs()
    
    def _load_or_create_index(self):
        """Load existing FAISS index or create a new one"""
        try:
            if os.path.exists(self.index_file):
                self.index = faiss.read_index(self.index_file)
                with open(self.metadata_file, 'rb') as f:
                    self.metadata = pickle.load(f)
                with open(self.documents_file, 'rb') as f:
                    self.documents = pickle.load(f)
                logger.info(f"Loaded existing index with {self.index.ntotal} documents")
            else:
                # Create new index (384 dimensions for all-MiniLM-L6-v2)
                self.index = faiss.IndexFlatL2(384)
                self.metadata = []
                self.documents = []
                logger.info("Created new FAISS index")
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            self.index = faiss.IndexFlatL2(384)
            self.metadata = []
            self.documents = []
    
    def _save_index(self):
        """Save FAISS index and metadata"""
        try:
            faiss.write_index(self.index, self.index_file)
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.metadata, f)
            with open(self.documents_file, 'wb') as f:
                pickle.dump(self.documents, f)
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    def _extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text chunks from PDF with metadata"""
        chunks = []
        
        try:
            # FIXED: Check file exists and is readable
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file not found: {pdf_path}")
                return chunks
            
            filename = os.path.basename(pdf_path)
            logger.info(f"Attempting to extract text from: {pdf_path}")
            
            with pdfplumber.open(pdf_path) as pdf:
                logger.info(f"PDF has {len(pdf.pages)} pages")
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    
                    if text and text.strip():  # Only add non-empty pages
                        logger.debug(f"Page {page_num + 1} extracted {len(text)} chars")
                        # Simple chunking by sentences
                        sentences = text.split('. ')
                        current_chunk = ""
                        
                        for sentence in sentences:
                            if len(current_chunk + sentence) < 1000:  # Max chunk size
                                current_chunk += sentence + ". "
                            else:
                                if current_chunk.strip():
                                    chunks.append({
                                        'text': current_chunk.strip(),
                                        'filename': filename,
                                        'page_number': page_num + 1,
                                        'chunk_id': len(chunks)
                                    })
                                current_chunk = sentence + ". "
                        
                        # Add remaining text
                        if current_chunk.strip():
                            chunks.append({
                                'text': current_chunk.strip(),
                                'filename': filename,
                                'page_number': page_num + 1,
                                'chunk_id': len(chunks)
                            })
                    else:
                        logger.warning(f"Page {page_num + 1} has no extractable text (possibly image-based)")
            
            logger.info(f"Extracted {len(chunks)} chunks from {pdf_path}")
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
        
        return chunks
    
    def ingest_pdf(self, pdf_path: str) -> bool:
        """Ingest a PDF into the RAG system"""
        try:
            # FIXED: Use cross-platform path
            pdf_path = os.path.abspath(pdf_path)  # Ensure absolute for safety
            
            # Extract text chunks
            chunks = self._extract_text_from_pdf(pdf_path)
            
            if not chunks:
                logger.warning(f"No text extracted from {pdf_path} - skipping ingestion")
                return False
            
            # Generate embeddings
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.embedder.encode(texts)
            
            # Add to FAISS index
            self.index.add(embeddings.astype('float32'))
            
            # Update metadata and documents
            self.metadata.extend(chunks)
            self.documents.extend(texts)
            
            # Save updated index
            self._save_index()
            
            logger.info(f"Successfully ingested {len(chunks)} chunks from {pdf_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error ingesting PDF {pdf_path}: {e}")
            return False
    
    def _ingest_sample_pdfs(self):
        """Ingest sample NebulaByte PDFs if index is empty"""
        if self.index.ntotal > 0:
            return  # Already have documents
            
        sample_pdfs_dir = 'sample_pdfs'
        
        # FIXED: Create samples first, then ingest
        self._create_sample_pdfs()
        
        # Ingest all sample PDFs
        for filename in os.listdir(sample_pdfs_dir):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(sample_pdfs_dir, filename)
                success = self.ingest_pdf(pdf_path)
                if success:
                    logger.info(f"Ingested sample: {filename}")
                else:
                    logger.warning(f"Failed to ingest sample: {filename}")
    
    def _create_sample_pdfs(self):
        """Create sample NebulaByte dialog PDFs using reportlab"""
        sample_pdfs_dir = 'sample_pdfs'
        os.makedirs(sample_pdfs_dir, exist_ok=True)
        
        sample_dialogs = [
            {'filename': 'ai_ethics_dialog.pdf', 'content': """NebulaByte AI Ethics Discussion

Participant A: What are the key ethical considerations when developing AI systems?

Participant B: I believe transparency is crucial. Users should understand how AI makes decisions that affect them. We also need to address bias in training data.

Participant A: Absolutely. Privacy is another major concern. How do we balance AI capabilities with user privacy?

Participant B: Data minimization principles should guide us. Collect only what is necessary and implement strong security measures."""},
            {'filename': 'machine_learning_trends.pdf', 'content': """NebulaByte ML Trends Analysis

Speaker 1: The field of machine learning is evolving rapidly. What trends are you seeing?

Speaker 2: Transformer architectures have revolutionized NLP. We are seeing applications beyond text now.

Speaker 1: How about few-shot and zero-shot learning?

Speaker 2: These approaches are game-changing. They allow models to adapt to new tasks with minimal examples."""},
            # Add the other 3 samples similarly...
            {'filename': 'neural_networks_basics.pdf', 'content': """NebulaByte Neural Networks Fundamentals

Instructor: Let's review the basics of neural networks.

Student: How do neurons in artificial networks compare to biological ones?

Instructor: Artificial neurons are simplified models. They receive inputs, apply weights, add bias, and pass through an activation function.

Student: What's the purpose of activation functions?

Instructor: They introduce non-linearity, allowing networks to learn complex patterns. Common ones include ReLU, sigmoid, and tanh."""},
            {'filename': 'deep_learning_applications.pdf', 'content': """NebulaByte Deep Learning Applications

Expert A: Deep learning has transformed many industries. What applications do you find most impactful?

Expert B: Computer vision in healthcare, particularly medical imaging diagnosis, has saved countless lives.

Expert A: Natural language processing has also seen remarkable progress.

Expert B: Yes, from machine translation to code generation. Large language models can now engage in complex reasoning tasks."""},
            {'filename': 'ai_future_predictions.pdf', 'content': """NebulaByte Future of AI Discussion

Futurist: Where do you see AI heading in the next decade?

Researcher: I expect continued growth in multimodal AI and improved reasoning capabilities.

Futurist: What about artificial general intelligence?

Researcher: AGI remains challenging. We may see more specialized AI systems working together rather than single general-purpose systems."""}
        ]
        
        for dialog in sample_dialogs:
            pdf_path = os.path.join(sample_pdfs_dir, dialog['filename'])
            
            if not os.path.exists(pdf_path):
                try:
                    # FIXED: Use reportlab to create PDF
                    c = canvas.Canvas(pdf_path, pagesize=letter)
                    width, height = letter
                    y = height - 50  # Start from top
                    
                    for line in dialog['content'].split('\n'):
                        if y < 50:  # New page if needed
                            c.showPage()
                            y = height - 50
                        c.drawString(50, y, line)
                        y -= 15  # Line spacing
                    
                    c.save()
                    logger.info(f"Created sample PDF: {dialog['filename']}")
                except Exception as e:
                    logger.error(f"Error creating sample PDF {dialog['filename']}: {e}")
    
    def retrieve_relevant_docs(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve most relevant documents for a query"""
        try:
            if self.index.ntotal == 0:
                return []
            
            # Generate query embedding
            query_embedding = self.embedder.encode([query])
            
            # Search FAISS index
            distances, indices = self.index.search(query_embedding.astype('float32'), k)
            
            # Get relevant documents with metadata
            relevant_docs = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.metadata):
                    doc = self.metadata[idx].copy()
                    doc['score'] = float(distances[0][i])
                    relevant_docs.append(doc)
            
            return relevant_docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query using RAG approach"""
        try:
            # Retrieve relevant documents
            relevant_docs = self.retrieve_relevant_docs(query)
            
            if not relevant_docs:
                return {
                    'answer': 'No relevant documents found in the knowledge base.',
                    'sources': [],
                    'agent': 'pdf_rag'
                }
            
            # Prepare context from retrieved documents
            context_parts = []
            for doc in relevant_docs[:3]:  # Use top 3 results
                context_parts.append(f"From {doc['filename']}, page {doc['page_number']}:\\n{doc['text']}")
            context = "\\n\\n".join(context_parts)
            
            # Generate answer using RAG
            rag_prompt = f"""Based on the following document excerpts, answer the user's question. If the information isn't available in the documents, say so clearly.

Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the available information."""
            
            # FIXED: Add retries for API calls
            max_retries = 3
            response = None
            for attempt in range(max_retries):
                try:
                    response = self.groq_client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided document context."},
                            {"role": "user", "content": rag_prompt}
                        ],
                        model=self.model,
                        temperature=0.3
                    )
                    break
                except Exception as api_err:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Groq API attempt {attempt + 1} failed: {api_err}. Retrying...")
                    time.sleep(1)
            
            answer = response.choices[0].message.content.strip()
            
            return {
                'answer': answer,
                'sources': relevant_docs[:3],
                'agent': 'pdf_rag',
                'num_sources': len(relevant_docs)
            }
            
        except Exception as e:
            logger.error(f"Error processing RAG query: {e}")
            return {
                'answer': f'Error processing query: {str(e)}',
                'sources': [],
                'agent': 'pdf_rag'
            }
