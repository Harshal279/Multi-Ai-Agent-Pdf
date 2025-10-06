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

logger = logging.getLogger(__name__)

class PDFRAGAgent:
    """Agent for PDF document processing and retrieval"""
    
    def __init__(self):
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.1-8b-instant"
        
        # Initialize embedding model
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # FAISS index and metadata storage
        self.index_file = 'rag_data/faiss_index.bin'
        self.metadata_file = 'rag_data/metadata.pkl'
        self.documents_file = 'rag_data/documents.pkl'
        
        # Create directory
        os.makedirs('rag_data', exist_ok=True)
        
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
            filename = os.path.basename(pdf_path)
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    
                    if text and text.strip():  # Only add non-empty pages
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
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
        
        return chunks
    
    def ingest_pdf(self, pdf_path: str) -> bool:
        """Ingest a PDF into the RAG system"""
        try:
            # Extract text chunks
            chunks = self._extract_text_from_pdf(pdf_path)
            
            if not chunks:
                logger.warning(f"No text extracted from {pdf_path}")
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
        """Ingest sample PDFs if index is empty"""
        if self.index.ntotal > 0:
            return  # Already have documents
            
        sample_pdfs_dir = 'sample_pdfs'
        
        if not os.path.exists(sample_pdfs_dir):
            logger.warning(f"Sample PDFs directory {sample_pdfs_dir} does not exist. Skipping sample PDF ingestion.")
            return
        
        # Ingest all sample PDFs
        for filename in os.listdir(sample_pdfs_dir):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(sample_pdfs_dir, filename)
                self.ingest_pdf(pdf_path)
    
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
            
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided document context."},
                    {"role": "user", "content": rag_prompt}
                ],
                model=self.model,
                temperature=0.3
            )
            
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
