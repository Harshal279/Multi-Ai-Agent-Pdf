import os
import json
import logging
from typing import Dict, Any, List
import requests
import xml.etree.ElementTree as ET
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ArxivAgent:
    """Agent for ArXiv paper search and summarization"""
    
    def __init__(self):
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.1-8b-instant"
        self.arxiv_api_base = "http://export.arxiv.org/api/query"
    
    def search_arxiv(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search ArXiv for papers"""
        try:
            # Format search query for ArXiv API
            search_query = f"all:{query}"
            params = {
                'search_query': search_query,
                'start': 0,
                'max_results': max_results,
                'sortBy': 'lastUpdatedDate',
                'sortOrder': 'descending'
            }
            
            response = requests.get(self.arxiv_api_base, params=params)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            namespace = {'atom': 'http://www.w3.org/2005/Atom',
                        'arxiv': 'http://arxiv.org/schemas/atom'}
            
            papers = []
            entries = root.findall('atom:entry', namespace)
            
            for entry in entries:
                title = entry.find('atom:title', namespace)
                summary = entry.find('atom:summary', namespace)
                published = entry.find('atom:published', namespace)
                arxiv_id = entry.find('atom:id', namespace)
                authors = entry.findall('atom:author', namespace)
                
                # Extract author names
                author_names = []
                for author in authors:
                    name = author.find('atom:name', namespace)
                    if name is not None:
                        author_names.append(name.text)
                
                papers.append({
                    'title': title.text.strip() if title is not None else '',
                    'summary': summary.text.strip() if summary is not None else '',
                    'published': published.text if published is not None else '',
                    'arxiv_id': arxiv_id.text if arxiv_id is not None else '',
                    'authors': author_names,
                    'pdf_url': arxiv_id.text.replace('abs', 'pdf') + '.pdf' if arxiv_id is not None else ''
                })
            
            return papers
            
        except Exception as e:
            logger.error(f"ArXiv search error: {e}")
            return []
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process ArXiv search query"""
        try:
            # Search ArXiv
            papers = self.search_arxiv(query)
            
            if not papers:
                return {
                    'answer': 'No ArXiv papers found for this query. Try different keywords or check the spelling.',
                    'sources': [],
                    'agent': 'arxiv'
                }
            
            # Prepare context from papers
            context_parts = []
            for i, paper in enumerate(papers[:3], 1):  # Use top 3 papers
                authors_str = ", ".join(paper['authors'][:3])  # First 3 authors
                if len(paper['authors']) > 3:
                    authors_str += " et al."
                
                context_parts.append(f"""Paper {i}:
Title: {paper['title']}
Authors: {authors_str}
Published: {paper['published'][:10]}  # Just the date
Summary: {paper['summary'][:500]}...
ArXiv ID: {paper['arxiv_id'].split('/')[-1] if '/' in paper['arxiv_id'] else paper['arxiv_id']}""")
            
            context = "\\n\\n".join(context_parts)
            
            # Generate summary using LLM
            summary_prompt = f"""Based on the following ArXiv papers, provide a comprehensive answer to the user's query about academic research.

Query: {query}

Papers:
{context}

Please synthesize the information from these papers to provide a helpful academic summary. Focus on key findings, methodologies, and implications."""
            
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful academic assistant that summarizes research papers."},
                    {"role": "user", "content": summary_prompt}
                ],
                model=self.model,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content.strip()
            
            return {
                'answer': answer,
                'sources': papers[:5],
                'agent': 'arxiv',
                'num_papers': len(papers)
            }
            
        except Exception as e:
            logger.error(f"Error processing ArXiv query: {e}")
            return {
                'answer': f'Error searching ArXiv: {str(e)}',
                'sources': [],
                'agent': 'arxiv'
            }