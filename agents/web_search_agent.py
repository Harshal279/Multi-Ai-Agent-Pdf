import os
import json
import logging
from typing import Dict, Any, List
from groq import Groq
import requests
from dotenv import load_dotenv
from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)

class WebSearchAgent:
    """Agent for web search using DuckDuckGo (with SerpAPI as fallback)"""
    
    def __init__(self):
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.1-8b-instant"
        self.serpapi_key = os.environ.get('SERPAPI_KEY')
        
    def search_duckduckgo(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo"""
        try:
            ddgs = DDGS()
            results = []
            
            for result in ddgs.text(query, max_results=max_results):
                results.append({
                    'title': result.get('title', ''),
                    'link': result.get('href', ''),
                    'snippet': result.get('body', ''),
                    'source': 'duckduckgo'
                })
            
            return results
            
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []
    
    def search_serpapi(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search using SerpAPI as fallback"""
        if not self.serpapi_key:
            return []
            
        try:
            url = "https://serpapi.com/search"
            params = {
                "engine": "google",
                "q": query,
                "api_key": self.serpapi_key,
                "num": max_results
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            results = []
            if 'organic_results' in data:
                for result in data['organic_results'][:max_results]:
                    results.append({
                        'title': result.get('title', ''),
                        'link': result.get('link', ''),
                        'snippet': result.get('snippet', ''),
                        'source': 'serpapi'
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"SerpAPI search error: {e}")
            return []
    
    def search_web(self, query: str) -> List[Dict[str, Any]]:
        """Search web using available methods"""
        # Try DuckDuckGo first (free)
        results = self.search_duckduckgo(query)
        
        # Fallback to SerpAPI if available and DuckDuckGo failed
        if not results and self.serpapi_key:
            results = self.search_serpapi(query)
        
        return results
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process web search query"""
        try:
            # Perform web search
            search_results = self.search_web(query)
            
            if not search_results:
                return {
                    'answer': 'No web search results found. Please check your internet connection or try a different query.',
                    'sources': [],
                    'agent': 'web_search'
                }
            
            # Prepare context from search results
            context_parts = []
            for i, result in enumerate(search_results[:5], 1):
                context_parts.append(f"Result {i}:\\nTitle: {result['title']}\\nSnippet: {result['snippet']}\\nURL: {result['link']}")
            context = "\\n\\n".join(context_parts)
            
            # Generate summary using LLM
            summary_prompt = f"""Based on the following web search results, provide a comprehensive answer to the user's query.

Query: {query}

Search Results:
{context}

Please synthesize the information from these sources to provide a helpful and accurate answer. Include relevant details and cite sources where appropriate."""
            
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that synthesizes information from web search results."},
                    {"role": "user", "content": summary_prompt}
                ],
                model=self.model,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content.strip()
            
            return {
                'answer': answer,
                'sources': search_results[:5],
                'agent': 'web_search',
                'num_results': len(search_results)
            }
            
        except Exception as e:
            logger.error(f"Error processing web search query: {e}")
            return {
                'answer': f'Error performing web search: {str(e)}',
                'sources': [],
                'agent': 'web_search'

            }
