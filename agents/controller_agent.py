import os
import json
import logging
from typing import Dict, List, Any
from dotenv import load_dotenv
from groq import Groq

logger = logging.getLogger(__name__)

class ControllerAgent:
    """Controller agent that decides which agents to call and synthesizes responses"""
    
    def __init__(self):
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.1-8b-instant"  # or "llama2-70b-4096"
        
    def route_query(self, query: str) -> Dict[str, Any]:
        """Decide which agents should handle the query"""
        
        routing_prompt = f"""
You are a controller agent that decides which specialized agents should handle a user query.

Available agents:
1. pdf_rag - For questions about uploaded PDF documents
2. web_search - For current information, news, recent developments
3. arxiv - For academic papers, research, scientific information

Query: "{query}"

Analyze the query and decide which agent(s) should handle it. Consider:
- If it mentions "recent", "latest", "news", "current" -> web_search
- If it mentions "paper", "research", "academic", "arxiv" -> arxiv  
- If it's about uploaded documents or asks to "summarize this" -> pdf_rag
- You can select multiple agents if needed

Respond with JSON in this format:
{{
    "agents": ["agent1", "agent2"],
    "reasoning": "Brief explanation of why these agents were selected"
}}
"""
        
        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a routing agent. Always respond with valid JSON."},
                    {"role": "user", "content": routing_prompt}
                ],
                model=self.model,
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            try:
                decision = json.loads(response_text)
                return decision
            except json.JSONDecodeError:
                # Fallback to rule-based routing if JSON parsing fails
                return self._rule_based_routing(query)
                
        except Exception as e:
            logger.error(f"Error in LLM routing: {e}")
            return self._rule_based_routing(query)
    
    def _rule_based_routing(self, query: str) -> Dict[str, Any]:
        """Fallback rule-based routing"""
        query_lower = query.lower()
        agents = []
        reasoning = "Fallback rule-based routing: "
        
        if any(word in query_lower for word in ['recent', 'latest', 'news', 'current', 'today']):
            agents.append('web_search')
            reasoning += "detected request for current information; "
            
        if any(word in query_lower for word in ['paper', 'research', 'academic', 'arxiv', 'study']):
            agents.append('arxiv')
            reasoning += "detected academic/research query; "
            
        if any(word in query_lower for word in ['document', 'pdf', 'summarize this', 'uploaded']):
            agents.append('pdf_rag')
            reasoning += "detected document-related query; "
        
        # Default to web search if no specific indicators
        if not agents:
            agents = ['web_search']
            reasoning += "defaulting to web search"
            
        return {
            "agents": agents,
            "reasoning": reasoning.strip('; ')
        }
    
    def synthesize_answer(self, query: str, agent_results: Dict[str, Any]) -> str:
        """Synthesize final answer from agent results"""
        
        synthesis_prompt = f"""
You are an AI assistant that synthesizes information from multiple specialized agents.

User Query: "{query}"

Agent Results:
{json.dumps(agent_results, indent=2)}

Based on the information provided by the agents, create a comprehensive and coherent answer to the user's query. 
If multiple agents provided information, integrate their responses seamlessly.
If an agent encountered an error, acknowledge it briefly but focus on available information.

Provide a clear, helpful response that directly addresses the user's question.
"""
        
        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that synthesizes information from multiple sources."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                model=self.model,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error synthesizing answer: {e}")
            # Fallback to simple concatenation
            fallback_answer = "Based on the available information:\\n\\n"
            for agent, result in agent_results.items():
                if isinstance(result, dict) and 'answer' in result:
                    fallback_answer += f"From {agent}: {result['answer']}\\n\\n"
                elif isinstance(result, str):
                    fallback_answer += f"From {agent}: {result}\\n\\n"
            
            return fallback_answer or "I apologize, but I encountered an error processing your request."