import json
import logging
import os
from datetime import datetime
from typing import Dict, Any

def setup_logging():
    """Setup application logging"""
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        handlers=[
            logging.FileHandler('logs/app.log'),
            logging.StreamHandler()
        ]
    )

def log_decision(query: str, decision: Dict[str, Any]):
    """Log controller agent decision"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'query': query,
        'decision': decision,
        'type': 'routing_decision'
    }
    
    # Append to decisions log file
    with open('logs/decisions.log', 'a') as f:
        f.write(json.dumps(log_entry) + "\n")

def log_agent_result(agent_name: str, query: str, result: Dict[str, Any]):
    """Log agent execution result"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'agent': agent_name,
        'query': query,
        'result': result,
        'type': 'agent_result'
    }
    
    with open('logs/agent_results.log', 'a') as f:
        f.write(json.dumps(log_entry) + "\n")