from typing import List, Dict, Any
from models.qa_manager import QAManager

qa_manager = QAManager()

# Mock data
policies_db = [
    {"id": 1, "title": "Leave Policy", "category": "Leave", "description": "Details about leave policy."},
    {"id": 2, "title": "Gratuity Policy", "category": "Benefits", "description": "Details about gratuity policy."},
    {"id": 3, "title": "Upskilling Policy", "category": "Development", "description": "Details about upskilling policy."}
]

def get_policies() -> List[Dict[str, Any]]:
    """Get all policies from the mock database."""
    return policies_db

def ask_question(question: str) -> str:
    """Process a question using the QA manager."""
    try:
        return qa_manager.ask_question(question)
    except Exception as e:
        raise Exception(f"Error processing question: {str(e)}")