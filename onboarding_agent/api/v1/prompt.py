from typing import List

from fastapi import APIRouter, HTTPException

from onboarding_agent.data.policy_mapping import policies_db
from onboarding_agent.db.schemas import PolicyBase, QuestionRequest
from onboarding_agent.service.qa_manager import QAManager

router = APIRouter()
qa_manager = QAManager()

@router.get("/policies", response_model=List[PolicyBase])
async def get_policies():
    """Get all policies from the policy mapping file."""
    return policies_db

@router.post("/ask")
async def ask_question(request: QuestionRequest):
    """Ask a question about HR policies. Ask questions from knowledge base."""
    try:
        answer, source, context = qa_manager.get_answer(request.question)
        return {
            "answer": answer,
            "source": source,
            "context": context
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))