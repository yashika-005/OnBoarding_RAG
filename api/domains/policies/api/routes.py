from fastapi import APIRouter, HTTPException
from typing import List
from ....models.qa_manager import QAManager
from .. import schemas, services

router = APIRouter()

@router.get("/", response_model=List[schemas.PolicyBase])
async def get_policies():
    """Get all policies."""
    return services.get_policies()

@router.post("/ask")
async def ask_question(request: schemas.QuestionRequest):
    """Ask a question about HR policies."""
    try:
        answer = services.ask_question(request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))