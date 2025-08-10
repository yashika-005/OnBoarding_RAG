from pydantic import BaseModel
from typing import Optional

class PolicyBase(BaseModel):
    id: int
    title: str
    category: str
    description: Optional[str] = None

class QuestionRequest(BaseModel):
    question: str