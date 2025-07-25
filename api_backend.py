from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from models.qa_manager import QAManager
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional


app = FastAPI()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
qa_manager = QAManager()

class PromptRequest(BaseModel):
    question: str

# Additional models for new endpoints
class FeedbackRequest(BaseModel):
    question: str
    answer: str
    feedback: str

class SearchRequest(BaseModel):
    query: str

class Policy(BaseModel):
    id: int
    title: str
    category: str
    description: Optional[str] = None

@app.post("/ask")
async def ask_question(request: PromptRequest):
    try:
        answer, source, context = qa_manager.get_answer(request.question)
        return JSONResponse({
            "answer": answer,
            "source": source,
            "context": context
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# 1. List all policies
@app.get("/policies", response_model=List[Policy])
async def get_policies():
    # Placeholder: Replace with real data source
    return [
        Policy(id=1, title="Leave Policy", category="Leave", description="Details about leave policy."),
        Policy(id=2, title="Gratuity Policy", category="Gratuity", description="Details about gratuity policy."),
        Policy(id=3, title="Upskilling Policy", category="Upskilling", description="Details about upskilling policy.")
    ]

# 2. Get policy by ID
@app.get("/policy/{policy_id}", response_model=Policy)
async def get_policy(policy_id: int):
    # Placeholder: Replace with real data source
    policies = await get_policies()
    for policy in policies:
        if policy.id == policy_id:
            return policy
    return JSONResponse(status_code=404, content={"error": "Policy not found"})

# 3. Search policies
@app.post("/search")
async def search_policies(request: SearchRequest):
    # Placeholder: Replace with real search logic
    policies = await get_policies()
    results = [p for p in policies if request.query.lower() in p.title.lower() or (p.description and request.query.lower() in p.description.lower())]
    return results

# 4. Submit feedback
@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    # Placeholder: Save feedback to a database or file
    return {"message": "Feedback received", "data": request.dict()}

# 5. Get user history (dummy, no user tracking implemented)
@app.get("/history")
async def get_history():
    # Placeholder: Return static history
    return [
        {"question": "What is the leave policy?", "answer": "You are entitled to 20 days leave per year."},
        {"question": "What is gratuity?", "answer": "Gratuity is a statutory benefit paid to employees."}
    ]

# 6. Upload new policy document (admin)
@app.post("/upload-policy")
async def upload_policy(file: UploadFile = File(...)):
    # Placeholder: Save file to disk or process
    filename = f"uploaded_{file.filename}"
    with open(filename, "wb") as f:
        f.write(await file.read())
    return {"filename": filename, "message": "File uploaded successfully."}

# 7. List policy categories
@app.get("/categories")
async def get_categories():
    # Placeholder: Return static categories
    return ["Leave", "Gratuity", "Upskilling"]

# 8. Get context for a question (dummy)
@app.get("/context/{question_id}")
async def get_context(question_id: int):
    # Placeholder: Return static context
    return {"question_id": question_id, "context": "Sample context for the question."}

@app.get("/")
async def root():
    return {"message": "HR Policy Q&A API is running."}
