# OnBoarding RAG

A Retrieval-Augmented Generation (RAG) system for HR policy question-answering, featuring intelligent multi-document search and interactive Q&A capabilities. This system helps organizations streamline employee onboarding and provide instant answers to HR policy queries using AI-powered document analysis.

## 🏗️ Architecture

The system uses a **3-tier intelligent Q&A approach**:

1. **Direct JSON Knowledge Base Match** - Fast exact/fuzzy matching for common queries
2. **Vector Database Semantic Search** - ChromaDB-powered similarity search across policy documents  
3. **LLM Fallback** - Groq-powered language model for complex reasoning and synthesis

### Tech Stack
- **Backend**: FastAPI with async support
- **Frontend**: Streamlit web interface
- **LLM**: Groq API (gemma2-9b-it model)
- **Vector DB**: ChromaDB with HuggingFace embeddings
- **Document Processing**: LangChain + PyPDF
- **Embeddings**: sentence-transformers/all-mpnet-base-v2

## ✨ Features

- **Multi-Document Q&A**: Handles questions across Leave, Gratuity, and Upskilling policies
- **Intelligent Source Selection**: Automatically routes queries to the most relevant policy document
- **Advanced Query Processing**: Keyword extraction, numerical matching, and context enhancement
- **Real-time Chat Interface**: Interactive Streamlit UI with chat history
- **RESTful API**: FastAPI endpoints for programmatic access
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **Source Attribution**: Shows which document and method provided each answer
- **Policy Management**: View and manage available HR policies

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Groq API Key (sign up at [console.groq.com](https://console.groq.com))

### Installation

```bash
# Clone the repository
git clone https://github.com/yashika-005/OnBoarding_RAG.git
cd OnBoarding_RAG

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
# Create a .env file in the root directory
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
```

### Usage

#### 1. Start the FastAPI Backend
```bash
# Start the API server
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# The API will be available at http://localhost:8000
# API documentation at http://localhost:8000/docs
```

#### 2. Launch the Streamlit Frontend
```bash
# In a new terminal, start the Streamlit app
streamlit run streamlit_app.py

# Open your browser to http://localhost:8501
```

#### 3. Run Tests
```bash
# Run the test suite
python -m pytest test/ -v

# Or run specific test files
python test/test_prompt.py
```

## 📡 API Endpoints

### GET `/api/v1/policies`
Returns all available HR policies.

**Response:**
```json
[
  {
    "id": 1,
    "title": "Leave Policy",
    "category": "Leave",
    "description": "Details about leave policy."
  }
]
```

### POST `/api/v1/ask`
Ask a question about HR policies.

**Request:**
```json
{
  "question": "How many days of annual leave do I get?"
}
```

**Response:**
```json
{
  "answer": "Based on the leave policy...",
  "source": "Vector Search + KB Context: leave",
  "context": "Relevant policy excerpts..."
}
```

## 📁 Project Structure

```
onboarding_agent/
├── api/v1/
│   └── prompt.py              # FastAPI routes and endpoints
├── core/
│   ├── config.py              # Application configuration and settings
│   └── utils.py               # Utility functions
├── data/
│   ├── document_policies/     # PDF policy documents
│   │   ├── Company Leave Policy_25.pdf
│   │   ├── Gratuity_Policy_2025.pdf
│   │   └── Upskilling_and_Continuous_Learning_Policy.pdf
│   ├── vector_db/             # ChromaDB vector stores
│   │   ├── document_chroma_store_gratuity/
│   │   ├── document_chroma_store_leave/
│   │   └── document_chroma_store_upskilling/
│   ├── knowledge_base.json    # Structured policy knowledge base
│   └── policy_mapping.py      # Policy metadata and mapping
├── db/
│   ├── models.py              # Database models
│   └── schemas.py             # Pydantic schemas for API
├── service/
│   ├── document_processor.py  # PDF processing and knowledge base loading
│   ├── qa_manager.py          # Core Q&A logic and LLM integration
│   └── vector_store.py        # Vector database management
├── test/
│   └── test_prompt.py         # Test suite
├── logs/
│   └── qa_log.txt             # Application logs
├── app.py                     # FastAPI application entry point
├── streamlit_app.py           # Streamlit web interface
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🔧 Configuration
Key settings in `core/config.py`:

- **LLM Model**: `gemma2-9b-it` (Groq)
- **Chunk Size**: 500 tokens with 100 token overlap
- **Top-K Matches**: 3 most relevant documents
- **Embedding Model**: `sentence-transformers/all-mpnet-base-v2`

## 🧪 How It Works

### Query Processing Flow

1. **Question Analysis**: Extract keywords, numbers, and HR-specific terms
2. **Direct Matching**: Search JSON knowledge base for exact/fuzzy matches
3. **Vector Search**: If no direct match, use semantic similarity search
4. **Context Enhancement**: Combine vector results with relevant KB sentences
5. **LLM Generation**: Generate comprehensive answers with proper reasoning
6. **Source Attribution**: Return answer with source information and context

### Supported Query Types

- **Policy Questions**: "What is the leave policy?"
- **Calculations**: "How much gratuity will I get after 5 years?"
- **Specific Details**: "Can I carry forward unused leaves?"
- **Comparative**: "What's the difference between sick leave and annual leave?"
- **Procedural**: "How do I apply for extended leave?"

## 🛠️ Development

### Adding New Policies

1. Add PDF document to `data/document_policies/`
2. Update `data/policy_mapping.py` with new policy metadata
3. Process document to create vector store
4. Update knowledge base JSON if needed

### Environment Variables

Create a `.env` file:
```env
GROQ_API_KEY=your_groq_api_key_here
LOG_LEVEL=INFO
ENABLE_DEBUG=false
---


