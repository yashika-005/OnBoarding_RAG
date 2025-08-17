import pytest

from unittest.mock import patch
from fastapi.testclient import TestClient

from onboarding_agent.app import app
from onboarding_agent.service.qa_manager import QAManager


@pytest.fixture
def client():
    """Create test client for FastAPI app"""
    return TestClient(app)


@pytest.fixture
def sample_questions():
    """Sample questions for testing"""
    return [
        "How many days of annual leave do I get?",
        "What is the gratuity calculation formula?",
        "Can I carry forward unused leaves?",
        "What training opportunities are available?"
    ]


class TestAPI:
    """Test API endpoints"""
    
    def test_get_policies(self, client):
        """Test GET /api/v1/policies endpoint"""
        response = client.get("/api/v1/policies")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        
        # Check policy structure
        policy = data[0]
        assert "id" in policy
        assert "title" in policy
        assert "category" in policy

    def test_policies_content(self, client):
        """Test policies contain expected content"""
        response = client.get("/api/v1/policies")
        data = response.json()
        
        titles = [p["title"] for p in data]
        assert "Leave Policy" in titles
        assert "Gratuity Policy" in titles
        assert "Upskilling Policy" in titles

    @patch('onboarding_agent.api.v1.prompt.qa_manager.get_answer')
    def test_ask_question_success(self, mock_get_answer, client):
        """Test successful question asking"""
        mock_get_answer.return_value = (
            "You get 21 days of annual leave per year.",
            "Vector Search: leave",
            "Leave policy context"
        )
        
        response = client.post("/api/v1/ask", json={"question": "How many leave days?"})
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "source" in data
        assert "context" in data
        assert "21 days" in data["answer"]

    def test_ask_question_validation(self, client):
        """Test request validation"""
        # Missing question field
        response = client.post("/api/v1/ask", json={})
        assert response.status_code == 422
        
        # Empty question should work
        response = client.post("/api/v1/ask", json={"question": ""})
        assert response.status_code == 200

    @patch('onboarding_agent.api.v1.prompt.qa_manager.get_answer')
    def test_ask_question_error(self, mock_get_answer, client):
        """Test error handling"""
        mock_get_answer.side_effect = Exception("Service error")
        
        response = client.post("/api/v1/ask", json={"question": "Test"})
        assert response.status_code == 500

    @patch('onboarding_agent.api.v1.prompt.qa_manager.get_answer')
    def test_multiple_questions(self, mock_get_answer, client, sample_questions):
        """Test multiple question types"""
        mock_get_answer.return_value = ("Test answer", "Test source", "Test context")
        
        for question in sample_questions:
            response = client.post("/api/v1/ask", json={"question": question})
            assert response.status_code == 200
            assert "answer" in response.json()


class TestQAManager:
    """Test QA Manager functionality"""
    
    @pytest.fixture
    def qa_manager(self):
        return QAManager()

    def test_initialization(self, qa_manager):
        """Test QA manager initialization"""
        assert qa_manager.llm is None
        assert qa_manager.vector_store is not None

    @patch.dict('os.environ', {'GROQ_API_KEY': 'test_key'})
    @patch('onboarding_agent.service.qa_manager.ChatGroq')
    def test_llm_initialization(self, mock_groq, qa_manager):
        """Test LLM initialization"""
        mock_groq.return_value = "mock_llm"
        
        result = qa_manager.initialize_llm()
        assert result is True
        assert qa_manager.llm == "mock_llm"

    def test_llm_no_api_key(self, qa_manager):
        """Test LLM initialization without API key"""
        with patch('onboarding_agent.core.config.settings.GROQ_API_KEY', None):
            result = qa_manager.initialize_llm()
            assert result is False

    def test_keyword_extraction(self, qa_manager):
        """Test keyword extraction"""
        text = "How many days of annual leave can I take?"
        keywords = qa_manager._extract_keywords(text)
        
        assert "days" in keywords
        assert "annual" in keywords
        assert "leave" in keywords
        # Stop words filtered out
        assert "how" not in keywords
        assert "can" not in keywords


@pytest.mark.parametrize("question,expected_type", [
    ("How many vacation days?", "leave"),
    ("What is gratuity calculation?", "gratuity"), 
    ("What training is available?", "upskilling"),
    ("Can I work from home?", "policy")
])
def test_question_types(question, expected_type):
    """Test different question types"""
    qa_manager = QAManager()
    keywords = qa_manager._extract_keywords(question.lower())
    
    # Basic validation that keywords are extracted
    assert len(keywords) > 0
    assert any(len(word) > 2 for word in keywords)


def test_malformed_requests():
    """Test error handling for bad requests"""
    client = TestClient(app)
    
    # Invalid JSON
    response = client.post(
        "/api/v1/ask",
        data="invalid json",
        headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
