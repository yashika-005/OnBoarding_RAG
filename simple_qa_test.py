#!/usr/bin/env python3

import os
import sys
import time
import pytest

# Add the project root to the path (parent directory of Main_app)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.qa_manager import QAManager
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory and all dependencies are installed")
    sys.exit(1)

@pytest.fixture(scope="module")
def qa_manager():
    """Initialize QA Manager for testing"""
    print("Initializing QA Manager...")
    manager = QAManager()
    
    if not manager.initialize_llm():
        pytest.fail("Error: Could not initialize LLM - check GROQ_API_KEY")
    
    return manager

@pytest.fixture(scope="module")
def test_questions():
    """Predefined test questions for HR policy Q&A"""
    return [
        "What is the gratuity policy?",
        "How many days of annual leave am I entitled to?",
        "What is the notice period for resignation?",
        "Can I encash my leave?",
        "What are the maternity leave benefits?",
        "How is gratuity calculated?",
        "What is the sick leave policy?",
        "Can I carry forward unused leave?",
        "What documents are required for leave application?",
        "What is the maximum gratuity amount?"
    ]

@pytest.mark.parametrize("question", [
    "What is the gratuity policy?",
    "How many days of annual leave am I entitled to?",
    "What is the notice period for resignation?",
    "Can I encash my leave?",
    "What are the maternity leave benefits?",
    "How is gratuity calculated?",
    "What is the sick leave policy?",
    "Can I carry forward unused leave?",
    "What documents are required for leave application?",
    "What is the maximum gratuity amount?"
])
def test_hr_policy_questions(qa_manager, question):
    """Test HR policy Q&A system with predefined questions"""
    print(f"\n[Question] {question}")
    print("-" * 50)
    
    start_time = time.time()
    try:
        answer, source, context = qa_manager.get_answer(question)
        end_time = time.time()
        
        # Assertions to validate the response
        assert answer is not None, "Answer should not be None"
        assert isinstance(answer, str), "Answer should be a string"
        assert len(answer.strip()) > 0, "Answer should not be empty"
        assert source is not None, "Source should not be None"
        
        print(f"Answer: {answer}")
        print(f"Source: {source}")
        print(f"Response time: {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        pytest.fail(f"Error getting answer: {str(e)}")
    
    print("-" * 50)

def test_all_questions_summary(qa_manager, test_questions):
    """Test all questions and provide summary statistics"""
    print("\n" + "="*60)
    print("HR POLICY Q&A SYSTEM - PYTEST TESTING")
    print("="*60)
    
    total_start_time = time.time()
    successful_queries = 0
    failed_queries = 0
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[Question {i}/{len(test_questions)}] {question}")
        print("-" * 50)
        
        start_time = time.time()
        try:
            answer, source, context = qa_manager.get_answer(question)
            end_time = time.time()
            
            print(f"Answer: {answer}")
            print(f"Source: {source}")
            print(f"Response time: {end_time - start_time:.2f} seconds")
            successful_queries += 1
            
        except Exception as e:
            print(f"Error getting answer: {str(e)}")
            failed_queries += 1
        
        print("-" * 50)
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    print(f"\n" + "="*60)
    print(f"TEST SUMMARY:")
    print(f"Total questions: {len(test_questions)}")
    print(f"Successful queries: {successful_queries}")
    print(f"Failed queries: {failed_queries}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Average time per question: {total_time / len(test_questions):.2f} seconds")
    print("="*60)
    
    # Assert that all queries were successful
    assert failed_queries == 0, f"{failed_queries} queries failed"

if __name__ == "__main__":
    # Run pytest programmatically if executed directly
    import sys
    pytest.main([__file__, "-v", "-s"])
