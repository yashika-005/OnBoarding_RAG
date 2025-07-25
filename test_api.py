import requests
import json

BASE_URL = "http://127.0.0.1:8000"
print("=== Starting API tests ===")
def debug_response(endpoint, response):
    print(f"\n[DEBUG] Endpoint: {endpoint}")
    print("[DEBUG] Request URL:", response.request.url)
    print("[DEBUG] Request Method:", response.request.method)
    if response.request.body:
        print("[DEBUG] Request Body:", response.request.body)
    print("[DEBUG] Status Code:", response.status_code)
    try:
        print("[DEBUG] Response JSON:")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"[DEBUG] Error parsing JSON: {e}")
        print("[DEBUG] Raw Response:", response.text)

def test_ask():
    url = f"{BASE_URL}/ask"
    data = {"question": "What is the sick leave policy?"}
    print(f"\nTesting /ask: {data['question']}")
    response = requests.post(url, json=data)
    debug_response("/ask", response)

def test_policies():
    url = f"{BASE_URL}/policies"
    print("\nTesting /policies")
    response = requests.get(url)
    debug_response("/policies", response)

def test_policy_by_id(policy_id):
    url = f"{BASE_URL}/policy/{policy_id}"
    print(f"\nTesting /policy/{{policy_id}} with id={policy_id}")
    response = requests.get(url)
    debug_response(f"/policy/{{policy_id}}", response)

def test_search():
    url = f"{BASE_URL}/search"
    data = {"query": "leave"}
    print("\nTesting /search")
    response = requests.post(url, json=data)
    debug_response("/search", response)

def test_feedback():
    url = f"{BASE_URL}/feedback"
    data = {"question": "What is gratuity?", "answer": "Gratuity is a statutory benefit.", "feedback": "Good answer."}
    print("\nTesting /feedback")
    response = requests.post(url, json=data)
    debug_response("/feedback", response)

def test_history():
    url = f"{BASE_URL}/history"
    print("\nTesting /history")
    response = requests.get(url)
    debug_response("/history", response)

def test_upload_policy():
    url = f"{BASE_URL}/upload-policy"
    files = {'file': ('test_policy.txt', b'This is a test policy file.', 'text/plain')}
    print("\nTesting /upload-policy")
    response = requests.post(url, files=files)
    debug_response("/upload-policy", response)

def test_categories():
    url = f"{BASE_URL}/categories"
    print("\nTesting /categories")
    response = requests.get(url)
    debug_response("/categories", response)

def test_context():
    url = f"{BASE_URL}/context/1"
    print("\nTesting /context/1")
    response = requests.get(url)
    debug_response("/context/1", response)

if __name__ == "__main__":
    test_policies()
    test_policy_by_id(1)
    test_search()
    test_feedback()
    test_history()
    test_upload_policy()
    test_categories()
    test_context()
