from datetime import datetime
from typing import List, Dict, Any

import requests
import streamlit as st

# API configuration
API_BASE_URL = "http://127.0.0.1:8000/api/v1"  # Correct FastAPI base path

# Page configuration
st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="🤖",
    layout="wide"
)

def check_api_status() -> tuple[bool, str]:
    """Check if API is running and return status with details"""
    try:
        response = requests.get(f"{API_BASE_URL}/policies", timeout=3)
        response.raise_for_status()
        return True, "✅ API is connected and running"
    except requests.exceptions.RequestException as e:
        return False, f"❌ API connection error: {str(e)}"

def fetch_policies() -> List[Dict[str, Any]]:
    """Fetch all policies from the API"""
    try:
        response = requests.get(f"{API_BASE_URL}/policies", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching policies: {e}")
        return []

def ask_question(question: str) -> Dict[str, Any]:
    """Send question to the API"""
    try:
        # Check if user is asking about number of policies
        if "how many policies" in question.lower() or "list all policies" in question.lower():
            policies = fetch_policies()
            if policies:
                # Format all policies into a nice string
                policy_list = "\n".join([
                    f"• {p.get('title', 'Untitled')} (ID: {p.get('id', 'N/A')})"
                    for p in policies
                ])
                return {
                    "answer": f"We have {len(policies)} policies available:\n{policy_list}",
                    "source": "System",
                    "context": "Listing all available policies"
                }
        
        # For other questions, use the QAManager
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json={"question": question},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting answer: {e}")
        return {"answer": "Sorry, I couldn't process your question right now."}

def display_chat():
    """Display chat interface"""
    st.title("💬 HR Policy Assistant")
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about HR policies..."):
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Get assistant response
        with st.spinner("Thinking..."):
            response = ask_question(prompt)
            
            # Add assistant response to chat
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response.get("answer", "I couldn't generate an answer.")
            })
        
        # Rerun to update the chat display
        st.rerun()

def display_policies():
    """Display available policies"""
    st.header("📚 Company Policies")
    
    # Refresh button
    if st.button("🔄 Refresh Policies"):
        st.session_state.policies = fetch_policies()
        st.session_state.last_updated = datetime.now()
    
    # Show last updated time
    if 'last_updated' in st.session_state and st.session_state.last_updated:
        st.caption(f"Last updated: {st.session_state.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Display policies
    if not st.session_state.policies:
        st.info("No policies found. Please check the API connection.")
        return
    
    # Display policies in cards
    cols = st.columns(3)
    for i, policy in enumerate(st.session_state.policies):
        with cols[i % 3]:
            with st.expander(f"📄 {policy.get('title', 'Untitled')}"):
                st.write(f"**Category:** {policy.get('category', 'N/A')}")
                if policy.get('description'):
                    st.write(policy['description'])
                st.caption(f"ID: {policy.get('id', 'N/A')}")

def main():
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.chat_history = []
        st.session_state.policies = []
        st.session_state.last_updated = None
    
    # Check API status
    api_status, status_message = check_api_status()
    
    # Sidebar with app info
    with st.sidebar:
        st.title("HR Policy Assistant")
        st.markdown("Ask questions about HR policies and get accurate answers.")
        
        # Display API status
        st.markdown("---")
        if api_status:
            st.success(status_message)
            # Only fetch policies if API is available and not already loaded
            if not st.session_state.policies:
                st.session_state.policies = fetch_policies()
                st.session_state.last_updated = datetime.now()
        else:
            st.error(status_message)
            st.markdown("""
            **To start the API server:**
            1. Open a new terminal
            2. Navigate to the onboarding_agent directory
            3. Run: `uvicorn app:app --reload`
            4. Keep the server running in the background
            5. Refresh this page
            """)
        
        # Navigation
        st.markdown("---")
        st.markdown("### Navigation")
        page = st.radio(
            "Go to:",
            ["💬 Chat", "📚 View Policies"],
            index=0 if st.session_state.get('current_page') != 'policies' else 1,
            disabled=not api_status
        )
        
        # Set current page based on selection
        st.session_state.current_page = "chat" if page == "💬 Chat" else "policies"
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("This assistant helps you navigate company HR policies.")
        st.markdown("*Powered by FastAPI & Streamlit*")
    
    # Display appropriate content based on API status
    if not api_status:
        st.warning("⚠️ Please start the API server to use this application.")
        st.markdown("""
        ### How to start the API server:
        1. Open a new terminal
        2. Navigate to the onboarding_agent directory
        3. Run: `uvicorn app:app --reload`
        4. Wait for the API to start
        5. Refresh this page
        """)
    else:
        # Main content area
        if st.session_state.get("current_page") == "policies":
            display_policies()
        else:
            display_chat()

if __name__ == "__main__":
    main()
