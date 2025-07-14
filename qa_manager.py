from typing import Tuple, Optional, List, Dict, Any, Union
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from langchain_core.language_models import BaseLanguageModel
from config.settings import LLM_MODEL, TOP_K_MATCHES, GROQ_API_KEY, LOG_FILE, LOG_LEVEL, POLICY_TYPES
from models.vector_store import VectorStoreManager
from utils.document_processor import load_knowledge_base
import re
from difflib import SequenceMatcher
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

class QAManager:
    def __init__(self):
        """Initialize the QA manager"""
        self.llm = None
        self.vector_store = VectorStoreManager()
        self.vector_store.load_existing_stores()
        logging.info("QA Manager initialized")

    def initialize_llm(self) -> bool:
        """Initialize the language model"""
        try:
            if not GROQ_API_KEY:
                logging.error("GROQ_API_KEY not found in environment variables")
                raise ValueError("GROQ_API_KEY not found in environment variables")
            
            # Set the API key in the environment
            os.environ["GROQ_API_KEY"] = GROQ_API_KEY
            
            self.llm = ChatGroq(
                model=LLM_MODEL,
                max_tokens=400,  # Reasonable limit for comprehensive answers
                temperature=0.1  # Lower temperature for focused responses
            )
            logging.info(f"LLM initialized successfully with model: {LLM_MODEL}")
            return True
        except Exception as e:
            logging.error(f"Error initializing LLM: {str(e)}")
            return False

    def get_answer(self, question: str) -> Tuple[str, str, Optional[str]]:
        """
        Get answer using 3-tier approach:
        1. Direct sentence match from JSON KB
        2. Vector DB semantic search + JSON context
        3. LLM fallback with all context
        """
        if not self.llm:
            if not self.initialize_llm():
                logging.error("Failed to initialize LLM")
                return "Error: Could not initialize LLM", "error", None

        logging.info(f"Processing question: {question}")
        question_lower = question.lower()
        kb = load_knowledge_base()
        
        # 1. Try direct sentence matching from JSON KB
        direct_match, match_source = self._find_direct_sentence_match(question_lower, kb)
        if direct_match and match_source:  # Only proceed if we have both match and source
            logging.info("Found direct match in knowledge base")
            # Enhance direct match with context and generate answer
            context = self._get_surrounding_context(direct_match, kb, match_source)
            answer = self._generate_llm_answer(question, context)
            return answer, "Direct Match + Context", context

        # 2. Try vector store semantic search
        logging.info("Attempting vector store search")
        docs, source = self.vector_store.get_relevant_documents(question)
        if docs:
            logging.info(f"Found relevant documents from: {source}")
            # Enhance vector results with relevant JSON KB sentences and tables
            enhanced_context = self._enhance_vector_results_with_kb(docs, kb, source)
            answer = self._get_answer_from_docs(question, enhanced_context)
            if answer:
                return answer, f"Vector Search + KB Context: {source}", enhanced_context

        # 3. Fallback to LLM with all available context
        logging.info("Using fallback with full context")
        all_context = self._get_all_kb_sentences(kb)
        answer = self._generate_llm_answer(question, all_context)
        return answer or "Unable to generate an answer", "LLM with Full Context", all_context

    def _find_direct_sentence_match(self, question: str, kb: dict) -> Tuple[Optional[str], Optional[str]]:
        """Find the best direct sentence match from the knowledge base using whole-sentence similarity"""
        best_match = None
        best_score = 0.65  # Higher threshold for whole-sentence matching
        match_source = None
        
        # Clean and normalize the question using keywords for better context
        clean_question = ' '.join(self._extract_keywords(question.lower()))
        logging.debug(f"Cleaned question: {clean_question}")
        
        for doc_name, doc_data in kb.items():
            if isinstance(doc_data, dict) and "sentences" in doc_data:
                for sentence_obj in doc_data["sentences"]:
                    # Clean and normalize the sentence using the same keyword extraction
                    clean_sentence = ' '.join(self._extract_keywords(sentence_obj["text"].lower()))
                    
                    # Use sequence matcher for whole-sentence similarity
                    similarity = SequenceMatcher(None, clean_question, clean_sentence).ratio()
                    
                    # Small boost for numerical matches to maintain accuracy with numbers
                    question_numbers = set(re.findall(r'\d+(?:\.\d+)?', question))
                    sentence_numbers = set(re.findall(r'\d+(?:\.\d+)?', sentence_obj["text"]))
                    if question_numbers and question_numbers.intersection(sentence_numbers):
                        similarity += 0.05  # Smaller boost than before
                    
                    if similarity > best_score:
                        best_score = similarity
                        best_match = sentence_obj["text"]
                        match_source = doc_name
                        logging.debug(f"New best match (score {similarity:.2f}): {best_match}")
        
        if best_match:
            logging.info(f"Found direct match with score {best_score:.2f}")
        else:
            logging.info("No direct match found above threshold")
            
        return best_match, match_source

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text including numbers, dates, and HR terms"""
        # Common HR-related terms across all policy types
        hr_terms = {
            "leave", "policy", "gratuity", "salary", "days", "months", "years",
            "employee", "employment", "benefits", "compensation", "annual",
            "sick", "maternity", "paternity", "paid", "unpaid", "training",
            "learning", "development", "skill", "course", "certification",
            "harassment", "discrimination", "workplace", "conduct", "upskilling"
        }
        
        # Extended stop words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "is", "are", "was", "were", "will", "be", "been", "being", "have",
            "has", "had", "do", "does", "did", "can", "could", "shall", "should",
            "may", "might", "must", "that", "this", "these", "those", "i", "you",
            "he", "she", "it", "we", "they", "what", "which", "who", "whom",
            "whose", "when", "where", "why", "how"
        }
        
        # Extract numbers and dates
        number_pattern = r'\d+(?:\.\d+)?'
        date_pattern = r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}'
        
        numbers = re.findall(number_pattern, text)
        dates = re.findall(date_pattern, text)
        
        # Extract words and filter
        words = text.lower().split()
        keywords = [word for word in words 
                   if word not in stop_words 
                   and (len(word) > 2 or word in hr_terms)]
        
        # Add numbers and dates to keywords
        keywords.extend(numbers)
        keywords.extend(dates)
        
        return list(set(keywords))  # Remove duplicates

    def _get_surrounding_context(self, matched_sentence: str, kb: dict, doc_name: str) -> str:
        """Get surrounding sentences for context"""
        if not doc_name or doc_name not in kb:
            return matched_sentence
            
        sentences = kb[doc_name]["sentences"]
        context_sentences = []
        
        # Find the matched sentence index
        match_idx = next((idx for idx, s in enumerate(sentences) 
                         if s["text"] == matched_sentence), -1)
        
        if match_idx != -1:
            # Get 2 sentences before and after for context
            start_idx = max(0, match_idx - 2)
            end_idx = min(len(sentences), match_idx + 3)
            
            context_sentences = [s["text"] for s in sentences[start_idx:end_idx]]
            
        return "\n".join(context_sentences) if context_sentences else matched_sentence

    def _enhance_vector_results_with_kb(self, docs: List[Document], kb: dict, source: str) -> str:
        """Enhance vector store results with relevant KB sentences and tables"""
        combined_text = [doc.page_content for doc in docs]
        
        # Find relevant KB document
        for doc_name, doc_data in kb.items():
            if source.lower() in doc_name.lower() and "sentences" in doc_data:
                # Add relevant sentences from KB
                kb_sentences = [s["text"] for s in doc_data["sentences"]]
                combined_text.extend(kb_sentences)
                
                # Add table data if available
                if "tables" in doc_data and doc_data["tables"]:
                    for table in doc_data["tables"]:
                        combined_text.append(f"Table from page {table['page']}: {table['text_representation']}")
        
        return "\n".join(combined_text)

    def _get_all_kb_sentences(self, kb: dict) -> str:
        """Get all sentences and tables from the knowledge base"""
        all_content = []
        
        for doc_name, doc_data in kb.items():
            if isinstance(doc_data, dict):
                # Add sentences
                if "sentences" in doc_data:
                    doc_sentences = [
                        f"{doc_data['doc_type'].upper()} Policy - {s['text']}"
                        for s in doc_data["sentences"]
                    ]
                    all_content.extend(doc_sentences)
                
                # Add tables
                if "tables" in doc_data and doc_data["tables"]:
                    for table in doc_data["tables"]:
                        table_text = f"Table from {doc_name} (Page {table['page']}):\n{table['text_representation']}"
                        all_content.append(table_text)
        
        return "\n".join(all_content)

    def _get_answer_from_docs(self, question: str, context: str) -> Optional[str]:
        """Get answer from documents using LLM"""
        if not self.llm:
            return None
            
        try:
            prompt = ChatPromptTemplate.from_template("""
            You are an HR expert. Answer this question based on the provided context.

            Context: {context}
            Question: {input}

            Instructions:
            - Provide a clear, informative answer (3-5 sentences)
            - Include key numbers, percentages, dates, and specific requirements
            - For calculations: show the formula and result
            - For policies: explain the main rule and any important exceptions
            - For procedures: outline the key steps briefly
            - If information is missing, state it clearly
            
            Base your answer strictly on the policy information provided above.
            """)
            
            # Format the prompt properly
            formatted_prompt = prompt.format_messages(
                context=context,
                input=question
            )
            
            result = self.llm.invoke(formatted_prompt)
            
            if hasattr(result, 'content'):
                return str(result.content)
            return str(result)
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    def _generate_llm_answer(self, question: str, context: Optional[str]) -> str:
        """Generate answer using LLM"""
        if not self.llm:
            logging.error("LLM not initialized")
            return "Error: LLM not initialized"
            
        try:
            # Create a proper chat template
            prompt = ChatPromptTemplate.from_template("""
            You are an HR expert. Answer this question about HR policies based on the provided context.

            {context_text}
            
            Question: {question}
            
            Instructions:
            - Provide a clear, informative answer (3-5 sentences)
            - Include key numbers, percentages, dates, and specific requirements
            - For calculations: show the formula and result
            - For policies: explain the main rule and any important exceptions
            - For procedures: outline the key steps briefly
            - If information is missing, state it clearly
            
            Base your answer strictly on the policy information provided above.
            """)
            
            # Format the prompt properly
            formatted_prompt = prompt.format_messages(
                context_text=f"Context: {context}" if context else "Note: No specific policy context available.",
                question=question
            )
            
            logging.info("Generating LLM answer")
            result = self.llm.invoke(formatted_prompt)
            answer = str(result.content) if hasattr(result, 'content') else str(result)
            logging.info("Answer generated successfully")
            return answer
            
        except Exception as e:
            error_msg = f"Error generating answer: {str(e)}"
            logging.error(error_msg)
            return error_msg
