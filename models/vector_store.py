from typing import List, Tuple, Optional, Sequence
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from config.settings import (
    GRATUITY_DB_PATH,
    LEAVE_DB_PATH,
    TOP_K_MATCHES,
    SIMILARITY_THRESHOLD
)
import logging
import os

class VectorStoreManager:
    def __init__(self):
        """Initialize the vector store manager"""
        # Use a reliable local embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.gratuity_db = None
        self.leave_db = None

    def create_vectorstore(self, documents: List[Document], store_path: str) -> Tuple[Chroma, int]:
        """Create a new vector store from documents"""
        try:
            # Delete existing DB if it exists
            if os.path.exists(store_path):
                import shutil
                shutil.rmtree(store_path)
                
            vectordb = Chroma.from_documents(
                documents,
                self.embeddings,
                persist_directory=store_path
            )
            vectordb.persist()
            return vectordb, len(documents)
        except Exception as e:
            logging.error(f"Error creating vector store: {e}")
            raise RuntimeError(f"Failed to create vector store: {str(e)}")

    def load_existing_stores(self) -> dict:
        """Load existing vector stores"""
        loaded_dbs = {"gratuity": False, "leave": False}
        
        try:
            if self._db_exists(GRATUITY_DB_PATH):
                self.gratuity_db = Chroma(
                    persist_directory=GRATUITY_DB_PATH,
                    embedding_function=self.embeddings
                )
                loaded_dbs["gratuity"] = True
        except Exception as e:
            logging.error(f"Error loading gratuity DB: {e}")

        try:
            if self._db_exists(LEAVE_DB_PATH):
                self.leave_db = Chroma(
                    persist_directory=LEAVE_DB_PATH,
                    embedding_function=self.embeddings
                )
                loaded_dbs["leave"] = True
        except Exception as e:
            logging.error(f"Error loading leave DB: {e}")

        return loaded_dbs

    def _db_exists(self, path: str) -> bool:
        """Check if a vector store exists and is valid"""
        return os.path.exists(path) and os.path.exists(os.path.join(path, "chroma.sqlite3"))

    def get_relevant_documents(self, query: str) -> Tuple[Optional[List[Document]], str]:
        """Get relevant documents from appropriate vector store based on the query"""
        query = query.strip()
        if not query:
            return None, "none"
            
        gratuity_results = self._search_db(self.gratuity_db, query)
        leave_results = self._search_db(self.leave_db, query)
        
        if not gratuity_results and not leave_results:
            return None, "none"
            
        # Compare relevance scores if we have results from both
        if gratuity_results and leave_results:
            gratuity_score = sum(score for _, score in gratuity_results) / len(gratuity_results)
            leave_score = sum(score for _, score in leave_results) / len(leave_results)
            
            if gratuity_score > leave_score:
                return [doc for doc, score in gratuity_results if score >= SIMILARITY_THRESHOLD], "gratuity"
            return [doc for doc, score in leave_results if score >= SIMILARITY_THRESHOLD], "leave"
        
        # Return whichever has results
        if gratuity_results:
            return [doc for doc, score in gratuity_results if score >= SIMILARITY_THRESHOLD], "gratuity"
        return [doc for doc, score in leave_results if score >= SIMILARITY_THRESHOLD], "leave"

    def _search_db(self, db: Optional[Chroma], query: str) -> List[Tuple[Document, float]]:
        """Search a specific vector store with error handling"""
        try:
            if db:
                return db.similarity_search_with_relevance_scores(
                    query,
                    k=TOP_K_MATCHES
                )
            return []
        except Exception as e:
            logging.error(f"Error searching vector store: {e}")
            return []
