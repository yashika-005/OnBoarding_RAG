from typing import List, Tuple, Optional, Sequence
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from config.settings import (
    POLICY_TYPES, TOP_K_MATCHES, SIMILARITY_THRESHOLD
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
        self.stores = {}  # Store all vector stores by policy type

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
        """Load existing vector stores for all policy types"""
        loaded_dbs = {}
        
        for policy_type, config in POLICY_TYPES.items():
            try:
                if self._db_exists(config["db_path"]):
                    store = Chroma(
                        persist_directory=config["db_path"],
                        embedding_function=self.embeddings
                    )
                    self.stores[policy_type] = store
                    loaded_dbs[policy_type] = True
                    logging.info(f"Loaded {policy_type} vector store")
                else:
                    loaded_dbs[policy_type] = False
            except Exception as e:
                logging.error(f"Error loading {policy_type} DB: {e}")
                loaded_dbs[policy_type] = False

        return loaded_dbs

    def _db_exists(self, path: str) -> bool:
        """Check if a vector store exists and is valid"""
        return os.path.exists(path) and os.path.exists(os.path.join(path, "chroma.sqlite3"))

    def get_relevant_documents(self, query: str) -> Tuple[Optional[List[Document]], str]:
        """Get relevant documents from appropriate vector store based on the query"""
        query = query.strip()
        if not query:
            return None, "none"
        
        # Determine which policy types are most relevant to the query
        relevant_policies = self._identify_relevant_policies(query)
        logging.info(f"Relevant policies for query '{query}': {relevant_policies}")
        
        if not relevant_policies:
            return None, "none"
        
        # Search across relevant stores
        all_results = []
        best_source = None
        best_score = 0
        
        for policy_type in relevant_policies:
            if policy_type in self.stores:
                logging.info(f"Searching {policy_type} store for query: {query}")
                results = self._search_db(self.stores[policy_type], query)
                logging.info(f"Found {len(results)} results in {policy_type} store")
                if results:
                    # Log the scores for debugging
                    scores = [score for _, score in results]
                    logging.info(f"Scores for {policy_type}: {scores}")
                    
                    # Calculate average score for this policy type
                    avg_score = sum(score for _, score in results) / len(results)
                    logging.info(f"Average score for {policy_type}: {avg_score}")
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_source = policy_type
                        all_results = results
            else:
                logging.warning(f"Store {policy_type} not found in loaded stores")
        
        if all_results:
            # Filter by similarity threshold
            logging.info(f"Filtering results with threshold {SIMILARITY_THRESHOLD}")
            filtered_results = [doc for doc, score in all_results if score >= SIMILARITY_THRESHOLD]
            logging.info(f"After filtering: {len(filtered_results)} results remain")
            
            if filtered_results:
                logging.info(f"Found relevant documents from: {best_source}")
                return filtered_results, best_source or "unknown"
            else:
                logging.info(f"No results above threshold {SIMILARITY_THRESHOLD}")
        
        logging.info("No relevant documents found")
        return None, "none"

    def _identify_relevant_policies(self, query: str) -> List[str]:
        """Identify which policy types are most relevant to the query"""
        query_lower = query.lower()
        relevant_policies = []
        
        for policy_type, config in POLICY_TYPES.items():
            # Check if any keywords match the query
            if any(keyword in query_lower for keyword in config["keywords"]):
                relevant_policies.append(policy_type)
        
        # If no specific matches, return all available stores
        if not relevant_policies:
            relevant_policies = list(self.stores.keys())
        
        return relevant_policies

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

    def get_store_for_policy_type(self, policy_type: str) -> Optional[Chroma]:
        """Get the vector store for a specific policy type"""
        return self.stores.get(policy_type)

    def add_documents_to_store(self, documents: List[Document], policy_type: str) -> bool:
        """Add documents to a specific policy store"""
        try:
            logging.info(f"Attempting to add {len(documents)} documents to {policy_type} store")
            
            if policy_type not in POLICY_TYPES:
                logging.error(f"Unknown policy type: {policy_type}")
                logging.error(f"Available policy types: {list(POLICY_TYPES.keys())}")
                return False
            
            store_path = POLICY_TYPES[policy_type]["db_path"]
            logging.info(f"Store path for {policy_type}: {store_path}")
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(store_path), exist_ok=True)
            logging.info(f"Directory created/verified: {os.path.dirname(store_path)}")
            
            # Create or update the store
            if policy_type in self.stores:
                logging.info(f"Adding to existing {policy_type} store")
                # Add to existing store
                self.stores[policy_type].add_documents(documents)
                self.stores[policy_type].persist()
            else:
                logging.info(f"Creating new {policy_type} store")
                # Create new store
                vectordb, _ = self.create_vectorstore(documents, store_path)
                self.stores[policy_type] = vectordb
            
            logging.info(f"Successfully added {len(documents)} documents to {policy_type} store")
            return True
            
        except Exception as e:
            logging.error(f"Error adding documents to {policy_type} store: {e}")
            logging.error(f"Exception type: {type(e).__name__}")
            import traceback
            logging.error(f"Full traceback: {traceback.format_exc()}")
            return False
