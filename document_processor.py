import json
import tempfile
import logging
import re
from datetime import datetime
from typing import List, Tuple, Dict, Any
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.settings import (
    KNOWLEDGE_BASE_JSON, CHUNK_SIZE, CHUNK_OVERLAP, LOG_FILE, LOG_LEVEL,
    POLICY_TYPES, TABLE_EXTRACTION_ENABLED, TABLE_MIN_ROWS, TABLE_MIN_COLS
)
from utils.policy_classifier import PolicyClassifier
import pandas as pd
import tabula
import io
import pdfplumber

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# Initialize the policy classifier
policy_classifier = PolicyClassifier()

def load_knowledge_base():
    """Load the knowledge base from JSON file"""
    try:
        with open(KNOWLEDGE_BASE_JSON, "r", encoding="utf-8") as f:
            kb = json.load(f)
            logging.info("Knowledge base loaded successfully")
            return kb
    except FileNotFoundError:
        logging.warning("Knowledge base file not found, returning empty dict")
        return {}
    except Exception as e:
        logging.error(f"Error loading knowledge base: {str(e)}")
        return {}

def save_knowledge_base(data: dict):
    """Save the knowledge base to JSON file"""
    try:
        with open(KNOWLEDGE_BASE_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logging.info("Knowledge base saved successfully")
    except Exception as e:
        logging.error(f"Error saving knowledge base: {str(e)}")
        raise

def detect_policy_type(filename: str, content: str) -> str:
    """Advanced policy type detection using multiple methods"""
    try:
        # Try to load pre-trained classifier first
        if not policy_classifier.trained:
            policy_classifier.load_trained_classifier()
        
        # Get file size for metadata analysis
        file_size = len(content.encode('utf-8'))
        
        # Use hybrid approach for classification
        policy_type, confidence = policy_classifier.get_classification_confidence(content, filename)
        
        logging.info(f"Policy classified as '{policy_type}' with confidence {confidence:.2f}")
        
        # If confidence is low, fall back to keyword-based detection
        if confidence < 0.3:
            logging.warning(f"Low confidence ({confidence:.2f}) in classification, using fallback")
            return _fallback_keyword_detection(filename, content)
        
        return policy_type
        
    except Exception as e:
        logging.error(f"Error in advanced policy detection: {str(e)}")
        return _fallback_keyword_detection(filename, content)

def _fallback_keyword_detection(filename: str, content: str) -> str:
    """Fallback to simple keyword-based detection"""
    filename_lower = filename.lower()
    content_lower = content.lower()
    
    # Check each policy type
    for policy_type, config in POLICY_TYPES.items():
        # Check filename
        if any(keyword in filename_lower for keyword in config["keywords"]):
            return policy_type
        
        # Check content
        if any(keyword in content_lower for keyword in config["keywords"]):
            return policy_type
    
    # Default fallback based on filename
    if "gratuity" in filename_lower:
        return "gratuity"
    elif "leave" in filename_lower:
        return "leave"
    elif "upskill" in filename_lower or "learning" in filename_lower:
        return "upskilling"
    elif "harassment" in filename_lower or "sexual" in filename_lower:
        return "harassment"
    else:
        return "general"

def extract_tables_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract tables from PDF using tabula-py"""
    tables = []
    
    try:
        if not TABLE_EXTRACTION_ENABLED:
            return tables
            
        # Extract tables from all pages
        all_tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
        
        for page_num, page_tables in enumerate(all_tables):
            for table_idx, table in enumerate(page_tables):
                if table.shape[0] >= TABLE_MIN_ROWS and table.shape[1] >= TABLE_MIN_COLS:
                    # Convert table to structured format
                    table_data = {
                        "page": page_num + 1,
                        "table_index": table_idx + 1,
                        "rows": table.shape[0],
                        "columns": table.shape[1],
                        "headers": table.columns.tolist(),
                        "data": table.values.tolist(),
                        "text_representation": table.to_string(index=False)
                    }
                    tables.append(table_data)
                    logging.info(f"Extracted table {table_idx + 1} from page {page_num + 1}")
        
        logging.info(f"Extracted {len(tables)} tables from PDF")
        return tables
        
    except Exception as e:
        logging.warning(f"Table extraction failed: {str(e)}")
        return []

def extract_meaningful_sentences(text: str) -> List[str]:
    """Extract meaningful sentences from text, handling punctuation, line breaks, and bullet/list markers."""
    import re
    logging.info("Extracting meaningful sentences (smart)")
    
    # Normalize line breaks
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # First, split on actual sentence endings (., !, ?) followed by whitespace
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Then, also split on bullet points and list markers
    final_sentences = []
    for sentence in raw_sentences:
        # Split each sentence on bullet/list markers
        bullet_splits = re.split(r'\n+[\u2022\u25CF\u25A0\u25E6\u2023\u2043\u2219\u25C6\u25CB\u25B6\u25AA\u25AB\u25A1\u25B2\u25BC\u25C7\u25C8\u25C9\u25CA\u25CC\u25CD\u25CE\u25CF\u25D0\u25D1\u25D2\u25D3\u25D4\u25D5\u25D6\u25D7\u25D8\u25D9\u25DA\u25DB\u25DC\u25DD\u25DE\u25DF\u25E0\u25E1\u25E2\u25E3\u25E4\u25E5\u25E6\u25E7\u25E8\u25E9\u25EA\u25EB\u25EC\u25ED\u25EE\u25EF\-\*\•\●\○\▪\‣\⁃\·\-]\s*', sentence)
        
        for split in bullet_splits:
            cleaned = split.strip()
            cleaned = re.sub(r'\s+', ' ', cleaned)  # Collapse multiple spaces
            cleaned = re.sub(r'\n+', ' ', cleaned)  # Replace line breaks with spaces
            
            # Only add if it's meaningful (has letters and is long enough)
            if len(cleaned) > 10 and any(char.isalpha() for char in cleaned):
                final_sentences.append(cleaned)
    
    logging.info(f"Extracted {len(final_sentences)} meaningful sentences (smart)")
    if not final_sentences:
        logging.warning("No meaningful sentences were extracted from the text!")
        
    return final_sentences

def clean_spaces(text: str) -> str:
    """Replace multiple spaces with a single space throughout the text."""
    import re
    # First, normalize all whitespace (spaces, tabs, etc.) to single spaces
    text = re.sub(r'\s+', ' ', text)
    # Then remove any leading/trailing spaces
    text = text.strip()
    return text

def clean_extracted_text(text: str) -> str:
    """Aggressively clean extracted PDF text to remove excessive spacing."""
    import re
    # Normalize line breaks first
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove excessive spaces between words (common in PDF extraction)
    text = re.sub(r' {2,}', ' ', text)
    
    # Remove spaces before punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    
    # Remove spaces after opening parentheses/brackets
    text = re.sub(r'\(\s+', '(', text)
    
    # Remove spaces before closing parentheses/brackets  
    text = re.sub(r'\s+\)', ')', text)
    
    # Normalize multiple spaces to single spaces
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()

def add_document_to_knowledge_base(doc_name: str, doc_text: str, tables: List[Dict] = None):
    """Add a document to the knowledge base with line-by-line sentences and tables"""
    logging.info(f"Adding document to knowledge base: {doc_name}")
    
    try:
        # Input validation
        if not doc_text or not doc_text.strip():
            raise ValueError("Empty document text provided")
            
        if len(doc_text.strip()) < 100:  # Arbitrary minimum length for a policy document
            raise ValueError(f"Document text too short ({len(doc_text)} chars)")
            
        kb = load_knowledge_base()
        
        # Determine document type using advanced classification
        logging.info(f"Starting policy type detection for {doc_name}")
        doc_type = detect_policy_type(doc_name, doc_text)
        logging.info(f"Document type determined: {doc_type}")
        
        # Validate that the detected policy type is valid
        if doc_type not in POLICY_TYPES:
            logging.warning(f"Detected policy type '{doc_type}' not in POLICY_TYPES, using 'general'")
            doc_type = "general"
        
        # Analyze document structure for better classification
        try:
            structure_analysis = policy_classifier.analyze_document_structure(doc_text)
            logging.info(f"Document structure analysis: {structure_analysis}")
        except Exception as e:
            logging.warning(f"Structure analysis failed: {e}")
            structure_analysis = {"error": str(e)}
        
        # Log the document text length for debugging
        logging.info(f"Document text length: {len(doc_text)} characters")
        logging.info(f"Preview of first 200 chars: {doc_text[:200]}...")
        
        # Clean up common PDF artifacts
        cleaned_text = doc_text.replace('\x0c', '\n')  # Form feed
        cleaned_text = re.sub(r'\s*\n\s*', '\n', cleaned_text)  # Clean up newlines
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)  # Normalize paragraph breaks
        cleaned_text = clean_extracted_text(cleaned_text)  # Aggressive space cleaning
        
        sentences = extract_meaningful_sentences(cleaned_text)
        
        # Validate we got meaningful content
        if not sentences:
            logging.error(f"No meaningful sentences extracted from {doc_name}!")
            logging.error(f"Cleaned text preview: {cleaned_text[:500]}")
            raise ValueError("No meaningful sentences could be extracted")
            
        # Additional validation of extracted sentences
        valid_sentences = []
        for sentence in sentences:
            # Skip likely headers, footers, and metadata
            if re.match(r'^(page|section|chapter|\d+|\s*$)', sentence.lower()):
                continue
            # Skip very short sentences that might be artifacts
            if len(sentence.split()) < 4:
                continue
            valid_sentences.append(sentence)
        
        if not valid_sentences:
            raise ValueError("No valid sentences after filtering")
            
        # Create simple numbered sentences list
        numbered_sentences = []
        for idx, sentence in enumerate(valid_sentences, 1):
            numbered_sentences.append({
                "line_no": idx,
                "text": sentence
            })
        
        # Store in knowledge base with enhanced structure
        kb[doc_name] = {
            "doc_type": doc_type,
            "sentences": numbered_sentences,
            "tables": tables or [],
            "structure_analysis": structure_analysis,
            "metadata": {
                "extraction_date": datetime.now().isoformat(),
                "total_sentences": len(numbered_sentences),
                "total_tables": len(tables) if tables else 0,
                "text_length": len(doc_text),
                "classification_method": "advanced" if policy_classifier.trained else "fallback"
            }
        }
        
        save_knowledge_base(kb)
        logging.info(f"Document {doc_name} added to knowledge base with {len(numbered_sentences)} valid sentences and {len(tables) if tables else 0} tables")
        return kb
        
    except Exception as e:
        logging.error(f"Error processing document {doc_name}: {str(e)}")
        logging.error(f"Exception type: {type(e).__name__}")
        import traceback
        logging.error(f"Full traceback: {traceback.format_exc()}")
        raise

def join_broken_lines(text):
    import re
    lines = text.split('\n')
    joined = []
    buffer = ""
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if buffer:
                joined.append(buffer)
                buffer = ""
            joined.append("")
        elif re.match(r'^[-•●○▪‣⁃·*]', stripped) or re.match(r'^[A-Z][a-z]+:', stripped) or re.match(r'^[A-Z][A-Za-z ]{2,}$', stripped):
            if buffer:
                joined.append(buffer)
                buffer = ""
            joined.append(stripped)
        else:
            if buffer:
                buffer += " " + stripped
            else:
                buffer = stripped
    if buffer:
        joined.append(buffer)
    return "\n".join(joined)


def extract_text_with_pdfplumber(pdf_path):
    all_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                lines = page_text.split('\n')
                all_text.extend(lines)
    return join_broken_lines('\n'.join(all_text))


def process_pdf(uploaded_file, doc_type: str = None):
    """Process a PDF file and return its text content and document chunks (no tables)."""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Extract text using pdfplumber
    combined_text = extract_text_with_pdfplumber(tmp_file_path)

    # Create a single Document object for the whole text
    from langchain.schema import Document
    docs = [Document(page_content=combined_text, metadata={"source": uploaded_file.name})]

    # No tables
    tables = []

    return combined_text, docs, tables
