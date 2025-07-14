import re
import logging
from typing import Dict, List, Tuple, Optional
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
import os
from pathlib import Path

class PolicyClassifier:
    """
    Advanced policy type classifier using multiple detection methods:
    1. Content-based classification (TF-IDF + ML)
    2. Structural analysis (document sections, headers)
    3. Semantic similarity with policy templates
    4. Metadata analysis (file properties, creation info)
    5. Hybrid approach combining all methods
    """
    
    def __init__(self):
        self.classifier = None
        self.vectorizer = None
        self.policy_templates = self._load_policy_templates()
        self.section_patterns = self._load_section_patterns()
        self.trained = False
        
    def _load_policy_templates(self) -> Dict[str, List[str]]:
        """Load policy-specific templates and common phrases"""
        return {
            "gratuity": [
                "gratuity calculation", "retirement benefits", "service period",
                "basic salary", "dearness allowance", "gratuity formula",
                "minimum 5 years", "maximum gratuity", "payment schedule",
                "employee contribution", "employer contribution", "vesting period"
            ],
            "leave": [
                "annual leave", "sick leave", "maternity leave", "paternity leave",
                "casual leave", "earned leave", "leave balance", "carry forward",
                "leave application", "approval process", "leave encashment",
                "leave calendar", "holiday schedule", "leave policy"
            ],
            "upskilling": [
                "training program", "learning development", "skill enhancement",
                "certification course", "professional development", "training budget",
                "learning path", "skill assessment", "training calendar",
                "certification program", "learning platform", "development plan"
            ],
            "harassment": [
                "sexual harassment", "workplace harassment", "discrimination",
                "code of conduct", "reporting procedure", "investigation process",
                "prevention policy", "complaint handling", "disciplinary action",
                "confidentiality", "retaliation protection", "awareness training"
            ]
        }
    
    def _load_section_patterns(self) -> Dict[str, List[str]]:
        """Load section headers and structural patterns for each policy type"""
        return {
            "gratuity": [
                r"gratuity\s+calculation", r"eligibility\s+criteria", r"payment\s+schedule",
                r"service\s+period", r"basic\s+salary", r"formula",
                r"minimum\s+period", r"maximum\s+amount", r"vesting"
            ],
            "leave": [
                r"leave\s+types", r"annual\s+leave", r"sick\s+leave", r"maternity",
                r"paternity", r"casual\s+leave", r"leave\s+balance", r"carry\s+forward",
                r"application\s+process", r"approval", r"encashment"
            ],
            "upskilling": [
                r"training\s+program", r"learning\s+development", r"skill\s+enhancement",
                r"certification", r"professional\s+development", r"training\s+budget",
                r"learning\s+path", r"skill\s+assessment", r"development\s+plan"
            ],
            "harassment": [
                r"sexual\s+harassment", r"workplace\s+harassment", r"discrimination",
                r"code\s+of\s+conduct", r"reporting\s+procedure", r"investigation",
                r"prevention", r"complaint", r"disciplinary", r"confidentiality"
            ]
        }
    
    def train_classifier(self, training_data: List[Tuple[str, str]]):
        """Train the ML classifier with labeled policy documents"""
        try:
            texts, labels = zip(*training_data)
            
            # Create pipeline with TF-IDF vectorizer and Naive Bayes
            self.classifier = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=1000,
                    ngram_range=(1, 3),
                    stop_words='english',
                    min_df=2
                )),
                ('clf', MultinomialNB())
            ])
            
            # Train the classifier
            self.classifier.fit(texts, labels)
            self.trained = True
            
            # Save the trained model
            model_path = Path("models/policy_classifier.pkl")
            model_path.parent.mkdir(exist_ok=True)
            with open(model_path, 'wb') as f:
                pickle.dump(self.classifier, f)
                
            logging.info("Policy classifier trained and saved successfully")
            
        except Exception as e:
            logging.error(f"Error training classifier: {str(e)}")
    
    def load_trained_classifier(self) -> bool:
        """Load a pre-trained classifier"""
        try:
            model_path = Path("models/policy_classifier.pkl")
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.classifier = pickle.load(f)
                self.trained = True
                logging.info("Pre-trained classifier loaded successfully")
                return True
            return False
        except Exception as e:
            logging.error(f"Error loading classifier: {str(e)}")
            return False
    
    def classify_by_content_analysis(self, text: str) -> Dict[str, float]:
        """Classify policy using content analysis and TF-IDF similarity"""
        if not self.trained:
            return {}
        
        try:
            # Get prediction probabilities
            proba = self.classifier.predict_proba([text])[0]
            classes = self.classifier.classes_
            
            # Create confidence scores
            scores = dict(zip(classes, proba))
            return scores
            
        except Exception as e:
            logging.error(f"Error in content analysis: {str(e)}")
            return {}
    
    def classify_by_structural_analysis(self, text: str) -> Dict[str, float]:
        """Classify policy based on document structure and section patterns"""
        scores = {}
        text_lower = text.lower()
        
        for policy_type, patterns in self.section_patterns.items():
            matches = 0
            total_patterns = len(patterns)
            
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    matches += 1
            
            # Calculate structural similarity score
            scores[policy_type] = matches / total_patterns if total_patterns > 0 else 0
        
        return scores
    
    def classify_by_semantic_similarity(self, text: str) -> Dict[str, float]:
        """Classify policy using semantic similarity with policy templates"""
        scores = {}
        text_words = set(re.findall(r'\b\w+\b', text.lower()))
        
        for policy_type, templates in self.policy_templates.items():
            template_words = set()
            for template in templates:
                template_words.update(re.findall(r'\b\w+\b', template.lower()))
            
            # Calculate Jaccard similarity
            intersection = len(text_words.intersection(template_words))
            union = len(text_words.union(template_words))
            
            scores[policy_type] = intersection / union if union > 0 else 0
        
        return scores
    
    def classify_by_metadata_analysis(self, filename: str, file_size: int = 0) -> Dict[str, float]:
        """Classify policy based on file metadata and properties"""
        scores = {}
        filename_lower = filename.lower()
        
        # File size analysis (different policy types have different typical sizes)
        size_scores = {
            "gratuity": 0.8 if 50000 < file_size < 200000 else 0.3,
            "leave": 0.8 if 30000 < file_size < 150000 else 0.3,
            "upskilling": 0.8 if 40000 < file_size < 180000 else 0.3,
            "harassment": 0.8 if 25000 < file_size < 120000 else 0.3
        }
        
        # Filename pattern analysis
        filename_patterns = {
            "gratuity": [r"gratuity", r"retirement", r"benefit"],
            "leave": [r"leave", r"vacation", r"holiday"],
            "upskilling": [r"upskill", r"training", r"learning", r"development"],
            "harassment": [r"harassment", r"conduct", r"discrimination", r"sexual"]
        }
        
        for policy_type, patterns in filename_patterns.items():
            pattern_matches = sum(1 for pattern in patterns if re.search(pattern, filename_lower))
            filename_score = pattern_matches / len(patterns) if patterns else 0
            scores[policy_type] = (filename_score + size_scores[policy_type]) / 2
        
        return scores
    
    def classify_by_hybrid_approach(self, text: str, filename: str, file_size: int = 0) -> str:
        """Combine all classification methods for robust policy detection"""
        # Get scores from all methods
        content_scores = self.classify_by_content_analysis(text)
        structural_scores = self.classify_by_structural_analysis(text)
        semantic_scores = self.classify_by_semantic_similarity(text)
        metadata_scores = self.classify_by_metadata_analysis(filename, file_size)
        
        # Weight the different methods
        weights = {
            'content': 0.4,      # ML-based classification
            'structural': 0.3,    # Document structure analysis
            'semantic': 0.2,      # Template similarity
            'metadata': 0.1       # File properties
        }
        
        # Combine scores
        combined_scores = {}
        all_policy_types = set(content_scores.keys()) | set(structural_scores.keys()) | \
                          set(semantic_scores.keys()) | set(metadata_scores.keys())
        
        for policy_type in all_policy_types:
            combined_score = (
                weights['content'] * content_scores.get(policy_type, 0) +
                weights['structural'] * structural_scores.get(policy_type, 0) +
                weights['semantic'] * semantic_scores.get(policy_type, 0) +
                weights['metadata'] * metadata_scores.get(policy_type, 0)
            )
            combined_scores[policy_type] = combined_score
        
        # Return the policy type with highest score
        if combined_scores:
            best_policy = max(combined_scores.items(), key=lambda x: x[1])
            logging.info(f"Policy classified as {best_policy[0]} with confidence {best_policy[1]:.2f}")
            return best_policy[0]
        
        return "general"
    
    def analyze_document_structure(self, text: str) -> Dict[str, any]:
        """Analyze document structure for better classification"""
        analysis = {
            "sections": [],
            "tables": 0,
            "lists": 0,
            "formulas": 0,
            "dates": 0,
            "numbers": 0
        }
        
        # Count structural elements
        analysis["tables"] = len(re.findall(r'table|tabulation', text.lower()))
        analysis["lists"] = len(re.findall(r'^\s*[\d\-•]\s+', text, re.MULTILINE))
        analysis["formulas"] = len(re.findall(r'[=+\-*/]', text))
        analysis["dates"] = len(re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text))
        analysis["numbers"] = len(re.findall(r'\d+', text))
        
        # Extract section headers
        section_patterns = [
            r'^\d+\.\s+[A-Z][^.\n]*',
            r'^[A-Z][A-Z\s]+\n',
            r'^\d+\.\d+\s+[A-Z][^.\n]*'
        ]
        
        for pattern in section_patterns:
            sections = re.findall(pattern, text, re.MULTILINE)
            analysis["sections"].extend(sections)
        
        return analysis
    
    def get_classification_confidence(self, text: str, filename: str) -> Tuple[str, float]:
        """Get policy type with confidence score"""
        # Use hybrid approach
        policy_type = self.classify_by_hybrid_approach(text, filename)
        
        # Calculate confidence based on agreement between methods
        content_scores = self.classify_by_content_analysis(text)
        structural_scores = self.classify_by_structural_analysis(text)
        semantic_scores = self.classify_by_semantic_similarity(text)
        
        # Get the score for the predicted policy type
        confidence = max([
            content_scores.get(policy_type, 0),
            structural_scores.get(policy_type, 0),
            semantic_scores.get(policy_type, 0)
        ])
        
        return policy_type, confidence 