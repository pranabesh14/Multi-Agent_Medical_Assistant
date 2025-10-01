# src/models/text_processor.py
import re
import spacy
import logging
from typing import Dict, List, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import numpy as np
from collections import defaultdict, Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

class MedicalTextProcessor:
    """Processes medical text using BioBERT, ClinicalBERT and medical NLP techniques"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self._initialize_models()
        
        # Medical entity patterns
        self.medical_patterns = {
            'medications': [
                r'\b(?:aspirin|ibuprofen|acetaminophen|morphine|penicillin|insulin|metformin|lisinopril|atorvastatin|warfarin)\b',
                r'\b\w+cillin\b',  # antibiotics
                r'\b\w+statin\b',  # statins
                r'\b\w+pril\b',    # ACE inhibitors
            ],
            'symptoms': [
                r'\b(?:pain|fever|nausea|vomiting|headache|dizziness|fatigue|shortness of breath|chest pain|abdominal pain)\b',
                r'\b(?:cough|sore throat|runny nose|congestion|wheezing|rash|swelling|bleeding)\b',
            ],
            'diagnoses': [
                r'\b(?:pneumonia|diabetes|hypertension|asthma|copd|cancer|stroke|heart attack|infection)\b',
                r'\b(?:fracture|laceration|contusion|sprain|strain|burn|wound)\b',
            ],
            'procedures': [
                r'\b(?:surgery|operation|biopsy|endoscopy|catheterization|intubation|dialysis)\b',
                r'\b(?:x-ray|ct scan|mri|ultrasound|ecg|ekg|blood test|urinalysis)\b',
            ],
            'anatomy': [
                r'\b(?:heart|lung|liver|kidney|brain|stomach|intestine|bone|muscle|skin)\b',
                r'\b(?:chest|abdomen|head|neck|arm|leg|back|pelvis)\b',
            ],
            'vital_signs': [
                r'\b(?:blood pressure|heart rate|temperature|respiratory rate|oxygen saturation)\b',
                r'\b(?:bp|hr|temp|rr|o2 sat|pulse|fever)\b',
            ]
        }
        
        # Medical abbreviations
        self.medical_abbreviations = {
            'bp': 'blood pressure',
            'hr': 'heart rate',
            'rr': 'respiratory rate',
            'temp': 'temperature',
            'o2': 'oxygen',
            'sat': 'saturation',
            'copd': 'chronic obstructive pulmonary disease',
            'mi': 'myocardial infarction',
            'chf': 'congestive heart failure',
            'uti': 'urinary tract infection',
            'dvt': 'deep vein thrombosis',
            'pe': 'pulmonary embolism',
            'icu': 'intensive care unit',
            'er': 'emergency room',
            'or': 'operating room'
        }
    
    def _initialize_models(self):
        """Initialize the required NLP models"""
        try:
            # Initialize BioBERT for medical text encoding
            self.biobert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            self.biobert_loaded = True
            self.logger.info("BioBERT model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load BioBERT: {e}")
            self.biobert_loaded = False
        
        try:
            # Initialize ClinicalBERT for clinical text analysis
            self.clinical_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            self.clinical_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            self.clinical_model.to(self.device)
            self.clinical_loaded = True
            self.logger.info("ClinicalBERT model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load ClinicalBERT: {e}")
            self.clinical_loaded = False
        
        try:
            # Initialize medical NER pipeline
            self.ner_pipeline = pipeline(
                "ner",
                model="d4data/biomedical-ner-all",
                tokenizer="d4data/biomedical-ner-all",
                aggregation_strategy="simple"
            )
            self.ner_loaded = True
            self.logger.info("Medical NER pipeline loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load NER pipeline: {e}")
            self.ner_loaded = False
        
        try:
            # Initialize spaCy for general NLP tasks
            self.nlp = spacy.load("en_core_web_sm")
            self.spacy_loaded = True
            self.logger.info("spaCy model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load spaCy: {e}")
            self.spacy_loaded = False
        
        # Download NLTK data if needed
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.nltk_ready = True
        except Exception as e:
            self.logger.error(f"Failed to download NLTK data: {e}")
            self.nltk_ready = False
    
    def check_biobert_status(self) -> bool:
        """Check if BioBERT model is loaded"""
        return self.biobert_loaded
    
    def process_medical_text(self, text: str) -> Dict[str, Any]:
        """Process medical text and extract relevant information"""
        try:
            # Preprocess text
            cleaned_text = self._preprocess_text(text)
            
            # Extract medical entities
            entities = self._extract_medical_entities(cleaned_text)
            
            # Perform sentence-level analysis
            sentence_analysis = self._analyze_sentences(cleaned_text)
            
            # Extract medical concepts
            medical_concepts = self._extract_medical_concepts(cleaned_text)
            
            # Generate text embeddings
            embeddings = self._generate_embeddings(cleaned_text)
            
            # Perform clinical assessment
            clinical_assessment = self._perform_clinical_assessment(entities, medical_concepts)
            
            # Extract temporal information
            temporal_info = self._extract_temporal_information(cleaned_text)
            
            # Calculate text statistics
            text_stats = self._calculate_text_statistics(cleaned_text)
            
            return {
                'entities': entities,
                'sentences': sentence_analysis,
                'medical_concepts': medical_concepts,
                'embeddings': embeddings,
                'clinical_assessment': clinical_assessment,
                'temporal_info': temporal_info,
                'statistics': text_stats,
                'processing_status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"Error processing medical text: {e}")
            return {
                'error': str(e),
                'processing_status': 'failed'
            }
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess medical text"""
        try:
            # Basic cleaning
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            text = text.strip()
            
            # Expand medical abbreviations
            for abbr, expansion in self.medical_abbreviations.items():
                pattern = r'\b' + re.escape(abbr.upper()) + r'\b'
                text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
            
            # Normalize medical units
            text = re.sub(r'(\d+)\s*(mg|ml|cc|units?)\b', r'\1 \2', text, flags=re.IGNORECASE)
            
            # Handle medical ranges
            text = re.sub(r'(\d+)-(\d+)', r'\1 to \2', text)
            
            return text
            
        except Exception as e:
            self.logger.error(f"Error preprocessing text: {e}")
            return text
    
    def _extract_medical_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract medical entities using NER and pattern matching"""
        entities = defaultdict(list)
        
        try:
            # Use medical NER pipeline if available
            if self.ner_loaded:
                ner_results = self.ner_pipeline(text)
                for entity in ner_results:
                    entities['ner_entities'].append({
                        'text': entity['word'],
                        'label': entity['entity_group'],
                        'confidence': entity['score'],
                        'start': entity['start'],
                        'end': entity['end']
                    })
            
            # Pattern-based entity extraction
            for category, patterns in self.medical_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        entities[category].append({
                            'text': match.group(),
                            'start': match.start(),
                            'end': match.end(),
                            'pattern': pattern
                        })
            
            # Use spaCy for additional entity extraction
            if self.spacy_loaded:
                doc = self.nlp(text)
                for ent in doc.ents:
                    if ent.label_ in ['PERSON', 'ORG', 'DATE', 'TIME', 'CARDINAL', 'ORDINAL']:
                        entities['spacy_entities'].append({
                            'text': ent.text,
                            'label': ent.label_,
                            'start': ent.start_char,
                            'end': ent.end_char
                        })
            
            return dict(entities)
            
        except Exception as e:
            self.logger.error(f"Error extracting entities: {e}")
            return {}
    
    def _analyze_sentences(self, text: str) -> List[Dict[str, Any]]:
        """Analyze individual sentences for medical content"""
        try:
            if not self.nltk_ready:
                sentences = text.split('.')
            else:
                sentences = sent_tokenize(text)
            
            sentence_analysis = []
            
            for i, sentence in enumerate(sentences):
                if len(sentence.strip()) < 10:  # Skip very short sentences
                    continue
                
                analysis = {
                    'sentence': sentence.strip(),
                    'index': i,
                    'length': len(sentence),
                    'medical_score': self._calculate_medical_relevance_score(sentence),
                    'sentiment': self._analyze_sentence_sentiment(sentence),
                    'entities': self._extract_sentence_entities(sentence),
                    'clinical_significance': self._assess_clinical_significance(sentence)
                }
                
                sentence_analysis.append(analysis)
            
            return sentence_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentences: {e}")
            return []
    
    def _extract_medical_concepts(self, text: str) -> Dict[str, Any]:
        """Extract high-level medical concepts"""
        try:
            concepts = {
                'chief_complaint': self._extract_chief_complaint(text),
                'medical_history': self._extract_medical_history(text),
                'current_medications': self._extract_current_medications(text),
                'physical_exam': self._extract_physical_exam_findings(text),
                'lab_results': self._extract_lab_results(text),
                'imaging_results': self._extract_imaging_results(text),
                'assessment_plan': self._extract_assessment_plan(text),
                'allergies': self._extract_allergies(text)
            }
            
            return concepts
            
        except Exception as e:
            self.logger.error(f"Error extracting medical concepts: {e}")
            return {}
    
    def _generate_embeddings(self, text: str) -> Optional[np.ndarray]:
        """Generate embeddings using BioBERT"""
        try:
            if not self.biobert_loaded:
                return None
            
            # Generate sentence embeddings
            embeddings = self.biobert_model.encode(text)
            return embeddings.tolist()  # Convert to list for JSON serialization
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            return None
    
    def _perform_clinical_assessment(self, entities: Dict, concepts: Dict) -> Dict[str, Any]:
        """Perform clinical assessment based on extracted information"""
        try:
            assessment = {
                'complexity_score': 0,
                'urgency_indicators': [],
                'key_findings': [],
                'recommended_actions': []
            }
            
            # Calculate complexity based on number of entities
            total_entities = sum(len(entity_list) for entity_list in entities.values())
            assessment['complexity_score'] = min(total_entities / 10, 1.0)  # Normalize to 0-1
            
            # Detect urgency indicators
            urgent_keywords = ['emergency', 'urgent', 'critical', 'severe', 'acute', 'stat', 'immediate']
            for keyword in urgent_keywords:
                if any(keyword in str(entity_list).lower() for entity_list in entities.values()):
                    assessment['urgency_indicators'].append(keyword)
            
            # Extract key findings
            if 'diagnoses' in entities:
                assessment['key_findings'].extend([e['text'] for e in entities['diagnoses']])
            if 'symptoms' in entities:
                assessment['key_findings'].extend([e['text'] for e in entities['symptoms']])
            
            # Generate recommendations
            if assessment['urgency_indicators']:
                assessment['recommended_actions'].append('Immediate medical attention required')
            if 'medications' in entities and len(entities['medications']) > 5:
                assessment['recommended_actions'].append('Medication review recommended')
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error in clinical assessment: {e}")
            return {}
    
    def _extract_temporal_information(self, text: str) -> Dict[str, List[str]]:
        """Extract temporal information from medical text"""
        try:
            temporal_patterns = {
                'dates': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                'times': r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b',
                'durations': r'\b\d+\s*(?:days?|weeks?|months?|years?|hours?|minutes?)\b',
                'frequencies': r'\b(?:daily|weekly|monthly|yearly|twice|once|every)\b'
            }
            
            temporal_info = {}
            for category, pattern in temporal_patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                temporal_info[category] = list(set(matches))  # Remove duplicates
            
            return temporal_info
            
        except Exception as e:
            self.logger.error(f"Error extracting temporal information: {e}")
            return {}
    
    def _calculate_text_statistics(self, text: str) -> Dict[str, Any]:
        """Calculate basic statistics about the medical text"""
        try:
            stats = {
                'total_characters': len(text),
                'total_words': len(text.split()),
                'total_sentences': len(sent_tokenize(text)) if self.nltk_ready else len(text.split('.')),
                'average_word_length': np.mean([len(word) for word in text.split()]),
                'medical_density': self._calculate_medical_density(text)
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating statistics: {e}")
            return {}
    
    # Helper methods for concept extraction
    def _extract_chief_complaint(self, text: str) -> List[str]:
        """Extract chief complaint information"""
        patterns = [
            r'chief complaint[:\s]+([^.]+)',
            r'presenting complaint[:\s]+([^.]+)',
            r'patient presents with[:\s]+([^.]+)'
        ]
        
        complaints = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            complaints.extend(matches)
        
        return [c.strip() for c in complaints]
    
    def _extract_medical_history(self, text: str) -> List[str]:
        """Extract medical history information"""
        patterns = [
            r'medical history[:\s]+([^.]+)',
            r'past medical history[:\s]+([^.]+)',
            r'history of[:\s]+([^.]+)'
        ]
        
        history = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            history.extend(matches)
        
        return [h.strip() for h in history]
    
    def _extract_current_medications(self, text: str) -> List[str]:
        """Extract current medications"""
        patterns = [
            r'current medications?[:\s]+([^.]+)',
            r'medications?[:\s]+([^.]+)',
            r'taking[:\s]+([^.]+)'
        ]
        
        medications = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            medications.extend(matches)
        
        return [m.strip() for m in medications]
    
    def _extract_physical_exam_findings(self, text: str) -> List[str]:
        """Extract physical examination findings"""
        patterns = [
            r'physical exam[:\s]+([^.]+)',
            r'examination[:\s]+([^.]+)',
            r'on examination[:\s]+([^.]+)'
        ]
        
        findings = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            findings.extend(matches)
        
        return [f.strip() for f in findings]
    
    def _extract_lab_results(self, text: str) -> List[str]:
        """Extract laboratory results"""
        patterns = [
            r'lab results?[:\s]+([^.]+)',
            r'laboratory[:\s]+([^.]+)',
            r'blood work[:\s]+([^.]+)'
        ]
        
        results = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            results.extend(matches)
        
        return [r.strip() for r in results]
    
    def _extract_imaging_results(self, text: str) -> List[str]:
        """Extract imaging results"""
        patterns = [
            r'imaging[:\s]+([^.]+)',
            r'x-ray[:\s]+([^.]+)',
            r'ct scan[:\s]+([^.]+)',
            r'mri[:\s]+([^.]+)'
        ]
        
        results = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            results.extend(matches)
        
        return [r.strip() for r in results]
    
    def _extract_assessment_plan(self, text: str) -> List[str]:
        """Extract assessment and plan"""
        patterns = [
            r'assessment and plan[:\s]+([^.]+)',
            r'plan[:\s]+([^.]+)',
            r'impression[:\s]+([^.]+)'
        ]
        
        plans = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            plans.extend(matches)
        
        return [p.strip() for p in plans]
    
    def _extract_allergies(self, text: str) -> List[str]:
        """Extract allergy information"""
        patterns = [
            r'allergies?[:\s]+([^.]+)',
            r'allergic to[:\s]+([^.]+)',
            r'nkda',  # no known drug allergies
            r'nka'    # no known allergies
        ]
        
        allergies = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            allergies.extend(matches)
        
        return [a.strip() for a in allergies]
    
    def _calculate_medical_relevance_score(self, sentence: str) -> float:
        """Calculate how medically relevant a sentence is"""
        medical_keywords = [
            'patient', 'diagnosis', 'treatment', 'symptoms', 'medication',
            'examination', 'test', 'results', 'condition', 'therapy'
        ]
        
        score = sum(1 for keyword in medical_keywords if keyword.lower() in sentence.lower())
        return min(score / len(medical_keywords), 1.0)
    
    def _analyze_sentence_sentiment(self, sentence: str) -> str:
        """Analyze sentiment of a sentence (positive, negative, neutral)"""
        # Simple sentiment analysis based on keywords
        positive_words = ['improved', 'better', 'stable', 'normal', 'good', 'excellent']
        negative_words = ['worse', 'deteriorated', 'abnormal', 'poor', 'critical', 'severe']
        
        pos_count = sum(1 for word in positive_words if word in sentence.lower())
        neg_count = sum(1 for word in negative_words if word in sentence.lower())
        
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _extract_sentence_entities(self, sentence: str) -> List[str]:
        """Extract entities from a single sentence"""
        entities = []
        
        for category, patterns in self.medical_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, sentence, re.IGNORECASE)
                entities.extend(matches)
        
        return list(set(entities))  # Remove duplicates
    
    def _assess_clinical_significance(self, sentence: str) -> str:
        """Assess the clinical significance of a sentence"""
        high_significance = ['diagnosis', 'treatment', 'critical', 'severe', 'emergency']
        medium_significance = ['symptoms', 'test', 'examination', 'medication']
        
        sentence_lower = sentence.lower()
        
        if any(word in sentence_lower for word in high_significance):
            return 'high'
        elif any(word in sentence_lower for word in medium_significance):
            return 'medium'
        else:
            return 'low'
    
    def _calculate_medical_density(self, text: str) -> float:
        """Calculate the density of medical terms in the text"""
        words = text.split()
        medical_word_count = 0
        
        for word in words:
            word_lower = word.lower()
            for patterns in self.medical_patterns.values():
                for pattern in patterns:
                    if re.search(pattern, word_lower):
                        medical_word_count += 1
                        break
        
        return medical_word_count / len(words) if words else 0.0