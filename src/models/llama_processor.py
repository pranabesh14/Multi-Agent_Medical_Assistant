import requests
import json
import logging
from typing import Dict, List, Any, Optional
import time

class LlamaProcessor:
    """Handles Llama model interactions via Ollama for medical queries and cross-modal analysis"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llama2"):
        self.base_url = base_url
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        
        # Medical prompt templates
        self.medical_prompts = {
            'diagnostic': """As a medical AI assistant, analyze the following medical data and provide diagnostic insights.
            Consider all available evidence including imaging findings, clinical notes, and patient symptoms.
            
            Medical Data: {data}
            Query: {query}
            
            Please provide:
            1. Clinical Assessment
            2. Differential Diagnosis (if applicable)
            3. Recommended Next Steps
            4. Key Evidence Used
            
            Response:""",
            
            'cross_modal': """You are a multimodal medical AI assistant. Analyze the provided medical images and text data together.
            
            Image Analysis Results: {image_data}
            Clinical Text Data: {text_data}
            User Query: {query}
            
            Provide an integrated analysis that correlates findings from both modalities.
            Focus on:
            1. Correlation between imaging and clinical findings
            2. Comprehensive diagnostic assessment
            3. Clinical recommendations
            4. Evidence-based reasoning
            
            Response:""",
            
            'general': """You are a helpful medical AI assistant. Answer the following medical question accurately and professionally.
            Always include appropriate medical disclaimers and recommend consulting healthcare professionals for serious concerns.
            
            Question: {query}
            Context: {context}
            
            Response:"""
        }
    
    def check_status(self) -> bool:
        """Check if Ollama service is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Ollama service not available: {e}")
            return False
    
    def generate_response(self, prompt: str, temperature: float = 0.3, max_tokens: int = 1000, timeout: int = 180) -> str:
        """Generate response using Llama model"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=timeout  # Use configurable timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response generated')
            else:
                self.logger.error(f"Ollama API error: {response.status_code}")
                return "Error: Could not generate response"
                
        except requests.exceptions.Timeout:
            self.logger.error(f"Ollama request timed out after {timeout} seconds")
            return "Analysis is taking longer than expected. Please try with a simpler query or ensure your image is clear and properly formatted."
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"
    
    def cross_modal_analysis(self, multimodal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform cross-modal analysis of medical images and text"""
        try:
            # Extract data
            images = multimodal_data.get('images', [])
            texts = multimodal_data.get('texts', [])
            query = multimodal_data.get('query', '')
            
            # Prepare image analysis summary
            image_summary = []
            for img in images:
                if 'analysis' in img:
                    image_summary.append(f"Image {img['filename']}: {json.dumps(img['analysis'])}")
            
            # Prepare text summary
            text_summary = []
            for txt in texts:
                if 'analysis' in txt:
                    text_summary.append(f"Document {txt['filename']}: {json.dumps(txt['analysis'])}")
                elif 'content' in txt:
                    # Use first 500 chars if no analysis available
                    content_preview = txt['content'][:500] + "..." if len(txt['content']) > 500 else txt['content']
                    text_summary.append(f"Document {txt['filename']}: {content_preview}")
            
            # Create cross-modal prompt
            prompt = self.medical_prompts['cross_modal'].format(
                image_data="\n".join(image_summary),
                text_data="\n".join(text_summary),
                query=query
            )
            
            # Generate response
            response = self.generate_response(prompt, temperature=0.2)
            
            # Extract evidence sources
            evidence = []
            for img in images:
                evidence.append({
                    'type': 'Image Analysis',
                    'source': img['filename'],
                    'content': str(img.get('analysis', 'No analysis available'))
                })
            
            for txt in texts:
                evidence.append({
                    'type': 'Clinical Document',
                    'source': txt['filename'],
                    'content': str(txt.get('analysis', txt.get('content', '')[:200] + "..."))
                })
            
            return {
                'response': response,
                'evidence': evidence,
                'recommendations': self._extract_recommendations(response)
            }
            
        except Exception as e:
            self.logger.error(f"Error in cross-modal analysis: {e}")
            return {
                'response': f"Error performing analysis: {str(e)}",
                'evidence': [],
                'recommendations': []
            }
    
    def generate_medical_response(self, query: str, conversation_history: List[Dict] = None, image_type: str = None) -> str:
        """Generate medical response with intelligent conversation context"""
        try:
            # Build context from conversation history with topic awareness
            context = ""
            if conversation_history:
                # Get relevant context based on topic similarity
                relevant_context = self._get_relevant_context(query, conversation_history)
                
                if relevant_context:
                    context_parts = []
                    for exchange in relevant_context:
                        context_parts.append(f"Previous Q: {exchange['user_message']}")
                        context_parts.append(f"Previous A: {exchange['assistant_response']}")
                    context = "\n".join(context_parts)
                else:
                    # No relevant context - treat as new topic
                    context = "Starting new medical topic"
            
            # Optimize prompt based on image type
            if image_type and "skin" in image_type.lower():
                # Shorter, more focused prompt for skin lesions
                prompt = f"""Analyze this skin lesion briefly and clearly:

Query: {query}
Context: {context if context else "No previous context"}

Provide a concise analysis covering:
1. Key visual features
2. ABCDE assessment (if applicable)
3. Risk level (low/medium/high concern)
4. Next steps recommendation

Keep response under 300 words. Include medical disclaimer."""
                
                # Use longer timeout for skin analysis
                timeout = 240
                max_tokens = 800
                
            elif image_type and "x-ray" in image_type.lower():
                # Standard prompt for X-rays
                prompt = self.medical_prompts['general'].format(
                    query=query,
                    context=context if context else "No previous context"
                )
                timeout = 180
                max_tokens = 1000
                
            else:
                # Standard medical prompt with context awareness
                if context == "Starting new medical topic":
                    prompt = f"""You are a medical AI assistant. Answer this medical question clearly and independently:

Question: {query}

This is a new medical topic, so provide a complete, standalone response. Do not reference previous conversations or try to connect to other medical conditions unless explicitly asked.

Provide accurate medical information including appropriate disclaimers."""
                else:
                    prompt = self.medical_prompts['general'].format(
                        query=query,
                        context=context
                    )
                
                timeout = 180
                max_tokens = 1000
            
            # Add medical disclaimer
            medical_disclaimer = "\n\nIMPORTANT DISCLAIMER: This information is for educational purposes only and should not replace professional medical advice. Always consult with qualified healthcare professionals for medical decisions."
            
            response = self.generate_response(prompt, temperature=0.3, max_tokens=max_tokens, timeout=timeout)
            
            # Add disclaimer to response if not present
            if "disclaimer" not in response.lower() and "educational" not in response.lower():
                response += medical_disclaimer
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating medical response: {e}")
            return f"I apologize, but I encountered an error while processing your request. This might be due to complexity of the analysis. Please try with a simpler question or contact technical support. Error: {str(e)}"
    
    def _get_relevant_context(self, current_query: str, conversation_history: List[Dict], max_history: int = 2) -> List[Dict]:
        """Get only relevant conversation history based on medical topic similarity"""
        try:
            if not conversation_history:
                return []
            
            # Extract medical topics from current query
            current_topics = self._extract_medical_topics(current_query)
            
            relevant_conversations = []
            
            # Look through recent conversation history (last 5 exchanges)
            recent_history = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
            
            for exchange in reversed(recent_history):  # Start from most recent
                # Extract topics from previous exchange
                previous_topics = self._extract_medical_topics(exchange['user_message'])
                
                # Check if topics are related
                if self._are_topics_related(current_topics, previous_topics):
                    relevant_conversations.insert(0, exchange)  # Add to beginning to maintain order
                    
                    # Limit the number of relevant exchanges
                    if len(relevant_conversations) >= max_history:
                        break
                else:
                    # If topics are not related, stop looking further back
                    break
            
            return relevant_conversations
            
        except Exception as e:
            self.logger.error(f"Error getting relevant context: {e}")
            # Return limited recent context as fallback
            return conversation_history[-1:] if conversation_history else []
    
    def _extract_medical_topics(self, text: str) -> List[str]:
        """Extract medical topics/conditions from text"""
        try:
            text_lower = text.lower()
            
            # Medical condition patterns
            medical_conditions = [
                # Gastrointestinal
                'irritable bowel syndrome', 'ibs', 'crohn', 'colitis', 'gastritis', 'ulcer', 'reflux', 'gerd',
                'diarrhea', 'constipation', 'nausea', 'vomiting', 'abdominal pain',
                
                # Neurological
                'brain cancer', 'brain tumor', 'glioblastoma', 'meningioma', 'stroke', 'seizure', 'epilepsy',
                'headache', 'migraine', 'dementia', 'alzheimer', 'parkinson',
                
                # Cardiovascular
                'heart attack', 'myocardial infarction', 'angina', 'hypertension', 'arrhythmia',
                'heart failure', 'cardiomyopathy', 'chest pain',
                
                # Respiratory
                'pneumonia', 'asthma', 'copd', 'bronchitis', 'tuberculosis', 'lung cancer',
                'shortness of breath', 'cough', 'wheeze',
                
                # Oncological
                'cancer', 'tumor', 'carcinoma', 'lymphoma', 'leukemia', 'metastasis', 'chemotherapy',
                'radiation therapy', 'biopsy',
                
                # Dermatological
                'skin cancer', 'melanoma', 'basal cell', 'squamous cell', 'mole', 'lesion', 'rash',
                'eczema', 'psoriasis', 'acne',
                
                # Musculoskeletal
                'arthritis', 'osteoporosis', 'fracture', 'sprain', 'back pain', 'joint pain',
                
                # Endocrine
                'diabetes', 'thyroid', 'hyperthyroid', 'hypothyroid', 'insulin', 'glucose',
                
                # General symptoms
                'fever', 'fatigue', 'weight loss', 'weight gain', 'pain', 'swelling'
            ]
            
            # Body systems
            body_systems = [
                'gastrointestinal', 'digestive', 'stomach', 'intestine', 'bowel',
                'neurological', 'brain', 'nervous system', 'spine',
                'cardiovascular', 'heart', 'cardiac', 'blood pressure',
                'respiratory', 'lung', 'breathing', 'pulmonary',
                'dermatological', 'skin', 'dermatology',
                'musculoskeletal', 'bone', 'joint', 'muscle',
                'endocrine', 'hormone', 'gland'
            ]
            
            found_topics = []
            
            # Find medical conditions
            for condition in medical_conditions:
                if condition in text_lower:
                    found_topics.append(condition)
            
            # Find body systems
            for system in body_systems:
                if system in text_lower:
                    found_topics.append(system)
            
            return found_topics
            
        except Exception as e:
            self.logger.error(f"Error extracting medical topics: {e}")
            return []
    
    def _are_topics_related(self, topics1: List[str], topics2: List[str]) -> bool:
        """Determine if two sets of medical topics are related"""
        try:
            if not topics1 or not topics2:
                return False
            
            # Direct overlap
            if set(topics1) & set(topics2):
                return True
            
            # System-based relationships
            system_groups = {
                'gastrointestinal': ['irritable bowel syndrome', 'ibs', 'crohn', 'colitis', 'gastritis', 'ulcer', 
                                   'reflux', 'gerd', 'diarrhea', 'constipation', 'nausea', 'vomiting', 
                                   'abdominal pain', 'gastrointestinal', 'digestive', 'stomach', 'intestine', 'bowel'],
                'neurological': ['brain cancer', 'brain tumor', 'glioblastoma', 'stroke', 'seizure', 'epilepsy',
                               'headache', 'migraine', 'dementia', 'alzheimer', 'parkinson', 'neurological', 
                               'brain', 'nervous system', 'spine'],
                'cardiovascular': ['heart attack', 'myocardial infarction', 'angina', 'hypertension', 'arrhythmia',
                                 'heart failure', 'cardiomyopathy', 'chest pain', 'cardiovascular', 'heart', 
                                 'cardiac', 'blood pressure'],
                'respiratory': ['pneumonia', 'asthma', 'copd', 'bronchitis', 'tuberculosis', 'lung cancer',
                              'shortness of breath', 'cough', 'wheeze', 'respiratory', 'lung', 'breathing', 'pulmonary'],
                'dermatological': ['skin cancer', 'melanoma', 'basal cell', 'squamous cell', 'mole', 'lesion', 
                                 'rash', 'eczema', 'psoriasis', 'acne', 'dermatological', 'skin', 'dermatology'],
                'oncological': ['cancer', 'tumor', 'carcinoma', 'lymphoma', 'leukemia', 'metastasis', 
                              'chemotherapy', 'radiation therapy', 'biopsy', 'brain cancer', 'lung cancer', 'skin cancer']
            }
            
            # Find which systems each topic set belongs to
            systems1 = set()
            systems2 = set()
            
            for system, conditions in system_groups.items():
                if any(topic in conditions for topic in topics1):
                    systems1.add(system)
                if any(topic in conditions for topic in topics2):
                    systems2.add(system)
            
            # Topics are related if they belong to the same system
            if systems1 & systems2:
                return True
            
            # Special case: cancer types are related to their organ systems
            cancer_terms = ['cancer', 'tumor', 'carcinoma', 'lymphoma', 'leukemia', 'metastasis']
            has_cancer1 = any(term in ' '.join(topics1) for term in cancer_terms)
            has_cancer2 = any(term in ' '.join(topics2) for term in cancer_terms)
            
            if has_cancer1 and has_cancer2:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking topic relationship: {e}")
            return False  # Default to treating as unrelated to avoid incorrect connections
    
    def generate_rag_response(self, query: str, retrieved_docs: List[Dict]) -> str:
        """Generate comprehensive response using RAG results"""
        try:
            # Prepare context from retrieved documents
            context_parts = []
            for i, doc in enumerate(retrieved_docs[:5], 1):  # Use top 5 results
                context_parts.append(f"Source {i} ({doc['source']}): {doc['content']}")
            
            context = "\n\n".join(context_parts)
            
            prompt = f"""Based on the following medical knowledge sources, provide a comprehensive answer to the user's question.

Medical Knowledge Sources:
{context}

User Question: {query}

Please provide a detailed, evidence-based response that:
1. Directly answers the question
2. Cites relevant sources
3. Provides additional context when helpful
4. Includes appropriate medical disclaimers

Response:"""
            
            response = self.generate_response(prompt, temperature=0.2, max_tokens=1500)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating RAG response: {e}")
            return f"Error generating comprehensive response: {str(e)}"
    
    def _extract_recommendations(self, response: str) -> List[str]:
        """Extract clinical recommendations from response"""
        recommendations = []
        
        # Simple extraction based on common patterns
        lines = response.split('\n')
        in_recommendations = False
        
        for line in lines:
            line = line.strip()
            
            # Look for recommendation sections
            if any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'next steps', 'follow-up']):
                in_recommendations = True
                if line and not any(keyword in line.lower() for keyword in ['recommend', 'suggest']):
                    recommendations.append(line)
            elif in_recommendations and line.startswith(('•', '-', '*', '1.', '2.', '3.')):
                recommendations.append(line.lstrip('•-*123456789. '))
            elif in_recommendations and not line:
                in_recommendations = False
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def validate_medical_query(self, query: str) -> Dict[str, Any]:
        """Validate and categorize medical query"""
        query_lower = query.lower()
        
        categories = []
        if any(term in query_lower for term in ['diagnose', 'diagnosis', 'symptoms', 'condition']):
            categories.append('diagnostic')
        if any(term in query_lower for term in ['treatment', 'therapy', 'medication', 'drug']):
            categories.append('treatment')
        if any(term in query_lower for term in ['x-ray', 'mri', 'ct scan', 'image', 'radiology']):
            categories.append('imaging')
        if any(term in query_lower for term in ['emergency', 'urgent', 'severe', 'critical']):
            categories.append('urgent')
        
        return {
            'categories': categories,
            'is_medical': len(categories) > 0,
            'requires_multimodal': 'imaging' in categories,
            'urgency_level': 'high' if 'urgent' in categories else 'normal'
        }