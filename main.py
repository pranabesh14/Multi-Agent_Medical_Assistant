import streamlit as st
import os
import tempfile
import pandas as pd
from pathlib import Path
import time
import json
from datetime import datetime

# Import custom modules
from src.models.llama_processor import LlamaProcessor
from src.models.image_processor import MedicalImageProcessor
from src.models.text_processor import MedicalTextProcessor
from src.rag.retrieval_system import MedicalRAGSystem
from src.utils.privacy_utils import DataDeidentifier
from src.utils.file_handler import FileHandler

class MultimodalMedicalAssistant:
    def __init__(self):
        self.initialize_components()
        
    def initialize_components(self):
        """Initialize all the required components"""
        try:
            # Initialize processors
            self.llama_processor = LlamaProcessor()
            self.image_processor = MedicalImageProcessor()
            self.text_processor = MedicalTextProcessor()
            self.rag_system = MedicalRAGSystem()
            self.deidentifier = DataDeidentifier()
            self.file_handler = FileHandler()
            
            # Initialize session state
            if 'medical_data' not in st.session_state:
                st.session_state.medical_data = {
                    'uploaded_files': [],
                    'processed_images': [],
                    'processed_texts': [],
                    'conversation_history': []
                }
        except Exception as e:
            st.error(f"Error initializing components: {str(e)}")

def main():
    st.set_page_config(
        page_title="Multimodal Medical Assistant",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize the assistant
    if 'assistant' not in st.session_state:
        st.session_state.assistant = MultimodalMedicalAssistant()
    
    assistant = st.session_state.assistant
    
    # Main header
    st.title("Multimodal Medical Assistant")
    st.markdown("**AI-powered clinical decision support with multimodal data processing**")
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        tab = st.selectbox(
            "Choose Function",
            ["Medical Chat", "System Status"]
        )
        
        # Add information about the simplified interface
        st.markdown("---")
        st.markdown("### 🏥 Medical Chat Features")
        st.markdown("""
        **The Medical Chat now handles everything:**
        
        🤖 **Intelligent Analysis**
        - Text medical questions
        - X-ray image analysis  
        - Skin lesion evaluation
        - Medical report review
        - Cross-modal reasoning
        
        🧠 **Smart Features**
        - Auto-detects content type
        - Routes to appropriate AI models
        - Context-aware responses
        - Topic change detection
        
        💬 **Simple to Use**
        - Single chat interface
        - Upload any medical file
        - Ask any medical question
        - Get comprehensive answers
        """)
        
        st.markdown("---")
        st.markdown("### ⚙️ System Status")
        st.markdown("""
        Check AI model availability and system performance.
        """)
    
    if tab == "Medical Chat":
        render_chat_tab(assistant)
    elif tab == "System Status":
        render_status_tab(assistant)

def render_data_upload_tab(assistant):
    """Render the data upload interface"""
    st.header("Medical Data Upload & Processing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Medical Images")
        uploaded_images = st.file_uploader(
            "Upload X-rays, MRIs, CT scans (DICOM, PNG, JPEG)",
            type=['dcm', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            key="image_uploader"
        )
        
        if uploaded_images:
            for uploaded_file in uploaded_images:
                if st.button(f"Process {uploaded_file.name}", key=f"process_img_{uploaded_file.name}"):
                    with st.spinner("Processing medical image..."):
                        try:
                            # Save uploaded file temporarily
                            temp_path = assistant.file_handler.save_temp_file(uploaded_file)
                            
                            # Process the image
                            result = assistant.image_processor.process_medical_image(temp_path)
                            
                            # Store in session state
                            st.session_state.medical_data['processed_images'].append({
                                'filename': uploaded_file.name,
                                'analysis': result,
                                'timestamp': datetime.now()
                            })
                            
                            st.success(f"Successfully processed {uploaded_file.name}")
                            st.json(result)
                            
                        except Exception as e:
                            st.error(f"Error processing image: {str(e)}")
    
    with col2:
        st.subheader("Upload Medical Documents")
        uploaded_docs = st.file_uploader(
            "Upload EHRs, discharge summaries, clinical notes (PDF, TXT)",
            type=['pdf', 'txt', 'doc', 'docx'],
            accept_multiple_files=True,
            key="doc_uploader"
        )
        
        if uploaded_docs:
            for uploaded_file in uploaded_docs:
                if st.button(f"Process {uploaded_file.name}", key=f"process_doc_{uploaded_file.name}"):
                    with st.spinner("Processing medical document..."):
                        try:
                            # Extract text from document
                            text_content = assistant.file_handler.extract_text(uploaded_file)
                            
                            # De-identify the text
                            deidentified_text = assistant.deidentifier.deidentify_text(text_content)
                            
                            # Process with medical NLP
                            processed_result = assistant.text_processor.process_medical_text(deidentified_text)
                            
                            # Add to RAG system
                            assistant.rag_system.add_document(deidentified_text, uploaded_file.name)
                            
                            # Store in session state
                            st.session_state.medical_data['processed_texts'].append({
                                'filename': uploaded_file.name,
                                'content': deidentified_text,
                                'analysis': processed_result,
                                'timestamp': datetime.now()
                            })
                            
                            st.success(f"Successfully processed {uploaded_file.name}")
                            st.json(processed_result)
                            
                        except Exception as e:
                            st.error(f"Error processing document: {str(e)}")

def render_analysis_tab(assistant):
    """Render the cross-modal analysis interface"""
    st.header("Cross-Modal Medical Analysis")
    
    # Check if we have both images and text data
    if not st.session_state.medical_data['processed_images']:
        st.warning("Please upload and process medical images first.")
        return
    
    if not st.session_state.medical_data['processed_texts']:
        st.warning("Please upload and process medical documents first.")
        return
    
    st.subheader("Available Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Processed Images:**")
        for img in st.session_state.medical_data['processed_images']:
            st.write(f"- {img['filename']} ({img['timestamp'].strftime('%H:%M:%S')})")
    
    with col2:
        st.write("**Processed Documents:**")
        for doc in st.session_state.medical_data['processed_texts']:
            st.write(f"- {doc['filename']} ({doc['timestamp'].strftime('%H:%M:%S')})")
    
    st.subheader("Cross-Modal Query")
    
    # Predefined query templates
    query_templates = [
        "Interpret this X-ray and correlate with patient symptoms",
        "Compare imaging findings with clinical notes",
        "Provide diagnostic suggestions based on multimodal evidence",
        "Summarize all available patient data",
        "Custom query"
    ]
    
    selected_template = st.selectbox("Select query template:", query_templates)
    
    if selected_template == "Custom query":
        user_query = st.text_area("Enter your medical query:")
    else:
        user_query = selected_template
        st.write(f"Query: {user_query}")
    
    if st.button("Analyze", key="cross_modal_analyze"):
        if user_query:
            with st.spinner("Performing cross-modal analysis..."):
                try:
                    # Gather all data for analysis
                    multimodal_data = {
                        'images': st.session_state.medical_data['processed_images'],
                        'texts': st.session_state.medical_data['processed_texts'],
                        'query': user_query
                    }
                    
                    # Perform cross-modal analysis
                    result = assistant.llama_processor.cross_modal_analysis(multimodal_data)
                    
                    st.subheader("Analysis Results")
                    st.write(result['response'])
                    
                    if 'evidence' in result:
                        st.subheader("Supporting Evidence")
                        for evidence in result['evidence']:
                            st.write(f"- **{evidence['type']}**: {evidence['content']}")
                    
                    if 'recommendations' in result:
                        st.subheader("Clinical Recommendations")
                        for rec in result['recommendations']:
                            st.write(f"- {rec}")
                            
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")

def render_rag_tab(assistant):
    """Render the RAG query interface"""
    st.header("Medical Knowledge Retrieval (RAG)")
    
    st.subheader("Query Medical Knowledge Base")
    
    # Query input
    query = st.text_input("Enter your medical question:")
    
    # Query type selection
    query_type = st.selectbox(
        "Query Type:",
        ["General Medical", "Diagnostic", "Treatment", "Drug Information", "Procedure"]
    )
    
    # Advanced options
    with st.expander("Advanced Options"):
        max_results = st.slider("Maximum results:", 1, 20, 5)
        similarity_threshold = st.slider("Similarity threshold:", 0.1, 1.0, 0.7)
    
    if st.button("Search Knowledge Base"):
        if query:
            with st.spinner("Searching medical knowledge base..."):
                try:
                    # Perform RAG search
                    results = assistant.rag_system.query(
                        query, 
                        max_results=max_results,
                        threshold=similarity_threshold
                    )
                    
                    if results:
                        st.subheader("Search Results")
                        
                        for i, result in enumerate(results, 1):
                            with st.expander(f"Result {i} (Score: {result['score']:.3f})"):
                                st.write("**Source:**", result['source'])
                                st.write("**Content:**", result['content'])
                                
                        # Generate comprehensive answer using Llama
                        comprehensive_answer = assistant.llama_processor.generate_rag_response(query, results)
                        
                        st.subheader("AI-Generated Summary")
                        st.write(comprehensive_answer)
                        
                    else:
                        st.warning("No relevant results found. Try adjusting your query or similarity threshold.")
                        
                except Exception as e:
                    st.error(f"Error during search: {str(e)}")

def render_chat_tab(assistant):
    """Render the simplified medical chat interface"""
    st.header("Medical Assistant Chat")
    st.markdown("**Ask medical questions or upload images/documents for analysis**")
    
    # Chat history display
    if st.session_state.medical_data['conversation_history']:
        st.subheader("Conversation History")
        for i, exchange in enumerate(st.session_state.medical_data['conversation_history']):
            with st.container():
                # Show new topic indicator
                if exchange.get('new_topic'):
                    st.info("🆕 **New Medical Topic Started**")
                
                st.write(f"**You ({exchange['timestamp']}):** {exchange['user_message']}")
                
                # Display uploaded content if present
                if 'uploaded_image' in exchange:
                    st.image(exchange['uploaded_image'], caption=f"📸 {exchange.get('detected_type', 'Medical Image')}", width=300)
                
                if 'uploaded_document' in exchange:
                    st.info(f"📄 Document: {exchange['document_name']} ({exchange.get('detected_type', 'Medical Document')})")
                
                st.write(f"**Assistant:** {exchange['assistant_response']}")
                
                # Show analysis details if available
                if 'analysis_details' in exchange:
                    with st.expander("📊 Analysis Details"):
                        details = exchange['analysis_details']
                        if 'detected_type' in details:
                            st.write(f"**Detected Type:** {details['detected_type']}")
                        if 'confidence' in details:
                            st.write(f"**Confidence:** {details['confidence']:.1%}")
                        if 'key_findings' in details:
                            st.write(f"**Key Findings:** {', '.join(details['key_findings'][:3])}")
                
                st.divider()
    
    # Simplified input section
    st.subheader("Ask or Upload")
    
    # Context management buttons
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.write("**Conversation Context:**")
    with col2:
        if st.button("🔄 New Topic", help="Start a new medical topic (clears previous context)"):
            # Mark that user wants to start a new topic
            st.session_state.new_topic_requested = True
            st.success("✅ Starting new medical topic - previous context cleared")
    with col3:
        if st.button("🗑️ Clear Chat", help="Clear entire conversation history"):
            st.session_state.medical_data['conversation_history'] = []
            st.success("✅ Chat history cleared")
            st.rerun()
    
    # Single text input area
    user_input = st.text_area(
        "Ask a medical question or describe what you're uploading:",
        height=100,
        placeholder="Examples:\n• What are the symptoms of pneumonia?\n• Please analyze this X-ray\n• Is this skin lesion concerning?\n• Review this medical report\n\n💡 Tip: Click 'New Topic' above if asking about a different condition"
    )
    
    # Single file uploader for all types
    uploaded_file = st.file_uploader(
        "📎 Upload any medical file (optional):",
        type=['jpg', 'jpeg', 'png', 'dcm', 'pdf', 'txt', 'doc', 'docx'],
        help="Supported: X-rays, CT scans, MRIs, skin photos, medical reports, clinical notes"
    )
    
    # Show file preview
    if uploaded_file:
        file_type = uploaded_file.type
        if file_type.startswith('image/'):
            st.image(uploaded_file, caption="📸 Uploaded Image", width=400)
        else:
            st.info(f"📄 Document uploaded: {uploaded_file.name}")
    
    # Send button
    if st.button("Send", type="primary", use_container_width=True):
        if user_input or uploaded_file:
            with st.spinner("🔍 Analyzing and generating response..."):
                try:
                    response_data = {}
                    detected_type = "General Query"
                    
                    # Automatic file processing and type detection
                    if uploaded_file:
                        st.info("📋 Step 1/3: Processing uploaded file...")
                        
                        # Auto-detect file type and content
                        detection_result = assistant._auto_detect_content_type(uploaded_file, user_input)
                        detected_type = detection_result['type']
                        confidence = detection_result['confidence']
                        
                        st.success(f"✅ Detected: {detected_type} (confidence: {confidence:.1%})")
                        
                        # Process based on detected type
                        if detection_result['category'] == 'image':
                            st.info("📋 Step 2/3: Analyzing medical image...")
                            
                            # Save and process image
                            temp_path = assistant.file_handler.save_temp_file(uploaded_file)
                            image_analysis = assistant.image_processor.process_medical_image(temp_path)
                            
                            response_data['image_analysis'] = image_analysis
                            response_data['uploaded_image'] = uploaded_file
                            response_data['detected_type'] = detected_type
                            response_data['confidence'] = confidence
                            
                        elif detection_result['category'] == 'document':
                            st.info("📋 Step 2/3: Processing medical document...")
                            
                            # Extract and process document text
                            extracted_text = assistant.file_handler.extract_text(uploaded_file)
                            deidentified_text = assistant.deidentifier.deidentify_text(extracted_text)
                            text_analysis = assistant.text_processor.process_medical_text(deidentified_text)
                            
                            response_data['document_analysis'] = text_analysis
                            response_data['document_content'] = deidentified_text[:1000] + "..." if len(deidentified_text) > 1000 else deidentified_text
                            response_data['uploaded_document'] = uploaded_file
                            response_data['detected_type'] = detected_type
                            response_data['confidence'] = confidence
                    
                    st.info("📋 Step 3/3: Generating intelligent response...")
                    
                    # Generate contextual query if none provided
                    if not user_input:
                        user_input = assistant._generate_contextual_query(detected_type, response_data)
                    
                    # Generate appropriate response based on content type
                    response = assistant._generate_intelligent_response(
                        user_input, 
                        detected_type, 
                        response_data,
                        st.session_state.medical_data['conversation_history']
                    )
                    
                    # Prepare conversation entry
                    conversation_entry = {
                        'user_message': user_input,
                        'assistant_response': response,
                        'timestamp': datetime.now().strftime('%H:%M:%S'),
                        'detected_type': detected_type
                    }
                    
                    # Mark if this is a new topic
                    if getattr(st.session_state, 'new_topic_requested', False):
                        conversation_entry['new_topic'] = True
                        st.session_state.new_topic_requested = False
                    
                    # Add file data if uploaded
                    if uploaded_file:
                        if response_data.get('image_analysis'):
                            conversation_entry['uploaded_image'] = uploaded_file
                            conversation_entry['analysis_details'] = {
                                'detected_type': detected_type,
                                'confidence': response_data.get('confidence', 0),
                                'key_findings': assistant._extract_key_findings(response_data['image_analysis'])
                            }
                        elif response_data.get('document_analysis'):
                            conversation_entry['uploaded_document'] = True
                            conversation_entry['document_name'] = uploaded_file.name
                            conversation_entry['analysis_details'] = {
                                'detected_type': detected_type,
                                'confidence': response_data.get('confidence', 0),
                                'key_findings': assistant._extract_key_findings(response_data['document_analysis'])
                            }
                    
                    # Add to conversation history
                    st.session_state.medical_data['conversation_history'].append(conversation_entry)
                    
                    # Display success and response
                    st.success("✅ Analysis complete!")
                    
                    # Show detection summary if file was uploaded
                    if uploaded_file:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Content Type", detected_type)
                        with col2:
                            st.metric("Confidence", f"{response_data.get('confidence', 0):.1%}")
                        with col3:
                            processing_time = "< 30s" if 'skin' not in detected_type.lower() else "< 90s"
                            st.metric("Processing Time", processing_time)
                    
                    # Display response
                    st.write("**Assistant Response:**")
                    st.write(response)
                    
                    # Rerun to update chat history
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error processing request: {str(e)}")
                    
                    # Add error to conversation for debugging
                    st.session_state.medical_data['conversation_history'].append({
                        'user_message': user_input if user_input else "[File uploaded]",
                        'assistant_response': f"I apologize, but I encountered an error: {str(e)}. Please try again with a simpler question or different file format.",
                        'timestamp': datetime.now().strftime('%H:%M:%S'),
                        'error': True
                    })
    
    # Quick example buttons (simplified)
    st.subheader("💡 Example Questions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🩺 General Medical"):
            example_question = "What are the common symptoms of pneumonia and how is it diagnosed?"
            st.session_state.medical_data['conversation_history'].append({
                'user_message': example_question,
                'assistant_response': assistant.llama_processor.generate_medical_response(example_question),
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'detected_type': 'General Medical Query'
            })
            st.rerun()
    
    with col2:
        if st.button("📸 Upload X-ray"):
            st.info("👆 Use the file uploader above to upload your X-ray image, then ask: 'Please analyze this X-ray'")
    
    with col3:
        if st.button("🔍 Upload Skin Photo"):
            st.info("👆 Use the file uploader above to upload your skin photo, then ask: 'Is this skin lesion concerning?'")
    
    with col4:
        if st.button("📄 Upload Report"):
            st.info("👆 Use the file uploader above to upload your medical report, then ask: 'Summarize this report'")
    
    # Help section
    with st.expander("ℹ️ How to Use This Medical Assistant"):
        st.markdown("""
        **This AI assistant automatically detects and analyzes:**
        
        📸 **Medical Images:**
        - X-rays (chest, limb, etc.)
        - Skin lesions and injuries
        - CT scans and MRIs
        - Other medical photos
        
        📄 **Medical Documents:**
        - Lab reports
        - Radiology reports  
        - Clinical notes
        - Discharge summaries
        
        💬 **Text Questions:**
        - Symptom explanations
        - Medical conditions
        - Treatment information
        - Drug interactions
        
        **Simply upload any file and ask your question - the AI will automatically:**
        1. ✅ Detect what type of content you uploaded
        2. ✅ Choose the right analysis method
        3. ✅ Provide appropriate medical insights
        4. ✅ Include relevant disclaimers and recommendations
        """)
    
    # Medical disclaimer (always visible)
    st.error("""
    ⚠️ **IMPORTANT MEDICAL DISCLAIMER**
    
    This AI assistant is for **educational purposes only**. It should NOT be used for medical diagnosis, treatment decisions, or emergency situations. Always consult qualified healthcare professionals for medical advice.
    
    **For emergencies:** Call your local emergency number immediately.
    """)


# Add these helper methods to the MultimodalMedicalAssistant class
def add_helper_methods_to_assistant():
    """Add the helper methods to the main assistant class"""
    
    def _auto_detect_content_type(self, uploaded_file, user_query=None):
        """Automatically detect the type of uploaded content"""
        try:
            file_name = uploaded_file.name.lower()
            file_type = uploaded_file.type
            
            # Image detection
            if file_type.startswith('image/') or file_name.endswith(('.jpg', '.jpeg', '.png', '.dcm')):
                # Try to detect medical image type from filename or user query
                if any(keyword in file_name for keyword in ['xray', 'x-ray', 'chest', 'lung']):
                    return {'type': 'Chest X-ray', 'category': 'image', 'confidence': 0.9}
                elif any(keyword in file_name for keyword in ['skin', 'lesion', 'mole', 'rash']):
                    return {'type': 'Skin Lesion', 'category': 'image', 'confidence': 0.9}
                elif any(keyword in file_name for keyword in ['ct', 'scan']):
                    return {'type': 'CT Scan', 'category': 'image', 'confidence': 0.8}
                elif any(keyword in file_name for keyword in ['mri']):
                    return {'type': 'MRI Scan', 'category': 'image', 'confidence': 0.8}
                elif file_name.endswith('.dcm'):
                    return {'type': 'DICOM Medical Image', 'category': 'image', 'confidence': 0.95}
                else:
                    # Analyze user query for hints
                    if user_query:
                        query_lower = user_query.lower()
                        if any(keyword in query_lower for keyword in ['x-ray', 'chest', 'lung', 'pneumonia']):
                            return {'type': 'X-ray Image', 'category': 'image', 'confidence': 0.7}
                        elif any(keyword in query_lower for keyword in ['skin', 'lesion', 'mole', 'melanoma', 'rash']):
                            return {'type': 'Skin Lesion', 'category': 'image', 'confidence': 0.7}
                    
                    return {'type': 'Medical Image', 'category': 'image', 'confidence': 0.6}
            
            # Document detection
            elif file_type.startswith('text/') or file_name.endswith(('.pdf', '.txt', '.doc', '.docx')):
                if any(keyword in file_name for keyword in ['lab', 'blood', 'test', 'result']):
                    return {'type': 'Lab Report', 'category': 'document', 'confidence': 0.9}
                elif any(keyword in file_name for keyword in ['radiology', 'imaging', 'scan']):
                    return {'type': 'Radiology Report', 'category': 'document', 'confidence': 0.9}
                elif any(keyword in file_name for keyword in ['discharge', 'summary']):
                    return {'type': 'Discharge Summary', 'category': 'document', 'confidence': 0.9}
                else:
                    return {'type': 'Medical Document', 'category': 'document', 'confidence': 0.7}
            
            else:
                return {'type': 'Unknown File Type', 'category': 'unknown', 'confidence': 0.3}
                
        except Exception as e:
            return {'type': 'File Processing Error', 'category': 'error', 'confidence': 0.0}
    
    def _generate_contextual_query(self, detected_type, response_data):
        """Generate appropriate query based on detected content type"""
        
        if 'x-ray' in detected_type.lower() or 'chest' in detected_type.lower():
            return "Please analyze this X-ray image. What are the key findings and any areas of concern?"
        
        elif 'skin' in detected_type.lower() or 'lesion' in detected_type.lower():
            return "Please analyze this skin lesion. Assess using ABCDE criteria and indicate the level of concern."
        
        elif 'ct' in detected_type.lower():
            return "Please analyze this CT scan image. What structures are visible and any abnormal findings?"
        
        elif 'mri' in detected_type.lower():
            return "Please analyze this MRI image. Describe the anatomy visible and any pathological findings."
        
        elif 'lab' in detected_type.lower():
            return "Please review this lab report. Highlight any abnormal values and their clinical significance."
        
        elif 'radiology' in detected_type.lower():
            return "Please summarize this radiology report. What are the key findings and recommendations?"
        
        elif 'discharge' in detected_type.lower():
            return "Please summarize this discharge summary. What was the diagnosis, treatment, and follow-up plan?"
        
        elif 'document' in detected_type.lower():
            return "Please analyze this medical document. Summarize the key medical information and findings."
        
        else:
            return "Please analyze the uploaded content and provide relevant medical insights."
    
    def _generate_intelligent_response(self, user_query, detected_type, response_data, conversation_history):
        """Generate intelligent response based on content type and context"""
        
        try:
            # Check if user requested new topic
            new_topic = getattr(st.session_state, 'new_topic_requested', False)
            if new_topic:
                # Clear the flag and use empty conversation history
                st.session_state.new_topic_requested = False
                conversation_history = []
            
            # Handle image analysis
            if response_data.get('image_analysis'):
                if 'skin' in detected_type.lower():
                    # Use optimized skin lesion processing
                    return self._generate_skin_lesion_response(user_query, response_data, conversation_history)
                else:
                    # Standard image analysis
                    return self._generate_image_response(user_query, detected_type, response_data, conversation_history)
            
            # Handle document analysis
            elif response_data.get('document_analysis'):
                return self._generate_document_response(user_query, detected_type, response_data, conversation_history)
            
            # Handle text-only queries
            else:
                return self.llama_processor.generate_medical_response(user_query, conversation_history)
                
        except Exception as e:
            return f"I encountered an error while analyzing your content. Please try asking a more specific question or contact support. Error: {str(e)}"
    
    def _generate_skin_lesion_response(self, user_query, response_data, conversation_history):
        """Generate optimized response for skin lesions"""
        
        try:
            # Create focused skin lesion prompt
            image_analysis = response_data['image_analysis']
            
            skin_prompt = f"""Skin lesion analysis request: {user_query}

Image Analysis Summary:
- Detected as: {response_data['detected_type']}
- Processing Status: {image_analysis.get('processing_status', 'completed')}
- Findings: {image_analysis.get('medical_findings', {})}

Provide focused response (under 250 words):
1. Visual assessment based on analysis
2. ABCDE criteria evaluation
3. Concern level (low/medium/high)
4. Immediate recommendations

Include appropriate medical disclaimers."""

            return self.llama_processor.generate_response(
                skin_prompt,
                temperature=0.2,
                max_tokens=500,
                timeout=90
            )
            
        except Exception:
            # Fallback response for skin lesions
            return """**Skin Lesion Analysis**

I've processed your skin lesion image. Here's important guidance:

**ABCDE Criteria Assessment:**
- **A**symmetry: Check if one half differs from the other
- **B**order: Look for irregular or poorly defined edges
- **C**olor: Note multiple colors or uneven distribution  
- **D**iameter: Measure if larger than 6mm (pencil eraser)
- **E**volving: Track any recent changes in size, shape, or color

**Recommendations:**
- Consult a dermatologist for professional evaluation
- Monitor for any changes and take photos for comparison
- Protect the area from sun exposure
- Seek immediate care if bleeding, rapid growth, or other concerning changes occur

**IMPORTANT:** This analysis is for educational purposes only. Skin lesion evaluation requires professional dermatological assessment. Any concerning features warrant immediate medical attention."""
    
    def _generate_image_response(self, user_query, detected_type, response_data, conversation_history):
        """Generate response for medical images (non-skin)"""
        
        image_analysis = response_data['image_analysis']
        
        multimodal_prompt = f"""Medical Image Analysis Request: {user_query}

Image Details:
- Type: {detected_type}
- Modality: {image_analysis.get('modality', 'Unknown')}
- Findings: {image_analysis.get('medical_findings', {})}
- Quality: {image_analysis.get('characteristics', {}).get('quality_category', 'Unknown')}

Provide comprehensive analysis including:
1. Image interpretation
2. Key findings and abnormalities
3. Clinical significance
4. Recommendations for next steps

Include appropriate medical disclaimers."""

        return self.llama_processor.generate_medical_response(
            multimodal_prompt,
            conversation_history,
            image_type=detected_type
        )
    
    def _generate_document_response(self, user_query, detected_type, response_data, conversation_history):
        """Generate response for medical documents"""
        
        document_analysis = response_data['document_analysis']
        document_content = response_data['document_content']
        
        document_prompt = f"""Medical Document Analysis Request: {user_query}

Document Type: {detected_type}
Content Preview: {document_content}

Document Analysis Summary:
- Entities Found: {document_analysis.get('entities', {})}
- Medical Concepts: {document_analysis.get('medical_concepts', {})}
- Clinical Assessment: {document_analysis.get('clinical_assessment', {})}

Please provide:
1. Document summary
2. Key medical findings
3. Important values or results
4. Clinical significance and recommendations

Include appropriate medical disclaimers."""

        return self.llama_processor.generate_medical_response(document_prompt, conversation_history)
    
    def _extract_key_findings(self, analysis_data):
        """Extract key findings from analysis data for display"""
        
        findings = []
        
        try:
            if 'medical_findings' in analysis_data:
                medical_findings = analysis_data['medical_findings'].get('findings', [])
                for finding in medical_findings[:3]:  # Top 3 findings
                    findings.append(finding.get('finding', 'Unknown finding'))
            
            elif 'entities' in analysis_data:
                entities = analysis_data['entities']
                for category, entity_list in entities.items():
                    if entity_list and len(findings) < 3:
                        findings.append(f"{category}: {entity_list[0].get('text', 'Unknown')}")
            
            return findings if findings else ['Analysis completed']
            
        except Exception:
            return ['Processing completed']
    
    # Add methods to the MultimodalMedicalAssistant class
    MultimodalMedicalAssistant._auto_detect_content_type = _auto_detect_content_type
    MultimodalMedicalAssistant._generate_contextual_query = _generate_contextual_query
    MultimodalMedicalAssistant._generate_intelligent_response = _generate_intelligent_response
    MultimodalMedicalAssistant._generate_skin_lesion_response = _generate_skin_lesion_response
    MultimodalMedicalAssistant._generate_image_response = _generate_image_response
    MultimodalMedicalAssistant._generate_document_response = _generate_document_response
    MultimodalMedicalAssistant._extract_key_findings = _extract_key_findings

# Call this function to add the methods
add_helper_methods_to_assistant()

def render_status_tab(assistant):
    """Render the system status interface"""
    st.header("System Status & Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Status")
        
        # Check model availability
        models_status = {
            "Llama (via Ollama)": assistant.llama_processor.check_status(),
            "MedCLIP": assistant.image_processor.check_medclip_status(),
            "BioBERT": assistant.text_processor.check_biobert_status(),
            "RAG System": assistant.rag_system.check_status()
        }
        
        for model, status in models_status.items():
            if status:
                st.success(f"{model}: Available")
            else:
                st.error(f"{model}: Not Available")
    
    with col2:
        st.subheader("Data Statistics")
        
        stats = {
            "Uploaded Images": len(st.session_state.medical_data['processed_images']),
            "Uploaded Documents": len(st.session_state.medical_data['processed_texts']),
            "Chat Messages": len(st.session_state.medical_data['conversation_history']),
            "RAG Documents": assistant.rag_system.get_document_count()
        }
        
        for stat, value in stats.items():
            st.metric(stat, value)
    
    # Clear data options
    st.subheader("Data Management")
    if st.button("Clear All Data", type="secondary"):
        st.session_state.medical_data = {
            'uploaded_files': [],
            'processed_images': [],
            'processed_texts': [],
            'conversation_history': []
        }
        assistant.rag_system.clear_documents()
        st.success("All data cleared!")
        st.rerun()

if __name__ == "__main__":
    main()