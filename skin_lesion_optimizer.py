# skin_lesion_optimizer.py - 

import streamlit as st
import time
from datetime import datetime
import asyncio
import threading

class SkinLesionOptimizer:
    """Optimizes skin lesion analysis to prevent timeouts and improve performance"""
    
    def __init__(self, assistant):
        self.assistant = assistant
        
    def quick_skin_analysis(self, image_path: str, user_query: str = None) -> dict:
        """Perform quick skin lesion analysis with timeout prevention"""
        
        try:
            # Step 1: Quick image processing (usually fast)
            st.info("Step 1/3: Processing image...")
            image_analysis = self.assistant.image_processor.process_medical_image(image_path)
            
            # Step 2: Extract key features without full AI analysis
            st.info("Step 2/3: Extracting features...")
            quick_features = self._extract_quick_features(image_analysis)
            
            # Step 3: Generate focused response
            st.info("Step 3/3: Generating analysis...")
            if user_query:
                simplified_query = self._simplify_skin_query(user_query, quick_features)
            else:
                simplified_query = self._create_default_skin_query(quick_features)
            
            # Use shorter, more focused prompt
            response = self._generate_quick_skin_response(simplified_query, quick_features)
            
            return {
                'success': True,
                'image_analysis': image_analysis,
                'quick_features': quick_features,
                'response': response,
                'processing_time': 'Optimized for speed'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'fallback_response': self._generate_fallback_response()
            }
    
    def _extract_quick_features(self, image_analysis: dict) -> dict:
        """Extract quick visual features without complex AI processing"""
        
        features = {
            'image_quality': 'good',  # Default assumption
            'has_findings': False,
            'basic_description': 'skin lesion image uploaded'
        }
        
        try:
            # Extract basic info from image analysis
            if image_analysis.get('processing_status') == 'success':
                characteristics = image_analysis.get('characteristics', {})
                
                features['image_quality'] = characteristics.get('quality_category', 'fair')
                features['has_findings'] = len(image_analysis.get('medical_findings', {}).get('findings', [])) > 0
                
                # Basic description based on findings
                findings = image_analysis.get('medical_findings', {}).get('findings', [])
                if findings:
                    top_finding = findings[0].get('finding', 'lesion')
                    features['basic_description'] = f"possible {top_finding}"
                
        except Exception:
            pass  # Use defaults
            
        return features
    
    def _simplify_skin_query(self, user_query: str, features: dict) -> str:
        """Simplify user query to prevent timeout"""
        
        # Keep only essential parts of the query
        simplified = user_query[:200]  # Limit length
        
        # Add quick context
        context = f"Image quality: {features['image_quality']}. "
        if features['has_findings']:
            context += f"Description: {features['basic_description']}. "
        
        return context + simplified
    
    def _create_default_skin_query(self, features: dict) -> str:
        """Create default query for skin lesion analysis"""
        
        base_query = "Analyze this skin lesion image. "
        
        if features['image_quality'] == 'poor':
            base_query += "Note: Image quality may limit analysis. "
        
        base_query += "Provide ABCDE assessment and recommendations."
        
        return base_query
    
    def _generate_quick_skin_response(self, query: str, features: dict) -> str:
        """Generate quick response optimized for speed"""
        
        try:
            # Very short, focused prompt
            prompt = f"""Skin lesion analysis request: {query}

Provide a brief response (under 200 words) with:
1. Visual assessment
2. ABCDE criteria (if applicable) 
3. Concern level (low/medium/high)
4. Next steps

Be concise and include medical disclaimer."""

            # Use shortest timeout and smaller response
            response = self.assistant.llama_processor.generate_response(
                prompt, 
                temperature=0.2, 
                max_tokens=400,  # Smaller response
                timeout=90       # Shorter timeout
            )
            
            return response
            
        except Exception as e:
            return self._generate_fallback_response()
    
    def _generate_fallback_response(self) -> str:
        """Generate fallback response when AI analysis fails"""
        
        return """**Skin Lesion Analysis - Quick Assessment**

I've received your skin lesion image but encountered processing difficulties. Here's general guidance:

**ABCDE Criteria to Consider:**
- **A**symmetry: Is one half different from the other?
- **B**order: Are edges irregular, scalloped, or poorly defined?
- **C**olor: Are there multiple colors or uneven distribution?
- **D**iameter: Is it larger than 6mm (pencil eraser size)?
- **E**volving: Has it changed in size, shape, or color?

**Immediate Action Needed If:**
- Any ABCDE criteria are present
- Bleeding, itching, or tenderness
- Recent changes in appearance
- Irregular or changing borders

**Next Steps:**
1. **Consult a dermatologist immediately** for professional evaluation
2. **Take photos** to track any changes
3. **Avoid sun exposure** to the area
4. **Don't delay** if you have concerns

**IMPORTANT:** This is a general guide only. Skin lesion evaluation requires professional dermatological assessment. Any concerning features warrant immediate medical attention.

**Emergency:** If bleeding, rapid growth, or severe changes occur, seek immediate medical care."""

def optimize_skin_lesion_chat():
    """Add to main.py to optimize skin lesion processing"""
    
    st.markdown("""
    ###  Skin Lesion Analysis Tips
    
    **For faster processing:**
    - Use clear, well-lit images
    - Keep questions short and specific
    - Focus on specific concerns (color, size, shape)
    
    **If analysis times out:**
    - Try simpler questions like "Is this concerning?"
    - Upload smaller image files
    - Ask about specific ABCDE criteria only
    """)

# Usage in main.py chat function
def handle_skin_lesion_timeout(assistant, image_path, user_query, image_type):
    """Handle skin lesion analysis with timeout prevention"""
    
    if "skin" in image_type.lower():
        st.warning("🔄 Skin lesion analysis detected. Using optimized processing...")
        
        optimizer = SkinLesionOptimizer(assistant)
        result = optimizer.quick_skin_analysis(image_path, user_query)
        
        if result['success']:
            return result['response']
        else:
            st.error("Analysis timed out. Using fallback response.")
            return result['fallback_response']
    
    # For non-skin images, use standard processing
    return None

if __name__ == "__main__":
    print("Skin Lesion Optimizer loaded successfully!")
    print("This module optimizes skin lesion analysis to prevent timeouts.")