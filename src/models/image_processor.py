# src/models/image_processor.py
import cv2
import numpy as np
import pydicom
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
import logging
from typing import Dict, List, Any, Tuple, Optional
import os
import tempfile

class MedicalImageProcessor:
    """Processes medical images using MedCLIP/BiomedCLIP for clinical image analysis"""
    
    def __init__(self, model_name: str = None):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use provided model or fallback to working CLIP model
        if model_name is None:
            model_name = "openai/clip-vit-base-patch32"  # Working fallback
        
        try:
            # Load CLIP model (fallback since BiomedCLIP not directly available)
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            self.model_loaded = True
            self.logger.info(f"CLIP model loaded successfully on {self.device}")
            if "openai/clip" in model_name:
                self.logger.warning("Using standard CLIP model as BiomedCLIP fallback")
            
        except Exception as e:
            self.logger.error(f"Failed to load CLIP model: {e}")
            self.model_loaded = False
            
        # Medical terminology for image analysis
        self.medical_labels = {
            'xray': [
                'normal chest x-ray', 'pneumonia', 'pleural effusion', 'pneumothorax',
                'cardiomegaly', 'pulmonary edema', 'lung nodule', 'rib fracture',
                'atelectasis', 'consolidation', 'interstitial lung disease'
            ],
            'ct': [
                'normal ct scan', 'brain hemorrhage', 'stroke', 'tumor', 'fracture',
                'pulmonary embolism', 'kidney stones', 'liver lesion', 'appendicitis'
            ],
            'mri': [
                'normal mri', 'brain tumor', 'multiple sclerosis', 'herniated disc',
                'torn ligament', 'joint effusion', 'bone marrow edema'
            ],
            'pathology': [
                'normal tissue', 'malignant tumor', 'benign tumor', 'inflammation',
                'necrosis', 'fibrosis', 'dysplasia', 'hyperplasia'
            ]
        }
        
        # Image preprocessing parameters
        self.image_size = (224, 224)
    
    def check_medclip_status(self) -> bool:
        """Check if MedCLIP model is loaded and ready"""
        return self.model_loaded
    
    def process_medical_image(self, image_path: str) -> Dict[str, Any]:
        """Process a medical image and return analysis results"""
        try:
            # Determine file type and load image
            if image_path.lower().endswith('.dcm'):
                image_array = self._load_dicom_image(image_path)
                image_type = 'dicom'
            else:
                image_array = self._load_standard_image(image_path)
                image_type = 'standard'
            
            if image_array is None:
                return {'error': 'Failed to load image'}
            
            # Preprocess image
            processed_image = self._preprocess_image(image_array)
            
            # Extract image features
            image_features = self._extract_image_features(processed_image)
            
            # Classify image type (X-ray, CT, MRI, etc.)
            detected_modality = self._classify_modality(processed_image)
            
            # Perform medical image analysis
            medical_analysis = self._analyze_medical_content(processed_image, detected_modality)
            
            # Extract additional image characteristics
            image_characteristics = self._extract_image_characteristics(image_array)
            
            # Generate comprehensive analysis
            analysis_result = {
                'modality': detected_modality,
                'medical_findings': medical_analysis,
                'image_features': image_features,
                'characteristics': image_characteristics,
                'image_type': image_type,
                'processing_status': 'success'
            }
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error processing medical image: {e}")
            return {
                'error': str(e),
                'processing_status': 'failed'
            }
    
    def _load_dicom_image(self, dicom_path: str) -> Optional[np.ndarray]:
        """Load and convert DICOM image to numpy array"""
        try:
            # Read DICOM file
            dicom_data = pydicom.dcmread(dicom_path)
            
            # Extract pixel array
            image_array = dicom_data.pixel_array
            
            # Normalize to 0-255 range
            if image_array.dtype != np.uint8:
                image_array = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX)
                image_array = image_array.astype(np.uint8)
            
            # Convert to 3-channel if grayscale
            if len(image_array.shape) == 2:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            
            return image_array
            
        except Exception as e:
            self.logger.error(f"Error loading DICOM image: {e}")
            return None
    
    def _load_standard_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load standard image formats (PNG, JPEG, etc.)"""
        try:
            # Load image using OpenCV
            image = cv2.imread(image_path)
            
            if image is None:
                # Try with PIL as fallback
                pil_image = Image.open(image_path)
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image
            
        except Exception as e:
            self.logger.error(f"Error loading standard image: {e}")
            return None
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for medical analysis"""
        try:
            # Resize image
            resized = cv2.resize(image, self.image_size)
            
            # Enhance contrast using CLAHE
            if len(resized.shape) == 3:
                # Convert to LAB color space for better contrast enhancement
                lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(resized)
            
            # Normalize pixel values
            normalized = enhanced.astype(np.float32) / 255.0
            
            return normalized
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {e}")
            return image
    
    def _extract_image_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract basic image features and statistics"""
        try:
            features = {}
            
            # Basic statistics
            features['mean_intensity'] = float(np.mean(image))
            features['std_intensity'] = float(np.std(image))
            features['min_intensity'] = float(np.min(image))
            features['max_intensity'] = float(np.max(image))
            
            # Shape information
            features['height'] = image.shape[0]
            features['width'] = image.shape[1]
            features['channels'] = image.shape[2] if len(image.shape) == 3 else 1
            
            # Convert to grayscale for additional analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = (image * 255).astype(np.uint8)
            
            # Texture features using Sobel edge detection
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            features['edge_density'] = float(np.mean(edge_magnitude))
            
            # Histogram features
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            features['histogram_entropy'] = float(-np.sum(hist * np.log2(hist + 1e-10)))
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting image features: {e}")
            return {}
    
    def _classify_modality(self, image: np.ndarray) -> str:
        """Classify the medical imaging modality"""
        if not self.model_loaded:
            return "unknown"
        
        try:
            # Prepare image for CLIP
            pil_image = Image.fromarray((image * 255).astype(np.uint8))
            
            # Define modality candidates
            modality_candidates = [
                "chest x-ray", "abdominal x-ray", "ct scan", "mri scan", 
                "ultrasound", "histopathology", "retinal photograph", "mammogram"
            ]
            
            # Process with CLIP
            inputs = self.processor(
                text=modality_candidates,
                images=pil_image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # Get the most likely modality
            predicted_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][predicted_idx].item()
            
            if confidence > 0.3:  # Threshold for confidence
                return modality_candidates[predicted_idx]
            else:
                return "unknown"
                
        except Exception as e:
            self.logger.error(f"Error classifying modality: {e}")
            return "unknown"
    
    def _analyze_medical_content(self, image: np.ndarray, modality: str) -> Dict[str, Any]:
        """Analyze medical content based on detected modality"""
        if not self.model_loaded:
            return {'error': 'Model not loaded'}
        
        try:
            # Prepare image for CLIP
            pil_image = Image.fromarray((image * 255).astype(np.uint8))
            
            # Select appropriate medical labels based on modality
            if 'x-ray' in modality.lower() or 'xray' in modality.lower():
                labels = self.medical_labels['xray']
            elif 'ct' in modality.lower():
                labels = self.medical_labels['ct']
            elif 'mri' in modality.lower():
                labels = self.medical_labels['mri']
            elif 'histopathology' in modality.lower() or 'pathology' in modality.lower():
                labels = self.medical_labels['pathology']
            else:
                labels = self.medical_labels['xray']  # Default to x-ray labels
            
            # Process with CLIP
            inputs = self.processor(
                text=labels,
                images=pil_image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # Get top predictions
            top_k = min(5, len(labels))
            top_probs, top_indices = torch.topk(probs[0], top_k)
            
            findings = []
            for i in range(top_k):
                idx = top_indices[i].item()
                prob = top_probs[i].item()
                if prob > 0.1:  # Threshold for relevance
                    findings.append({
                        'finding': labels[idx],
                        'confidence': float(prob),
                        'severity': 'high' if prob > 0.7 else 'medium' if prob > 0.4 else 'low'
                    })
            
            # Determine overall assessment
            normal_findings = [f for f in findings if 'normal' in f['finding'].lower()]
            abnormal_findings = [f for f in findings if 'normal' not in f['finding'].lower()]
            
            if normal_findings and normal_findings[0]['confidence'] > 0.5:
                overall_assessment = 'normal'
            elif abnormal_findings:
                overall_assessment = 'abnormal'
            else:
                overall_assessment = 'indeterminate'
            
            return {
                'findings': findings,
                'overall_assessment': overall_assessment,
                'modality_analyzed': modality,
                'total_findings': len(findings)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing medical content: {e}")
            return {'error': str(e)}
    
    def _extract_image_characteristics(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract additional characteristics of the medical image"""
        try:
            characteristics = {}
            
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Image quality metrics
            # 1. Contrast measurement
            characteristics['contrast'] = float(np.std(gray))
            
            # 2. Brightness measurement
            characteristics['brightness'] = float(np.mean(gray))
            
            # 3. Sharpness measurement using Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            characteristics['sharpness'] = float(np.var(laplacian))
            
            # 4. Noise estimation
            noise_std = np.std(cv2.GaussianBlur(gray, (5, 5), 0) - gray)
            characteristics['noise_level'] = float(noise_std)
            
            # 5. Image quality score (composite)
            quality_score = (characteristics['contrast'] * characteristics['sharpness']) / (characteristics['noise_level'] + 1e-10)
            characteristics['quality_score'] = float(quality_score)
            
            # Determine quality category
            if quality_score > 1000:
                characteristics['quality_category'] = 'excellent'
            elif quality_score > 500:
                characteristics['quality_category'] = 'good'
            elif quality_score > 100:
                characteristics['quality_category'] = 'fair'
            else:
                characteristics['quality_category'] = 'poor'
            
            return characteristics
            
        except Exception as e:
            self.logger.error(f"Error extracting image characteristics: {e}")
            return {}
    
    def compare_images(self, image1_path: str, image2_path: str) -> Dict[str, Any]:
        """Compare two medical images for similarities and differences"""
        try:
            # Process both images
            result1 = self.process_medical_image(image1_path)
            result2 = self.process_medical_image(image2_path)
            
            if 'error' in result1 or 'error' in result2:
                return {'error': 'Failed to process one or both images'}
            
            # Compare modalities
            modality_match = result1['modality'] == result2['modality']
            
            # Compare findings
            findings1 = result1.get('medical_findings', {}).get('findings', [])
            findings2 = result2.get('medical_findings', {}).get('findings', [])
            
            common_findings = []
            different_findings = []
            
            for f1 in findings1:
                for f2 in findings2:
                    if f1['finding'] == f2['finding']:
                        common_findings.append({
                            'finding': f1['finding'],
                            'confidence_diff': abs(f1['confidence'] - f2['confidence'])
                        })
            
            # Calculate similarity score
            similarity_score = len(common_findings) / max(len(findings1), len(findings2), 1)
            
            return {
                'modality_match': modality_match,
                'common_findings': common_findings,
                'different_findings': different_findings,
                'similarity_score': similarity_score,
                'comparison_summary': f"Images are {similarity_score:.2%} similar in findings"
            }
            
        except Exception as e:
            self.logger.error(f"Error comparing images: {e}")
            return {'error': str(e)}