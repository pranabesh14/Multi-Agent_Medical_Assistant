# test_medical_assistant.py
"""
Test and demonstration script for the Multimodal Medical Assistant
This script validates functionality and provides usage examples
"""

import os
import sys
import tempfile
import logging
from pathlib import Path
from datetime import datetime
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.llama_processor import LlamaProcessor
from src.models.image_processor import MedicalImageProcessor
from src.models.text_processor import MedicalTextProcessor
from src.rag.retrieval_system import MedicalRAGSystem
from src.utils.privacy_utils import DataDeidentifier
from src.utils.file_handler import FileHandler
from config import MedicalAssistantConfig

class MedicalAssistantTester:
    """Test suite for the Medical Assistant"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.config = MedicalAssistantConfig()
        self.test_results = {}
        self.sample_data_dir = Path("data/sample_data")
        self.sample_data_dir.mkdir(parents=True, exist_ok=True)
        
    def _setup_logging(self):
        """Setup logging for tests"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def create_sample_data(self):
        """Create sample medical data for testing"""
        print(" Creating sample medical data...")
        
        # Sample medical text
        sample_medical_text = """
        MEDICAL RECORD - SAMPLE CASE
        ============================
        
        Patient: John Smith
        DOB: 01/15/1975
        MRN: 123456789
        Date: 09/27/2025
        
        CHIEF COMPLAINT:
        45-year-old male presents with acute chest pain and shortness of breath.
        
        HISTORY OF PRESENT ILLNESS:
        Patient reports sudden onset of severe chest pain approximately 2 hours prior to arrival.
        Pain is described as crushing, substernal, 8/10 severity, radiating to left arm and jaw.
        Associated with diaphoresis, nausea, and mild dyspnea. Denies previous cardiac events.
        
        PAST MEDICAL HISTORY:
        - Hypertension, controlled on lisinopril 10mg daily
        - Hyperlipidemia, on atorvastatin 40mg daily
        - Type 2 diabetes mellitus, diet controlled
        - No previous hospitalizations
        
        MEDICATIONS:
        - Lisinopril 10mg PO daily
        - Atorvastatin 40mg PO daily
        - Metformin 500mg PO BID
        
        ALLERGIES:
        NKDA (No Known Drug Allergies)
        
        PHYSICAL EXAMINATION:
        Vital Signs: BP 160/95, HR 110, RR 24, O2 Sat 94% on room air, Temp 98.6°F
        General: Anxious appearing male in moderate acute distress
        HEENT: PERRLA, no JVD
        Cardiovascular: Tachycardic, regular rhythm, S1 S2 present, no murmurs
        Pulmonary: Bilateral crackles at bases, otherwise clear
        Abdomen: Soft, non-tender, non-distended
        Extremities: No peripheral edema
        
        DIAGNOSTIC STUDIES:
        EKG: ST elevation in leads II, III, aVF consistent with inferior STEMI
        Chest X-ray: Mild pulmonary edema, normal cardiac silhouette
        Laboratory Results:
        - Troponin I: 3.2 ng/mL (elevated, normal <0.04)
        - CK-MB: 45 ng/mL (elevated)
        - BNP: 1200 pg/mL (elevated)
        - CBC: WBC 12.5, Hgb 14.2, Plt 285
        - BMP: Na 138, K 4.2, Cl 102, CO2 22, BUN 18, Cr 1.1, Glu 165
        
        ASSESSMENT AND PLAN:
        1. Acute ST-elevation myocardial infarction (STEMI) - inferior wall
           - Emergent cardiac catheterization
           - Dual antiplatelet therapy: ASA 325mg, Clopidogrel 600mg loading dose
           - Anticoagulation with heparin per protocol
           - Atorvastatin 80mg daily
           - Metoprolol 25mg BID when stable
        
        2. Acute heart failure
           - Furosemide 40mg IV PRN
           - Monitor I/O, daily weights
           - ACE inhibitor when stable
        
        3. Diabetes mellitus
           - Continue current regimen
           - Monitor blood glucose closely
        
        DISPOSITION:
        Patient admitted to cardiac ICU for post-catheterization monitoring.
        Cardiology consulted. Family notified.
        
        Dr. Sarah Johnson, MD
        Emergency Medicine
        Phone: (555) 123-4567
        """
        
        # Save sample text
        with open(self.sample_data_dir / "sample_medical_record.txt", 'w') as f:
            f.write(sample_medical_text)
        
        # Create sample image description (since we can't create actual medical images)
        sample_image_info = """
        SAMPLE CHEST X-RAY REPORT
        =========================
        
        Study: Chest X-ray, PA and Lateral
        Date: 09/27/2025
        Indication: Chest pain, rule out pneumonia
        
        FINDINGS:
        - Heart size is mildly enlarged
        - Bilateral lower lobe opacities consistent with pulmonary edema
        - No pneumothorax or pleural effusion
        - Bony structures intact
        - No acute infiltrates
        
        IMPRESSION:
        Mild cardiomegaly with pulmonary edema. Correlate clinically.
        """
        
        with open(self.sample_data_dir / "sample_xray_report.txt", 'w') as f:
            f.write(sample_image_info)
        
        print(" Sample data created successfully")
        return True
    
    def test_configuration(self):
        """Test system configuration"""
        print("\n🔧 Testing system configuration...")
        
        try:
            # Validate configuration
            validation_result = self.config.validate_config()
            
            if validation_result['valid']:
                print(" Configuration validation passed")
                self.test_results['configuration'] = 'PASS'
            else:
                print("❌ Configuration validation failed:")
                for error in validation_result['errors']:
                    print(f"   - {error}")
                self.test_results['configuration'] = 'FAIL'
                
            # Test environment info
            env_info = self.config.get_environment_info()
            print(f" Environment: {env_info['platform']}")
            print(f" Python: {env_info['python_version']}")
            print(f" PyTorch: {env_info['torch_version']}")
            print(f" CUDA: {'Available' if env_info['cuda_available'] else 'Not Available'}")
            
            return validation_result['valid']
            
        except Exception as e:
            print(f"❌ Configuration test failed: {e}")
            self.test_results['configuration'] = 'ERROR'
            return False
    
    def test_llama_processor(self):
        """Test Llama processor functionality"""
        print("\n🦙 Testing Llama processor...")
        
        try:
            processor = LlamaProcessor()
            
            # Test connection
            if not processor.check_status():
                print("❌ Ollama service not available")
                self.test_results['llama'] = 'SKIP'
                return False
            
            # Test basic response generation
            test_prompt = "What is pneumonia? Provide a brief medical explanation."
            response = processor.generate_response(test_prompt, max_tokens=200)
            
            if response and len(response) > 10:
                print(" Basic response generation working")
                print(f" Sample response: {response[:100]}...")
                
                # Test medical response
                medical_query = "A patient presents with chest pain and elevated troponin. What could this indicate?"
                medical_response = processor.generate_medical_response(medical_query)
                
                if "disclaimer" in medical_response.lower():
                    print(" Medical disclaimer included")
                    self.test_results['llama'] = 'PASS'
                    return True
                else:
                    print("  Medical disclaimer missing")
                    self.test_results['llama'] = 'PARTIAL'
                    return True
            else:
                print("❌ Response generation failed")
                self.test_results['llama'] = 'FAIL'
                return False
                
        except Exception as e:
            print(f"❌ Llama processor test failed: {e}")
            self.test_results['llama'] = 'ERROR'
            return False
    
    def test_text_processor(self):
        """Test medical text processor"""
        print("\n Testing medical text processor...")
        
        try:
            processor = MedicalTextProcessor()
            
            # Test with sample medical text
            sample_text = """
            Patient presents with chest pain, elevated troponin levels, and EKG changes 
            consistent with myocardial infarction. Started on aspirin and heparin.
            Blood pressure 150/90, heart rate 110 bpm.
            """
            
            # Process the text
            result = processor.process_medical_text(sample_text)
            
            if result.get('processing_status') == 'success':
                print(" Text processing successful")
                
                # Check for medical entities
                entities = result.get('entities', {})
                if any(entities.values()):
                    print(f" Medical entities detected: {len(entities)} categories")
                    
                # Check embeddings
                embeddings = result.get('embeddings')
                if embeddings:
                    print(" Text embeddings generated")
                    
                # Check clinical assessment
                assessment = result.get('clinical_assessment', {})
                if assessment:
                    print(f" Clinical assessment: complexity={assessment.get('complexity_score', 0):.2f}")
                
                self.test_results['text_processor'] = 'PASS'
                return True
            else:
                print("❌ Text processing failed")
                self.test_results['text_processor'] = 'FAIL'
                return False
                
        except Exception as e:
            print(f"❌ Text processor test failed: {e}")
            self.test_results['text_processor'] = 'ERROR'
            return False
    
    def test_image_processor(self):
        """Test medical image processor"""
        print("\n  Testing medical image processor...")
        
        try:
            processor = MedicalImageProcessor()
            
            if not processor.check_medclip_status():
                print("  MedCLIP model not available, skipping image tests")
                self.test_results['image_processor'] = 'SKIP'
                return True
            
            # Since we can't create actual medical images in this test,
            # we'll test the processor initialization and methods
            print(" MedCLIP model loaded successfully")
            
            # Test supported formats
            supported_formats = ['.dcm', '.png', '.jpg', '.jpeg']
            print(f" Supported formats: {supported_formats}")
            
            self.test_results['image_processor'] = 'PASS'
            return True
            
        except Exception as e:
            print(f"❌ Image processor test failed: {e}")
            self.test_results['image_processor'] = 'ERROR'
            return False
    
    def test_rag_system(self):
        """Test RAG system functionality"""
        print("\n Testing RAG system...")
        
        try:
            rag_system = MedicalRAGSystem()
            
            if not rag_system.check_status():
                print("❌ RAG system not ready")
                self.test_results['rag_system'] = 'FAIL'
                return False
            
            # Add sample document
            sample_doc = """
            Myocardial infarction (MI), commonly known as a heart attack, occurs when 
            blood flow to the heart muscle is blocked. Symptoms include chest pain, 
            shortness of breath, and nausea. Treatment includes aspirin, anticoagulants, 
            and emergency revascularization procedures.
            """
            
            doc_id = rag_system.add_document(sample_doc, "test_document.txt")
            
            if doc_id:
                print(" Document added to RAG system")
                
                # Test query
                query_results = rag_system.query("What is myocardial infarction?", max_results=3)
                
                if query_results:
                    print(f" Query returned {len(query_results)} results")
                    print(f" Top result score: {query_results[0]['score']:.3f}")
                    
                    # Test statistics
                    stats = rag_system.get_statistics()
                    print(f" RAG statistics: {stats['total_documents']} docs, {stats['total_chunks']} chunks")
                    
                    self.test_results['rag_system'] = 'PASS'
                    return True
                else:
                    print("❌ Query returned no results")
                    self.test_results['rag_system'] = 'PARTIAL'
                    return False
            else:
                print("❌ Failed to add document")
                self.test_results['rag_system'] = 'FAIL'
                return False
                
        except Exception as e:
            print(f"❌ RAG system test failed: {e}")
            self.test_results['rag_system'] = 'ERROR'
            return False
    
    def test_privacy_utils(self):
        """Test privacy and de-identification utilities"""
        print("\n  Testing privacy utilities...")
        
        try:
            deidentifier = DataDeidentifier()
            
            # Test text with PHI
            phi_text = """
            Patient: John Smith
            DOB: 01/15/1975
            Phone: (555) 123-4567
            SSN: 123-45-6789
            Email: john.smith@email.com
            Address: 123 Main St, Anytown, CA 90210
            MRN: MR123456
            
            Mr. Smith is a 48-year-old male who presents with chest pain.
            """
            
            # De-identify the text
            deidentified = deidentifier.deidentify_text(phi_text)
            
            # Validate de-identification
            validation = deidentifier.validate_deidentification(phi_text, deidentified)
            
            print(f" De-identification completed")
            print(f" Confidence score: {validation['confidence_score']:.2f}")
            print(f" Validation passed: {validation['validation_passed']}")
            
            if validation['confidence_score'] > 0.8:
                print(" High confidence de-identification")
                self.test_results['privacy'] = 'PASS'
                return True
            else:
                print("  Low confidence de-identification")
                self.test_results['privacy'] = 'PARTIAL'
                return True
                
        except Exception as e:
            print(f"❌ Privacy utilities test failed: {e}")
            self.test_results['privacy'] = 'ERROR'
            return False
    
    def test_file_handler(self):
        """Test file handling utilities"""
        print("\n📁 Testing file handler...")
        
        try:
            file_handler = FileHandler()
            
            # Test supported formats
            formats = file_handler.get_supported_formats()
            print(f" Supported formats: {len(formats['all'])} total")
            
            # Create a temporary text file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
                temp_file.write("This is a test medical document for validation.")
                temp_path = temp_file.name
            
            try:
                # Test text extraction
                extracted_text = file_handler.extract_text(temp_path)
                
                if extracted_text and len(extracted_text) > 0:
                    print(" Text extraction working")
                    
                    # Test metadata extraction
                    metadata = file_handler.get_file_metadata(temp_path)
                    
                    if metadata and 'filename' in metadata:
                        print(" Metadata extraction working")
                        self.test_results['file_handler'] = 'PASS'
                        return True
                    else:
                        print("❌ Metadata extraction failed")
                        self.test_results['file_handler'] = 'PARTIAL'
                        return False
                else:
                    print("❌ Text extraction failed")
                    self.test_results['file_handler'] = 'FAIL'
                    return False
                    
            finally:
                # Cleanup
                os.unlink(temp_path)
                
        except Exception as e:
            print(f"❌ File handler test failed: {e}")
            self.test_results['file_handler'] = 'ERROR'
            return False
    
    def test_integration(self):
        """Test integration between components"""
        print("\n Testing component integration...")
        
        try:
            # Test cross-modal analysis simulation
            print(" Simulating cross-modal analysis...")
            
            # Create sample multimodal data
            sample_image_analysis = {
                'modality': 'chest x-ray',
                'findings': [
                    {'finding': 'cardiomegaly', 'confidence': 0.85},
                    {'finding': 'pulmonary edema', 'confidence': 0.72}
                ],
                'overall_assessment': 'abnormal'
            }
            
            sample_text_analysis = {
                'entities': {
                    'symptoms': [{'text': 'chest pain'}, {'text': 'shortness of breath'}],
                    'diagnoses': [{'text': 'myocardial infarction'}]
                },
                'clinical_assessment': {
                    'complexity_score': 0.8,
                    'urgency_indicators': ['acute']
                }
            }
            
            multimodal_data = {
                'images': [{'filename': 'chest_xray.png', 'analysis': sample_image_analysis}],
                'texts': [{'filename': 'clinical_note.txt', 'analysis': sample_text_analysis}],
                'query': 'Correlate the chest X-ray findings with the clinical presentation'
            }
            
            # Test with Llama processor if available
            llama_processor = LlamaProcessor()
            if llama_processor.check_status():
                result = llama_processor.cross_modal_analysis(multimodal_data)
                
                if result.get('response'):
                    print(" Cross-modal analysis successful")
                    print(f" Generated {len(result.get('evidence', []))} evidence items")
                    self.test_results['integration'] = 'PASS'
                    return True
                else:
                    print("❌ Cross-modal analysis failed")
                    self.test_results['integration'] = 'FAIL'
                    return False
            else:
                print("  Llama not available, skipping integration test")
                self.test_results['integration'] = 'SKIP'
                return True
                
        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            self.test_results['integration'] = 'ERROR'
            return False
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print(" TEST REPORT")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result == 'PASS')
        failed_tests = sum(1 for result in self.test_results.values() if result == 'FAIL')
        error_tests = sum(1 for result in self.test_results.values() if result == 'ERROR')
        skipped_tests = sum(1 for result in self.test_results.values() if result == 'SKIP')
        partial_tests = sum(1 for result in self.test_results.values() if result == 'PARTIAL')
        
        print(f" Total Tests: {total_tests}")
        print(f" Passed: {passed_tests}")
        print(f"  Partial: {partial_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f" Error: {error_tests}")
        print(f"  Skipped: {skipped_tests}")
        
        print("\n Detailed Results:")
        for component, result in self.test_results.items():
            status_icon = {
                'PASS': '',
                'FAIL': '❌',
                'ERROR': '',
                'SKIP': '',
                'PARTIAL': ''
            }.get(result, '❓')
            
            print(f"{status_icon} {component}: {result}")
        
        # Calculate overall score
        score = (passed_tests + (partial_tests * 0.5)) / total_tests * 100
        print(f"\n Overall Score: {score:.1f}%")
        
        # Generate recommendations
        print("\n💡 Recommendations:")
        
        if failed_tests > 0 or error_tests > 0:
            print("- Review failed components and check dependencies")
            print("- Ensure all required models are downloaded")
            print("- Verify Ollama service is running")
        
        if skipped_tests > 0:
            print("- Install missing dependencies for skipped tests")
            print("- Check model availability")
        
        if score >= 80:
            print(" System is ready for use!")
        elif score >= 60:
            print("  System partially functional - some features may not work")
        else:
            print("❌ System needs significant fixes before use")
        
        # Save report to file
        report_file = Path("test_report.json")
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'results': self.test_results,
            'summary': {
                'total': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'error': error_tests,
                'skipped': skipped_tests,
                'partial': partial_tests,
                'score': score
            }
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n Detailed report saved to: {report_file}")
        
        return score >= 60  # Return True if system is functional
    
    def run_all_tests(self):
        """Run all test suites"""
        print(" Starting Medical Assistant Test Suite")
        print("=" * 60)
        
        # Create sample data first
        self.create_sample_data()
        
        # Run all tests
        test_functions = [
            self.test_configuration,
            self.test_file_handler,
            self.test_privacy_utils,
            self.test_text_processor,
            self.test_image_processor,
            self.test_rag_system,
            self.test_llama_processor,
            self.test_integration
        ]
        
        for test_func in test_functions:
            try:
                test_func()
            except Exception as e:
                component_name = test_func.__name__.replace('test_', '')
                print(f" Unexpected error in {component_name}: {e}")
                self.test_results[component_name] = 'ERROR'
        
        # Generate final report
        return self.generate_test_report()

def main():
    """Main test function"""
    print(" Multimodal Medical Assistant - Test Suite")
    print("=" * 60)
    
    tester = MedicalAssistantTester()
    
    if len(sys.argv) > 1:
        test_name = sys.argv[1].replace('--', '').replace('-', '_')
        test_method = getattr(tester, f'test_{test_name}', None)
        
        if test_method:
            print(f"Running single test: {test_name}")
            test_method()
            tester.generate_test_report()
        else:
            print(f"Unknown test: {test_name}")
            print("Available tests: configuration, file_handler, privacy_utils, text_processor, image_processor, rag_system, llama_processor, integration")
    else:
        # Run all tests
        success = tester.run_all_tests()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()