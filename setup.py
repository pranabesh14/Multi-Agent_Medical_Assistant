# setup.py
"""
Setup script for the Multimodal Medical Assistant
This script installs dependencies and configures the environment
"""

import subprocess
import sys
import os
import platform
from pathlib import Path
import requests
import zipfile
import shutil

class MedicalAssistantSetup:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.python_executable = sys.executable
        self.platform = platform.system().lower()
        
    def print_header(self):
        """Print setup header"""
        print("=" * 60)
        print(" MULTIMODAL MEDICAL ASSISTANT SETUP")
        print("=" * 60)
        print(f"Platform: {platform.platform()}")
        print(f"Python: {sys.version}")
        print(f"Base Directory: {self.base_dir}")
        print("=" * 60)
    
    def check_python_version(self):
        """Check if Python version is compatible"""
        print("\n📋 Checking Python version...")
        
        version = sys.version_info
        if version.major != 3 or version.minor < 8:
            print(" ERROR: Python 3.8 or higher is required")
            print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
            return False
        
        print(f" Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    
    def install_base_requirements(self):
        """Install base Python requirements"""
        print("\n📦 Installing Python dependencies...")
        
        requirements_file = self.base_dir / "requirements.txt"
        if not requirements_file.exists():
            print(" ERROR: requirements.txt not found")
            return False
        
        try:
            # Update pip first
            subprocess.check_call([
                self.python_executable, "-m", "pip", "install", "--upgrade", "pip"
            ])
            
            # Install requirements
            subprocess.check_call([
                self.python_executable, "-m", "pip", "install", "-r", str(requirements_file)
            ])
            
            print(" Python dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f" ERROR installing dependencies: {e}")
            return False
    
    def setup_ollama(self):
        """Setup Ollama for Llama model"""
        print("\n🦙 Setting up Ollama for Llama models...")
        
        # Check if Ollama is already installed
        try:
            result = subprocess.run(["ollama", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(" Ollama is already installed")
                return self.pull_llama_model()
        except FileNotFoundError:
            pass
        
        print("📥 Ollama not found. Installation required...")
        
        if self.platform == "windows":
            print("🪟 Windows detected - Please download Ollama from: https://ollama.ai/download")
            print("   After installation, run this setup script again.")
            return False
        elif self.platform == "darwin":  # macOS
            print("🍎 macOS detected - Installing Ollama...")
            try:
                subprocess.check_call(["brew", "install", "ollama"])
                print(" Ollama installed via Homebrew")
                return self.pull_llama_model()
            except (subprocess.CalledProcessError, FileNotFoundError):
                print(" Failed to install via Homebrew")
                print("   Please install manually from: https://ollama.ai/download")
                return False
        else:  # Linux
            print("🐧 Linux detected - Installing Ollama...")
            try:
                # Download and install Ollama
                install_script = "curl -fsSL https://ollama.ai/install.sh | sh"
                subprocess.check_call(install_script, shell=True)
                print(" Ollama installed successfully")
                return self.pull_llama_model()
            except subprocess.CalledProcessError:
                print(" Failed to install Ollama")
                print("   Please install manually from: https://ollama.ai/download")
                return False
    
    def pull_llama_model(self):
        """Pull the Llama model using Ollama"""
        print("📥 Downloading Llama2 model (this may take a while)...")
        
        try:
            # Start Ollama service
            subprocess.Popen(["ollama", "serve"], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            
            # Wait a moment for service to start
            import time
            time.sleep(3)
            
            # Pull the model
            subprocess.check_call(["ollama", "pull", "llama2"])
            print(" Llama2 model downloaded successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f" ERROR downloading Llama model: {e}")
            return False
    
    def download_spacy_models(self):
        """Download required spaCy models"""
        print("\n🧠 Downloading spaCy models...")
        
        models = [
            "en_core_web_sm",
            "en_core_sci_sm"  # Scientific model
        ]
        
        for model in models:
            try:
                print(f"📥 Downloading {model}...")
                subprocess.check_call([
                    self.python_executable, "-m", "spacy", "download", model
                ])
                print(f" {model} downloaded successfully")
            except subprocess.CalledProcessError:
                print(f"  Warning: Could not download {model}")
                if model == "en_core_sci_sm":
                    print("   Scientific model is optional but recommended")
                    print("   Install manually: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_sm-0.5.0.tar.gz")
    
    def setup_medical_models(self):
        """Setup medical AI models"""
        print("\n Setting up medical AI models...")
        
        # This will download models on first use
        models_to_verify = [
            "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            "dmis-lab/biobert-base-cased-v1.1",
            "emilyalsentzer/Bio_ClinicalBERT",
            "d4data/biomedical-ner-all"
        ]
        
        print(" Medical models will be downloaded automatically on first use:")
        for model in models_to_verify:
            print(f"   - {model}")
        
        print(" Medical models configured")
        return True
    
    def create_directories(self):
        """Create necessary directories"""
        print("\n📁 Creating directory structure...")
        
        directories = [
            "data",
            "data/documents",
            "data/embeddings", 
            "data/medical_kb",
            "models",
            "logs",
            "logs/application",
            "logs/audit",
            "temp"
        ]
        
        for dir_name in directories:
            dir_path = self.base_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f" Created: {dir_path}")
        
        return True
    
    def create_sample_data(self):
        """Create sample medical data for testing"""
        print("\n📄 Creating sample data...")
        
        sample_text = """
        MEDICAL RECORD - SAMPLE DATA FOR TESTING
        =======================================
        
        Patient: [SAMPLE_PATIENT]
        DOB: [DATE]
        MRN: [MRN_12345]
        
        CHIEF COMPLAINT:
        Patient presents with chest pain and shortness of breath.
        
        HISTORY OF PRESENT ILLNESS:
        45-year-old male presents to the emergency department with acute onset chest pain 
        that started 2 hours ago. Pain is described as crushing, substernal, radiating 
        to left arm. Associated with diaphoresis and nausea. No previous cardiac history.
        
        PHYSICAL EXAMINATION:
        Vital Signs: BP 150/90, HR 110, RR 22, O2 Sat 95% on room air
        General: Anxious appearing male in moderate distress
        Cardiovascular: Tachycardic, regular rhythm, no murmurs
        Pulmonary: Clear to auscultation bilaterally
        
        DIAGNOSTIC TESTS:
        EKG: ST elevation in leads II, III, aVF
        Chest X-ray: Clear lung fields, normal cardiac silhouette
        Troponin: Elevated at 2.5 ng/mL (normal <0.04)
        
        ASSESSMENT AND PLAN:
        Acute ST-elevation myocardial infarction (STEMI)
        - Activate cardiac catheterization lab
        - Administer aspirin, clopidogrel, atorvastatin
        - Heparin per protocol
        - Serial cardiac enzymes
        
        This is sample data for testing the medical assistant.
        """
        
        sample_file = self.base_dir / "data" / "sample_medical_record.txt"
        with open(sample_file, 'w') as f:
            f.write(sample_text)
        
        print(f" Created sample medical record: {sample_file}")
        return True
    
    def create_environment_file(self):
        """Create .env file with default settings"""
        print("\n  Creating environment configuration...")
        
        env_content = """# Multimodal Medical Assistant Environment Configuration

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
LLAMA_MODEL=llama2

# Model Configuration
BIOMEDCLIP_MODEL=microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
BIOBERT_MODEL=dmis-lab/biobert-base-cased-v1.1
CLINICAL_BERT_MODEL=emilyalsentzer/Bio_ClinicalBERT
MEDICAL_NER_MODEL=d4data/biomedical-ner-all

# Logging
LOG_LEVEL=INFO

# Security (Change these in production!)
SECRET_KEY=your-secret-key-change-in-production
REQUIRE_HTTPS=False

# Development Settings
DEBUG=True
"""
        
        env_file = self.base_dir / ".env"
        if not env_file.exists():
            with open(env_file, 'w') as f:
                f.write(env_content)
            print(f" Created environment file: {env_file}")
        else:
            print("  .env file already exists, skipping...")
        
        return True
    
    def verify_installation(self):
        """Verify that everything is installed correctly"""
        print("\n🔍 Verifying installation...")
        
        # Check Python packages
        required_packages = [
            'streamlit', 'torch', 'transformers', 'sentence_transformers',
            'langchain', 'faiss', 'opencv-python', 'pydicom', 'spacy'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                print(f" {package}")
            except ImportError:
                missing_packages.append(package)
                print(f" {package}")
        
        # Check Ollama
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                print(" Ollama service")
            else:
                print(" Ollama service (not running)")
        except:
            print(" Ollama service (not accessible)")
        
        # Check spaCy models
        try:
            import spacy
            try:
                spacy.load("en_core_web_sm")
                print(" spaCy English model")
            except OSError:
                print(" spaCy English model")
        except ImportError:
            print(" spaCy not installed")
        
        if missing_packages:
            print(f"\n Missing packages: {', '.join(missing_packages)}")
            return False
        
        print("\n🎉 Installation verification complete!")
        return True
    
    def print_next_steps(self):
        """Print next steps for the user"""
        print("\n" + "=" * 60)
        print("🎉 SETUP COMPLETE!")
        print("=" * 60)
        print("\nNEXT STEPS:")
        print("1. Ensure Ollama is running: 'ollama serve'")
        print("2. Start the medical assistant: 'streamlit run main.py'")
        print("3. Open your browser to: http://localhost:8501")
        print("\n DOCUMENTATION:")
        print("- Upload medical images (X-rays, CT scans, MRIs)")
        print("- Upload medical documents (PDFs, text files)")
        print("- Ask questions about medical data")
        print("- Perform cross-modal analysis")
        print("\n  IMPORTANT DISCLAIMERS:")
        print("- This is for educational/research purposes only")
        print("- Not for clinical decision making")
        print("- Always consult healthcare professionals")
        print("- Ensure patient data is properly de-identified")
        print("\n TROUBLESHOOTING:")
        print("- Check logs in the 'logs' directory")
        print("- Ensure all models are downloaded")
        print("- Verify Ollama is running on port 11434")
        print("=" * 60)
    
    def run_setup(self):
        """Run the complete setup process"""
        self.print_header()
        
        steps = [
            ("Checking Python version", self.check_python_version),
            ("Creating directories", self.create_directories),
            ("Installing requirements", self.install_base_requirements),
            ("Setting up Ollama", self.setup_ollama),
            ("Downloading spaCy models", self.download_spacy_models),
            ("Setting up medical models", self.setup_medical_models),
            ("Creating sample data", self.create_sample_data),
            ("Creating environment file", self.create_environment_file),
            ("Verifying installation", self.verify_installation)
        ]
        
        failed_steps = []
        
        for step_name, step_function in steps:
            try:
                if not step_function():
                    failed_steps.append(step_name)
            except Exception as e:
                print(f" ERROR in {step_name}: {e}")
                failed_steps.append(step_name)
        
        if failed_steps:
            print(f"\n Setup completed with warnings. Failed steps: {', '.join(failed_steps)}")
            print("   You may need to complete these steps manually.")
        
        self.print_next_steps()
        return len(failed_steps) == 0

def main():
    """Main setup function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--verify-only":
        # Only run verification
        setup = MedicalAssistantSetup()
        setup.verify_installation()
    else:
        # Run full setup
        setup = MedicalAssistantSetup()
        success = setup.run_setup()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()