# config.py
import os
from typing import Dict, Any
from pathlib import Path

class MedicalAssistantConfig:
    """Configuration settings for the Multimodal Medical Assistant"""
    
    # Base directories
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"
    TEMP_DIR = BASE_DIR / "temp"
    
    # Model configurations
    MODELS = {
        'llama': {
            'base_url': os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
            'model_name': os.getenv('LLAMA_MODEL', 'llama2'),
            'timeout': 120,
            'max_tokens': 2000,
            'temperature': 0.3
        },
        'biomedclip': {
            'model_name': os.getenv('BIOMEDCLIP_MODEL', 'openai/clip-vit-base-patch32'),
            'cache_dir': MODELS_DIR / "biomedclip",
            'device': 'auto'  # 'auto', 'cpu', 'cuda'
        },
        'biobert': {
            'model_name': os.getenv('BIOBERT_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'),
            'cache_dir': MODELS_DIR / "biobert",
            'max_length': 512
        },
        'clinical_bert': {
            'model_name': os.getenv('CLINICAL_BERT_MODEL', 'emilyalsentzer/Bio_ClinicalBERT'),
            'cache_dir': MODELS_DIR / "clinical_bert",
            'max_length': 512
        },
        'medical_ner': {
            'model_name': os.getenv('MEDICAL_NER_MODEL', 'd4data/biomedical-ner-all'),
            'cache_dir': MODELS_DIR / "medical_ner"
        }
    }
    
    # File processing settings
    FILE_PROCESSING = {
        'max_file_size': {
            'image': 50 * 1024 * 1024,  # 50MB
            'document': 100 * 1024 * 1024,  # 100MB
            'dicom': 200 * 1024 * 1024  # 200MB
        },
        'supported_formats': {
            'images': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.dcm'],
            'documents': ['.pdf', '.txt', '.doc', '.docx']
        },
        'image_preprocessing': {
            'resize_dimensions': (224, 224),
            'normalize': True,
            'enhance_contrast': True
        },
        'temp_file_cleanup_hours': 24
    }
    
    # RAG system settings
    RAG_SETTINGS = {
        'chunk_size': 1000,
        'chunk_overlap': 200,
        'max_results': 10,
        'similarity_threshold': 0.7,
        'use_hybrid_search': True,
        'vector_store_type': 'faiss',  
        'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2'
    }
    
    # Privacy and de-identification settings
    PRIVACY_SETTINGS = {
        'auto_deidentify': True,
        'preserve_clinical_context': True,
        'phi_detection_confidence': 0.8,
        'date_shift_range_days': 365,
        'audit_logging': True,
        'cache_replacements': True
    }
    
    # UI/UX settings
    STREAMLIT_CONFIG = {
        'page_title': 'Multimodal Medical Assistant',
        'page_icon': '🏥',
        'layout': 'wide',
        'sidebar_state': 'expanded',
        'theme': 'light',
        'max_upload_size': 200  # MB
    }
    
    # Medical analysis settings
    MEDICAL_ANALYSIS = {
        'clinical_significance_threshold': 0.6,
        'urgency_keywords': [
            'emergency', 'urgent', 'critical', 'severe', 'acute', 
            'stat', 'immediate', 'life-threatening', 'cardiac arrest'
        ],
        'medical_specialties': [
            'cardiology', 'radiology', 'pathology', 'emergency',
            'internal_medicine', 'surgery', 'neurology', 'oncology'
        ],
        'confidence_thresholds': {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
    }
    
    # Logging configuration
    LOGGING = {
        'level': os.getenv('LOG_LEVEL', 'INFO'),
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_rotation': 'midnight',
        'backup_count': 7,
        'max_file_size': '10MB'
    }
    
    # Database settings (for future expansion)
    DATABASE = {
        'type': os.getenv('DB_TYPE', 'sqlite'),
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'name': os.getenv('DB_NAME', 'medical_assistant'),
        'user': os.getenv('DB_USER', ''),
        'password': os.getenv('DB_PASSWORD', ''),
        'connection_timeout': 30
    }
    
    # API settings (for future API deployment)
    API_SETTINGS = {
        'host': os.getenv('API_HOST', '0.0.0.0'),
        'port': int(os.getenv('API_PORT', '8000')),
        'debug': os.getenv('DEBUG', 'False').lower() == 'true',
        'cors_origins': os.getenv('CORS_ORIGINS', '*').split(','),
        'rate_limit': {
            'requests_per_minute': 60,
            'burst_limit': 100
        }
    }
    
    # Security settings
    SECURITY = {
        'secret_key': os.getenv('SECRET_KEY', 'your-secret-key-here'),
        'encryption_algorithm': 'AES-256-GCM',
        'session_timeout_minutes': 60,
        'max_login_attempts': 3,
        'require_https': os.getenv('REQUIRE_HTTPS', 'False').lower() == 'true'
    }
    
    # Performance settings
    PERFORMANCE = {
        'max_concurrent_requests': 10,
        'request_timeout_seconds': 300,
        'memory_limit_mb': 4096,
        'gpu_memory_fraction': 0.8,
        'batch_size': {
            'text_processing': 32,
            'image_processing': 8,
            'embedding_generation': 16
        }
    }
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        directories = [
            cls.DATA_DIR,
            cls.MODELS_DIR,
            cls.LOGS_DIR,
            cls.TEMP_DIR,
            cls.DATA_DIR / "documents",
            cls.DATA_DIR / "embeddings",
            cls.DATA_DIR / "medical_kb",
            cls.LOGS_DIR / "application",
            cls.LOGS_DIR / "audit"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_model_config(cls, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model"""
        return cls.MODELS.get(model_name, {})
    
    @classmethod
    def get_file_config(cls) -> Dict[str, Any]:
        """Get file processing configuration"""
        return cls.FILE_PROCESSING
    
    @classmethod
    def get_rag_config(cls) -> Dict[str, Any]:
        """Get RAG system configuration"""
        return cls.RAG_SETTINGS
    
    @classmethod
    def get_privacy_config(cls) -> Dict[str, Any]:
        """Get privacy and de-identification configuration"""
        return cls.PRIVACY_SETTINGS
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Validate configuration settings"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check if required environment variables are set
        required_env_vars = []
        for var in required_env_vars:
            if not os.getenv(var):
                validation_results['errors'].append(f"Required environment variable {var} not set")
                validation_results['valid'] = False
        
        # Check if directories can be created
        try:
            cls.create_directories()
        except Exception as e:
            validation_results['errors'].append(f"Cannot create directories: {str(e)}")
            validation_results['valid'] = False
        
        # Validate file size limits
        for file_type, size in cls.FILE_PROCESSING['max_file_size'].items():
            if size <= 0:
                validation_results['warnings'].append(f"Invalid file size limit for {file_type}: {size}")
        
        # Check Ollama connection (if configured)
        try:
            import requests
            ollama_url = cls.MODELS['llama']['base_url']
            response = requests.get(f"{ollama_url}/api/tags", timeout=5)
            if response.status_code != 200:
                validation_results['warnings'].append("Ollama service not accessible")
        except Exception:
            validation_results['warnings'].append("Cannot verify Ollama connection")
        
        return validation_results
    
    @classmethod
    def get_environment_info(cls) -> Dict[str, Any]:
        """Get environment information"""
        import platform
        import torch
        
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'base_directory': str(cls.BASE_DIR),
            'config_valid': cls.validate_config()['valid']
        }

# Create directories on import
MedicalAssistantConfig.create_directories()