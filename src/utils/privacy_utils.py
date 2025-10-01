# src/utils/privacy_utils.py
import re
import hashlib
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import random
import string

class DataDeidentifier:
    """Handles de-identification and privacy protection for medical data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize de-identification patterns
        self._initialize_patterns()
        
        # Cache for consistent replacements
        self.replacement_cache = {}
        
        # HIPAA Safe Harbor identifiers to remove/replace
        self.phi_categories = [
            'names', 'addresses', 'dates', 'phone_numbers', 'fax_numbers',
            'email_addresses', 'ssn', 'mrn', 'account_numbers', 'certificate_numbers',
            'vehicle_identifiers', 'device_identifiers', 'web_urls', 'ip_addresses',
            'biometric_identifiers', 'photos', 'other_unique_identifiers'
        ]
    
    def _initialize_patterns(self):
        """Initialize regex patterns for PHI detection"""
        
        # Name patterns (basic - would need more sophisticated NER for production)
        self.name_patterns = [
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # First Last
            r'\b[A-Z][a-z]+,\s*[A-Z][a-z]+\b',  # Last, First
            r'\bDr\.\s+[A-Z][a-z]+\b',  # Dr. Name
            r'\bMr\.\s+[A-Z][a-z]+\b',  # Mr. Name
            r'\bMs\.\s+[A-Z][a-z]+\b',  # Ms. Name
            r'\bMrs\.\s+[A-Z][a-z]+\b'  # Mrs. Name
        ]
        
        # Address patterns
        self.address_patterns = [
            r'\b\d+\s+[A-Z][a-z]+\s+(St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Ct|Court)\b',
            r'\b[A-Z][a-z]+,\s*[A-Z]{2}\s+\d{5}(-\d{4})?\b',  # City, State ZIP
            r'\b\d{5}(-\d{4})?\b'  # ZIP codes
        ]
        
        # Date patterns (various formats)
        self.date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY or MM-DD-YYYY
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',    # YYYY/MM/DD or YYYY-MM-DD
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b',  # Month DD, YYYY
            r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b'  # DD Month YYYY
        ]
        
        # Phone number patterns
        self.phone_patterns = [
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # XXX-XXX-XXXX or XXX.XXX.XXXX or XXX XXX XXXX
            r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}\b',    # (XXX) XXX-XXXX
            r'\b1[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'  # 1-XXX-XXX-XXXX
        ]
        
        # Email patterns
        self.email_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        ]
        
        # Social Security Number patterns
        self.ssn_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # XXX-XX-XXXX
            r'\b\d{9}\b'  # XXXXXXXXX (9 consecutive digits)
        ]
        
        # Medical Record Number patterns
        self.mrn_patterns = [
            r'\bMRN:?\s*\d+\b',
            r'\bMedical Record Number:?\s*\d+\b',
            r'\bPatient ID:?\s*\d+\b',
            r'\bAccount:?\s*\d+\b'
        ]
        
        # URL patterns
        self.url_patterns = [
            r'https?://[^\s]+',
            r'www\.[^\s]+',
            r'ftp://[^\s]+'
        ]
        
        # IP Address patterns
        self.ip_patterns = [
            r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        ]
        
        # Age patterns (ages > 89 need special handling per HIPAA)
        self.age_patterns = [
            r'\b\d{2,3}\s*(?:year|yr|y)s?\s*old\b',
            r'\bage\s*:?\s*\d{2,3}\b',
            r'\b\d{2,3}\s*y/?o\b'
        ]
    
    def deidentify_text(self, text: str, preserve_clinical_context: bool = True) -> str:
        """
        De-identify medical text while preserving clinical context
        
        Args:
            text: Input medical text
            preserve_clinical_context: Whether to preserve clinical meaning while de-identifying
        """
        try:
            deidentified_text = text
            
            # Track what was replaced for audit log
            replacements = {
                'names': 0,
                'dates': 0,
                'addresses': 0,
                'phone_numbers': 0,
                'emails': 0,
                'ssn': 0,
                'mrn': 0,
                'urls': 0,
                'ip_addresses': 0,
                'ages': 0
            }
            
            # Replace Names
            for pattern in self.name_patterns:
                deidentified_text, count = self._replace_with_placeholder(
                    deidentified_text, pattern, 'NAME', preserve_clinical_context
                )
                replacements['names'] += count
            
            # Replace Dates (with date shifting if preserving context)
            for pattern in self.date_patterns:
                if preserve_clinical_context:
                    deidentified_text, count = self._replace_dates_with_shift(deidentified_text, pattern)
                else:
                    deidentified_text, count = self._replace_with_placeholder(
                        deidentified_text, pattern, 'DATE'
                    )
                replacements['dates'] += count
            
            # Replace Addresses
            for pattern in self.address_patterns:
                deidentified_text, count = self._replace_with_placeholder(
                    deidentified_text, pattern, 'ADDRESS'
                )
                replacements['addresses'] += count
            
            # Replace Phone Numbers
            for pattern in self.phone_patterns:
                deidentified_text, count = self._replace_with_placeholder(
                    deidentified_text, pattern, 'PHONE'
                )
                replacements['phone_numbers'] += count
            
            # Replace Email Addresses
            for pattern in self.email_patterns:
                deidentified_text, count = self._replace_with_placeholder(
                    deidentified_text, pattern, 'EMAIL'
                )
                replacements['emails'] += count
            
            # Replace SSNs
            for pattern in self.ssn_patterns:
                deidentified_text, count = self._replace_with_placeholder(
                    deidentified_text, pattern, 'SSN'
                )
                replacements['ssn'] += count
            
            # Replace MRNs
            for pattern in self.mrn_patterns:
                deidentified_text, count = self._replace_with_placeholder(
                    deidentified_text, pattern, 'MRN'
                )
                replacements['mrn'] += count
            
            # Replace URLs
            for pattern in self.url_patterns:
                deidentified_text, count = self._replace_with_placeholder(
                    deidentified_text, pattern, 'URL'
                )
                replacements['urls'] += count
            
            # Replace IP Addresses
            for pattern in self.ip_patterns:
                deidentified_text, count = self._replace_with_placeholder(
                    deidentified_text, pattern, 'IPADDRESS'
                )
                replacements['ip_addresses'] += count
            
            # Handle Ages > 89 (HIPAA requirement)
            deidentified_text, age_count = self._handle_ages(deidentified_text)
            replacements['ages'] += age_count
            
            # Log de-identification summary
            total_replacements = sum(replacements.values())
            if total_replacements > 0:
                self.logger.info(f"De-identified {total_replacements} PHI elements: {replacements}")
            
            return deidentified_text
            
        except Exception as e:
            self.logger.error(f"Error during de-identification: {e}")
            return text  # Return original text if de-identification fails
    
    def _replace_with_placeholder(self, text: str, pattern: str, placeholder_type: str, 
                                 preserve_context: bool = False) -> Tuple[str, int]:
        """Replace matched patterns with placeholders"""
        count = 0
        
        def replacement_func(match):
            nonlocal count
            count += 1
            
            matched_text = match.group()
            
            # Use cache for consistent replacements
            if matched_text in self.replacement_cache:
                return self.replacement_cache[matched_text]
            
            if preserve_context and placeholder_type in ['NAME']:
                # Generate realistic but fake replacement
                replacement = self._generate_realistic_replacement(matched_text, placeholder_type)
            else:
                # Use generic placeholder
                replacement = f"[{placeholder_type}_{count}]"
            
            self.replacement_cache[matched_text] = replacement
            return replacement
        
        modified_text = re.sub(pattern, replacement_func, text, flags=re.IGNORECASE)
        return modified_text, count
    
    def _replace_dates_with_shift(self, text: str, pattern: str) -> Tuple[str, int]:
        """Replace dates with shifted dates to preserve temporal relationships"""
        count = 0
        
        # Generate a consistent date shift for this session
        if not hasattr(self, 'date_shift_days'):
            self.date_shift_days = random.randint(-365, 365)  # Shift by up to 1 year
        
        def date_replacement_func(match):
            nonlocal count
            count += 1
            
            date_str = match.group()
            
            # Try to parse and shift the date
            try:
                # Simple date parsing (would need more robust parsing for production)
                if '/' in date_str:
                    parts = date_str.split('/')
                    if len(parts) == 3:
                        month, day, year = parts
                        if len(year) == 2:
                            year = '20' + year if int(year) < 50 else '19' + year
                        
                        original_date = datetime(int(year), int(month), int(day))
                        shifted_date = original_date + timedelta(days=self.date_shift_days)
                        
                        return shifted_date.strftime('%m/%d/%Y')
                
                # If parsing fails, use placeholder
                return f"[DATE_{count}]"
                
            except:
                return f"[DATE_{count}]"
        
        modified_text = re.sub(pattern, date_replacement_func, text)
        return modified_text, count
    
    def _handle_ages(self, text: str) -> Tuple[str, int]:
        """Handle ages according to HIPAA Safe Harbor (ages > 89 must be aggregated)"""
        count = 0
        
        def age_replacement_func(match):
            nonlocal count
            
            age_text = match.group()
            
            # Extract numeric age
            age_nums = re.findall(r'\d+', age_text)
            if age_nums:
                age = int(age_nums[0])
                if age > 89:
                    count += 1
                    # Replace with "over 89" as per HIPAA
                    return re.sub(r'\d+', 'over 89', age_text)
            
            return age_text
        
        modified_text = re.sub(
            r'\b\d{2,3}\s*(?:year|yr|y)s?\s*old\b|\bage\s*:?\s*\d{2,3}\b|\b\d{2,3}\s*y/?o\b',
            age_replacement_func,
            text,
            flags=re.IGNORECASE
        )
        
        return modified_text, count
    
    def _generate_realistic_replacement(self, original: str, placeholder_type: str) -> str:
        """Generate realistic but fake replacements to preserve context"""
        
        if placeholder_type == 'NAME':
            # Generate fake names that maintain similar characteristics
            fake_names = [
                'John Smith', 'Jane Doe', 'Michael Johnson', 'Sarah Wilson',
                'David Brown', 'Lisa Davis', 'Robert Miller', 'Maria Garcia',
                'James Anderson', 'Jennifer Taylor', 'William Moore', 'Patricia Jackson'
            ]
            
            # Use hash to ensure consistency
            hash_value = int(hashlib.md5(original.encode()).hexdigest(), 16)
            return fake_names[hash_value % len(fake_names)]
        
        elif placeholder_type == 'PHONE':
            # Generate fake phone number with same format
            return '555-' + ''.join([str(random.randint(0, 9)) for _ in range(7)])
        
        elif placeholder_type == 'EMAIL':
            domains = ['example.com', 'test.org', 'sample.net']
            username = 'user' + str(random.randint(1000, 9999))
            domain = random.choice(domains)
            return f"{username}@{domain}"
        
        return f"[{placeholder_type}]"
    
    def validate_deidentification(self, original_text: str, deidentified_text: str) -> Dict[str, Any]:
        """Validate that de-identification was successful"""
        try:
            validation_results = {
                'phi_detected': False,
                'remaining_phi': [],
                'confidence_score': 0.0,
                'validation_passed': True
            }
            
            # Check for remaining PHI patterns
            all_patterns = (
                self.name_patterns + self.address_patterns + self.date_patterns +
                self.phone_patterns + self.email_patterns + self.ssn_patterns +
                self.mrn_patterns + self.url_patterns + self.ip_patterns
            )
            
            remaining_phi = []
            for pattern in all_patterns:
                matches = re.findall(pattern, deidentified_text, re.IGNORECASE)
                if matches:
                    remaining_phi.extend(matches)
            
            # Check for placeholder consistency
            placeholders = re.findall(r'\[(\w+)_\d+\]', deidentified_text)
            placeholder_counts = {}
            for placeholder in placeholders:
                placeholder_counts[placeholder] = placeholder_counts.get(placeholder, 0) + 1
            
            # Calculate confidence score
            original_phi_count = len(re.findall('|'.join(all_patterns), original_text, re.IGNORECASE))
            remaining_phi_count = len(remaining_phi)
            
            if original_phi_count > 0:
                confidence_score = max(0, (original_phi_count - remaining_phi_count) / original_phi_count)
            else:
                confidence_score = 1.0
            
            validation_results.update({
                'phi_detected': len(remaining_phi) > 0,
                'remaining_phi': remaining_phi,
                'confidence_score': confidence_score,
                'validation_passed': confidence_score >= 0.95,
                'placeholder_counts': placeholder_counts
            })
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error validating de-identification: {e}")
            return {
                'phi_detected': True,
                'validation_passed': False,
                'error': str(e)
            }
    
    def generate_audit_log(self, original_text: str, deidentified_text: str) -> Dict[str, Any]:
        """Generate audit log for de-identification process"""
        try:
            audit_log = {
                'timestamp': datetime.now().isoformat(),
                'original_length': len(original_text),
                'deidentified_length': len(deidentified_text),
                'reduction_ratio': (len(original_text) - len(deidentified_text)) / len(original_text),
                'phi_categories_processed': [],
                'validation_results': self.validate_deidentification(original_text, deidentified_text)
            }
            
            # Count PHI categories found and processed
            for category, patterns in {
                'names': self.name_patterns,
                'dates': self.date_patterns,
                'addresses': self.address_patterns,
                'phones': self.phone_patterns,
                'emails': self.email_patterns,
                'ssn': self.ssn_patterns,
                'mrn': self.mrn_patterns,
                'urls': self.url_patterns,
                'ips': self.ip_patterns
            }.items():
                
                found_count = 0
                for pattern in patterns:
                    found_count += len(re.findall(pattern, original_text, re.IGNORECASE))
                
                if found_count > 0:
                    audit_log['phi_categories_processed'].append({
                        'category': category,
                        'count': found_count
                    })
            
            return audit_log
            
        except Exception as e:
            self.logger.error(f"Error generating audit log: {e}")
            return {'error': str(e)}
    
    def create_reidentification_map(self, original_text: str, deidentified_text: str) -> Dict[str, str]:
        """Create a secure mapping for potential re-identification (for authorized use only)"""
        try:
            # This should only be used in controlled environments with proper security
            reidentification_map = {}
            
            # Create mapping of placeholders back to original values
            for original, replacement in self.replacement_cache.items():
                if replacement.startswith('[') and replacement.endswith(']'):
                    # Hash the original value for security
                    hashed_original = hashlib.sha256(original.encode()).hexdigest()
                    reidentification_map[replacement] = hashed_original
            
            return reidentification_map
            
        except Exception as e:
            self.logger.error(f"Error creating re-identification map: {e}")
            return {}
    
    def clear_cache(self):
        """Clear the replacement cache"""
        self.replacement_cache.clear()
        if hasattr(self, 'date_shift_days'):
            delattr(self, 'date_shift_days')