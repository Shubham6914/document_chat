"""
Extraction Service Module
Improved field extraction with better accuracy for legal documents.
"""

import json
from typing import Dict, List, Optional, Any
from openai import OpenAI
from src.utils.logger import app_logger as logger

from ..utils.exceptions import ExtractionError
from config.settings import settings
from config.prompts import prompts


class ExtractionService:
    """
    Enhanced extraction service with improved accuracy for legal documents.
    Addresses critical errors in tenant identification, security deposit, and lease terms.
    """
    
    def __init__(self):
        """Initialize the extraction service with Requesty API."""
        self.client = OpenAI(
            api_key=settings.requesty_api_key,
            base_url=settings.requesty_base_url
        )
        self.model = settings.model_name
        self.temperature = settings.extraction_temperature
        self.max_retries = settings.extraction_max_retries
        
        logger.info("ExtractionService initialized with enhanced accuracy features")
    
    def extract_fields(
        self,
        document_text: str,
        field_schema: Dict[str, Any],
        document_type: str = "lease_agreement"
    ) -> Dict[str, Any]:
        """
        Extract structured fields from document text with improved accuracy.
        
        Args:
            document_text: Full text of the document
            field_schema: Schema defining fields to extract
            document_type: Type of document
            
        Returns:
            Dictionary with extracted fields, confidence scores, and sources
            
        Raises:
            ExtractionError: If extraction fails
        """
        logger.info(f"Extracting fields from {document_type} with enhanced accuracy")
        
        try:
            # Build extraction prompt using centralized prompts
            prompt = self._build_extraction_prompt(
                document_text,
                field_schema,
                document_type
            )
            
            # Call LLM with retries
            for attempt in range(self.max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": prompts.EXTRACTION_SYSTEM_PROMPT
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        temperature=self.temperature,
                        max_tokens=settings.max_tokens,
                        response_format={"type": "json_object"}
                    )
                    
                    # Parse response
                    result = json.loads(response.choices[0].message.content)
                    
                    # Post-process and validate
                    validated_result = self._post_process_extraction(
                        result,
                        field_schema,
                        document_text
                    )
                    
                    logger.info(f"Successfully extracted {len(validated_result.get('fields', {}))} fields")
                    return validated_result
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Attempt {attempt + 1}: JSON decode error: {str(e)}")
                    if attempt == self.max_retries - 1:
                        raise ExtractionError(f"Failed to parse LLM response: {str(e)}")
                        
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1}: Extraction error: {str(e)}")
                    if attempt == self.max_retries - 1:
                        raise ExtractionError(f"Extraction failed after {self.max_retries} attempts: {str(e)}")
            
        except Exception as e:
            logger.error(f"Field extraction failed: {str(e)}")
            raise ExtractionError(f"Failed to extract fields: {str(e)}")
    
    def _build_extraction_prompt(
        self,
        document_text: str,
        field_schema: Dict[str, Any],
        document_type: str
    ) -> str:
        """
        Build extraction prompt using centralized templates.
        
        Args:
            document_text: Document text
            field_schema: Field schema
            document_type: Document type
            
        Returns:
            Formatted prompt string
        """
        # Truncate document if too long (keep first and last parts)
        max_chars = 15000
        if len(document_text) > max_chars:
            half = max_chars // 2
            document_text = document_text[:half] + "\n\n[... middle section truncated ...]\n\n" + document_text[-half:]
        
        fields_description = self._format_field_schema(field_schema)
        
        return prompts.format_extraction_prompt(
            document_text=document_text,
            fields_description=fields_description,
            document_type=document_type
        )
    
    def _format_field_schema(self, field_schema: Dict[str, Any]) -> str:
        """
        Format field schema for prompt with extraction hints.
        
        Args:
            field_schema: Field schema dictionary
            
        Returns:
            Formatted schema string with hints
        """
        if "lease_summary_template" in field_schema:
            # Use the new comprehensive template
            required_fields = field_schema["lease_summary_template"].get("required_fields", {})
            optional_fields = field_schema["lease_summary_template"].get("optional_fields", {})
            
            lines = ["REQUIRED FIELDS:"]
            for field_name, field_info in required_fields.items():
                description = field_info.get("description", "")
                field_type = field_info.get("type", "string")
                hints = field_info.get("extraction_hints", [])
                
                lines.append(f"\n- **{field_name}** ({field_type}) [REQUIRED]")
                lines.append(f"  Description: {description}")
                if hints:
                    lines.append(f"  Keywords: {', '.join(hints)}")
            
            lines.append("\n\nOPTIONAL FIELDS:")
            for field_name, field_info in optional_fields.items():
                description = field_info.get("description", "")
                field_type = field_info.get("type", "string")
                
                lines.append(f"\n- **{field_name}** ({field_type}) [OPTIONAL]")
                lines.append(f"  Description: {description}")
            
            return "\n".join(lines)
        
        elif "fields" in field_schema:
            # Legacy format
            lines = []
            for field_name, field_info in field_schema["fields"].items():
                description = field_info.get("description", "")
                field_type = field_info.get("type", "string")
                required = field_info.get("required", False)
                
                req_marker = "[REQUIRED]" if required else "[OPTIONAL]"
                lines.append(f"- {field_name} ({field_type}) {req_marker}: {description}")
            
            return "\n".join(lines)
        
        return "Extract all relevant fields from the document."
    
    def _post_process_extraction(
        self,
        result: Dict[str, Any],
        field_schema: Dict[str, Any],
        document_text: str
    ) -> Dict[str, Any]:
        """
        Post-process and validate extracted fields with critical error fixes.
        
        Args:
            result: Raw extraction result
            field_schema: Field schema
            document_text: Original document text
            
        Returns:
            Validated and corrected result
        """
        if "fields" not in result:
            raise ExtractionError("Invalid extraction result: missing 'fields' key")
        
        validated = {
            "fields": {},
            "extraction_notes": result.get("extraction_notes", ""),
            "warnings": result.get("warnings", []),
            "validation_warnings": []
        }
        
        # Apply critical fixes
        fields = result["fields"]
        
        # FIX 1: Tenant identification - distinguish legal entity from trade name
        if "tenant" in fields:
            tenant_value = fields["tenant"].get("value", "")
            if tenant_value and tenant_value != "Not specified":
                # Check if this looks like a trade name
                if self._is_likely_trade_name(tenant_value, document_text):
                    validated["validation_warnings"].append(
                        f"WARNING: '{tenant_value}' may be a trade name, not the legal tenant. "
                        "Please verify the legal entity name."
                    )
                    # Try to find the actual legal tenant
                    legal_tenant = self._find_legal_tenant(document_text)
                    if legal_tenant:
                        fields["tenant"]["value"] = legal_tenant
                        fields["tenant"]["confidence"] = max(0.7, fields["tenant"].get("confidence", 0.5))
                        validated["validation_warnings"].append(
                            f"Corrected tenant to legal entity: {legal_tenant}"
                        )
        
        # FIX 2: Security deposit - ensure it's explicitly stated, not calculated
        if "security_deposit" in fields:
            deposit_value = fields["security_deposit"].get("value", "")
            source = fields["security_deposit"].get("source", "")
            
            if deposit_value and deposit_value != "Not specified":
                # Check if the source explicitly mentions security deposit
                if not self._is_explicit_security_deposit(source):
                    fields["security_deposit"]["confidence"] = min(
                        0.5,
                        fields["security_deposit"].get("confidence", 0.5)
                    )
                    validated["validation_warnings"].append(
                        "Security deposit confidence reduced: not explicitly stated in source"
                    )
        
        # FIX 3: Lease term vs commencement date distinction
        if "lease_start_date" in fields or "lease_term" in fields:
            # Ensure we distinguish between lease term (duration) and dates
            if "lease_term" in fields:
                term_value = fields["lease_term"].get("value", "")
                # Lease term should be a duration, not a date
                if self._looks_like_date(term_value):
                    validated["validation_warnings"].append(
                        f"WARNING: Lease term '{term_value}' looks like a date, not a duration"
                    )
        
        # Validate all fields
        schema_fields = self._get_schema_fields(field_schema)
        
        for field_name, field_data in fields.items():
            # Ensure required structure
            if not isinstance(field_data, dict):
                field_data = {"value": str(field_data), "confidence": 0.5}
            
            # Ensure all required keys
            if "value" not in field_data:
                field_data["value"] = "Not specified"
            if "confidence" not in field_data:
                field_data["confidence"] = 0.5
            if "source" not in field_data:
                field_data["source"] = ""
            
            # Validate confidence score
            confidence = field_data.get("confidence", 0.5)
            if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
                field_data["confidence"] = 0.5
                validated["validation_warnings"].append(
                    f"Invalid confidence score for '{field_name}', set to 0.5"
                )
            
            validated["fields"][field_name] = field_data
        
        # Check for missing required fields
        for field_name, field_info in schema_fields.items():
            if field_info.get("required", False):
                if field_name not in validated["fields"] or validated["fields"][field_name].get("value") == "Not specified":
                    validated["validation_warnings"].append(
                        f"Required field '{field_name}' not found or is not specified"
                    )
        
        return validated
    
    def _is_likely_trade_name(self, name: str, document_text: str) -> bool:
        """
        Check if a name is likely a trade name rather than a legal entity.
        
        Args:
            name: Name to check
            document_text: Full document text
            
        Returns:
            True if likely a trade name
        """
        # Common indicators of trade names
        trade_indicators = [
            "cafe", "restaurant", "bar", "grill", "bistro", "diner",
            "shop", "store", "boutique", "market", "company", "co.",
            "llc", "inc", "corp", "ltd"
        ]
        
        name_lower = name.lower()
        
        # Check if name contains trade indicators
        for indicator in trade_indicators[:9]:  # First 9 are business types
            if indicator in name_lower:
                # Look for "dba" or "doing business as" near this name
                context_window = 500
                name_pos = document_text.lower().find(name_lower)
                if name_pos != -1:
                    context = document_text[max(0, name_pos - context_window):name_pos + context_window].lower()
                    if "dba" in context or "doing business as" in context or "d/b/a" in context:
                        return True
        
        return False
    
    def _find_legal_tenant(self, document_text: str) -> Optional[str]:
        """
        Try to find the legal tenant name in the document.
        
        Args:
            document_text: Full document text
            
        Returns:
            Legal tenant name if found, None otherwise
        """
        import re
        
        # Look for patterns like "Tenant: [Name]" or "Lessee: [Name]"
        patterns = [
            r'Tenant[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+and\s+[A-Z][a-z]+\s+[A-Z][a-z]+)?)',
            r'Lessee[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+and\s+[A-Z][a-z]+\s+[A-Z][a-z]+)?)',
            r'between\s+[^,]+,\s+and\s+([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+and\s+[A-Z][a-z]+\s+[A-Z][a-z]+)?)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, document_text)
            if match:
                potential_tenant = match.group(1).strip()
                # Verify it looks like a person's name (not a business name)
                if not any(word in potential_tenant.lower() for word in ["cafe", "restaurant", "llc", "inc", "corp"]):
                    return potential_tenant
        
        return None
    
    def _is_explicit_security_deposit(self, source_text: str) -> bool:
        """
        Check if source text explicitly mentions security deposit.
        
        Args:
            source_text: Source text snippet
            
        Returns:
            True if explicitly mentioned
        """
        if not source_text:
            return False
        
        source_lower = source_text.lower()
        explicit_terms = ["security deposit", "deposit of", "deposit in the amount"]
        
        return any(term in source_lower for term in explicit_terms)
    
    def _looks_like_date(self, text: str) -> bool:
        """
        Check if text looks like a date.
        
        Args:
            text: Text to check
            
        Returns:
            True if looks like a date
        """
        import re
        
        if not text:
            return False
        
        # Common date patterns
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # ISO format
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # MM/DD/YYYY
            r'(January|February|March|April|May|June|July|August|September|October|November|December)',
            r'\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)',
        ]
        
        for pattern in date_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _get_schema_fields(self, field_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract fields from schema regardless of format.
        
        Args:
            field_schema: Field schema
            
        Returns:
            Dictionary of fields
        """
        if "lease_summary_template" in field_schema:
            all_fields = {}
            required = field_schema["lease_summary_template"].get("required_fields", {})
            optional = field_schema["lease_summary_template"].get("optional_fields", {})
            
            for field_name, field_info in required.items():
                all_fields[field_name] = {**field_info, "required": True}
            
            for field_name, field_info in optional.items():
                all_fields[field_name] = {**field_info, "required": False}
            
            return all_fields
        
        elif "fields" in field_schema:
            return field_schema["fields"]
        
        return {}
    
    def extract_with_validation(
        self,
        document_text: str,
        field_schema: Dict[str, Any],
        document_type: str = "lease_agreement"
    ) -> Dict[str, Any]:
        """
        Extract fields with additional validation pass.
        
        Args:
            document_text: Document text
            field_schema: Field schema
            document_type: Document type
            
        Returns:
            Validated extraction result
        """
        # First extraction pass
        result = self.extract_fields(document_text, field_schema, document_type)
        
        # Additional validation for critical fields
        critical_fields = ["tenant", "landlord", "security_deposit", "lease_start_date", "lease_term"]
        
        for field_name in critical_fields:
            if field_name in result["fields"]:
                field_data = result["fields"][field_name]
                confidence = field_data.get("confidence", 0)
                
                # Flag low confidence on critical fields
                if confidence < 0.7:
                    result["validation_warnings"].append(
                        f"Low confidence ({confidence:.2f}) on critical field: {field_name}"
                    )
        
        return result
