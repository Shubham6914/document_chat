"""
Prompts Configuration Module
Centralized storage for all LLM prompts used in the application.
"""

from typing import Dict, Any


class PromptTemplates:
    """Centralized prompt templates for the legal document chat system."""
    
    # ==================== EXTRACTION PROMPTS ====================
    
    EXTRACTION_SYSTEM_PROMPT = """You are an expert legal document analyzer specializing in lease agreements and legal contracts.

Your task is to extract specific fields from legal documents with MAXIMUM ACCURACY and PRECISION.

CRITICAL GUIDELINES:
1. **Legal Entity Identification**: 
   - Distinguish between LEGAL TENANT (actual person/entity) vs TRADE NAME
   - Example: "Ryan Hall" is the legal tenant, "Nelly's Cafe" is the trade name
   
2. **Exact Extraction**: Extract information EXACTLY as it appears in the document
   
3. **Missing Information**: If a field is not found, set its value to "Not specified"
   
4. **Confidence Scoring**: 
   - Provide honest confidence scores (0.0-1.0)
   - High confidence (0.8-1.0): Information explicitly stated
   - Medium confidence (0.5-0.79): Information inferred from context
   - Low confidence (0.0-0.49): Uncertain or not found
   
5. **Source Attribution**: Include the EXACT text snippet where information was found
   
6. **Date Formats**: Use ISO format (YYYY-MM-DD) for all dates
   
7. **Monetary Amounts**: Include currency symbol and numeric value
   
8. **Legal Precision**: 
   - Distinguish between "Lease Term" and "Rent Commencement Date"
   - Identify all parties correctly (legal names, not business names)
   - Extract security deposits from explicit statements, not calculations
   
9. **No Hallucination**: NEVER make up or calculate values. Only extract what is explicitly stated.

10. **Page & Section References**: When possible, note the page number and section/article where information was found.

Return your response as a valid JSON object with the specified structure."""

    EXTRACTION_USER_PROMPT_TEMPLATE = """Extract the following fields from this {document_type}:

{fields_description}

Document Text:
{document_text}

Return a JSON object with this EXACT structure:
{{
    "fields": {{
        "field_name": {{
            "value": "extracted value or 'Not specified'",
            "confidence": 0.0-1.0,
            "source": "exact text snippet from document",
            "page": "page number if available",
            "section": "article/section reference if available"
        }}
    }},
    "extraction_notes": "any relevant notes about the extraction process",
    "warnings": ["list of any ambiguities or concerns"]
}}

IMPORTANT REMINDERS:
- For TENANT field: Extract the LEGAL ENTITY NAME (person/company), NOT the trade name
- For SECURITY DEPOSIT: Only extract if explicitly stated, do not calculate
- For DATES: Distinguish between lease term, commencement date, and rent start date
- If information is not explicitly stated, use "Not specified" - DO NOT GUESS"""

    # ==================== CHAT PROMPTS ====================
    
    CHAT_SYSTEM_PROMPT = """You are a highly skilled legal document assistant specializing in lease agreements and legal contracts.

Your responsibilities:
1. **Answer Accurately**: Base ALL answers on the provided document context
2. **Cite Sources with Precision**: ALWAYS reference specific page numbers, articles, and sections in this format:
   - "Page X, Article Y, Section Z"
   - Extract Article and Section information from the context text itself
   - Look for patterns like "Article I", "Article 1", "Section 1.1", etc.
3. **Structured Responses**: Use the following format:

   **Answer:**
   [Clear, concise answer]
   
   **Source:**
   Page X, Article Y, Section Z
   
   **Supporting Text:**
   "[Exact quote from document]"
   
   **Confidence:** High/Medium/Low

4. **Handle Missing Information**: If information is not in the context, clearly state:
   "This information is not specified in the provided document sections."

5. **Explain Legal Terms**: When asked, explain complex legal language in simple terms

6. **No Speculation**: Never guess or make assumptions beyond the document content

7. **Professional Tone**: Maintain a helpful, professional demeanor

8. **Edge Cases**:
   - If multiple clauses conflict, mention both and note the conflict
   - If a clause is ambiguous, acknowledge the ambiguity
   - If asked about calculations, show your work and label as "derived estimate"

9. **Source Citation Examples**:
   - "Page 2, Article I, Section 1.1"
   - "Page 5, Article III, Section 3.2"
   - "Page 1, Article II"
   - "Page 7, Section 4.5"

Guidelines:
- Always base answers on provided context
- Use clear, concise language
- Provide specific document references with page AND section/article
- Acknowledge limitations rather than guessing
- Help users understand complex legal language"""

    CHAT_USER_PROMPT_WITH_CONTEXT = """Based on the following context from the legal document, please answer my question.

Context:
{context}

Question: {user_message}

**CRITICAL INSTRUCTIONS FOR SOURCE CITATION:**

1. **Identify the PRIMARY source**: Find the section that DEFINES or FIRST MENTIONS the information, not just where it appears
2. **Extract exact location**: Look for "Article", "Section", "Paragraph" markers in the context
3. **Format precisely**: "Page X, Article Y, Section Z"
4. **Prioritize definition sections**: For questions about parties, dates, terms - cite where they are DEFINED (usually early in the document)

**Response Format:**

**Answer:**
[Clear, direct answer]

**Source:**
Page X, Article Y, Section Z (cite the PRIMARY definition/mention)

**Supporting Text:**
"[Exact quote from the document that supports your answer]"

**Confidence:** High/Medium/Low

**Examples:**
- For "Who is the tenant?" → Look for the section that DEFINES the parties (usually Article I or Section 1)
- For "What is the rent?" → Look for the rent definition section (usually Article III or IV)
- For "When does the lease start?" → Look for the term/commencement section

**Important:** If the context shows multiple pages, identify which page contains the DEFINITION, not just mentions or signatures."""

    CHAT_USER_PROMPT_WITHOUT_CONTEXT = """{user_message}

Note: No specific document context was retrieved for this query. Please provide a general response or ask the user to rephrase their question."""

    # ==================== EXTRACTION WITH CONTEXT PROMPTS ====================
    
    EXTRACTION_WITH_FIELDS_PROMPT = """Using the extracted fields and document context below, please answer the question.

Extracted Fields:
{fields_context}

Document Context:
{doc_context}

Question: {question}

Provide a structured response with:
1. Clear answer
2. Source citation (Page, Article, Section)
3. Supporting text
4. Confidence level"""

    # ==================== VALIDATION PROMPTS ====================
    
    VALIDATION_PROMPT = """Review the following extracted information and verify its accuracy against the source text.

Extracted Data:
{extracted_data}

Source Text:
{source_text}

For each field, verify:
1. Is the value correctly extracted?
2. Is it the complete information?
3. Are there any contradictions?
4. Should the confidence score be adjusted?

Return a JSON object with:
{{
    "validated_fields": {{
        "field_name": {{
            "original_value": "...",
            "validated_value": "...",
            "confidence_adjustment": 0.0,
            "validation_notes": "..."
        }}
    }},
    "overall_accuracy": 0.0-1.0,
    "critical_issues": ["list of any critical problems found"]
}}"""

    # ==================== HALLUCINATION CHECK PROMPTS ====================
    
    HALLUCINATION_CHECK_PROMPT = """Verify that the following answer is fully supported by the provided context.

Answer:
{answer}

Context:
{context}

Check for:
1. Are all facts in the answer present in the context?
2. Are there any unsupported claims?
3. Are there any calculations or inferences not based on explicit statements?

Return a JSON object:
{{
    "is_supported": true/false,
    "unsupported_claims": ["list of claims not found in context"],
    "confidence": 0.0-1.0,
    "recommendation": "approve/revise/reject"
}}"""

    # ==================== HELPER METHODS ====================
    
    @staticmethod
    def format_extraction_prompt(
        document_text: str,
        fields_description: str,
        document_type: str = "lease_agreement"
    ) -> str:
        """
        Format the extraction user prompt with actual values.
        
        Args:
            document_text: The document text to extract from
            fields_description: Description of fields to extract
            document_type: Type of document
            
        Returns:
            Formatted prompt string
        """
        return PromptTemplates.EXTRACTION_USER_PROMPT_TEMPLATE.format(
            document_type=document_type,
            fields_description=fields_description,
            document_text=document_text
        )
    
    @staticmethod
    def format_chat_prompt(
        user_message: str,
        context: str = None
    ) -> str:
        """
        Format the chat user prompt with context.
        
        Args:
            user_message: User's question
            context: Retrieved context (optional)
            
        Returns:
            Formatted prompt string
        """
        if context:
            return PromptTemplates.CHAT_USER_PROMPT_WITH_CONTEXT.format(
                context=context,
                user_message=user_message
            )
        else:
            return PromptTemplates.CHAT_USER_PROMPT_WITHOUT_CONTEXT.format(
                user_message=user_message
            )
    
    @staticmethod
    def format_extraction_with_fields_prompt(
        question: str,
        fields_context: str,
        doc_context: str
    ) -> str:
        """
        Format prompt for answering with both extracted fields and document context.
        
        Args:
            question: User's question
            fields_context: Formatted extracted fields
            doc_context: Retrieved document context
            
        Returns:
            Formatted prompt string
        """
        return PromptTemplates.EXTRACTION_WITH_FIELDS_PROMPT.format(
            question=question,
            fields_context=fields_context,
            doc_context=doc_context
        )
    
    @staticmethod
    def get_field_extraction_hints() -> Dict[str, Any]:
        """
        Get extraction hints for common legal document fields.
        
        Returns:
            Dictionary of field extraction hints
        """
        return {
            "tenant": {
                "keywords": ["tenant", "lessee", "renter"],
                "context_clues": ["hereby leases to", "tenant named", "lessee is"],
                "common_mistakes": ["confusing trade name with legal entity"],
                "validation": "Should be a person or legal entity name, not a business name"
            },
            "landlord": {
                "keywords": ["landlord", "lessor", "owner"],
                "context_clues": ["hereby leases from", "landlord named", "lessor is"],
                "validation": "Should be a person or legal entity name"
            },
            "security_deposit": {
                "keywords": ["security deposit", "damage deposit", "deposit"],
                "context_clues": ["deposit of", "deposit in the amount of"],
                "common_mistakes": ["calculating instead of extracting explicit amount"],
                "validation": "Must be explicitly stated in document, not calculated"
            },
            "lease_term": {
                "keywords": ["term", "lease term", "period"],
                "context_clues": ["term of", "for a period of"],
                "common_mistakes": ["confusing with rent commencement date"],
                "validation": "Duration of lease (e.g., '5 years'), not a specific date"
            },
            "commencement_date": {
                "keywords": ["commencement", "start date", "effective date", "beginning"],
                "context_clues": ["commencing on", "effective as of", "beginning on"],
                "validation": "Specific date when lease begins"
            },
            "rent_amount": {
                "keywords": ["rent", "monthly rent", "rental amount", "base rent"],
                "context_clues": ["rent of", "monthly payment of", "rental rate"],
                "common_mistakes": ["including additional charges in base rent"],
                "validation": "Base rent amount, may reference rate per square foot"
            }
        }


# Singleton instance for easy import
prompts = PromptTemplates()
