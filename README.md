# Legal Document Chat Application

A Streamlit-based application for analyzing legal documents with AI-powered extraction and interactive chat capabilities.

## Features

### 1. Document Processing & Field Extraction
- Upload legal documents (PDF, DOCX, TXT)
- Automatic extraction of structured fields from lease agreements
- Key fields extracted:
  - Parties (Landlord, Tenant)
  - Property details (Address, Size)
  - Financial terms (Rent, Security Deposit)
  - Dates (Lease Start, End, Term)
  - Special clauses and conditions

### 2. Chat System with Source Attribution
- Ask questions about the document in natural language
- Get accurate answers with precise source citations
- Citations include page numbers and section references
- Format: "Extracted from Page X, Section Y"
- Single, most relevant source per answer

### 3. Edge Case Handling
The system handles various challenges:
- **Ambiguous or missing fields**: Returns "Not specified" with low confidence
- **Conflicting clauses**: Identifies and reports conflicts
- **Complex nested clauses**: Extracts hierarchical information
- **Multiple possible answers**: Returns most relevant with confidence score
- **Long sections**: Intelligent chunking preserves context

## Setup

### Prerequisites
- Python 3.8+
- API Keys:
  - Requesty API key (OpenAI-compatible)
  - Pinecone API key

### Installation

1. Clone the repository and navigate to the project:
```bash
cd legal_document_chat
```

2. Create `.env` file from example:
```bash
cp .env.example .env
```

3. Edit `.env` and add your API keys:
```
REQUESTY_API_KEY=your_requesty_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=us-east-1
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

### Option 1: Using the run script
```bash
./run_local.sh
```

### Option 2: Direct command
```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`

## Usage

### 1. Upload Document
- Click "Upload Document" in the sidebar
- Select a PDF, DOCX, or TXT file
- Wait for processing and field extraction

### 2. View Extracted Fields
- Navigate to "Extracted Fields" tab
- Review automatically extracted information
- Check confidence scores for each field

### 3. Chat with Document
- Go to "Chat" tab
- Ask questions about the document
- View answers with source citations

### Example Queries
```
- What is the monthly rent amount?
- When does the lease term start and end?
- What are the tenant's maintenance responsibilities?
- Can the landlord terminate the lease early?
```

## Project Structure

```
legal_document_chat/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── run_local.sh                    # Run script
├── config/
│   ├── settings.py                 # Configuration settings
│   ├── prompts.py                  # LLM prompts
│   └── lease_template.json         # Field extraction template
├── src/
│   ├── document_processor/         # Document loading and chunking
│   ├── extraction/                 # Field extraction service
│   ├── retrieval/                  # Vector store and search
│   ├── chat/                       # Chat service
│   └── utils/                      # Helper utilities
└── data/
    ├── processed/                  # Uploaded documents
    └── vector_db/                  # Vector database storage
```

## Technology Stack

- **Frontend**: Streamlit
- **LLM**: Requesty API (OpenAI-compatible)
- **Vector Database**: Pinecone
- **Document Processing**: PyPDF2, pdfplumber, python-docx
- **Embeddings**: text-embedding-3-small

## Key Implementation Details

### Chunking Strategy
- Semantic chunking with section detection
- Extracts Article and Section metadata
- Chunk size: 1000 characters with 200 character overlap
- Preserves document structure for accurate citations

### Citation System
- Single, most relevant source per answer
- Precise page and section references
- Format: "Extracted from Page X, Article Y, Section Z"
- Relevance score displayed as percentage

### Field Extraction
- Template-based extraction using LLM
- Confidence scoring for each field
- Handles missing or ambiguous information
- Validates extracted data against document

## Edge Cases Handled

1. **Ambiguous Fields**: Returns "Not specified" with confidence score
2. **Conflicting Clauses**: Identifies conflicts and reports both
3. **Complex Structures**: Extracts nested information hierarchically
4. **Multiple Answers**: Returns most relevant with confidence
5. **Long Sections**: Intelligent chunking maintains context

## Deliverables

✅ **Streamlit App** with:
- File upload (single document)
- Structured lease summary output
- Interactive chat with source citations

✅ **Source Citations** in format:
- "Extracted from Page X, Section Y"

✅ **Edge Case Analysis** with handling strategies

## Notes

- First-time setup requires API keys in `.env` file
- Documents are processed and stored in vector database
- Clear vector store from sidebar to re-process documents
- Supports PDF, DOCX, and TXT formats
