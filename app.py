"""
Legal Document Chat Application
Streamlit UI for interacting with legal documents.
"""

import streamlit as st
import json
from pathlib import Path
from src.utils.logger import app_logger as logger

from src.document_processor import DocumentProcessor
from src.extraction import ExtractionService
from src.retrieval import VectorStore, RetrievalService
from src.chat import ChatService
from config.settings import settings


# Page configuration
st.set_page_config(
    page_title=settings.app_title,
    page_icon=settings.app_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.vector_store = None
        st.session_state.retrieval_service = None
        st.session_state.chat_service = None
        st.session_state.document_processor = DocumentProcessor()
        st.session_state.extraction_service = ExtractionService()
        st.session_state.current_document = None
        st.session_state.extracted_fields = None
        st.session_state.chat_history = []


def initialize_components():
    """Initialize vector store and chat components."""
    try:
        with st.spinner("Initializing components..."):
            if not st.session_state.initialized:
                # Initialize vector store and services
                st.session_state.vector_store = VectorStore()
                st.session_state.retrieval_service = RetrievalService(st.session_state.vector_store)
                st.session_state.chat_service = ChatService(st.session_state.retrieval_service)
                st.session_state.initialized = True
                
                # Check if documents already exist in vector store
                check_existing_documents()
                
                st.success("✅ Components initialized successfully!")
    except Exception as e:
        st.error(f"❌ Failed to initialize components: {str(e)}")
        logger.error(f"Initialization error: {str(e)}")


def check_existing_documents():
    """Check if documents already exist in vector store and restore session state."""
    try:
        if st.session_state.vector_store:
            stats = st.session_state.vector_store.get_stats()
            vector_count = stats.get('total_vector_count', 0)
            
            if vector_count > 0:
                # Documents exist in vector store
                logger.info(f"Found {vector_count} existing vectors in store")
                
                # Check if there are processed documents in the data folder
                processed_dir = Path("data/processed")
                if processed_dir.exists():
                    pdf_files = list(processed_dir.glob("*.pdf"))
                    docx_files = list(processed_dir.glob("*.docx"))
                    txt_files = list(processed_dir.glob("*.txt"))
                    all_files = pdf_files + docx_files + txt_files
                    
                    if all_files:
                        # Load the most recent document
                        most_recent = max(all_files, key=lambda p: p.stat().st_mtime)
                        
                        # Create a minimal document object for session state
                        st.session_state.current_document = {
                            'file_name': most_recent.name,
                            'file_path': str(most_recent),
                            'file_type': most_recent.suffix.lstrip('.'),
                            'page_count': 'N/A'  # We don't need to reload the full document
                        }
                        
                        st.info(f"📄 Restored session: {most_recent.name} ({vector_count} vectors in store)")
                        logger.info(f"Restored document: {most_recent.name}")
    except Exception as e:
        logger.warning(f"Could not check existing documents: {str(e)}")


def load_field_schema():
    """Load field schema from config."""
    try:
        schema_path = Path("config/lease_template.json")
        if schema_path.exists():
            with open(schema_path, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        logger.error(f"Failed to load field schema: {str(e)}")
        return None


def sidebar():
    """Render sidebar with document upload and management."""
    st.sidebar.title(f"{settings.app_icon} Legal Document Chat")
    st.sidebar.markdown("---")
    
    # Document upload section
    st.sidebar.header("📄 Upload Document")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a legal document",
        type=['pdf', 'docx', 'txt'],
        help="Upload a lease agreement or legal contract"
    )
    
    if uploaded_file:
        if st.sidebar.button("Process Document", type="primary"):
            process_document(uploaded_file)
    
    st.sidebar.markdown("---")
    
    # Document info
    if st.session_state.current_document:
        st.sidebar.header("📋 Current Document")
        doc = st.session_state.current_document
        st.sidebar.info(f"""
        **File:** {doc['file_name']}  
        **Type:** {doc['file_type'].upper()}  
        **Pages/Sections:** {doc.get('page_count', doc.get('paragraph_count', 'N/A'))}
        """)
        
        if st.sidebar.button("Clear Document"):
            clear_document()
    
    st.sidebar.markdown("---")
    
    # Vector store stats
    if st.session_state.vector_store:
        st.sidebar.header("📊 Vector Store Stats")
        stats = st.session_state.vector_store.get_stats()
        if stats:
            st.sidebar.metric("Total Vectors", stats.get('total_vector_count', 0))
    
    st.sidebar.markdown("---")
    st.sidebar.caption("Powered by Requesty API & Pinecone")


def process_document(uploaded_file):
    """Process uploaded document."""
    try:
        with st.spinner("Processing document..."):
            # Save uploaded file temporarily
            temp_path = Path("data/processed") / uploaded_file.name
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # Load document
            document = st.session_state.document_processor.load_document(temp_path)
            st.session_state.current_document = document
            
            # Create chunks
            chunks = st.session_state.document_processor.chunk_document(
                document,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap
            )
            
            # Add to vector store
            st.session_state.vector_store.add_documents(chunks)
            
            st.success(f"✅ Document processed! Created {len(chunks)} chunks.")
            
            # Extract fields
            with st.spinner("Extracting structured fields..."):
                schema = load_field_schema()
                if schema:
                    extracted = st.session_state.extraction_service.extract_fields(
                        document['full_text'],
                        schema,
                        document_type="lease_agreement"
                    )
                    st.session_state.extracted_fields = extracted
                    st.success("✅ Fields extracted successfully!")
            
    except Exception as e:
        st.error(f"❌ Error processing document: {str(e)}")
        logger.error(f"Document processing error: {str(e)}")


def clear_document():
    """Clear current document and reset state."""
    if st.session_state.current_document:
        file_name = st.session_state.current_document['file_name']
        st.session_state.vector_store.delete_document(file_name)
    
    st.session_state.current_document = None
    st.session_state.extracted_fields = None
    st.session_state.chat_history = []
    st.session_state.chat_service.clear_history()
    st.success("✅ Document cleared!")
    st.rerun()


def display_extracted_fields():
    """Display extracted fields in a nice format."""
    if not st.session_state.extracted_fields:
        return
    
    st.header("📋 Extracted Fields")
    
    fields = st.session_state.extracted_fields.get('fields', {})
    
    if not fields:
        st.info("No fields extracted yet.")
        return
    
    # Display in columns
    col1, col2 = st.columns(2)
    
    field_items = list(fields.items())
    mid = len(field_items) // 2
    
    with col1:
        for field_name, field_data in field_items[:mid]:
            display_field(field_name, field_data)
    
    with col2:
        for field_name, field_data in field_items[mid:]:
            display_field(field_name, field_data)
    
    # Display warnings if any
    warnings = st.session_state.extracted_fields.get('validation_warnings', [])
    if warnings:
        with st.expander("⚠️ Validation Warnings"):
            for warning in warnings:
                st.warning(warning)


def display_field(field_name: str, field_data: dict):
    """Display a single extracted field."""
    value = field_data.get('value', 'N/A')
    confidence = field_data.get('confidence', 0)
    
    # Color code by confidence
    if confidence >= 0.8:
        color = "🟢"
    elif confidence >= 0.5:
        color = "🟡"
    else:
        color = "🔴"
    
    st.markdown(f"**{field_name}** {color}")
    st.text(f"Value: {value}")
    st.caption(f"Confidence: {confidence:.2%}")
    st.markdown("---")


def chat_interface():
    """Render chat interface."""
    st.header("💬 Chat with Document")
    
    # Check if vector store has documents even if current_document is not set
    has_vectors = False
    if st.session_state.vector_store:
        stats = st.session_state.vector_store.get_stats()
        has_vectors = stats.get('total_vector_count', 0) > 0
    
    if not st.session_state.current_document and not has_vectors:
        st.info("👆 Please upload a document first to start chatting.")
        return
    
    if not st.session_state.current_document and has_vectors:
        st.warning("⚠️ Documents exist in vector store but session was reset. You can still chat, or re-upload to see document details.")
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message and message["sources"]:
                    st.markdown("---")
                    st.markdown("**📚 Source:**")
                    
                    # Display only the first (most relevant) source
                    source = message["sources"][0]
                    citation = source.get('citation', source.get('file_name', 'Unknown'))
                    relevance = source.get('relevance_score', 0)
                    
                    # Format relevance score properly
                    if isinstance(relevance, (int, float)):
                        relevance_str = f"{relevance:.1f}%"
                    else:
                        relevance_str = str(relevance)
                    
                    # Display citation with relevance
                    st.markdown(f"Extracted from **{citation}** ({relevance_str} relevance)")
                    
                    # Show text preview if available
                    if source.get('text_preview'):
                        with st.expander("View excerpt"):
                            st.caption(source['text_preview'])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the document..."):
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chat_service.chat(
                        prompt,
                        include_context=True,
                        use_hybrid_search=True
                    )
                    
                    st.markdown(response['response'])
                    
                    # Display only the top source
                    if response.get('sources'):
                        st.markdown("---")
                        st.markdown("**📚 Source:**")
                        
                        # Display only the first (most relevant) source
                        source = response['sources'][0]
                        citation = source.get('citation', source.get('file_name', 'Unknown'))
                        relevance = source.get('relevance_score', 0)
                        
                        # Format relevance score properly
                        if isinstance(relevance, (int, float)):
                            relevance_str = f"{relevance:.1f}%"
                        else:
                            relevance_str = str(relevance)
                        
                        # Display citation with relevance
                        st.markdown(f"Extracted from **{citation}** ({relevance_str} relevance)")
                        
                        # Show text preview if available
                        if source.get('text_preview'):
                            with st.expander("View excerpt"):
                                st.caption(source['text_preview'])
                    
                    # Display confidence if available
                    if response.get('confidence'):
                        st.caption(f"🎯 Confidence: {response['confidence']}")
                    
                    # Add to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response['response'],
                        "sources": response.get('sources', [])
                    })
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.error(f"Chat error: {str(e)}")


def main():
    """Main application."""
    initialize_session_state()
    
    # Sidebar
    sidebar()
    
    # Main content
    st.title(f"{settings.app_icon} {settings.app_title}")
    st.markdown("Upload a legal document and chat with it using AI-powered analysis.")
    
    # Initialize components
    if not st.session_state.initialized:
        initialize_components()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📋 Extracted Fields", "ℹ️ About"])
    
    with tab1:
        chat_interface()
    
    with tab2:
        display_extracted_fields()
    
    with tab3:
        st.markdown("""
        ## About This Application
        
        This Legal Document Chat Assistant helps you:
        
        - 📄 **Upload** legal documents (PDF, DOCX, TXT)
        - 🔍 **Extract** structured fields automatically
        - 💬 **Chat** with your documents using natural language
        - 📊 **Analyze** lease agreements and contracts
        
        ### Features
        
        - **AI-Powered Extraction**: Automatically extracts key fields from lease agreements
        - **Semantic Search**: Find relevant information using natural language queries
        - **Context-Aware Responses**: Get accurate answers based on document content
        - **Source Citations**: See exactly where information comes from
        
        ### Technology Stack
        
        - **LLM**: Requesty API (OpenAI-compatible)
        - **Vector Database**: Pinecone
        - **Framework**: LangChain, Streamlit
        - **Document Processing**: PyPDF2, pdfplumber, python-docx
        
        ### How to Use
        
        1. Upload a legal document using the sidebar
        2. Wait for processing and field extraction
        3. Ask questions in the chat interface
        4. View extracted fields in the "Extracted Fields" tab
        
        ---
        
        Made with ❤️ using Streamlit and Requesty API
        """)


if __name__ == "__main__":
    main()
