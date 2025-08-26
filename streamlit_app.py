import streamlit as st
import time
from typing import List, Dict, Any
import os
from datetime import datetime
from pathlib import Path
import shutil

# Import your existing RAG chatbot
from rag_chatbot import TunisiaPolytechnicRAGBot

# Page configuration
st.set_page_config(
    page_title="ChatEPT - École Polytechnique de Tunisie",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f4e79;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    
    .bot-message {
        background-color: #000;
        border-left-color: #28a745;
    }
    
    .source-box {
        background-color: #f5f5f5;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 0.5rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
    st.session_state.chatbot_initialized = False

if "api_key" not in st.session_state:
    st.session_state.api_key = None

if "documents_count" not in st.session_state:
    st.session_state.documents_count = 0

# Sidebar configuration
with st.sidebar:
    st.image("https://via.placeholder.com/200x100/1f4e79/white?text=EPT", width=200)
    st.title("⚙️ Configuration")
    
    # API Key input
    if not st.session_state.api_key:
        st.subheader("🔑 API Configuration")
        api_key_input = st.text_input(
            "OpenRouter API Key:",
            type="password",
            help="Enter your OpenRouter API key to use the chatbot"
        )
        
        if st.button("Connect"):
            if api_key_input:
                st.session_state.api_key = api_key_input
                st.success("API Key saved!")
                st.rerun()
            else:
                st.error("Please enter a valid API key")
        
        st.info("💡 Get your free API key from [OpenRouter](https://openrouter.ai/)")
        st.stop()
    else:
        st.success("🔑 API Key connected")
        if st.button("Reset API Key"):
            st.session_state.api_key = None
            st.session_state.chatbot = None
            st.session_state.chatbot_initialized = False
            st.rerun()
    
    st.divider()
    
    # Document upload section
    st.subheader("📄 Document Management")
    
    # Show current documents count
    documents_dir = Path("documents")
    if documents_dir.exists():
        pdf_files = list(documents_dir.glob("*.pdf"))
        st.session_state.documents_count = len(pdf_files)
        st.info(f"📚 Current documents: {len(pdf_files)}")
        
        # List current documents
        if pdf_files:
            with st.expander("View Documents"):
                for pdf in pdf_files:
                    st.text(f"📄 {pdf.name}")
    else:
        st.info("📚 No documents directory found")
    
    uploaded_files = st.file_uploader(
        "Upload PDF documents:",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload PDF documents to expand the knowledge base"
    )
    
    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                # Create documents directory if it doesn't exist
                documents_dir.mkdir(exist_ok=True)
                
                success_count = 0
                for uploaded_file in uploaded_files:
                    try:
                        # Save uploaded file
                        file_path = documents_dir / uploaded_file.name
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Add to chatbot if initialized
                        if st.session_state.chatbot:
                            st.session_state.chatbot.add_document(str(file_path))
                        
                        success_count += 1
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                
                if success_count > 0:
                    st.success(f"✅ Processed {success_count} documents!")
                    st.session_state.documents_count += success_count
                    
                    # Reinitialize chatbot to reload vector store
                    if st.session_state.chatbot_initialized:
                        st.session_state.chatbot_initialized = False
                        st.rerun()
    
    st.divider()
    
    # Statistics
    st.subheader("📊 Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Messages", len(st.session_state.messages))
    with col2:
        st.metric("Documents", st.session_state.documents_count)
    
    # Vector store status
    if st.session_state.chatbot and st.session_state.chatbot.vectorstore:
        chunks_count = st.session_state.chatbot.vectorstore.index.ntotal
        st.metric("Document Chunks", chunks_count)
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", type="secondary"):
        st.session_state.messages = []
        st.rerun()
    
    # Rebuild knowledge base button
    if st.button("🔄 Rebuild Knowledge Base", help="Rebuild the vector store from documents"):
        if st.session_state.chatbot:
            with st.spinner("Rebuilding knowledge base..."):
                # Force rebuild by removing vectorstore and reinitializing
                vectorstore_path = Path("vectorstore")
                if vectorstore_path.exists():
                    shutil.rmtree(vectorstore_path)
                st.session_state.chatbot_initialized = False
                st.success("Knowledge base will be rebuilt on next query")
                st.rerun()

# Main content area
st.markdown("<h1 class='main-header'>🎓 ChatEPT - École Polytechnique de Tunisie</h1>", 
            unsafe_allow_html=True)

st.markdown("""
**Welcome to ChatEPT!** 🤖 I'm here to help you with questions about École Polytechnique de Tunisie. 
Ask me about admissions, courses, facilities, regulations, or any other information you need.
""")

# Initialize chatbot
@st.cache_resource
def initialize_chatbot(api_key):
    """Initialize the TunisiaPolytechnicRAGBot"""
    try:
        return TunisiaPolytechnicRAGBot(api_key, pdf_directory="documents/")
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {str(e)}")
        return None

if not st.session_state.chatbot_initialized and st.session_state.api_key:
    with st.spinner("Initializing ChatEPT... This may take a moment to load the knowledge base."):
        st.session_state.chatbot = initialize_chatbot(st.session_state.api_key)
        if st.session_state.chatbot:
            st.session_state.chatbot_initialized = True
            # Update documents count
            if st.session_state.chatbot.vectorstore:
                chunks_count = st.session_state.chatbot.vectorstore.index.ntotal
                st.success(f"✅ ChatEPT initialized with {chunks_count} document chunks!")
            else:
                st.warning("⚠️ ChatEPT initialized without knowledge base. Add documents to enable RAG functionality.")
        else:
            st.error("Failed to initialize ChatEPT. Please check your API key and try again.")

# Chat interface
chat_container = st.container()

# Display chat messages
with chat_container:
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>🤖 ChatEPT:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
            
            # Display sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources", expanded=False):
                    for source in message["sources"]:
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>📄 Document:</strong> {source}<br>
                        </div>
                        """, unsafe_allow_html=True)

# Chat input
with st.container():
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_input = st.text_input(
            "Ask your question about EPT:",
            placeholder="e.g., What are the admission requirements for engineering programs?",
            key="user_input"
        )
    
    with col2:
        send_button = st.button("Send", type="primary", key="send_button")

# Handle user input
if send_button and user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Show typing indicator
    with st.spinner("ChatEPT is thinking..."):
        if st.session_state.chatbot and st.session_state.chatbot_initialized:
            try:
                # Get response from RAG chatbot
                start_time = time.time()
                response = st.session_state.chatbot.get_response(user_input)
                end_time = time.time()
                
                # Extract sources from response if available
                sources = []
                if "📚 Sources:" in response:
                    response_parts = response.split("📚 Sources:")
                    main_response = response_parts[0].strip()
                    sources_text = response_parts[1].strip()
                    # Extract source lines
                    source_lines = [line.strip()[2:] for line in sources_text.split('\n') if line.strip().startswith('-')]
                    sources = source_lines
                    response = main_response
                
                # Add bot response to chat history
                message_data = {
                    "role": "assistant", 
                    "content": response,
                    "timestamp": datetime.now(),
                    "response_time": f"{end_time - start_time:.2f}s"
                }
                
                if sources:
                    message_data["sources"] = sources
                
                st.session_state.messages.append(message_data)
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg,
                    "timestamp": datetime.now()
                })
        else:
            # Chatbot not initialized
            error_msg = "Please ensure the chatbot is properly initialized with a valid API key."
            st.session_state.messages.append({
                "role": "assistant", 
                "content": error_msg,
                "timestamp": datetime.now()
            })
    
    # Clear input and rerun to show new messages
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>
        ChatEPT v1.0 | Built with Streamlit & LangChain | 
        École Polytechnique de Tunisie © 2024
    </small>
</div>
""", unsafe_allow_html=True)

# Quick action buttons
st.markdown("### 🚀 Quick Questions")
quick_questions = [
    "What are the admission requirements?",
    "Tell me about the engineering programs",
    "What facilities are available on campus?",
    "How can I apply for scholarships?"
]

cols = st.columns(len(quick_questions))
for i, question in enumerate(quick_questions):
    with cols[i]:
        if st.button(question, key=f"quick_{i}"):
            st.session_state.messages.append({"role": "user", "content": question})
            
            # Process the quick question
            with st.spinner("Processing..."):
                if st.session_state.chatbot and st.session_state.chatbot_initialized:
                    try:
                        response = st.session_state.chatbot.get_response(question)
                        
                        # Extract sources from response if available
                        sources = []
                        if "📚 Sources:" in response:
                            response_parts = response.split("📚 Sources:")
                            main_response = response_parts[0].strip()
                            sources_text = response_parts[1].strip()
                            source_lines = [line.strip()[2:] for line in sources_text.split('\n') if line.strip().startswith('-')]
                            sources = source_lines
                            response = main_response
                        
                        message_data = {
                            "role": "assistant",
                            "content": response,
                            "timestamp": datetime.now()
                        }
                        
                        if sources:
                            message_data["sources"] = sources
                        
                        st.session_state.messages.append(message_data)
                    except Exception as e:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Sorry, I encountered an error: {str(e)}",
                            "timestamp": datetime.now()
                        })
                else:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Please ensure the chatbot is properly initialized.",
                        "timestamp": datetime.now()
                    })
            st.rerun()