# Configuration file for Tunisia Polytechnic RAG Chatbot
import os
from dotenv import load_dotenv

# Load environment variables if .env file exists
load_dotenv()

# ========================================
# API Configuration
# ========================================

# Your OpenRouter API key
# Replace 'your_openrouter_api_key_here' with your actual API key
api = "api_key"

# Alternative: Use environment variable (recommended for security)
# Uncomment the line below if you prefer to use environment variables
# api = os.getenv('OPENROUTER_API_KEY')

# ========================================
# Model Configuration
# ========================================

# Default LLM model
DEFAULT_MODEL = "deepseek/deepseek-chat"

# Embedding model (free HuggingFace model)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Model parameters
MAX_TOKENS = 800
TEMPERATURE = 0.3
TOP_P = 0.9

# ========================================
# RAG Configuration
# ========================================

# Document chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval parameters
SIMILARITY_TOP_K = 4

# ========================================
# Directory Configuration
# ========================================

# Directories for documents and vector store
PDF_DIRECTORY = "documents"
VECTORSTORE_DIRECTORY = "vectorstore"

# ========================================
# Optional: Advanced Configuration
# ========================================

# Embeddings device (cpu or cuda)
EMBEDDINGS_DEVICE = "cpu"

# Text splitter separators
TEXT_SEPARATORS = ["\n\n", "\n", " ", ""]

# Number of sources to show in responses
MAX_SOURCES_SHOWN = 2

# Enable/disable features
ENABLE_SOURCE_CITATION = True
ENABLE_RESPONSE_TIME = True
ENABLE_STATUS_COMMAND = True

# ========================================
# Logging Configuration (optional)
# ========================================

LOG_LEVEL = "INFO"

LOG_FILE = "logs/chatbot.log"
