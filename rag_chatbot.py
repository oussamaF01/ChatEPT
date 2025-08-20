import os
import time
from typing import List, Dict
from pathlib import Path

# LangChain imports
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

# OpenAI for LLM integration
from openai import OpenAI
import numpy as np

# Try to import from config, fallback to environment variable or input
try:
    from config import api
except ImportError:
    print("⚠️  config.py not found. Using environment variable or manual input.")
    api = os.getenv('OPENROUTER_API_KEY')

class TunisiaPolytechnicRAGBot:
    def __init__(self, api_key: str, pdf_directory: str = "documents/"):
        """Initialize RAG chatbot with PDF knowledge base"""
        self.api_key = api_key
        self.pdf_directory = Path(pdf_directory)
        
        # Initialize OpenAI client
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        
        self.vectorstore = None
        
        # Initialize embeddings (using free HuggingFace model)
        print("📚 Loading embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Setup the knowledge base
        self._setup_knowledge_base()
    
    def _load_documents(self) -> List[Document]:
        """Load and process PDF documents"""
        print(f"📁 Loading documents from {self.pdf_directory}...")
        
        documents = []
        
        if not self.pdf_directory.exists():
            print(f"⚠️  Directory {self.pdf_directory} not found. Creating it...")
            self.pdf_directory.mkdir(parents=True, exist_ok=True)
            print("📋 Please add your PDF files to this directory and restart.")
            return documents
        
        # Load all PDF files in directory
        pdf_files = list(self.pdf_directory.glob("*.pdf"))
        
        if not pdf_files:
            print("⚠️  No PDF files found in directory.")
            return documents
        
        for pdf_file in pdf_files:
            print(f"📄 Loading: {pdf_file.name}")
            try:
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()
                documents.extend(docs)
                print(f"✅ Loaded {len(docs)} pages from {pdf_file.name}")
            except Exception as e:
                print(f"❌ Error loading {pdf_file.name}: {e}")
        
        return documents
    
    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks for better retrieval"""
        print("✂️  Splitting documents into chunks...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"📝 Created {len(chunks)} document chunks")
        
        return chunks
    
    def _setup_knowledge_base(self):
        """Setup the vector store knowledge base"""
        # Check if vector store already exists
        vectorstore_path = "vectorstore"
        if os.path.exists(vectorstore_path):
            try:
                print("📁 Loading existing vector store...")
                self.vectorstore = FAISS.load_local(vectorstore_path, self.embeddings, allow_dangerous_deserialization=True)
                print("✅ Vector store loaded from disk!")
                return
            except Exception as e:
                print(f"⚠️  Error loading existing vector store: {e}")
                print("Creating new vector store...")
        
        # Load documents
        documents = self._load_documents()
        
        if not documents:
            print("⚠️  No documents loaded. The bot will work without RAG.")
            return
        
        # Split documents
        chunks = self._split_documents(documents)
        
        # Create vector store
        print("🔍 Creating vector store...")
        try:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
            print("✅ Vector store created successfully!")
            
            # Save vector store for future use
            self.vectorstore.save_local(vectorstore_path)
            print(f"💾 Vector store saved to {vectorstore_path}")
            
        except Exception as e:
            print(f"❌ Error creating vector store: {e}")
    
    def _retrieve_relevant_docs(self, question: str, k: int = 4) -> List[Document]:
        """Retrieve relevant documents for a question"""
        if not self.vectorstore:
            return []
        
        try:
            docs = self.vectorstore.similarity_search(question, k=k)
            return docs
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
    
    def _format_context(self, docs: List[Document]) -> str:
        """Format retrieved documents as context"""
        if not docs:
            return "No relevant information found in the documents."
        
        context_parts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown source')
            source_name = Path(source).name if source != 'Unknown source' else source
            page = doc.metadata.get('page', 'Unknown page')
            
            context_parts.append(f"Document {i+1} ({source_name}, Page {page + 1 if isinstance(page, int) else page}):\n{doc.page_content}")
        
        return "\n\n".join(context_parts)
    
    def _generate_response(self, question: str, context: str) -> str:
        """Generate response using OpenRouter LLM"""
        prompt = f"""
You are a helpful assistant specialized in Tunisia Polytechnic School (École Polytechnique de Tunisie).
Use the following context from official documents to answer the question accurately.

Context:
{context}

Question: {question}

Instructions:
- Answer based primarily on the provided context
- If the context doesn't contain enough information, say so clearly
- Provide specific details when available
- Be concise but thorough
- If relevant, mention which document or section the information comes from

Answer:
"""
        
        try:
            response = self.client.chat.completions.create(
                model="deepseek/deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.3,
                top_p=0.9
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def get_response(self, question: str) -> str:
        """Get response using RAG or fallback to general response"""
        try:
            if self.vectorstore:
                # Retrieve relevant documents
                docs = self._retrieve_relevant_docs(question)
                
                if docs:
                    # Format context
                    context = self._format_context(docs)
                    
                    # Generate response with context
                    answer = self._generate_response(question, context)
                    
                    # Add source information
                    source_info = "\n\n📚 Sources:"
                    unique_sources = set()
                    for doc in docs[:2]:  # Show top 2 sources
                        source = doc.metadata.get('source', 'Unknown')
                        source_name = Path(source).name if source != 'Unknown' else source
                        if source_name not in unique_sources:
                            unique_sources.add(source_name)
                            page = doc.metadata.get('page', 'Unknown page')
                            source_info += f"\n- {source_name} (Page {page + 1 if isinstance(page, int) else page})"
                    
                    if unique_sources:
                        answer += source_info
                    
                    return answer
                else:
                    return self._get_general_response(question)
            else:
                # Fallback to general response
                return self._get_general_response(question)
                
        except Exception as e:
            print(f"Error getting RAG response: {e}")
            return self._get_general_response(question)
    
    def _get_general_response(self, question: str) -> str:
        """Fallback method for general responses without RAG"""
        system_prompt = """
        You are a helpful assistant specialized in Tunisia Polytechnic School (École Polytechnique de Tunisie).
        Answer questions accurately and mention that for specific details, 
        official documents should be consulted.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="deepseek/deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                max_tokens=500,
                temperature=0.3
            )
            return response.choices[0].message.content + "\n\n💡 Note: For more detailed information, please refer to official documents."
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    def add_document(self, pdf_path: str):
        """Add a new PDF document to the knowledge base"""
        try:
            # Load new document
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)
            
            # Add to existing vector store
            if self.vectorstore:
                self.vectorstore.add_documents(chunks)
                # Save updated vector store
                self.vectorstore.save_local("vectorstore")
                print(f"✅ Added {pdf_path} to knowledge base and saved")
            else:
                # Create new vector store if none exists
                self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
                self.vectorstore.save_local("vectorstore")
                print(f"✅ Created new knowledge base with {pdf_path}")
                
        except Exception as e:
            print(f"❌ Error adding document: {e}")
    
    def run(self):
        """Main chat loop"""
        print("🎓 Tunisia Polytechnic School RAG Chatbot")
        print("Ask me anything about École Polytechnique de Tunisie!")
        print("I'll search through official documents to give you accurate answers.")
        print("Commands: 'exit', 'quit' to end | 'status' for system info")
        print("-" * 60)
        
        # Show system status
        if self.vectorstore:
            doc_count = self.vectorstore.index.ntotal
            print(f"📊 Knowledge Base: {doc_count} document chunks loaded")
        else:
            print("⚠️  No knowledge base loaded - using general responses only")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\n🧑 You: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ["exit", "quit","bye"]:
                    print("👋 Goodbye! Good luck with your studies!")
                    break
                
                if user_input.lower() == "status":
                    if self.vectorstore:
                        doc_count = self.vectorstore.index.ntotal
                        print(f"📊 Knowledge Base: {doc_count} document chunks")
                        print(f"📁 PDF Directory: {self.pdf_directory}")
                    else:
                        print("⚠️  No knowledge base loaded")
                    continue
                
                print("\n🤖 Bot: ", end="")
                start_time = time.time()
                response = self.get_response(user_input)
                end_time = time.time()
                
                print(response)
                print(f"\n⏱️  Response time: {end_time - start_time:.2f}s")
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ An error occurred: {e}")

# Main execution
if __name__ == "__main__":
    # Required packages check
    required_packages = [
        "langchain", "faiss-cpu", "sentence-transformers", 
        "pypdf", "openai", "transformers", "torch"
    ]
    
    print("📦 Required packages:", ", ".join(required_packages))
    print("Install with: pip install " + " ".join(required_packages))
    print("-" * 60)
    
    # Get API key with multiple fallback options
    api_key = None
    
    # Try to get from imported config
    if 'api' in globals() and api:
        api_key = api
        print("✅ Using API key from config.py")
    
    # Try environment variable
    if not api_key:
        api_key = os.getenv('OPENROUTER_API_KEY')
        if api_key:
            print("✅ Using API key from environment variable")
    
    # Manual input as last resort
    if not api_key:
        api_key = input("Enter your OpenRouter API key: ")
    
    if not api_key:
        print("❌ API key is required!")
        exit(1)
    
    # Create documents directory if it doesn't exist
    pdf_dir = "documents"
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
        print(f"📁 Created directory: {pdf_dir}")
        print("📋 Please add your PDF files to this directory.")
    
    # Create and run the RAG chatbot
    try:
        bot = TunisiaPolytechnicRAGBot(api_key, pdf_dir)
        bot.run()
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {e}")
        print("💡 Make sure all required packages are installed")
        import traceback
        traceback.print_exc()