import streamlit as st
import PyPDF2
import docx
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import os
import tempfile
from io import BytesIO

# Configure the page
st.set_page_config(
    page_title="Document Analysis Assistant",
    page_icon="📚",
    layout="wide"
)

# Initialize Gemini AI
try:
    genai.configure(api_key=st.secrets['GOOGLE_API_KEY'])
    model = 'models/embedding-001'
    chat_model = genai.GenerativeModel('gemini-1.5-flash')
except:
    st.error("Please configure your Google API key in Streamlit secrets")
    st.stop()

# Document extraction functions
def extract_from_pdf(file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def extract_from_docx(file):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return ""

def extract_from_txt(file):
    """Extract text from TXT file"""
    try:
        text = file.read().decode('utf-8')
        return text
    except Exception as e:
        st.error(f"Error reading TXT: {str(e)}")
        return ""

def extract_text_from_file(file):
    """Extract text based on file type"""
    file_type = file.name.split('.')[-1].lower()
    
    if file_type == 'pdf':
        return extract_from_pdf(file)
    elif file_type == 'docx':
        return extract_from_docx(file)
    elif file_type == 'txt':
        return extract_from_txt(file)
    else:
        st.error(f"Unsupported file type: {file_type}")
        return ""

def create_embeddings():
    """Create Google Gemini embeddings"""
    return GoogleGenerativeAIEmbeddings(
        model=model, 
        task_type="SEMANTIC_SIMILARITY"
    )

def create_vector_database(texts):
    """Create FAISS vector database from texts"""
    try:
        embeddings = create_embeddings()
        vector_db = FAISS.from_texts(texts, embeddings)
        return vector_db
    except Exception as e:
        st.error(f"Error creating vector database: {str(e)}")
        return None

def get_relevant_context(vector_db, query, k=5):
    """Retrieve relevant context from vector database"""
    if vector_db:
        try:
            docs = vector_db.similarity_search(query, k=k)
            return docs
        except Exception as e:
            st.error(f"Error retrieving context: {str(e)}")
            return []
    return []

def generate_response(prompt):
    """Generate response using Gemini"""
    try:
        response = chat_model.generate_content(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        yield "Sorry, I encountered an error while processing your request."

# Initialize session state
if "documents" not in st.session_state:
    st.session_state.documents = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# Main UI
st.title("📚 Document Analysis Assistant")
st.markdown("*An ethical tool for document research and analysis*")

# Sidebar for file upload and controls
with st.sidebar:
    st.header("📁 Document Management")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        help="Upload PDF, DOCX, or TXT files for analysis"
    )
    
    if uploaded_files:
        st.write(f"**{len(uploaded_files)} file(s) selected:**")
        for file in uploaded_files:
            st.write(f"• {file.name}")
    
    # Process documents button
    if st.button("🔄 Process Documents", disabled=not uploaded_files):
        with st.spinner("Processing documents..."):
            all_texts = []
            processed_files = []
            
            for file in uploaded_files:
                st.info(f"Processing: {file.name}")
                text = extract_text_from_file(file)
                
                if text.strip():
                    # Split text into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len
                    )
                    chunks = text_splitter.split_text(text)
                    all_texts.extend(chunks)
                    processed_files.append(file.name)
            
            if all_texts:
                # Create vector database
                st.session_state.vector_db = create_vector_database(all_texts)
                st.session_state.processed_files = processed_files
                
                if st.session_state.vector_db:
                    st.success(f"✅ Processed {len(processed_files)} documents successfully!")
                    st.info(f"📊 Created {len(all_texts)} text chunks for analysis")
                else:
                    st.error("❌ Failed to create vector database")
            else:
                st.error("❌ No valid text found in uploaded documents")
    
    # Display processed files
    if st.session_state.processed_files:
        st.header("📋 Processed Files")
        for file_name in st.session_state.processed_files:
            st.write(f"✓ {file_name}")
    
    # Control buttons
    st.header("🛠️ Controls")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.success("Chat history cleared!")
    
    if st.button("🗂️ Reset All"):
        st.session_state.messages = []
        st.session_state.vector_db = None
        st.session_state.processed_files = []
        st.success("All data reset!")

# Main chat interface
st.header("💬 Document Q&A")

# Display ethical use guidelines
with st.expander("📋 Ethical Use Guidelines", expanded=False):
    st.markdown("""
    **This tool is designed for legitimate research and learning purposes:**
    
    ✅ **Appropriate Uses:**
    - Research and literature review
    - Document summarization and analysis
    - Learning from educational materials
    - Content organization and note-taking
    - Understanding complex documents
    
    ❌ **Inappropriate Uses:**
    - Academic dishonesty or plagiarism
    - Copying content without attribution
    - Circumventing learning processes
    - Violating copyright or terms of use
    
    **Remember:** Always cite your sources and use this tool to enhance your understanding, not replace your own work.
    """)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
user_query = st.chat_input("Ask a question about your documents...")

if user_query:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("user"):
        st.write(user_query)
    
    # Generate AI response
    with st.chat_message("assistant"):
        if not st.session_state.vector_db:
            st.warning("⚠️ Please upload and process documents first to get contextual answers.")
            response = "I'd be happy to help analyze your documents, but it looks like you haven't uploaded any documents yet. Please use the sidebar to upload PDF, DOCX, or TXT files, then click 'Process Documents' to get started."
            st.write(response)
        else:
            # Get relevant context
            relevant_docs = get_relevant_context(st.session_state.vector_db, user_query, k=5)
            
            # Build context from relevant documents
            context = ""
            if relevant_docs:
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Create prompt with ethical guidelines
            system_prompt = """You are a document analysis assistant designed to help users understand and analyze their documents ethically. 

Guidelines:
- Provide accurate, helpful analysis based on the provided documents
- Encourage critical thinking and learning
- Always remind users to cite sources appropriately
- Do not provide answers that could facilitate academic dishonesty
- Focus on helping users understand concepts rather than just providing answers

Based on the following document excerpts, please answer the user's question:

"""
            
            full_prompt = system_prompt + f"\nContext from documents:\n{context}\n\nUser question: {user_query}\n\nResponse:"
            
            # Stream response
            response_placeholder = st.empty()
            full_response = ""
            
            for chunk in generate_response(full_prompt):
                full_response += chunk
                response_placeholder.write(full_response)
    
    # Add assistant response to chat history
    if 'full_response' in locals():
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Footer with additional information
st.markdown("---")
st.markdown("""
**Note:** This application is designed to assist with legitimate document analysis and research. 
Always ensure you comply with your institution's academic integrity policies and properly cite all sources.
""")