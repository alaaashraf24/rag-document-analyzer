import streamlit as st
import PyPDF2
import docx
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import os
import tempfile
from io import BytesIO

# Configure the page
st.set_page_config(
    page_title="Document Analysis Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with harmonious blue palette
st.markdown("""
<style>
    /* Color Palette Variables */
    :root {
        --dark-blue: #012a4a;      /* Darkest blue */
        --navy-blue: #013a63;      /* Deep navy */
        --medium-blue: #01497c;    /* Medium dark blue */
        --steel-blue: #014f86;     /* Steel blue */
        --ocean-blue: #2a6f97;     /* Ocean blue */
        --teal-blue: #2c7da0;      /* Teal blue */
        --sky-blue: #468faf;       /* Sky blue */
        --light-blue: #61a5c2;     /* Light blue */
        --powder-blue: #89c2d9;    /* Powder blue */
        --pale-blue: #a9d6e5;      /* Pale blue */
    }
    
    /* Main background - Light blue gradient */
    .stApp {
        background: linear-gradient(135deg, var(--pale-blue) 0%, rgba(255,255,255,0.9) 50%, var(--powder-blue) 100%);
        min-height: 100vh;
    }
    
    /* Sidebar styling - Keep dark but readable */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--steel-blue) 0%, var(--ocean-blue) 100%);
        border-right: 2px solid var(--light-blue);
    }
    
    /* Main container - Pure white background */
    .main .block-container {
        background: rgba(255, 255, 255, 1.0);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(1, 42, 74, 0.15);
        backdrop-filter: blur(10px);
        margin-top: 2rem;
        border: 2px solid var(--light-blue);
    }
    
    /* Headers styling - Dark blue on white background */
    h1 {
        color: var(--steel-blue) !important;
        background: none !important;
        -webkit-text-fill-color: var(--steel-blue) !important;
        text-align: center;
        font-size: 3rem !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
    }
    
    h2 {
        color: var(--dark-blue) !important;
        font-weight: 600;
        border-bottom: 3px solid var(--ocean-blue);
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    h3 {
        color: var(--medium-blue) !important;
        font-weight: 500;
    }
    
    /* All text in main area - Dark for readability */
    .main .block-container {
        color: #1a1a1a !important;
    }
    
    .main .block-container p {
        color: #1a1a1a !important;
        line-height: 1.6;
    }
    
    .main .block-container .stMarkdown {
        color: #1a1a1a !important;
    }
    
    .main .block-container div {
        color: #1a1a1a !important;
    }
    
    /* Sidebar headers and text - Light colors on dark background */
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
        color: white !important;
        background: none !important;
        -webkit-text-fill-color: white !important;
    }
    
    .css-1d391kg .stMarkdown {
        color: var(--pale-blue) !important;
    }
    
    .css-1d391kg .stMarkdown p {
        color: var(--pale-blue) !important;
    }
    
    .css-1d391kg .stMarkdown div {
        color: var(--pale-blue) !important;
    }
    
    /* Button styling - Updated to make all buttons consistent */
    .stButton > button {
        background: linear-gradient(45deg, var(--ocean-blue), var(--teal-blue));
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(42, 111, 151, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(42, 111, 151, 0.4);
        background: linear-gradient(45deg, var(--teal-blue), var(--sky-blue));
    }
    
    /* Specific styling for Process Documents button in sidebar */
    .css-1d391kg .stButton > button {
        background: linear-gradient(45deg, var(--ocean-blue), var(--teal-blue));
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(42, 111, 151, 0.3);
        width: 100%;
    }
    
    .css-1d391kg .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(42, 111, 151, 0.4);
        background: linear-gradient(45deg, var(--teal-blue), var(--sky-blue));
    }
    
    /* File uploader styling */
    .stFileUploader {
        background: linear-gradient(135deg, var(--pale-blue) 0%, rgba(255,255,255,0.8) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px dashed var(--ocean-blue);
        margin: 1rem 0;
    }
    
    .stFileUploader label {
        color: var(--steel-blue) !important;
        font-weight: 600;
    }
    
    /* Chat messages */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        backdrop-filter: blur(5px);
        border-left: 4px solid var(--ocean-blue);
        color: #2c3e50 !important;
    }
    
    .stChatMessage [data-testid="chatAvatarIcon-user"] {
        background: var(--ocean-blue);
    }
    
    .stChatMessage [data-testid="chatAvatarIcon-assistant"] {
        background: var(--teal-blue);
    }
    
    /* Success/Info messages */
    .stSuccess {
        background: linear-gradient(45deg, var(--teal-blue), var(--sky-blue)) !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
    }
    
    .stInfo {
        background: linear-gradient(45deg, var(--ocean-blue), var(--light-blue)) !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
    }
    
    .stError {
        background: linear-gradient(45deg, #d63031, #e17055) !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
    }
    
    .stWarning {
        background: linear-gradient(45deg, #fdcb6e, #e17055) !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid var(--light-blue);
        padding: 0.5rem 1rem;
        background: rgba(255, 255, 255, 0.95);
        color: var(--dark-blue);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--ocean-blue);
        box-shadow: 0 0 10px rgba(42, 111, 151, 0.3);
    }
    
    /* Chat input dark theme styling */
    .stChatInput {
        background: var(--dark-blue) !important;
        border-radius: 15px !important;
        padding: 1rem !important;
        margin: 1rem 0 !important;
        border: 2px solid var(--ocean-blue) !important;
    }
    
    .stChatInput > div {
        background: var(--dark-blue) !important;
    }
    
    .stChatInput > div > div {
        background: var(--dark-blue) !important;
    }
    
    .stChatInput > div > div > div {
        background: var(--dark-blue) !important;
    }
    
    .stChatInput > div > div > div > div {
        background: var(--dark-blue) !important;
    }
    
    .stChatInput > div > div > div > div > input {
        background: var(--navy-blue) !important;
        border: 2px solid var(--ocean-blue) !important;
        border-radius: 25px !important;
        color: white !important;
        padding: 0.75rem 1.5rem !important;
        font-size: 1rem !important;
    }
    
    .stChatInput > div > div > div > div > input::placeholder {
        color: white !important;
        opacity: 0.8 !important;
    }
    
    .stChatInput > div > div > div > div > input:focus {
        border-color: var(--teal-blue) !important;
        box-shadow: 0 0 15px rgba(42, 111, 151, 0.4) !important;
        color: white !important;
    }
    
    /* Chat input button styling */
    .stChatInput button {
        background: linear-gradient(45deg, var(--ocean-blue), var(--teal-blue)) !important;
        border: none !important;
        border-radius: 50% !important;
        color: white !important;
        transition: all 0.3s ease !important;
    }
    
    .stChatInput button:hover {
        background: linear-gradient(45deg, var(--teal-blue), var(--sky-blue)) !important;
        transform: scale(1.1) !important;
        box-shadow: 0 4px 15px rgba(42, 111, 151, 0.4) !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(45deg, var(--ocean-blue), var(--teal-blue)) !important;
        color: white !important;
        border-radius: 10px !important;
        font-weight: 600;
    }
    
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.98) !important;
        border-radius: 0 0 10px 10px;
        color: #2c3e50 !important;
    }
    
    /* Progress bar */
    .stProgress .st-bo {
        background: linear-gradient(45deg, var(--ocean-blue), var(--teal-blue)) !important;
    }
    
    /* Custom title styling - Blue but readable */
    .main-title {
        color: var(--steel-blue) !important;
        background: none !important;
        -webkit-background-clip: none !important;
        -webkit-text-fill-color: var(--steel-blue) !important;
        text-align: center;
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: none;
    }
    
    .subtitle {
        text-align: center;
        color: var(--medium-blue) !important;
        font-size: 1.2rem;
        font-weight: 400;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    /* Spinner styling */
    .stSpinner {
        color: var(--ocean-blue) !important;
    }
    
    /* Markdown text in main area */
    .main .block-container .stMarkdown {
        color: #2c3e50;
    }
    
    /* Ensure all text in main area is readable */
    .main .block-container * {
        color: #2c3e50;
    }
    
    /* Override for headers to keep gradient */
    .main .block-container h1, 
    .main .block-container h2, 
    .main .block-container h3 {
        color: var(--steel-blue) !important;
    }
    
    /* Special styling for the main title */
    .main-title {
        background: linear-gradient(45deg, var(--ocean-blue), var(--sky-blue)) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }
    
    /* Footer styling */
    .main .block-container hr + div {
        color: var(--medium-blue) !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Gemini AI
try:
    genai.configure(api_key=st.secrets['GOOGLE_API_KEY'])
    model = 'models/embedding-001'
    chat_model = genai.GenerativeModel('gemini-1.5-flash')
except:
    st.error("🔑 Please configure your Google API key in Streamlit secrets")
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
        st.error(f"❌ Error reading PDF: {str(e)}")
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
        st.error(f"❌ Error reading DOCX: {str(e)}")
        return ""

def extract_from_txt(file):
    """Extract text from TXT file"""
    try:
        text = file.read().decode('utf-8')
        return text
    except Exception as e:
        st.error(f"❌ Error reading TXT: {str(e)}")
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
        st.error(f"❌ Unsupported file type: {file_type}")
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
        st.error(f"❌ Error creating vector database: {str(e)}")
        return None

def get_relevant_context(vector_db, query, k=5):
    """Retrieve relevant context from vector database"""
    if vector_db:
        try:
            docs = vector_db.similarity_search(query, k=k)
            return docs
        except Exception as e:
            st.error(f"❌ Error retrieving context: {str(e)}")
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
        st.error(f"❌ Error generating response: {str(e)}")
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

# Custom title
st.markdown('<h1 class="main-title">📚 RAG Document Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Transform your documents into intelligent conversations</p>', unsafe_allow_html=True)

# Sidebar for file upload and controls
with st.sidebar:
    st.markdown("## 📁 Document Management")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "🔄 Upload Documents",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        help="Upload PDF, DOCX, or TXT files for analysis"
    )
    
    if uploaded_files:
        st.markdown(f"**📋 {len(uploaded_files)} file(s) selected:**")
        for file in uploaded_files:
            st.markdown(f"• {file.name}")
    
    # Process documents button
    if st.button("🔄 Process Documents", disabled=not uploaded_files):
        with st.spinner("🔄 Processing documents..."):
            all_texts = []
            processed_files = []
            
            progress_bar = st.progress(0)
            for i, file in enumerate(uploaded_files):
                st.info(f"📄 Processing: {file.name}")
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
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
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
        st.markdown("## 📋 Processed Files")
        for file_name in st.session_state.processed_files:
            st.markdown(f"✅ {file_name}")
    
    # Control buttons
    st.markdown("## 🛠️ Controls")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.success("✅ Chat cleared!")
    
    with col2:
        if st.button("🔄 Reset All"):
            st.session_state.messages = []
            st.session_state.vector_db = None
            st.session_state.processed_files = []
            st.success("✅ All reset!")

# Main chat interface
st.markdown("## 💬 Document Q&A")

# Display ethical use guidelines
with st.expander("📋 Ethical Use Guidelines", expanded=False):
    st.markdown("""
    ### 🎯 **This tool is designed for legitimate research and learning purposes:**
    
    #### ✅ **Appropriate Uses:**
    - 📚 Research and literature review
    - 📝 Document summarization and analysis
    - 🎓 Learning from educational materials
    - 📁 Content organization and note-taking
    - 🔍 Understanding complex documents
    
    #### ❌ **Inappropriate Uses:**
    - 🚫 Academic dishonesty or plagiarism
    - 📋 Copying content without attribution
    - ⚠️ Circumventing learning processes
    - ⚖️ Violating copyright or terms of use
    
    ### 💡 **Remember:** Always cite your sources and use this tool to enhance your understanding, not replace your own work.
    """)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
user_query = st.chat_input("💭 Ask a question about your documents...")

if user_query:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("user"):
        st.write(user_query)
    
    # Generate AI response
    with st.chat_message("assistant"):
        if not st.session_state.vector_db:
            st.warning("⚠️ Please upload and process documents first to get contextual answers.")
            response = "I'd be happy to help analyze your documents, but it looks like you haven't uploaded any documents yet. Please use the sidebar to upload PDF, DOCX, or TXT files, then click '🔄 Process Documents' to get started."
            st.write(response)
        else:
            # Get relevant context
            with st.spinner("🤔 Thinking..."):
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
<div style="text-align: center; color: #7f8c8d; font-style: italic;">
    <p>🚀 <strong>RAG-Powered Document Analysis</strong> | Built with Streamlit, Google Gemini & FAISS</p>
    <p>💡 This application is designed to assist with legitimate document analysis and research.</p>
    <p>📚 Always ensure you comply with your institution's academic integrity policies and properly cite all sources.</p>
</div>
""", unsafe_allow_html=True)