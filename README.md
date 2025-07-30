# 📚 RAG Document Analyzer

A powerful **Retrieval-Augmented Generation (RAG)** application that transforms your documents into intelligent, interactive conversations. Built with Streamlit, Google Gemini AI, and FAISS vector database for accurate document analysis and question-answering.

## 🌟 Features

- **Multi-Format Support**: Process PDF, DOCX, and TXT files
- **Intelligent Q&A**: Ask questions and get contextual answers from your documents
- **RAG Architecture**: Combines document retrieval with AI generation for accurate responses
- **Beautiful UI**: Modern, responsive interface with intuitive design
- **Real-time Processing**: Stream responses for better user experience
- **Ethical Guidelines**: Built-in safeguards for responsible AI usage

## 🔧 Technology Stack

- **Frontend**: Streamlit
- **AI Model**: Google Gemini 1.5 Flash
- **Embeddings**: Google Generative AI Embeddings
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Document Processing**: PyPDF2, python-docx
- **Text Processing**: LangChain

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Google API Key for Gemini AI

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/alaaashraf24/rag-document-analyzer.git
   cd rag-document-analyzer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your Google API Key**
   
   **Option 1: Streamlit Secrets (Recommended for deployment)**
   Create a `.streamlit/secrets.toml` file:
   ```toml
   GOOGLE_API_KEY = "your_google_api_key_here"
   ```

   **Option 2: Environment Variable**
   Create a `.env` file:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

### Getting Your Google API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated key and use it in your configuration

## 📖 How to Use

1. **Upload Documents**: Use the sidebar to upload PDF, DOCX, or TXT files
2. **Process Documents**: Click "🔄 Process Documents" to create embeddings
3. **Ask Questions**: Type your questions in the chat interface
4. **Get Answers**: Receive contextual answers based on your documents

### Example Usage

```
Upload: research_paper.pdf, meeting_notes.docx
Question: "What are the main findings in the research paper?"
Answer: [AI provides summary based on document content]
```

## 🎯 Use Cases

- **📚 Research & Literature Review**: Analyze academic papers and research documents
- **📝 Document Summarization**: Extract key insights from lengthy documents
- **🎓 Educational Support**: Understand complex materials and textbooks
- **📁 Content Organization**: Query and organize large document collections
- **🔍 Information Extraction**: Find specific information across multiple documents

## ⚡ Key Features Explained

### RAG Architecture
The application uses Retrieval-Augmented Generation to:
1. **Chunk Documents**: Break documents into manageable pieces
2. **Create Embeddings**: Generate vector representations of text chunks
3. **Store in FAISS**: Use efficient similarity search
4. **Retrieve Context**: Find relevant chunks for user queries
5. **Generate Answers**: Use Google Gemini with retrieved context

### Ethical AI Usage
- Built-in guidelines for responsible usage
- Designed to enhance learning, not replace it
- Encourages proper citation and academic integrity
- Prevents misuse for academic dishonesty

## 🛠️ Configuration

### Supported File Types
- **PDF**: `.pdf` files
- **Word Documents**: `.docx` files  
- **Text Files**: `.txt` files

### Customizable Parameters
- **Chunk Size**: 1000 characters (configurable in code)
- **Chunk Overlap**: 200 characters
- **Similarity Search**: Top 5 relevant chunks (k=5)
- **Model**: Google Gemini 1.5 Flash

## 📁 Project Structure

```
rag-document-analyzer/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .streamlit/
│   └── secrets.toml      # Streamlit secrets (create this)
└── .env                  # Environment variables (optional)
```

## 🔒 Security & Privacy

- **API Keys**: Store securely using Streamlit secrets or environment variables
- **Document Privacy**: Documents are processed locally and not stored permanently  
- **No Data Retention**: Chat history is session-based only
- **Secure Processing**: Uses Google's secure AI APIs

## 🚀 Deployment

### Deploy on Streamlit Cloud

1. Push your code to GitHub
2. Connect your GitHub repo to [Streamlit Cloud](https://streamlit.io/cloud)
3. Add your `GOOGLE_API_KEY` in the Streamlit Cloud secrets
4. Deploy with one click!

### Local Development

```bash
# Install in development mode
pip install -e .

# Run with debug mode
streamlit run app.py --logger.level=debug
```

## 📊 System Requirements

- **Memory**: Minimum 4GB RAM (8GB recommended for large documents)
- **Storage**: 1GB free space for dependencies
- **Internet**: Required for Google AI API calls
- **Browser**: Modern web browser (Chrome, Firefox, Safari, Edge)

**Built with ❤️ using Streamlit and Google Gemini AI**

*Transform your documents into intelligent conversations today!*