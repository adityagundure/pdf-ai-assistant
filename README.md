# pdf-ai-assistant

An AI-powered application that reads PDF documents and generates accurate, context-aware answers to user questions. This tool simplifies information retrieval from large documents using modern NLP models.

# PDF Chatbot (Streamlit + LangChain)

A simple Streamlit app that lets you chat with one or more uploaded PDF documents.  
It uses LangChain to load PDFs, split them into chunks, embed them, store them in FAISS, and run a conversational retrieval chain with OpenAI.

## Features
- Upload multiple PDF files
- Chunking of text for better retrieval
- FAISS vector store for fast semantic search
- Conversational retrieval with chat history
- Streamlit UI for easy usage

## Technologies
- Python
- Streamlit
- LangChain
- FAISS
- OpenAI (via `OpenAIEmbeddings` and `ChatOpenAI`)

## Files
.
├── app.py     # Main application file
└── README.md

## Setup

1. Create a Python environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows

pip install -r requirements.txt
export OPENAI_API_KEY="your_openai_api_key_here"   # macOS / Linux
setx OPENAI_API_KEY "your_openai_api_key_here"     # Windows (or use env vars in your runner)
streamlit run app.py

