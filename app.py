import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

st.title("Chatbot on PDF")

os.environ["OPENAI_API_KEY"] = "your openai api key here"

# Session state for conversation
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# File uploader
uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    documents = []

    # Load all PDFs
    for file in uploaded_files:
        with open(file.name, "wb") as f:
            f.write(file.read())
        loader = PyPDFLoader(file.name)
        documents.extend(loader.load())

     # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

     # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Store in FAISS
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Create conversational retrieval chain
    st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

# Chat UI
if st.session_state.qa_chain:
    user_input = st.chat_input("Ask something about your PDFs...")
    if user_input:
        response = st.session_state.qa_chain(
            {"question": user_input, "chat_history": st.session_state.chat_history}
        )    

        # Save chat history
        st.session_state.chat_history.append((user_input, response["answer"]))

    # Display chat messages
    for q, a in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)        
