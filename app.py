__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import tempfile

# 2026 Compatible Imports
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader, UnstructuredImageLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# THESE ARE THE CRITICAL CHANGES FOR 2026
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- PAGE SETUP ---
st.set_page_config(page_title="Multi-Format RAG Assistant", page_icon="📁", layout="wide")
st.title("📁 Document AI Assistant (PDF, Excel, Images)")

# --- ACCESS SECRETS ---
groq_api_key = st.secrets.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("Please add your GROQ_API_KEY to Streamlit Secrets!")
    st.stop()

# --- INITIALIZE EMBEDDINGS ---
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- FILE PROCESSING ROUTER ---
def process_file(uploaded_file):
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    # Save file to a safe temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        # Route to appropriate loader based on extension
        if file_extension == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif file_extension in [".xlsx", ".xls"]:
            loader = UnstructuredExcelLoader(tmp_path, mode="elements")
        elif file_extension in [".png", ".jpg", ".jpeg"]:
            loader = UnstructuredImageLoader(tmp_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        docs = loader.load()
        
        # Split text into bite-sized chunks for the LLM
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)

        # Store in Vector Database (RAM)
        vector_store = Chroma.from_documents(
            documents=chunks, 
            embedding=load_embeddings()
        )
        return vector_store.as_retriever()

    finally:
        # Clean up temp file immediately after reading
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- SIDEBAR & FILE UPLOAD ---
with st.sidebar:
    st.header("File Upload Center")
    uploaded_file = st.file_uploader(
        "Upload your document", 
        type=["pdf", "xlsx", "xls", "png", "jpg", "jpeg"]
    )
    
    if st.button("Analyze Document") and uploaded_file:
        with st.spinner("Reading and indexing the file... This might take a moment for images/Excel..."):
            try:
                st.session_state.retriever = process_file(uploaded_file)
                st.success("Analysis Complete! Start chatting.")
            except Exception as e:
                st.error(f"Processing Error: {e}")

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User prompt
if prompt := st.chat_input("Ask a question about your uploaded file..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "retriever" in st.session_state:
        with st.chat_message("assistant"):
            try:
                llm = ChatGroq(
                    groq_api_key=groq_api_key, 
                    model_name="llama3-8b-8192",
                    temperature=0.2
                )
                
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state.retriever
                )
                
                response = qa_chain.invoke(prompt)
                answer = response["result"]
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error with Groq API: {e}")
    else:
        st.info("👈 Please upload and 'Analyze' a file using the sidebar first.")
