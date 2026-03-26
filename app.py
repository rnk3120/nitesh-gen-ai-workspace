import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# --- PAGE SETUP ---
st.set_page_config(page_title="Groq RAG Assistant", page_icon="📄", layout="wide")
st.title("📄 High-Speed PDF Chat (Groq + RAG)")

# --- ACCESS SECRETS ---
# Note: Set 'GROQ_API_KEY' in Streamlit Cloud -> Settings -> Secrets
groq_api_key = st.secrets.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("Please add your GROQ_API_KEY to Streamlit Secrets!")
    st.stop()

# --- INITIALIZE EMBEDDINGS ---
@st.cache_resource
def load_embeddings():
    # Downloads a small model once and caches it to save memory
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- PDF PROCESSING LOGIC ---
def process_pdf(uploaded_file):
    # 1. Create a safe temporary file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        # 2. Load and Split PDF
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)

        # 3. Create Vector Store in RAM
        vector_store = Chroma.from_documents(
            documents=chunks, 
            embedding=load_embeddings()
        )
        return vector_store.as_retriever()
    
    finally:
        # 4. Cleanup the temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- SIDEBAR & FILE UPLOAD ---
with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")
    
    if st.button("Process Document") and uploaded_file:
        with st.spinner("Analyzing PDF..."):
            st.session_state.retriever = process_pdf(uploaded_file)
            st.success("Analysis Complete!")

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question about the PDF..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
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
        st.info("👈 Please upload and 'Process' a PDF to start chatting.")
