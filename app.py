__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import tempfile
import base64
from groq import Groq

# LangChain & AI Imports
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Multi-Tool", page_icon="🤖", layout="wide")
st.title("🤖 RAG & Vision Assistant")
st.markdown("---")

# Retrieve API Key from Secrets
groq_api_key = st.secrets.get("GROQ_API_KEY")
if not groq_api_key:
    st.error("Please add GROQ_API_KEY to your Streamlit Secrets.")
    st.stop()

# Initialize Groq Client for Vision
client = Groq(api_key=groq_api_key)

# --- HELPER FUNCTIONS ---

@st.cache_resource
def get_embeddings():
    """Loads the embedding model once and caches it."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def describe_image(image_bytes):
    """Uses Llama 3.2 Vision to describe images without Tesseract."""
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    try:
        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please describe this image in detail and extract any visible text."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Vision Error: {str(e)}"

def process_document(uploaded_file):
    """Handles PDF and Excel processing for RAG."""
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(tmp_path)
        else:
            loader = UnstructuredExcelLoader(tmp_path)
            
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        
        vector_store = Chroma.from_documents(
            documents=chunks, 
            embedding=get_embeddings(),
            collection_name="user_data"
        )
        return vector_store.as_retriever()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- SIDEBAR UI ---
with st.sidebar:
    st.header("📤 Upload Files")
    uploaded_file = st.file_uploader(
        "Upload Image, PDF, or Excel", 
        type=["pdf", "xlsx", "xls", "jpg", "png", "jpeg"]
    )
    
    if st.button("Submit & Process") and uploaded_file:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        
        if file_ext in [".jpg", ".png", ".jpeg"]:
            with st.spinner("AI is analyzing the image..."):
                desc = describe_image(uploaded_file.getvalue())
                st.session_state.image_context = desc
                st.success("Image analyzed!")
        else:
            with st.spinner("Indexing document..."):
                st.session_state.retriever = process_document(uploaded_file)
                st.success("Document ready!")

# --- MAIN CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous image description if available
if "image_context" in st.session_state:
    with st.expander("🖼️ Last Image Analysis", expanded=True):
        st.write(st.session_state.image_context)

# Chat History Display
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if user_input := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            llm = ChatGroq(api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
            
            # Context logic
            context = ""
            if "retriever" in st.session_state:
                # Get relevant snippets from RAG
                rel_docs = st.session_state.retriever.get_relevant_documents(user_input)
                context = "\n".join([doc.page_content for doc in rel_docs])
            
            # Combine with Image info if user asks
            full_prompt = f"Context: {context}\n\nImage Info: {st.session_state.get('image_context', '')}\n\nQuestion: {user_input}"
            
            response = llm.invoke(full_prompt)
            st.markdown(response.content)
            st.session_state.messages.append({"role": "assistant", "content": response.content})
            
        except Exception as e:
            st.error(f"Chat Error: {e}")
