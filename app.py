import os
import base64
import shutil
import pandas as pd
import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage

# --- 1. INITIAL CONFIG ---
st.set_page_config(page_title="Universal AI Studio 2026", layout="wide", page_icon="⚡")

# Verify API Key
if "GROQ_API_KEY" not in st.secrets:
    st.error("Please add your GROQ_API_KEY to the Streamlit Secrets dashboard.")
    st.stop()

gen_api_key = st.secrets["GROQ_API_KEY"]

# Initialize Session State
for key in ["chat_history", "df", "retriever", "active_mode", "img_b64"]:
    if key not in st.session_state:
        st.session_state[key] = [] if "history" in key else None

# --- 2. THE ENGINES ---
@st.cache_resource
def get_embeddings():
    # Downloads once and keeps in memory for speed
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_pdf(file_path):
    """Wipes old DB and creates a fresh one for the new PDF"""
    if os.path.exists("./chroma_db"): 
        shutil.rmtree("./chroma_db")
    
    loader = PyPDFLoader(file_path)
    data = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(data)
    
    # Create in-memory vector database
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=get_embeddings()
    )
    return vector_db.as_retriever(search_kwargs={"k": 3})

# --- 3. SIDEBAR (Uploader & Reset) ---
with st.sidebar:
    st.title("📂 File Center")
    uploaded_file = st.file_uploader("Upload PDF, Image, or CSV", type=["pdf", "csv", "png", "jpg", "jpeg"])
    
    if uploaded_file:
        ext = uploaded_file.name.split('.')[-1].lower()
        
        if ext in ['png', 'jpg', 'jpeg']:
            st.session_state.active_mode = "IMAGE"
            st.session_state.img_b64 = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
            st.success("Image uploaded!")
            
        elif ext == 'pdf':
            st.session_state.active_mode = "PDF"
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.retriever = process_pdf("temp.pdf")
            st.success("PDF processed!")
            
        elif ext == 'csv':
            st.session_state.active_mode = "CSV"
            st.session_state.df = pd.read_csv(uploaded_file)
            st.success("CSV loaded!")

    st.divider()
    if st.button("🗑️ Clear Everything"):
        if os.path.exists("./chroma_db"): shutil.rmtree("./chroma_db")
        st.session_state.clear()
        st.rerun()

# --- 4. CHAT AREA ---
st.title("🤖 Intelligent Assistant")

# Display previous messages
for m in st.session_state.chat_history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask me about the file or anything else..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # We use Llama 3.3 for the main brain
        llm_main = ChatGroq(api_key=gen_api_key, model="llama-3.3-70b-versatile", temperature=0.3)
        
        # 🟢 LOGIC: Is this a General question or a File question?
        # A quick check to ensure we don't hallucinate context
        intent_check = llm_main.invoke(
            f"If the question is general (like coding help, greetings, or web dev) answer 'GENERAL'. "
            f"If it asks for data from a file, answer 'FILE'. Question: {prompt}"
        ).content.upper()

        # 🟢 ROUTING
        if "GENERAL" in intent_check or st.session_state.active_mode is None:
            # Route: General Brain
            response = llm_main.invoke(prompt).content
        
        elif st.session_state.active_mode == "IMAGE":
            # Route: Vision Engine (Llama 4 Scout)
            v_llm = ChatGroq(api_key=gen_api_key, model="meta-llama/llama-4-scout-17b-16e-instruct")
            msg = HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{st.session_state.img_b64}"}}
            ])
            response = v_llm.invoke([msg]).content

        elif st.session_state.active_mode == "PDF":
            # Route: Document Search (RAG)
            context = ""
            if st.session_state.retriever:
                docs = st.session_state.retriever.invoke(prompt)
                context = "\n".join([d.page_content for d in docs])
            
            strict_prompt = f"Using ONLY this PDF context:\n{context}\n\nQuestion: {prompt}"
            response = llm_main.invoke(strict_prompt).content
            
        else:
            response = "I'm ready! Please upload a file to begin specific analysis."

        st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
