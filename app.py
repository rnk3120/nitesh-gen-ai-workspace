import os
import base64
import re
import shutil
import pandas as pd
import streamlit as st
import plotly.express as px
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi

# --- 1. CONFIG & SESSION ---
st.set_page_config(page_title="Secure AI Analyst 2026", layout="wide", page_icon="🔐")

# Initialize session state keys
for key in ["chat_history", "df", "retriever", "active_mode", "img_b64"]:
    if key not in st.session_state:
        st.session_state[key] = [] if "history" in key else None

if "active_mode" not in st.session_state or st.session_state.active_mode is None:
    st.session_state.active_mode = "Chat"

# --- 2. ENGINES ---
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_text_sources(source, source_type):
    try:
        # CLEANUP: Delete any old database folders to prevent data mixing
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
        
        # LOAD
        if source_type == "URL":
            if "youtu" in source:
                vid_id = re.search(r'(?:v=|\/|be\/)([0-9A-Za-z_-]{11})', source).group(1)
                transcript = YouTubeTranscriptApi.get_transcript(vid_id)
                data = [Document(page_content=" ".join([t['text'] for t in transcript]))]
            else:
                data = WebBaseLoader(source).load()
        else:
            loader = PyPDFLoader(source) if source.endswith(".pdf") else Docx2txtLoader(source) if source.endswith(".docx") else TextLoader(source)
            data = loader.load()
        
        # SPLIT & INDEX (Ephemeral/In-Memory for security)
        chunks = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50).split_documents(data)
        vector_db = Chroma.from_documents(
            documents=chunks, 
            embedding=get_embeddings()
        )
        return vector_db.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# --- 3. SIDEBAR (Auto-Routing) ---
with st.sidebar:
    st.title("📂 Secure Uploader")
    file = st.file_uploader("Upload File", type=["csv", "xlsx", "pdf", "docx", "png", "jpg", "jpeg"])
    
    if file:
        ext = file.name.split('.')[-1].lower()
        if ext in ['csv', 'xlsx']:
            st.session_state.active_mode = "📊 Data Analysis"
            st.session_state.df = pd.read_csv(file) if ext == 'csv' else pd.read_excel(file)
        elif ext in ['png', 'jpg', 'jpeg']:
            st.session_state.active_mode = "🖼️ Vision"
            st.session_state.img_b64 = base64.b64encode(file.getvalue()).decode("utf-8")
        elif ext in ['pdf', 'docx']:
            st.session_state.active_mode = "📄 Documents"
            with open("temp_file", "wb") as f: f.write(file.getbuffer())
            st.session_state.retriever = process_text_sources("temp_file", "File")

    if st.button("🗑️ Reset Application"):
        if os.path.exists("./chroma_db"): shutil.rmtree("./chroma_db")
        st.session_state.clear()
        st.rerun()

# --- 4. CHAT AREA ---
st.title(f"🤖 AI {st.session_state.active_mode}")

# Display history
for m in st.session_state.chat_history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask about your file..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        bot_res = {"role": "assistant", "content": ""}
        
        # DATA MODE
        if st.session_state.active_mode == "📊 Data Analysis" and st.session_state.df is not None:
            llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
            df_info = f"Columns: {list(st.session_state.df.columns)}"
            ans = llm.invoke(f"{df_info}\nQuestion: {prompt}").content
            
        # VISION MODE
        elif st.session_state.active_mode == "🖼️ Vision":
            llm = ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct")
            msg = HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{st.session_state.img_b64}"}}
            ])
            ans = llm.invoke([msg]).content

        # DOCUMENT MODE (Anti-Hallucination)
        else:
            llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
            context = ""
            if st.session_state.retriever:
                relevant_docs = st.session_state.retriever.invoke(prompt)
                context = "\n".join([d.page_content for d in relevant_docs])
            
            strict_prompt = f"""
            SYSTEM: Answer the question using ONLY the context below. 
            If the answer is not in the context, say: "I cannot find this in your PDF."
            Do NOT mention any copyright or other companies.
            
            CONTEXT: {context}
            USER: {prompt}
            """
            ans = llm.invoke(strict_prompt).content

        st.markdown(ans)
        bot_res["content"] = ans
        st.session_state.chat_history.append(bot_res)
