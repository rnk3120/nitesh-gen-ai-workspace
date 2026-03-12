import os
import base64
import re
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv

# LangChain & AI Tools
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

# --- 1. CONFIG & SESSION ---
st.set_page_config(page_title="AI Auto-Analyst 2026", layout="wide", page_icon="🧠")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "df" not in st.session_state:
    st.session_state.df = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "active_mode" not in st.session_state:
    st.session_state.active_mode = "Chat"

# --- 2. ENGINES ---
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_text_sources(source, source_type):
    try:
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
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(data)
        return Chroma.from_documents(chunks, get_embeddings()).as_retriever()
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def build_smart_visuals(df, x_col, y_cols, subplots_status):
    if subplots_status.strip().lower() == "yes" or len(y_cols) > 1:
        fig = make_subplots(rows=len(y_cols), cols=1, subplot_titles=y_cols, vertical_spacing=0.1)
        for i, col in enumerate(y_cols):
            if pd.api.types.is_numeric_dtype(df[col]):
                fig.add_trace(go.Scatter(x=df[x_col], y=df[col], name=col, mode='lines+markers'), row=i+1, col=1)
            else:
                counts = df[col].value_counts().reset_index()
                fig.add_trace(go.Bar(x=counts['index'], y=counts[col], name=col), row=i+1, col=1)
        fig.update_layout(height=350 * len(y_cols), template="plotly_white", showlegend=False)
        return fig
    return px.line(df, x=x_col, y=y_cols[0], markers=True, template="plotly_white")

# --- 3. SIDEBAR (Auto-Uploader) ---
with st.sidebar:
    st.title("📂 Universal Uploader")
    file = st.file_uploader("Drop any file (Image, Excel, PDF, CSV)", type=["csv", "xlsx", "pdf", "docx", "png", "jpg", "jpeg"])
    
    if file:
        ext = file.name.split('.')[-1].lower()
        if ext in ['csv', 'xlsx']:
            st.session_state.active_mode = "📊 Data Analysis"
            st.session_state.df = pd.read_csv(file) if ext == 'csv' else pd.read_excel(file)
            st.success(f"Mode: Data Analysis ({file.name})")
        elif ext in ['png', 'jpg', 'jpeg']:
            st.session_state.active_mode = "🖼️ Vision"
            st.session_state.img_b64 = base64.b64encode(file.getvalue()).decode("utf-8")
            st.image(file, caption="Image Ready")
        elif ext in ['pdf', 'docx']:
            st.session_state.active_mode = "📄 Documents"
            with open(file.name, "wb") as f: f.write(file.getbuffer())
            st.session_state.retriever = process_text_sources(file.name, "File")
            st.success(f"Mode: Document Analysis ({file.name})")

    url_in = st.text_input("🔗 Or enter a URL (Web/YouTube)")
    if st.button("Analyze Link"):
        st.session_state.active_mode = "🌐 Web/YouTube"
        st.session_state.retriever = process_text_sources(url_in, "URL")

    if st.button("🗑️ Clear Everything"):
        for key in st.session_state.keys(): del st.session_state[key]
        st.rerun()

# --- 4. CHAT AREA ---
st.title(f"🤖 AI Analyst ({st.session_state.active_mode})")

for m in st.session_state.chat_history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "fig" in m and m["fig"]: st.plotly_chart(m["fig"], use_container_width=True)

if prompt := st.chat_input("What should I do with this file?"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        bot_res = {"role": "assistant", "content": "", "fig": None}
        
        # ROUTE 1: DATA
        if st.session_state.active_mode == "📊 Data Analysis":
            llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
            df = st.session_state.df
            expl = llm.invoke(f"Columns: {list(df.columns)}\nUser: {prompt}").content
            st.markdown(expl)
            bot_res["content"] = expl
            
            intent = llm.invoke(f"Columns: {list(df.columns)}\nUser: '{prompt}'\nReturn: X | Y1,Y2 | Subplots(Yes/No). Else: 'NONE'").content
            if "NONE" not in intent:
                try:
                    x, y, sub = intent.split("|")
                    y_list = [col.strip() for col in y.split(",") if col.strip() in df.columns]
                    fig = build_smart_visuals(df, x.strip(), y_list, sub)
                    st.plotly_chart(fig, use_container_width=True)
                    bot_res["fig"] = fig
                except: pass

        # ROUTE 2: VISION
        elif st.session_state.active_mode == "🖼️ Vision":
            # Using the stable 2026 Llama 4 Vision model
            v_llm = ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct")
            msg = HumanMessage(content=[{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{st.session_state.img_b64}"}}])
            ans = v_llm.invoke([msg]).content
            st.markdown(ans)
            bot_res["content"] = ans

        # ROUTE 3: RAG / CHAT
        else:
            t_llm = ChatGroq(model_name="llama-3.1-8b-instant")
            ctx = ""
            if st.session_state.retriever:
                docs = st.session_state.retriever.invoke(prompt)
                ctx = "\n".join([d.page_content for d in docs])
            ans = t_llm.invoke(f"Context: {ctx}\n\nQuestion: {prompt}").content
            st.markdown(ans)
            bot_res["content"] = ans

        st.session_state.chat_history.append(bot_res)
