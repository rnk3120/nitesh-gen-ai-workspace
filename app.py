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
st.set_page_config(page_title="Specialized AI Workspace 2026", layout="wide", page_icon="🚀")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "df" not in st.session_state:
    st.session_state.df = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None

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
    
    y_col = y_cols[0]
    return px.line(df, x=x_col, y=y_col, markers=True, template="plotly_white")

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    mode = st.radio("Choose Mode", ["📊 Data Analysis", "📄 Documents", "🌐 Web/YouTube", "🖼️ Images"])
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    if mode == "📊 Data Analysis":
        up = st.file_uploader("Upload Excel/CSV", type=["csv", "xlsx"])
        if up: st.session_state.df = pd.read_csv(up) if up.name.endswith('.csv') else pd.read_excel(up)

    elif mode == "📄 Documents":
        doc_up = st.file_uploader("Upload PDF/Doc", type=["pdf", "docx", "txt"])
        if doc_up:
            with open(doc_up.name, "wb") as f: f.write(doc_up.getbuffer())
            st.session_state.retriever = process_text_sources(doc_up.name, "File")

    elif mode == "🌐 Web/YouTube":
        url_in = st.text_input("Enter URL")
        if st.button("Read Link"):
            st.session_state.retriever = process_text_sources(url_in, "URL")

    elif mode == "🖼️ Images":
        img_up = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
        if img_up:
            st.session_state.img_b64 = base64.b64encode(img_up.getvalue()).decode("utf-8")
            st.image(img_up, caption="Image Ready")

# --- 4. MAIN CHAT ---
st.title("🤖 Specialized AI Assistant")

for m in st.session_state.chat_history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "fig" in m and m["fig"]: st.plotly_chart(m["fig"], use_container_width=True)

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        bot_res = {"role": "assistant", "content": "", "fig": None}
        
        # --- 📊 DATA MODE: Llama 3.3 70B (Best for Logic/Math) ---
        if mode == "📊 Data Analysis" and st.session_state.df is not None:
            data_llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
            df = st.session_state.df
            
            expl = data_llm.invoke(f"Columns: {list(df.columns)}\nUser: {prompt}\nExplain simply.").content
            st.markdown(expl)
            bot_res["content"] = expl
            
            intent = data_llm.invoke(f"Columns: {list(df.columns)}\nUser: '{prompt}'\nIf chart needed return: X | Y1,Y2... | Subplots(Yes/No). Else: 'NONE'").content
            if "NONE" not in intent:
                try:
                    x_c, y_c, sub = intent.split("|")
                    y_list = [y.strip() for y in y_c.split(",") if y.strip() in df.columns]
                    if y_list:
                        fig = build_smart_visuals(df, x_c.strip(), y_list, sub)
                        st.plotly_chart(fig, use_container_width=True)
                        bot_res["fig"] = fig
                except: pass

        # --- 🖼️ IMAGE MODE: Llama 4 Scout (Stable 2026 Vision) ---
        elif mode == "🖼️ Images" and "img_b64" in st.session_state:
            # We use the full ID for the new Llama 4 production model
            vision_llm = ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct")
            msg = HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{st.session_state.img_b64}"}}
            ])
            ans = vision_llm.invoke([msg]).content
            st.markdown(ans)
            bot_res["content"] = ans

        # --- 📄 DOCS/WEB MODE: Llama 3.1 8B (Super Fast Retrieval) ---
        else:
            text_llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1)
            context = ""
            if st.session_state.retriever:
                docs = st.session_state.retriever.invoke(prompt)
                context = "\n".join([d.page_content for d in docs])
            ans = text_llm.invoke(f"Context: {context}\n\nQuestion: {prompt}").content
            st.markdown(ans)
            bot_res["content"] = ans

        st.session_state.chat_history.append(bot_res)