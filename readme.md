# 🤖 Conversational RAG Research Assistant

A professional-grade Retrieval-Augmented Generation (RAG) application built in 2026. This tool allows users to upload PDF documents and have context-aware, "grounded" conversations with their data using High-Speed LPU inference.

## 🚀 Key Features
- **Semantic Document Ingestion:** Uses Recursive Character Splitting to maintain context across document chunks.
- **Vectorized Memory:** Powered by ChromaDB and OpenAI Embeddings for high-accuracy document retrieval.
- **Ultra-Fast Inference:** Integrated with **Groq (Llama 3.3)** for near-instant responses.
- **Conversational Intelligence:** Full session-based chat history management, allowing the AI to remember previous context within a session.
- **Anti-Hallucination:** System prompts are engineered to ensure the AI only answers based on the provided PDF context.

## 🛠️ Tech Stack
- **Orchestration:** LangChain (LCEL)
- **LLM:** Groq (Llama-3.3-70b-versatile)
- **Vector Store:** ChromaDB
- **Embeddings:** OpenAI (text-embedding-3-small)
- **UI:** Streamlit

## 📦 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/rnk3120/nitesh-gen-ai-workspace.git
cd nitesh-gen-ai-workspace


# Windows
python -m venv gen-env
gen-env\Scripts\activate

# Mac/Linux
python3 -m venv gen-env
source gen-env/bin/activate

### 2. Install dependencies
pip install -r requirements.txt

GROQ_API_KEY=your_groq_key_here
OPENAI_API_KEY=your_openai_key_here

### 3. Run the app
streamlit run app.py


