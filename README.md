# nitesh-gen-ai-workspace

# ⚡ Universal AI Studio 2026
A high-performance, multimodal AI web application built with **Streamlit**, **LangChain**, and **Groq**. This app intelligently switches between specialized models to handle Data Analysis, PDF Document Intelligence (RAG), and Computer Vision.

## 🚀 Features
- **Auto-Intent Routing:** Automatically detects if you are asking about an uploaded file or a general topic (like coding).
- **Vision Intelligence:** Powered by `Llama-4-Scout` for real-time image description and analysis.
- **Document Brain:** Uses RAG (Retrieval-Augmented Generation) with `ChromaDB` to answer questions about PDFs with zero hallucinations.
- **Data Analysis:** Quick CSV/Excel column mapping and insights.
- **Zero-Memory Sessions:** Automatically wipes data between sessions for privacy and security.

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **AI Orchestration:** LangChain
- **Models:** Meta Llama 3.3 (Reasoning) & Llama 4 Scout (Vision) via Groq Cloud
- **Vector Store:** ChromaDB
- **Embeddings:** HuggingFace (all-MiniLM-L6-v2)

## 📦 Installation & Setup
1. **Clone the repo:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
   cd YOUR_REPO_NAME
   
2. **download requirements**
   pip install -r requirements.txt

3. **Create a .env file for local use or add to Streamlit Secrets:**
   GROQ_API_KEY=your_api_key_here

4. **Run the app:**
   streamlit run app.py

---

### 2. The Repository Summary (About Section)
On the right side of your GitHub repository page, click the **cog icon (⚙️)** next to "About" and paste this summary:

> **"A multimodal AI workspace using Llama 3.3 & Llama 4 on Groq. Features auto-intent routing for PDF analysis, Image Vision, and General Coding assistance with zero hallucinations."**

---

### 3. Topics (Tags)
Add these tags to your repository so other developers can find it:
`python` `streamlit` `langchain` `groq` `llama3` `llama4` `rag` `computer-vision`

---

### 4. Final Checklist for your GitHub
To ensure the app never fails to host, double-check that your file structure looks exactly like this:
* `app.py` (The full code I gave you)
* `requirements.txt` (The list of libraries)
* `README.md` (The documentation above)
* `.gitignore` (Optional: add `chroma_db/` to this file so you don't upload your database by accident)

**Would you like me to generate a specific "User Guide" PDF that you can also upload to the repository for others to read?**


   <img width="1919" height="879" alt="image" src="https://github.com/user-attachments/assets/85988750-2d90-457f-9ff4-7c1431d41973" />
