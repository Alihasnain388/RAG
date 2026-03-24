# 📚 RAG Chatbot

A **Retrieval-Augmented Generation (RAG) Chatbot** built with Streamlit that answers questions based on uploaded PDF and DOCX files. It uses **hybrid search (BM25 + vector embeddings + Cross-Encoder reranking)** and an LLM to generate accurate, context-aware responses.

---

## 🚀 Features

- 📂 Upload multiple files:
  - PDF
  - DOCX  

- 📄 Automatic text extraction from documents

- ✂️ Intelligent document chunking using RecursiveCharacterTextSplitter

- 🔍 **Hybrid Search Pipeline**:
  - BM25 keyword-based retrieval
  - Vector similarity search using Pinecone + HuggingFace embeddings
  - Cross-Encoder reranking for improved relevance

- 🤖 LLM-powered answers using Groq (LLaMA 3.1)

- 📌 Displays answers with source references

- 🌐 Deployed and accessible via Streamlit Cloud

---

## 🛠️ Tech Stack

- Python 3.10+
- Streamlit (UI & deployment)
- LangChain ecosystem:
  - langchain_community
  - langchain_text_splitters
  - langchain_huggingface
  - langchain_pinecone
  - langchain_groq
- Pinecone (Vector Database)
- HuggingFace Embeddings (all-MiniLM-L6-v2)
- BM25 (rank_bm25)
- Cross-Encoder (sentence-transformers)
- python-docx
- python-dotenv

---

## ⚙️ Setup

 1. Clone Repository
```bash
git clone <repo-url>
cd <repo-folder>

2. Install Dependencies
pip install -r requirements.txt

3. Environment Variables

Create a .env file in the root directory:

PINECONE_API_KEY=your_api_key
PINECONE_INDEX_NAME=your_index_name
GROQ_API_KEY=your_groq_api_key

4. Run the App
streamlit run app.py

Link= https://alihasnain388-rag-app-tc1hur.streamlit.app/
