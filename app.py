import streamlit as st
import os
from PIL import Image
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from docx import Document as DocxDocument
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from io import BytesIO
import tempfile

# ✅ Groq LLM
from langchain_groq import ChatGroq

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()

st.title("📚 RAG Chatbot")

# -------------------------------
# Pinecone Setup
# -------------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# -------------------------------
# Multi-file upload
# -------------------------------
uploaded_files = st.file_uploader(
    "Upload Files (PDF, DOCX, or Image)",
    type=["pdf", "docx", "jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

if uploaded_files:
    documents = []

    for uploaded_file in uploaded_files:
        source_name = uploaded_file.name
        try:
            # ---------------- PDF ----------------
            if uploaded_file.type == "application/pdf":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    loader = PyPDFLoader(tmp.name)
                    for doc in loader.load():
                        doc.metadata["source"] = source_name
                        documents.append(doc)
                st.success(f"PDF uploaded: {source_name}")

            # ---------------- DOCX ----------------
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                docx_doc = DocxDocument(BytesIO(uploaded_file.read()))
                full_text = "\n".join([p.text for p in docx_doc.paragraphs])
                documents.append(Document(page_content=full_text, metadata={"source": source_name}))
                st.success(f"DOCX uploaded: {source_name}")

            # ---------------- IMAGE (OCR DISABLED) ----------------
            else:
                st.warning(f"OCR not supported in deployed version: {source_name}")

        except Exception as e:
            st.error(f"Failed to process {source_name}: {e}")

    # -------------------------------
    # Process Documents
    # -------------------------------
    if len(documents) > 0:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        st.write(f"Total chunks: {len(chunks)}")

        # Embeddings
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        vectorstore = PineconeVectorStore(index=index, embedding=embedding_model)

        # ❌ REMOVED delete_all=True (important fix)
        vectorstore.add_documents(chunks)

        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        # BM25
        corpus = [doc.page_content.split() for doc in chunks]
        bm25 = BM25Okapi(corpus)

        # Reranker
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # -------------------------------
        # Groq LLM Setup
        # -------------------------------
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY")
        )

        # -------------------------------
        # Hybrid Search
        # -------------------------------
        def hybrid_search(query, k=5):
            tokenized_query = query.split()

            bm25_scores = bm25.get_scores(tokenized_query)
            bm25_results = sorted(
                zip(chunks, bm25_scores),
                key=lambda x: x[1],
                reverse=True
            )[:k]

            bm25_docs = [doc for doc, _ in bm25_results]
            vector_docs = retriever.invoke(query)[:k]

            # Combine
            seen = set()
            combined = []

            for doc in bm25_docs + vector_docs:
                if doc.page_content not in seen:
                    combined.append(doc)
                    seen.add(doc.page_content)

            # Rerank
            pairs = [(query, doc.page_content) for doc in combined]
            scores = reranker.predict(pairs)

            reranked = sorted(
                zip(combined, scores),
                key=lambda x: x[1],
                reverse=True
            )

            return [doc for doc, _ in reranked[:3]]

        # -------------------------------
        # Query Input
        # -------------------------------
        query = st.text_input("Ask a question:")

        if query:
            with st.spinner("Generating answer..."):
                docs = hybrid_search(query)

                if not docs:
                    st.warning("No relevant info found.")
                else:
                    context = "\n\n".join([doc.page_content for doc in docs])
                    sources = ", ".join([doc.metadata.get("source", "Unknown") for doc in docs])

                    prompt = f"""
You are a helpful assistant.

Use ONLY the provided context.
If answer not found, say "I don't know".

Context:
{context}

Question:
{query}
"""

                    response = llm.invoke(prompt)

                    st.write("### Answer")
                    st.write(response.content)

                    st.write("### Sources")
                    st.write(sources)
