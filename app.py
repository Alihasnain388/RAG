import streamlit as st
import os
from PIL import Image
import numpy as np
import cv2
import pytesseract
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from docx import Document as DocxDocument
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from io import BytesIO
import tempfile

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()

st.title("📚 RAG Chatbot")

# Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = os.getenv(
    "TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# -------------------------------
# Pinecone Setup
# -------------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# -------------------------------
# OCR Function
# -------------------------------
def extract_text_from_image(uploaded_image):
    image = Image.open(uploaded_image)
    img = np.array(image)

    if len(img.shape) == 2:
        gray = img
    elif len(img.shape) == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError("Unsupported image format")

    return pytesseract.image_to_string(gray)

# -------------------------------
# Multi-file upload
# -------------------------------
uploaded_files = st.file_uploader(
    "Upload Files (PDF, DOCX, or Image) - You can select multiple",
    type=["pdf", "docx", "jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

if uploaded_files:
    documents = []

    for uploaded_file in uploaded_files:
        source_name = uploaded_file.name
        try:
            if uploaded_file.type == "application/pdf":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    loader = PyPDFLoader(tmp.name)
                    for doc in loader.load():
                        doc.metadata["source"] = source_name
                        documents.append(doc)
                st.success(f"PDF uploaded successfully: {source_name}")

            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                docx_doc = DocxDocument(BytesIO(uploaded_file.read()))
                full_text = "\n".join([p.text for p in docx_doc.paragraphs])
                documents.append(Document(page_content=full_text, metadata={"source": source_name}))
                st.success(f"Word file uploaded successfully: {source_name}")

            else:
                text = extract_text_from_image(uploaded_file)
                if text.strip():
                    documents.append(Document(page_content=text, metadata={"source": source_name}))
                    st.success(f"Text extracted from image: {source_name}")
                else:
                    st.warning(f"No text detected in image: {source_name}")

        except Exception as e:
            st.error(f"Failed to process {source_name}: {e}")

    # -------------------------------
    # Process all uploaded files together
    # -------------------------------
    if len(documents) > 0:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        st.write(f"Total text chunks from all files: {len(chunks)}")

        # Embeddings
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = PineconeVectorStore(index=index, embedding=embedding_model)
        vectorstore.index.delete(delete_all=True)
        vectorstore.add_documents(chunks)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        # BM25
        corpus = [doc.page_content.split() for doc in chunks]
        bm25 = BM25Okapi(corpus)

        # Cross-Encoder reranker
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # LLM
        llm = OllamaLLM(model="llama2")

        # -------------------------------
        # Hybrid Search
        # -------------------------------
        def hybrid_search(query, k=5):
            tokenized_query = query.split()
            bm25_scores = bm25.get_scores(tokenized_query)
            bm25_results = sorted(zip(chunks, bm25_scores), key=lambda x: x[1], reverse=True)[:k]
            bm25_docs = [doc for doc, _ in bm25_results]

            vector_docs = retriever.invoke(query)[:k]

            # Combine and remove duplicates
            seen_texts = set()
            combined_docs = []
            for doc in bm25_docs + vector_docs:
                if doc.page_content not in seen_texts:
                    combined_docs.append(doc)
                    seen_texts.add(doc.page_content)

            # Rerank
            pairs = [(query, doc.page_content) for doc in combined_docs]
            scores = reranker.predict(pairs)
            reranked = sorted(zip(combined_docs, scores), key=lambda x: x[1], reverse=True)

            return [doc for doc, _ in reranked[:3]]

        # -------------------------------
        # Ask Question
        # -------------------------------
        query = st.text_input("Ask a question about your uploaded files:")

        if query:
            with st.spinner("Generating answer..."):
                docs = hybrid_search(query)
                if not docs:
                    st.warning("No relevant information found in uploaded files.")
                else:
                    context = "\n\n".join([doc.page_content for doc in docs])
                    sources = ", ".join([doc.metadata.get("source", "Unknown") for doc in docs])

                    prompt = f"""
You are a helpful assistant.

Use ONLY the provided context to answer.
If answer not found, say "I don't know".

Context:
{context}

Question:
{query}
"""
                    response = llm.invoke(prompt)

                    st.write("**Answer:**", response)
                    st.write("**Sources:**", sources)