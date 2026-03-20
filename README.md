📚 RAG Chatbot

A Retrieval-Augmented Generation (RAG) Chatbot built with Streamlit that can answer questions based on uploaded PDF, DOCX, or image files. It uses OCR, hybrid search (BM25 + embeddings + Cross-Encoder reranker), and LLM for context-aware answers.

Features

Upload multiple files: PDF, DOCX, JPG, PNG, WEBP

Extract text from documents and images

Split large documents into chunks for better retrieval

Hybrid search:

BM25 keyword search

Vector embeddings via Pinecone + HuggingFace

Cross-Encoder reranker for better accuracy

Answer questions using Ollama LLaMA2 with source references

Display results in a user-friendly Streamlit interface

Requirements

Python 3.10+

Streamlit

Pillow

OpenCV (cv2)

pytesseract

python-docx

langchain and related modules:

langchain_community, langchain_text_splitters, langchain_huggingface, langchain_ollama, langchain_core, langchain_pinecone

Pinecone client

BM25 (rank_bm25)

Sentence Transformers (CrossEncoder)

python-dotenv

Setup

Clone the repository

git clone <repo-url>
cd <repo-folder>

Install dependencies

pip install -r requirements.txt

Download Ollama: https://ollama.com/
Downalod pytesseract: https://pypi.org/project/pytesseract/

Set up environment variables

Create a .env file in the project root:

PINECONE_API_KEY=<your_pinecone_api_key>
PINECONE_ENV=<your_pinecone_environment>
PINECONE_INDEX_NAME=<your_index_name>

Run Streamlit app

streamlit run app.py

How it Works

Upload files

PDF: Loaded via PyPDFLoader

DOCX: Loaded via python-docx

Images: Text extracted via Tesseract OCR

Document processing

Split documents into chunks using RecursiveCharacterTextSplitter

Add chunks to Pinecone vector store for embeddings

Hybrid search

BM25 keyword-based scoring

Vector embeddings retrieval (via Pinecone)

Combine and deduplicate results

Rerank using Cross-Encoder

Answer generation

Prompt Ollama LLaMA2 with top retrieved chunks

Provide answer along with sources

Usage

Open the Streamlit app in your browser.

Upload one or multiple files (PDF, DOCX, or images).

Enter a question in the text input box.

Get the answer and sources extracted from uploaded documents.
