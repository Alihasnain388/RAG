import hashlib
import os
import tempfile
import uuid
from io import BytesIO
from pathlib import Path

import streamlit as st
from docx import Document as DocxDocument
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openpyxl import load_workbook
from pinecone import Pinecone
from pptx import Presentation
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


load_dotenv()

st.set_page_config(page_title="RAG Chatbot")
st.title("RAG Chatbot")


def get_config_value(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value

    try:
        value = st.secrets.get(name)
    except Exception:
        return None

    return value or None


PINECONE_API_KEY = get_config_value("PINECONE_API_KEY")
INDEX_NAME = get_config_value("PINECONE_INDEX_NAME")
GROQ_API_KEY = get_config_value("GROQ_API_KEY")

missing_config = [
    name
    for name, value in {
        "PINECONE_API_KEY": PINECONE_API_KEY,
        "PINECONE_INDEX_NAME": INDEX_NAME,
        "GROQ_API_KEY": GROQ_API_KEY,
    }.items()
    if not value
]

if missing_config:
    st.error("Missing required configuration: " + ", ".join(missing_config))
    st.stop()
