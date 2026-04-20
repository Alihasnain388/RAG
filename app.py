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


@st.cache_resource(show_spinner=False)
def get_pinecone_index(api_key: str, index_name: str):
    pc = Pinecone(api_key=api_key)
    return pc.Index(index_name)


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def get_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


@st.cache_resource(show_spinner=False)
def get_llm(api_key: str):
    return ChatGroq(model="llama-3.1-8b-instant", api_key=api_key)


def ensure_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    return st.session_state.session_id


def uploaded_files_signature(uploaded_files) -> str:
    hasher = hashlib.sha256()

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        hasher.update(uploaded_file.name.encode("utf-8", errors="ignore"))
        hasher.update(str(len(file_bytes)).encode("utf-8"))
        hasher.update(hashlib.sha256(file_bytes).digest())

    return hasher.hexdigest()


def extract_pdf(uploaded_file) -> list[Document]:
    source_name = uploaded_file.name
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = source_name
            if "page" in doc.metadata:
                doc.metadata["page"] = doc.metadata["page"] + 1

        return docs
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def extract_docx(uploaded_file) -> list[Document]:
    docx_doc = DocxDocument(BytesIO(uploaded_file.getvalue()))
    parts = [paragraph.text for paragraph in docx_doc.paragraphs if paragraph.text.strip()]

    for table in docx_doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                parts.append(row_text)

    return [
        Document(
            page_content="\n".join(parts),
            metadata={"source": uploaded_file.name},
        )
    ]


def extract_pptx(uploaded_file) -> list[Document]:
    presentation = Presentation(BytesIO(uploaded_file.getvalue()))
    docs = []

    for slide_number, slide in enumerate(presentation.slides, start=1):
        parts = []

        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False) and shape.text.strip():
                parts.append(shape.text.strip())

            if getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_text:
                        parts.append(row_text)

        if getattr(slide, "has_notes_slide", False):
            notes_frame = getattr(slide.notes_slide, "notes_text_frame", None)
            notes_text = notes_frame.text.strip() if notes_frame and notes_frame.text else ""
            if notes_text:
                parts.append("Speaker notes:\n" + notes_text)

        slide_text = "\n".join(parts).strip()
        if slide_text:
            docs.append(
                Document(
                    page_content=slide_text,
                    metadata={"source": uploaded_file.name, "slide": slide_number},
                )
            )

    return docs


def extract_documents(uploaded_files) -> list[Document]:
    documents = []

    for uploaded_file in uploaded_files:
        source_name = uploaded_file.name
        extension = Path(source_name).suffix.lower()

        try:
            if uploaded_file.type == "application/pdf" or extension == ".pdf":
                file_docs = extract_pdf(uploaded_file)
                st.success(f"PDF uploaded: {source_name}")
            elif extension == ".docx":
                file_docs = extract_docx(uploaded_file)
                st.success(f"DOCX uploaded: {source_name}")
            elif extension == ".pptx":
                file_docs = extract_pptx(uploaded_file)
                st.success(f"PPTX uploaded: {source_name}")
            else:
                st.warning(f"Unsupported file skipped: {source_name}")
                continue

            documents.extend(file_docs)
        except Exception as exc:
            st.error(f"Failed to process {source_name}: {exc}")

    return [doc for doc in documents if doc.page_content.strip()]


def source_label(doc: Document) -> str:
    source = doc.metadata.get("source", "Unknown")
    if "page" in doc.metadata:
        return f"{source} (page {doc.metadata['page']})"
    if "slide" in doc.metadata:
        return f"{source} (slide {doc.metadata['slide']})"
    return source


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"Source: {source_label(doc)}\nContent:\n{doc.page_content}" for doc in docs
    )


def build_namespace(signature: str) -> str:
    session_id = ensure_session_id()
    return f"rag-{session_id[:12]}-{signature[:16]}"


def clear_previous_namespace(index, namespace: str | None):
    if not namespace:
        return

    try:
        index.delete(delete_all=True, namespace=namespace)
    except Exception:
        pass


uploaded_files = st.file_uploader(
    "Upload Files (PDF, DOCX, PPTX)",
    type=["pdf", "docx", "pptx"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload one or more PDF, DOCX, or PPTX files to start chatting.")
    st.stop()

index = get_pinecone_index(PINECONE_API_KEY, INDEX_NAME)
embedding_model = get_embedding_model()
reranker = get_reranker()
llm = get_llm(GROQ_API_KEY)

signature = uploaded_files_signature(uploaded_files)
namespace = build_namespace(signature)

if st.session_state.get("processed_signature") != signature:
    with st.spinner("Processing uploaded files..."):
        documents = extract_documents(uploaded_files)

        if not documents:
            st.warning("No readable text found in the uploaded files.")
            st.stop()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        chunks = [chunk for chunk in chunks if chunk.page_content.strip()]

        if not chunks:
            st.warning("No readable chunks could be created from the uploaded files.")
            st.stop()

        previous_namespace = st.session_state.get("namespace")
        if previous_namespace and previous_namespace != namespace:
            clear_previous_namespace(index, previous_namespace)

        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embedding_model,
            namespace=namespace,
        )
        ids = [f"{namespace}-{idx}" for idx in range(len(chunks))]
        vectorstore.add_documents(chunks, ids=ids)

        st.session_state.processed_signature = signature
        st.session_state.namespace = namespace
        st.session_state.chunks = chunks

chunks = st.session_state.get("chunks", [])
if not chunks:
    st.warning("Upload documents before asking a question.")
    st.stop()

st.write(f"Total chunks: {len(chunks)}")

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embedding_model,
    namespace=st.session_state.namespace,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
bm25 = BM25Okapi([doc.page_content.split() for doc in chunks])


def hybrid_search(query: str, k: int = 5) -> list[Document]:
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_results = sorted(
        zip(chunks, bm25_scores),
        key=lambda item: item[1],
        reverse=True,
    )[:k]

    bm25_docs = [doc for doc, _ in bm25_results]
    vector_docs = retriever.invoke(query)[:k]

    seen = set()
    combined = []

    for doc in bm25_docs + vector_docs:
        key = (
            doc.metadata.get("source"),
            doc.metadata.get("page"),
            doc.metadata.get("slide"),
            doc.page_content,
        )
        if key not in seen:
            combined.append(doc)
            seen.add(key)

    if not combined:
        return []

    pairs = [(query, doc.page_content) for doc in combined]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(combined, scores), key=lambda item: item[1], reverse=True)

    return [doc for doc, _ in reranked[:3]]


query = st.text_input("Ask a question:").strip()

if query:
    with st.spinner("Generating answer..."):
        docs = hybrid_search(query)

        if not docs:
            st.warning("No relevant info found.")
        else:
            context = format_context(docs)
            sources = sorted({source_label(doc) for doc in docs})

            prompt = f"""
You are a helpful assistant.

Use ONLY the provided context.
If the answer is not found in the context, say "I don't know".

Context:
{context}

Question:
{query}
"""

            response = llm.invoke(prompt)

            st.write("### Answer")
            st.write(response.content)

            st.write("### Sources")
            for source in sources:
                st.write(f"- {source}")
