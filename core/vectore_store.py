import threading
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import chromadb

embeddings_model = "sentence-transformers/all-MiniLM-L6-v2"

_embedding_instance = None
_embedding_lock = threading.Lock()


def get_embedding():
    global _embedding_instance
    if _embedding_instance is None:
        with _embedding_lock:
            if _embedding_instance is None:
                print("Loading embedding model...")
                _embedding_instance = HuggingFaceEmbeddings(
                    model_name=embeddings_model,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
                print("Embedding model loaded.")
    return _embedding_instance


def vector_store(transcript: str) -> Chroma:
    print("Building vector store...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(transcript)
    documents = [
        Document(page_content=chunk, metadata={"chunk_id": i})
        for i, chunk in enumerate(chunks)
    ]
    embedding = get_embedding()

    # EphemeralClient = pure in-memory, no SQLite, no Windows thread-lock freeze
    client = chromadb.EphemeralClient()
    vector_db = Chroma(
        client=client,
        collection_name="meeting_session",
        embedding_function=embedding,
    )
    if documents:
        vector_db.add_documents(documents)
    print("Vector store built.....")
    return vector_db


# ✅ FIX: load_vector_store now requires the transcript to build a meaningful store,
#    since we use EphemeralClient (in-memory only — no persistence between sessions).
def load_vector_store(transcript: str) -> Chroma:
    if not transcript:
        raise ValueError(
            "A transcript must be provided. EphemeralClient is in-memory only "
            "and cannot persist data between sessions."
        )
    return vector_store(transcript)


def get_retriever(vector_db: Chroma, k: int = 4):
    return vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )