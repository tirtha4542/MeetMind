import os
import threading
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Pinecone configuration
INDEX_NAME = "meeting-index"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

embeddings_model = "sentence-transformers/all-MiniLM-L6-v2"

_embedding_instance = None
_embedding_lock = threading.Lock()

def get_embedding():
    global _embedding_instance
    if _embedding_instance is None:
        with _embedding_lock:
            if _embedding_instance is None:
                _embedding_instance = HuggingFaceEmbeddings(
                    model_name=embeddings_model,
                    encode_kwargs={"normalize_embeddings": True},
                )
    return _embedding_instance

def vector_store(transcript: str = None) -> PineconeVectorStore:
    embedding = get_embedding()
    
    # Initialize PineconeVectorStore
    vector_db = PineconeVectorStore(
        index_name=INDEX_NAME, 
        embedding=embedding,
        pinecone_api_key=PINECONE_API_KEY
    )
    
    if transcript and transcript.strip():
        print("Adding transcript to Pinecone...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(transcript)
        documents = [
            Document(page_content=chunk, metadata={"chunk_id": i})
            for i, chunk in enumerate(chunks)
        ]
        vector_db.add_documents(documents)
        print("Transcript successfully indexed in Pinecone.")
        
    return vector_db

def load_vector_store():
    # Load the existing index
    return vector_store(transcript=None)

def get_retriever(vector_db: PineconeVectorStore, k: int = 4):
    return vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )