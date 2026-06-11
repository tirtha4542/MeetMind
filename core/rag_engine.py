from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os

# ✅ FIX: corrected import path — removed wrong 'core.' prefix
from core.vectore_store import vector_store, load_vector_store, get_retriever


def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        temperature=0.4,
        max_tokens=2048,
        api_key=os.getenv("MISTRAL_API_KEY"),
    )


def format_docs(docs):
    formatted = []
    for doc in docs:
        content = doc.page_content
        metadata = doc.metadata
        formatted.append(f"Chunk ID: {metadata.get('chunk_id', 'N/A')}\nContent: {content}\n")
    return "\n".join(formatted)


def get_base_prompt():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant that answers questions based on meeting transcripts. "
            "Context for the meeting transcript is: {context}",
        ),
        (
            "human",
            "Answer the following question based on the retrieved meeting transcript\n\n{question}",
        ),
    ])


def build_rag_chain(transcript: str):
    print("Building vector store for RAG...")
    vs = vector_store(transcript)
    print("Getting retriever...")
    retriever = get_retriever(vs)
    print("RAG chain ready.")
    llm = get_llm()
    prompt = get_base_prompt()

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def load_rag_chain(transcript: str):
    # ✅ FIX: accept transcript so vector store is actually populated
    vs = load_vector_store(transcript)
    retriever = get_retriever(vs)
    llm = get_llm()
    prompt = get_base_prompt()

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def ask_question(rag_chain, question: str) -> str:
    print(f"Answering question: {question}")
    answer = rag_chain.invoke(question)
    print(f"Answer: {answer}")
    return answer