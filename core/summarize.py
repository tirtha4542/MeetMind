from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import os
load_dotenv()
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


def get_llm():
    return ChatMistralAI(model='mistral-small-latest', temperature=0.4, max_tokens=2048, api_key=os.getenv("MISTRAL_API_KEY"))
def split_transcribe(transcript: str, chunk_size: int = 1000) -> list:
    """Split transcript into manageable chunks for summarization."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
    return splitter.split_text(transcript)
def summarize(transcript: str) -> str:
    """Summarize the transcript using Mistral."""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that summarizes meeting transcripts."),
        ("human", "Summarize the following transcript:\n\n{transcript}")
    ])
    map_chain = prompt | llm | StrOutputParser()
    chunks = split_transcribe(transcript)
    chunk_summary = [map_chain.invoke({"transcript": chunk}) for chunk in chunks]
    combined = "\n".join(chunk_summary)
    combine_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert meeting summarizer. Combine these partial summaries into one final professional meeting summary in bullet points."),("human", "Combine the following summaries into a concise summary:\n\n{text}")]) 
    combine_chain = (
        RunnablePassthrough() | RunnablePassthrough(lambda x: {"text": x}) | combine_prompt | llm | StrOutputParser()
    )  
    final_summary = combine_chain.invoke("\n".join(chunk_summary))
    return final_summary
def generate_title(transcript: str) -> str:
    """Generate a title for the transcript using Mistral."""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that generates concise titles for meeting transcripts."),
        ("human", "Generate a concise title for the following transcript:\n\n{text}")
    ])
    title_chain = (
        RunnablePassthrough() | RunnablePassthrough(lambda x: {"text": x}) | prompt | llm | StrOutputParser()
    )
    title = title_chain.invoke(transcript)
    return title