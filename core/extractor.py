from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import os
load_dotenv()
load_dotenv()
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
def get_llm():
    return ChatMistralAI(model='mistral-small-latest', temperature=0.4, max_tokens=2048, api_key=os.getenv("MISTRAL_API_KEY"))

def build_chain(system_prompt:str):
    llm = get_llm()
    return (RunnablePassthrough() | RunnablePassthrough(lambda x: {"text": x}) | ChatPromptTemplate.from_messages([
        ("system", system_prompt),  
        ("human", "{text}")])) | llm | StrOutputParser()
def extract_action_items(transcript: str) -> str:
    chain = build_chain("You are a helpful assistant that extracts action items from meeting transcripts. Extract the action items from the following transcript and list them in bullet points.")
    return chain.invoke(transcript)
def extract_decisions(transcript: str) -> str:
    chain = build_chain("You are a helpful assistant that extracts decisions from meeting transcripts. Extract the decisions from the following transcript and list them in bullet points.")
    return chain.invoke(transcript)
def extract_questions(transcript: str) -> str:
    chain = build_chain("You are a helpful assistant that extracts questions from meeting transcripts. Extract the questions from the following transcript and list them in bullet points.")
    return chain.invoke(transcript)
