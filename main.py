from dotenv import load_dotenv
from core.summarize import summarize, generate_title
from core.rag_engine import build_rag_chain, ask_question
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace') if hasattr(sys.stdout, 'buffer') else sys.stdout

load_dotenv()
from utils.audio_processor import process_input
from core.extractor import extract_action_items, extract_decisions, extract_questions
from core.transcribe import transcribe_all

def run_pipeline(source:str)->dict:
    print("Starting AI video meeting assistant pipeline...")

    chunks = process_input(source)
    transcript = transcribe_all(chunks)
    print("Transcript generated.....")
    summary = summarize(transcript)
    print("Summary generated.....")
    title = generate_title(transcript)
    print("Title generated.....")
    action_items = extract_action_items(transcript)
    print("Action items extracted.....")
    decisions = extract_decisions(transcript)
    print("Decisions extracted.....")
    questions = extract_questions(transcript)
    print("Questions extracted.....")
    rag_chain = build_rag_chain(transcript)
    print("RAG chain built.....")
    result = {
        "title": title,
        "summary": summary,
        "action_items": action_items,
        "decisions": decisions,
        "questions": questions,
        "rag_chain": rag_chain
    }
    return result

if __name__ == "__main__":
    source = input("Enter the path to the video file or YouTube URL: ")
    result = run_pipeline(source)
    print("Pipeline completed. Here are the results:")
    print(f"Title: {result['title']}")
    print(f"Summary: {result['summary']}")
    print(f"Action Items: {result['action_items']}")
    print(f"Decisions: {result['decisions']}")
    print(f"Questions: {result['questions']}")

    print("\nYou can now ask questions about the meeting transcript using the RAG chain.")
    rag_chain = result['rag_chain']
    while True:
        question = input("\nEnter your question (or 'exit' to quit): ")
        if question.lower() == 'exit':
            print("Exiting. Goodbye!")
            break
        answer = ask_question(rag_chain, question)
        print(f"Answer: {answer}")
        