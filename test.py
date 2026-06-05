from core.summarize import generate_title, summarize
from utils.audio_processor import process_input
from core.transcribe import transcribe_all 
source = "https://www.youtube.com/watch?v=_0i9Q5R23u4"
chunks = process_input(source)
transcript = transcribe_all(chunks)
print(transcript)
print("Transcript generated successfully. Now generating title and summary...")
title = generate_title(transcript)
print(title)
summarized = summarize(transcript)
print(summarized)

print("Title and summary generated successfully.")
