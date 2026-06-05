from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uuid
import threading
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles      # ← ADD
from fastapi.responses import FileResponse       
load_dotenv()

# ── Import your pipeline modules ─────────────────────────────────────────────
from utils.audio_processor import process_input
from core.transcribe import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="MeetMind API")
app.mount("/static", StaticFiles(directory="static"), name="static")   # ← ADD

@app.get("/")                                                           # ← ADD
def serve_ui():                                                         # ← ADD
    return FileResponse("static/index.html")                           

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory job store ───────────────────────────────────────────────────────
# Structure per job_id:
#   status : "processing" | "done" | "error"
#   step   : current human-readable step string
#   result : dict with title, summary, action_items, decisions, questions
#   rag    : the built rag_chain object (kept in memory)
#   error  : error message string (on failure)
jobs: dict = {}


# ── Request / Response models ─────────────────────────────────────────────────
class ProcessRequest(BaseModel):
    source: str   # YouTube URL or local file path

class AskRequest(BaseModel):
    job_id: str
    question: str


# ── Background pipeline ───────────────────────────────────────────────────────
def run_pipeline(job_id: str, source: str):
    """Runs the full pipeline in a background thread and updates jobs[job_id]."""
    try:
        def update(step: str):
            jobs[job_id]["step"] = step
            print(f"[{job_id}] {step}")

        update("Downloading & processing audio...")
        chunks = process_input(source)

        update("Transcribing audio with Whisper...")
        transcript = transcribe_all(chunks)

        update("Generating summary...")
        summary = summarize(transcript)

        update("Generating title...")
        title = generate_title(transcript)

        update("Extracting action items...")
        action_items = extract_action_items(transcript)

        update("Extracting decisions...")
        decisions = extract_decisions(transcript)

        update("Extracting questions...")
        questions = extract_questions(transcript)

        update("Building RAG chain...")
        rag_chain = build_rag_chain(transcript)

        # ── Mark done ──────────────────────────────────────────────────────────
        jobs[job_id].update({
            "status": "done",
            "step": "Done!",
            "result": {
                "title": title,
                "summary": summary,
                "action_items": action_items,
                "decisions": decisions,
                "questions": questions,
            },
            "rag": rag_chain,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        jobs[job_id].update({
            "status": "error",
            "step": "Error",
            "error": str(e),
        })


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/process")
def start_process(req: ProcessRequest):
    """Start processing a YouTube URL or local file. Returns a job_id immediately."""
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "processing",
        "step": "Starting...",
        "result": None,
        "rag": None,
        "error": None,
    }
    # Run pipeline in a daemon thread so FastAPI stays responsive
    t = threading.Thread(target=run_pipeline, args=(job_id, req.source), daemon=True)
    t.start()
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    """Poll this endpoint to get the current status & results once done."""
    job = jobs.get(job_id)
    if not job:
        return {"status": "error", "error": "Job not found"}

    response = {
        "status": job["status"],
        "step":   job["step"],
        "error":  job.get("error"),
    }

    # Include results when done so the UI can render them immediately
    if job["status"] == "done" and job["result"]:
        response.update(job["result"])

    return response


@app.post("/ask")
def ask(req: AskRequest):
    """Ask a RAG question against the processed transcript."""
    job = jobs.get(req.job_id)
    if not job:
        return {"answer": "Job not found."}
    if job["status"] != "done":
        return {"answer": "Processing is not finished yet."}
    if not job.get("rag"):
        return {"answer": "RAG chain not available."}

    try:
        answer = ask_question(job["rag"], req.question)
        return {"answer": answer}
    except Exception as e:
        return {"answer": f"Error: {str(e)}"}


# ── Run locally ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)