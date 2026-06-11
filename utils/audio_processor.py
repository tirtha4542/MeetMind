import os
import subprocess
import yt_dlp

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Path to your exported cookies.txt file (place in project root)
COOKIES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cookies.txt")


def download_youtube_audio(url: str) -> str | None:
    output_template = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")

    cookies_path = os.path.normpath(COOKIES_FILE)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "quiet": False,
        "noplaylist": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
    }

    if os.path.exists(cookies_path):
        print(f"[COOKIES] Using cookies file: {cookies_path}")
        ydl_opts["cookiefile"] = cookies_path
    else:
        print("[WARN] cookies.txt not found - trying Edge browser cookies as fallback...")
        ydl_opts["cookiesfrombrowser"] = ("edge",)

    try:
        print("[DOWNLOAD] Downloading audio via yt-dlp...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded_file = ydl.prepare_filename(info)
            base, _ = os.path.splitext(downloaded_file)
            wav_file = base + ".wav"
            return wav_file
    except Exception as e:
        print(f"\n[ERROR] {e}")
        return None


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to mono 16kHz WAV using ffmpeg."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000", output_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    """Split WAV into chunks using ffmpeg — no pydub needed."""
    chunk_secs = chunk_minutes * 60
    chunks = []
    i = 0
    while True:
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(i * chunk_secs),
                "-i", wav_path,
                "-t", str(chunk_secs),
                "-ac", "1", "-ar", "16000",
                chunk_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if (
            result.returncode != 0
            or not os.path.exists(chunk_path)
            or os.path.getsize(chunk_path) < 1000
        ):
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
            break

        chunks.append(chunk_path)
        i += 1

    return chunks


def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    if not wav_path or not os.path.exists(wav_path):
        raise FileNotFoundError(
            f"Audio file not found: {wav_path}\n\n"
            "nsig extraction is failing for this YouTube player version.\n"
            "Please update yt-dlp nightly build:\n"
            "  pip install -U --pre yt-dlp\n"
            "Then try again."
        )

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created")
    return chunks