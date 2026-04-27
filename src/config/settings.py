import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

GROQ_MODEL = "llama-3.3-70b-versatile"
TAVILY_MAX_RESULTS = 3
RECURSION_LIMIT = 30

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "doc-result"
OUTPUT_DIR.mkdir(exist_ok=True)


def validate_keys() -> None:
    missing = [k for k, v in {"GROQ_API_KEY": GROQ_API_KEY, "TAVILY_API_KEY": TAVILY_API_KEY}.items() if not v]
    if missing:
        raise EnvironmentError(f"Missing required env vars in .env: {', '.join(missing)}")
