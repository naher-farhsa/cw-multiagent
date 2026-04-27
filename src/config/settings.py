"""
settings.py — Central configuration for the Content-Writing Multi-Agent system.

Responsibilities:
  - Load environment variables from the .env file (API keys)
  - Define model names, search limits, and recursion caps
  - Resolve the absolute path for the output directory and create it if absent
  - Expose a validate_keys() helper so main.py can fail fast if keys are missing

All other modules import their constants from here; do NOT hard-code values
in individual agent or tool files.
"""

import os
from pathlib import Path
from dotenv import load_dotenv  # reads key=value pairs from .env into os.environ

# ── Load .env ──────────────────────────────────────────────────────────────────
# load_dotenv() looks for a .env file in the current working directory (or any
# parent directory) and injects the variables into the process environment.
# This must run before any os.getenv() calls.
load_dotenv()

# ── API Keys ───────────────────────────────────────────────────────────────────
# Fetched from the environment; will be None if the variable is not set.
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY: str | None = os.getenv("TAVILY_API_KEY")

# ── Model / Search Settings ────────────────────────────────────────────────────
# The Groq-hosted Llama-3.3 model used for all LLM calls.
GROQ_MODEL: str = "llama-3.3-70b-versatile"

# Maximum number of search results Tavily should return per query.
TAVILY_MAX_RESULTS: int = 3

# Hard cap on how many LangGraph node executions can happen in one run.
# Prevents runaway loops (supervisor ↔ agents cycling indefinitely).
RECURSION_LIMIT: int = 30

# ── Output Directory ───────────────────────────────────────────────────────────
# __file__ is  …/src/config/settings.py
# .parents[2]  is  …/  (the project root, two levels up)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# All generated Markdown articles are written into this folder.
OUTPUT_DIR: Path = PROJECT_ROOT / "doc-result"
OUTPUT_DIR.mkdir(exist_ok=True)  # create the folder on import if it doesn't exist


# ── Validation ─────────────────────────────────────────────────────────────────
def validate_keys() -> None:
    """Check that all required API keys are present before starting the workflow.

    Iterates over the keys that must be set in the .env file and raises a
    descriptive ``EnvironmentError`` listing every missing variable so the user
    knows exactly what to add.

    Raises:
        EnvironmentError: If one or more API key environment variables are empty
                          or not set.

    Example .env file::

        GROQ_API_KEY=gsk_...
        TAVILY_API_KEY=tvly-...
    """
    # Build a dict of name → value for every key we depend on
    required = {
        "GROQ_API_KEY": GROQ_API_KEY,
        "TAVILY_API_KEY": TAVILY_API_KEY,
    }
    # Collect the names of any key whose value is falsy (None or empty string)
    missing = [name for name, value in required.items() if not value]

    if missing:
        raise EnvironmentError(
            f"Missing required environment variables in .env: {', '.join(missing)}"
        )
