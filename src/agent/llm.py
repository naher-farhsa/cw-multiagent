"""
llm.py — Shared LangChain-Groq LLM instance.

A single ``ChatGroq`` object is created here at import time and imported by
every agent that needs to make LLM calls.  Centralising it means:

  * API key and model name are configured in exactly one place (settings.py).
  * All agents share the same model configuration; changing the model only
    requires editing settings.py.
  * ``temperature=0`` ensures deterministic, consistent outputs — important
    for a structured supervisor that must output a valid routing decision.

Usage (from any agent file)::

    from src.agent.llm import llm
"""

from langchain_groq import ChatGroq  # Groq-hosted LLM via LangChain integration
from src.config.settings import GROQ_MODEL, GROQ_API_KEY

# ── Instantiate the shared LLM ─────────────────────────────────────────────────
# temperature=0  → fully deterministic; no random sampling
# api_key        → passed explicitly (read from .env via settings.py)
# model          → Llama-3.3-70b hosted on Groq's fast inference layer
llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0)
