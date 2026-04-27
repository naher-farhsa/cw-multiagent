from langchain_groq import ChatGroq
from src.config.settings import GROQ_MODEL, GROQ_API_KEY

llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0)
