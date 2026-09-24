import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MY_EMAIL = os.getenv("MY_EMAIL", "").lower()
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_store")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    temperature=0.2,  
)