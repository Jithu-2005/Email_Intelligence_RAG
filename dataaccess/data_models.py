import os
import chromadb
from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.sql import func

from config.session import Base

class AgentQueryLog(Base):
    __tablename__ = "agent_query_logs"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    tool_used = Column(String)        
    sources = Column(String)          
    answer = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_store")
_client = chromadb.PersistentClient(path=CHROMA_PATH)


def get_collection():
    """The one shared collection that every module reads from and writes to."""
    return _client.get_or_create_collection(name="emails")


def get_thread_text(thread_id: str, max_chars: int = 6000) -> str:
    """All chunks of one thread in date order, each prefixed with its [email_id]."""
    res = get_collection().get(where={"thread_id": thread_id})
    if not res["ids"]:
        return ""
    rows = sorted(
        zip(res["documents"], res["metadatas"]),
        key=lambda r: (r[1]["timestamp"], r[1]["email_id"], r[1]["chunk_index"]),
    )
    text = "\n\n".join(f"[{m['email_id']}] {doc}" for doc, m in rows)
    return text[:max_chars]
