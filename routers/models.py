import re
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from langchain_core.messages import HumanMessage, ToolMessage

from config.session import get_db
from dataaccess.data_models import AgentQueryLog, get_collection
from business_logic.models import AskRequest, IngestRequest
from business_logic.email_logic import load_emails, index_emails
from agent_graph import app_graph

router = APIRouter()


@router.post("/ingest", tags=["Mailbox"], summary="Fetch emails, chunk and embed them into the vector store")
def ingest(request: IngestRequest):
    try:
        emails = load_emails(request.source, limit=request.limit)
        chunks = index_emails(emails)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ingestion failed: {e}")
    return {"emails_loaded": len(emails), "chunks_indexed": chunks}


@router.post("/ask", tags=["Agent"], summary="Ask about your emails — the agent picks the right tool(s)")
def ask(request: AskRequest, db: Session = Depends(get_db)):
    try:
        result = app_graph.invoke({"messages": [HumanMessage(content=request.question)]})
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"The agent could not answer right now: {e}")

    messages = result["messages"]
    final_answer = messages[-1].content
    tools_used = [m.name for m in messages if isinstance(m, ToolMessage)]
    sources = list(dict.fromkeys(re.findall(r"\[(msg-[\w]+)\]", final_answer)))  # unique, in order

    log = AgentQueryLog(
        question=request.question,
        tool_used=", ".join(tools_used) if tools_used else "none",
        sources=", ".join(sources) if sources else "none",
        answer=final_answer,
    )
    db.add(log)
    db.commit()

    return {"question": request.question, "tools_used": tools_used, "sources": sources, "answer": final_answer}


@router.get("/logs", tags=["Agent"], summary="See every question asked, which tools ran, and which emails were cited")
def get_logs(db: Session = Depends(get_db)):
    return db.query(AgentQueryLog).order_by(AgentQueryLog.id.desc()).limit(100).all()


@router.get("/health", tags=["System"], summary="Check the API is alive")
def health_check():
    return {"status": "ok", "indexed_chunks": get_collection().count()}
