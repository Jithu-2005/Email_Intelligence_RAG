from contextlib import asynccontextmanager
from fastapi import FastAPI

from config.session import engine, Base
from dataaccess.data_models import get_collection
from business_logic.email_logic import load_emails, index_emails
from routers.models import router

Base.metadata.create_all(bind=engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if get_collection().count() == 0:
        index_emails(load_emails("sample"))
    yield


app = FastAPI(
    title="MailMind — Email Intelligence RAG",
    description="Ask questions about a mailbox in plain English. The agent searches the indexed emails, "
                "then answers, summarizes, extracts details, or drafts replies — always with citations.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
