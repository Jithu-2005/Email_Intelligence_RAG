from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str


class IngestRequest(BaseModel):
    source: str = "sample"
    limit: int = 100
