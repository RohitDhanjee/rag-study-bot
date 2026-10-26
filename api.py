import time

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag import answer

app = FastAPI(title="Computer Architecture RAG Chatbot")


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    k: int = Field(default=3, ge=1, le=10)


class Source(BaseModel):
    page: int
    text: str


class Answer(BaseModel):
    answer: str
    sources: list[Source]
    latency: float


@app.get("/health")
def health():
    """Liveness check: deployment platforms poll this endpoint."""
    return {"status": "ok"}


@app.post("/ask", response_model=Answer)
def ask(q: Question):
    start = time.time()
    reply, results = answer(q.question, k=q.k)
    return Answer(
        answer=reply,
        sources=[Source(page=c["page"], text=c["text"]) for _, c in results],
        latency=round(time.time() - start, 2),
    )
