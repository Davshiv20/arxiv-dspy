import app.config  # noqa: F401 — triggers dspy.configure on import
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

from app.routes import search_router, papers_router, agent_router
from app.services.agent import set_embedder


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    set_embedder(app.state.embedder)
    yield


app = FastAPI(title="Arxiv Research API", lifespan=lifespan)

app.include_router(search_router)
app.include_router(papers_router)
app.include_router(agent_router)
