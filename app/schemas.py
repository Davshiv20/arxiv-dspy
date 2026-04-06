from datetime import datetime
from pydantic import BaseModel


# --- Search ---

class SearchRequest(BaseModel):
    query: str
    max_results: int = 5


class PaperResult(BaseModel):
    title: str
    authors: list[str]
    summary: str
    published: datetime | str
    pdf_url: str


class SearchResponse(BaseModel):
    papers: list[PaperResult]


# --- Load ---

class LoadRequest(BaseModel):
    pdf_url: str
    title: str | None = None


class LoadResponse(BaseModel):
    paper_id: str
    num_chunks: int
    title: str | None = None
    already_existed: bool = False


# --- Stored papers ---

class PaperInfo(BaseModel):
    paper_id: str
    title: str | None
    pdf_url: str
    num_chunks: int = 0
    created_at: datetime | str | None = None


class PapersListResponse(BaseModel):
    papers: list[PaperInfo]


# --- Summarize ---

class SummarizeResponse(BaseModel):
    paper_id: str
    summary: str


# --- Q&A ---

class QARequest(BaseModel):
    question: str


class QAResponse(BaseModel):
    paper_id: str
    question: str
    answer: str
    context: list[str]


# --- Agent ---

class AgentRequest(BaseModel):
    question: str


class AgentResponse(BaseModel):
    question: str
    answer: str
    reasoning: str | None = None


# --- DSPy typed signature models ---

class QAInput(BaseModel):
    context: str
    question: str


class QAOutput(BaseModel):
    answer: str
    confidence: str


class SummaryInput(BaseModel):
    document: str


class SummaryOutput(BaseModel):
    summary: str
    key_findings: list[str]


class ResearchInput(BaseModel):
    question: str


class ResearchOutput(BaseModel):
    answer: str
    reasoning: str
