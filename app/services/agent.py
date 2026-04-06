import hashlib
import dspy
from sentence_transformers import SentenceTransformer
from app.services.arxiv import search_arxiv as _search_arxiv
from app.services.ingestion import load_paper, chunk_text
from app.services.store import add_paper, retrieve_chunks
from app.schemas import ResearchInput, ResearchOutput

_last_paper_id: str | None = None
_embedder: SentenceTransformer | None = None


def set_embedder(embedder: SentenceTransformer):
    global _embedder
    _embedder = embedder


def search_papers(query: str) -> str:
    """Search arxiv for papers relevant to a query. Returns titles and summaries."""
    papers = _search_arxiv(query, max_results=3)
    output = []
    for p in papers:
        output.append(f"Title: {p['title']}\nSummary: {p['summary'][:300]}\nURL: {p['pdf_url']}")
    return "\n\n".join(output)


def load_and_index_paper(pdf_url: str) -> str:
    """Download a paper from a PDF URL, extract its text, and index it for retrieval."""
    global _last_paper_id
    paper_id = hashlib.md5(pdf_url.encode()).hexdigest()[:10]
    _last_paper_id = paper_id
    text = load_paper(pdf_url)
    chunks = chunk_text(text)
    num = add_paper(_embedder, paper_id, pdf_url, None, text, chunks)
    if num == 0:
        return f"Paper already indexed as paper_id={paper_id}. Ready for retrieval."
    return f"Paper loaded and indexed as paper_id={paper_id}. {num} chunks ready."


def retrieve_from_paper(query: str) -> str:
    """Retrieve relevant chunks from the most recently indexed paper based on a query."""
    if _last_paper_id is None:
        return "No paper loaded yet. Use load_and_index_paper first."
    chunks = retrieve_chunks(_embedder, _last_paper_id, query, k=4)
    return "\n\n".join(chunks)


class ResearchSignature(dspy.Signature):
    """You are a research assistant. Search for papers, load them, and answer questions."""
    input: ResearchInput = dspy.InputField(desc="The research question")
    output: ResearchOutput = dspy.OutputField(desc="Answer with reasoning")


class ResearchAgent(dspy.Module):
    def __init__(self):
        self.agent = dspy.ReAct(
            ResearchSignature,
            tools=[search_papers, load_and_index_paper, retrieve_from_paper],
        )

    def forward(self, question: str) -> dspy.Prediction:
        return self.agent(input=ResearchInput(question=question))
