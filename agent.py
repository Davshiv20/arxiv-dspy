import dspy
from papers import search_arxiv as _search_arxiv
from ingestion import load_paper, chunk_text
from retriever import build_index, retrieve


def search_papers(query: str) -> str:
    """Search arxiv for papers relevant to a query. Returns titles and summaries."""
    papers = _search_arxiv(query, max_results=3)
    output = []
    for p in papers:
        output.append(f"Title: {p['title']}\nSummary: {p['summary'][:300]}\nURL: {p['pdf_url']}")
    return "\n\n".join(output)


def load_and_index_paper(pdf_url: str) -> str:
    """Download a paper from a PDF URL, extract its text, and index it for retrieval."""
    text = load_paper(pdf_url)
    chunks = chunk_text(text)
    build_index(chunks)
    return f"Paper loaded and indexed. {len(chunks)} chunks ready for retrieval."


def retrieve_from_paper(query: str) -> str:
    """Retrieve relevant chunks from the indexed paper based on a query."""
    chunks = retrieve(query, k=4)
    return "\n\n".join(chunks)


class ResearchAgent(dspy.Module):
    def __init__(self):
        self.agent = dspy.ReAct(
            "question -> answer",
            tools=[search_papers, load_and_index_paper, retrieve_from_paper]
        )

    def forward(self, question: str) -> dspy.Prediction:
        return self.agent(question=question)


# fastapi server
# chainOfthought vs ReAct
# skills
# tracings - Langfuse, arrise
