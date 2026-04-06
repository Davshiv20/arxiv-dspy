from app.config import supabase  # noqa: F401 — triggers dspy.configure
from sentence_transformers import SentenceTransformer
from app.services.arxiv import search_arxiv
from app.services.ingestion import load_paper, chunk_text
from app.services.store import add_paper, retrieve_chunks, make_paper_id
from app.services.rag import RAG, Summarizer

embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")


def pick_paper(papers: list) -> dict:
    print("\nFound papers:")
    for i, p in enumerate(papers):
        print(f"  [{i}] {p['title']}")
    choice = int(input("\nPick a paper [0-{}]: ".format(len(papers) - 1)))
    return papers[choice]


def main():
    topic = input("What topic do you want to explore? ")
    papers = search_arxiv(topic, max_results=5)

    paper = pick_paper(papers)
    print(f"\nLoading: {paper['title']}")

    text = load_paper(paper["pdf_url"])
    print(f"Extracted {len(text)} characters")
    chunks = chunk_text(text)
    paper_id = make_paper_id(paper["pdf_url"])
    num = add_paper(embedder, paper_id, paper["pdf_url"], paper["title"], text, chunks)
    if num:
        print(f"Created {num} chunks, stored in Supabase")
    else:
        print("Paper already indexed, reusing existing chunks")

    print("\n--- SUMMARY ---")
    summarizer = Summarizer()
    summary = summarizer(document=text)
    print(summary.output.summary)

    print("\n--- Q&A (type 'exit' to quit) ---")
    rag = RAG(embedder)
    while True:
        question = input("\nQ: ")
        if question.lower() == "exit":
            break
        answer = rag(question=question, paper_id=paper_id)
        print(f"A: {answer.output.answer}")


if __name__ == "__main__":
    main()
