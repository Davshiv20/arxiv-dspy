import config
from papers import search_arxiv
from ingestion import load_paper, chunk_text
from retriever import build_index
from rag import RAG, Summarizer


def pick_paper(papers: list) -> dict:
    print("\nFound papers:")
    for i, p in enumerate(papers):
        print(f"  [{i}] {p['title']}")
    choice = int(input("\nPick a paper [0-{}]: ".format(len(papers) - 1)))
    return papers[choice]


def main():
    # 1. User drives the search
    topic = input("What topic do you want to explore? ")
    papers = search_arxiv(topic, max_results=5)

    # 2. User picks a paper
    paper = pick_paper(papers)
    print(f"\nLoading: {paper['title']}")

    # 3. Load + chunk + index
    text = load_paper(paper["pdf_url"])
    print(f"Extracted {len(text)} characters")
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks, building FAISS index...")
    build_index(chunks)

    # 4. Summarize
    print("\n--- SUMMARY ---")
    summarizer = Summarizer()
    summary = summarizer(document=text)
    print(summary.summary)

    # 5. Interactive Q&A
    print("\n--- Q&A (type 'exit' to quit) ---")
    rag = RAG()
    while True:
        question = input("\nQ: ")
        if question.lower() == "exit":
            break
        answer = rag(question=question)
        print(f"A: {answer.answer}")


if __name__ == "__main__":
    main()
