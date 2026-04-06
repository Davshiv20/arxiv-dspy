import arxiv

client = arxiv.Client(page_size=10, delay_seconds=3.0, num_retries=3)

MAX_RESULTS_CAP = 10


def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    max_results = min(max_results, MAX_RESULTS_CAP)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    results = []
    for result in client.results(search):
        results.append({
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "summary": result.summary,
            "published": result.published,
            "pdf_url": result.pdf_url,
        })

    return results
