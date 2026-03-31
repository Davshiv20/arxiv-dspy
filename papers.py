import arxiv

client = arxiv.Client()
def search_arxiv(query: str, max_results:int =5):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    results = []
    for result in client.results(search):
        results.append({
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "summary": result.summary,
            "published": result.published,
            "pdf_url": result.pdf_url
        })
    
    return results