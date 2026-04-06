from fastapi import APIRouter
from app.schemas import SearchRequest, SearchResponse, PaperResult
from app.services.arxiv import search_arxiv

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    results = search_arxiv(req.query, req.max_results)
    return SearchResponse(papers=[PaperResult(**r) for r in results])
