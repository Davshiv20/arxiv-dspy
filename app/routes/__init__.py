from app.routes.search import router as search_router
from app.routes.papers import router as papers_router
from app.routes.agent import router as agent_router

__all__ = ["search_router", "papers_router", "agent_router"]
