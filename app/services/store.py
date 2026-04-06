import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import supabase


def make_paper_id(pdf_url: str) -> str:
    return hashlib.md5(pdf_url.encode()).hexdigest()[:10]


def add_paper(embedder: SentenceTransformer, paper_id: str, pdf_url: str, title: str | None, raw_text: str, chunks: list[str]) -> int:
    if has_paper(paper_id):
        return 0

    supabase.table("papers").insert({
        "id": paper_id,
        "title": title,
        "pdf_url": pdf_url,
        "raw_text": raw_text,
    }).execute()

    embeddings = embedder.encode(chunks, convert_to_numpy=True).astype(np.float32)

    rows = [
        {
            "paper_id": paper_id,
            "content": chunk,
            "embedding": embedding.tolist(),
        }
        for chunk, embedding in zip(chunks, embeddings)
    ]

    for i in range(0, len(rows), 500):
        supabase.table("chunks").insert(rows[i:i + 500]).execute()

    return len(chunks)


def retrieve_chunks(embedder: SentenceTransformer, paper_id: str, query: str, k: int = 5) -> list[str]:
    query_embedding = embedder.encode([query], convert_to_numpy=True).astype(np.float32)[0]

    result = supabase.rpc("match_chunks", {
        "query_embedding": query_embedding.tolist(),
        "target_paper_id": paper_id,
        "match_count": k,
    }).execute()

    return [row["content"] for row in result.data]


def has_paper(paper_id: str) -> bool:
    result = supabase.table("papers").select("id").eq("id", paper_id).limit(1).execute()
    return len(result.data) > 0


def get_paper(paper_id: str) -> dict | None:
    result = supabase.table("papers").select("*").eq("id", paper_id).limit(1).execute()
    if result.data:
        return result.data[0]
    return None


def get_paper_text(paper_id: str) -> str | None:
    paper = get_paper(paper_id)
    if paper:
        return paper["raw_text"]
    return None


def list_papers() -> list[dict]:
    result = supabase.table("papers").select("id, title, pdf_url, created_at, chunks(count)").order("created_at", desc=True).execute()
    papers = []
    for row in result.data:
        papers.append({
            "paper_id": row["id"],
            "title": row["title"],
            "pdf_url": row["pdf_url"],
            "num_chunks": row["chunks"][0]["count"] if row.get("chunks") else 0,
            "created_at": row["created_at"],
        })
    return papers
