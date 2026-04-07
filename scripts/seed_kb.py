"""
Seed the Supabase knowledge base with a curated set of RAG / retrieval papers.

Usage:
    1. Make sure SQL migration has been applied (section col, content_tsv, hybrid_match_chunks)
    2. Start the server:    uvicorn app.server:app --reload
    3. In another shell:    python scripts/seed_kb.py

What it does:
    1. Deletes all rows from `papers` (cascades to chunks) via the Supabase client
    2. POSTs /papers/load for each URL below, re-using the server's embedder + ingestion pipeline
    3. Prints a compact report of what was loaded
"""
import sys
import time
from pathlib import Path

# Make the project root importable regardless of how this script is invoked
# (`python scripts/seed_kb.py` vs `python -m scripts.seed_kb`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from app.config import supabase

SERVER = "http://127.0.0.1:8000"

# Curated RAG / retrieval cluster — arxiv PDFs, public, thematically linked
PAPERS: list[tuple[str, str]] = [
    ("Attention Is All You Need",                       "https://arxiv.org/pdf/1706.03762"),
    ("BERT: Pre-training of Deep Bidirectional Transformers", "https://arxiv.org/pdf/1810.04805"),
    ("Dense Passage Retrieval for Open-Domain QA (DPR)","https://arxiv.org/pdf/2004.04906"),
    ("REALM: Retrieval-Augmented Language Model",       "https://arxiv.org/pdf/2002.08909"),
    ("Retrieval-Augmented Generation (RAG)",            "https://arxiv.org/pdf/2005.11401"),
    ("Fusion-in-Decoder (FiD)",                         "https://arxiv.org/pdf/2007.01282"),
    ("ColBERT: Efficient Passage Retrieval",            "https://arxiv.org/pdf/2004.12832"),
    ("HyDE: Hypothetical Document Embeddings",          "https://arxiv.org/pdf/2212.10496"),
    ("Self-RAG",                                        "https://arxiv.org/pdf/2310.11511"),
    ("MultiHop-RAG Benchmark",                          "https://arxiv.org/pdf/2401.15391"),
]


def wipe():
    print("[wipe] deleting all papers (cascades to chunks)...")
    # .neq on a never-matching value is the supabase-py idiom for "delete all"
    supabase.table("papers").delete().neq("id", "__never__").execute()
    print("[wipe] done")


def load_one(client: httpx.Client, title: str, url: str) -> dict:
    t0 = time.time()
    r = client.post(f"{SERVER}/papers/load", json={"pdf_url": url, "title": title}, timeout=120)
    dt = time.time() - t0
    if r.status_code != 200:
        return {"title": title, "ok": False, "status": r.status_code, "error": r.text[:200], "elapsed": dt}
    data = r.json()
    return {
        "title": title,
        "ok": True,
        "paper_id": data.get("paper_id"),
        "num_chunks": data.get("num_chunks"),
        "already_existed": data.get("already_existed", False),
        "elapsed": dt,
    }


def main():
    # Sanity check the server is up before wiping anything
    try:
        httpx.get(f"{SERVER}/docs", timeout=3)
    except Exception as e:
        print(f"[error] server not reachable at {SERVER}: {e}", file=sys.stderr)
        print("Start it with:  uvicorn app.server:app --reload", file=sys.stderr)
        sys.exit(1)

    wipe()

    print(f"[load] ingesting {len(PAPERS)} papers...")
    results = []
    with httpx.Client() as client:
        for title, url in PAPERS:
            print(f"  → {title}")
            res = load_one(client, title, url)
            results.append(res)
            if res["ok"]:
                print(f"    id={res['paper_id']}  chunks={res['num_chunks']}  ({res['elapsed']:.1f}s)")
            else:
                print(f"    FAILED status={res['status']}: {res['error']}")

    # Summary
    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]
    total_chunks = sum(r.get("num_chunks") or 0 for r in ok)
    print()
    print(f"[done] {len(ok)}/{len(results)} papers loaded, {total_chunks} chunks total")
    if fail:
        print(f"[done] {len(fail)} failures:")
        for r in fail:
            print(f"   - {r['title']}: {r['error']}")


if __name__ == "__main__":
    main()
