import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

index = None
stored_chunks = []


def build_index(chunks: list[str]):
    global index, stored_chunks
    stored_chunks = chunks
    embeddings = model.encode(chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))


def retrieve(query: str, k: int = 5) -> list[str]:
    query_vec = model.encode([query], convert_to_numpy=True).astype(np.float32)
    _, indices = index.search(query_vec, k)
    return [stored_chunks[i] for i in indices[0] if i < len(stored_chunks)]
