import requests
import fitz  # pymupdf
import io


def load_paper(pdf_url: str) -> str:
    response = requests.get(pdf_url)
    response.raise_for_status()
    doc = fitz.open(stream=io.BytesIO(response.content), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
