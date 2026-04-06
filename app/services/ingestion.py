import re
import io
import requests
import fitz
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_CHUNK_SIZE = 1000       # characters (~200-250 tokens for most models)
DEFAULT_OVERLAP_PIECES = 2      # number of sentences to overlap between chunks

# Section headers: all-caps headings, or Roman numeral + title (common in arXiv)
_SECTION_PATTERN = re.compile(
    r'\n(?=([A-Z][A-Z\s]{2,}|(?:[IVX]+\.\s+)[A-Z][^\n]{0,60})\n)'
)

# References/bibliography markers — extended to catch more variants
_REFERENCES_PATTERN = re.compile(
    r'\n\s*(References|REFERENCES|Bibliography|BIBLIOGRAPHY|Works Cited|Further Reading)\s*\n',
    re.IGNORECASE,
)


# ── PDF Loading ──────────────────────────────────────────────────────────────

def load_paper(pdf_url: str) -> str:
    """Download a PDF from a URL and extract its raw text."""
    response = requests.get(pdf_url, timeout=30)
    response.raise_for_status()

    doc = fitz.open(stream=io.BytesIO(response.content), filetype="pdf")
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()


# ── Cleaning ─────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Fix common PDF extraction artifacts."""
    # Rejoin hyphenated line breaks (e.g. "compu-\ntation" → "computation")
    text = re.sub(r'-\n(\w)', r'\1', text)
    # Remove lone page numbers (a number on its own line)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    # Collapse 3+ newlines into a paragraph break
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip leading/trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def _strip_references(text: str) -> str:
    """Remove the references section and everything after it."""
    match = _REFERENCES_PATTERN.search(text)
    if match:
        return text[:match.start()]
    return text


# ── Section Splitting ────────────────────────────────────────────────────────

def _split_into_sections(text: str) -> list[str]:
    """
    Split the paper at detected section headers.
    Keeps each section as a contiguous block so chunks never span
    across e.g. Abstract → Introduction or Methods → Results.
    """
    parts = _SECTION_PATTERN.split(text)
    # _SECTION_PATTERN uses a lookahead so split() returns alternating
    # [before, header, after, header, ...]; collapse back into sections.
    sections = []
    current = parts[0]
    for part in parts[1:]:
        if _SECTION_PATTERN.match("\n" + part):
            if current.strip():
                sections.append(current.strip())
            current = part
        else:
            current += "\n" + part
    if current.strip():
        sections.append(current.strip())

    return sections if sections else [text]


# ── Chunking ─────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """
    Sentence-tokenize using NLTK, which correctly handles abbreviations
    like 'et al.', 'Fig.', 'e.g.', 'vs.' that naive regex splits on.
    """
    return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]


def _merge_into_chunks(
    pieces: list[str],
    target_size: int,
    overlap_pieces: int,
) -> list[str]:
    """
    Merge sentence/paragraph pieces into chunks of ~target_size characters.
    Overlap is piece-based (not character-based) to avoid mid-sentence cuts.
    """
    chunks = []
    i = 0

    while i < len(pieces):
        current: list[str] = []
        size = 0

        while i < len(pieces) and size + len(pieces[i]) + 1 <= target_size:
            current.append(pieces[i])
            size += len(pieces[i]) + 1
            i += 1

        # If a single piece exceeds target_size, take it alone rather than skip it
        if not current:
            current.append(pieces[i])
            i += 1

        chunks.append(" ".join(current))

        # Step back by overlap_pieces for the next chunk window
        if i < len(pieces):
            i = max(i - overlap_pieces, i - len(current) + 1)

    return chunks


def _chunk_section(
    section: str,
    chunk_size: int,
    overlap_pieces: int,
) -> list[str]:
    """
    Chunk a single section:
    - Keep small paragraphs whole
    - Sentence-tokenize large paragraphs before merging
    """
    paragraphs = [p.strip() for p in section.split("\n\n") if p.strip()]

    pieces: list[str] = []
    for para in paragraphs:
        if len(para) > chunk_size:
            pieces.extend(_split_sentences(para))
        else:
            pieces.append(para)

    return _merge_into_chunks(pieces, chunk_size, overlap_pieces)


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap_pieces: int = DEFAULT_OVERLAP_PIECES,
) -> list[str]:
    """
    Full pipeline:
      1. Clean PDF artifacts
      2. Strip references section
      3. Split by section headers (section-aware boundaries)
      4. Within each section: paragraph split → sentence split → merge with overlap

    Args:
        text:           Raw text extracted from a PDF.
        chunk_size:     Target chunk size in characters. ~1000 chars ≈ 200-250 tokens.
        overlap_pieces: Number of sentences shared between consecutive chunks.

    Returns:
        List of text chunks ready for embedding.
    """
    text = _clean_text(text)
    text = _strip_references(text)
    sections = _split_into_sections(text)

    chunks: list[str] = []
    for section in sections:
        chunks.extend(_chunk_section(section, chunk_size, overlap_pieces))

    return chunks