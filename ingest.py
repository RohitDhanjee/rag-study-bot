import re
from collections import Counter
from pathlib import Path
from pypdf import PdfReader


def load_pdf(path):
    """Extract clean text from every PDF page, keeping the page number."""
    reader = PdfReader(path)
    raw_pages = [(page.extract_text() or "").splitlines() for page in reader.pages]

    # A line repeated on more than half of the pages is a header or footer
    counts = Counter(line.strip() for lines in raw_pages for line in set(lines))
    repeated = {line for line, n in counts.items() if n > len(raw_pages) / 2}

    pages = []
    for i, lines in enumerate(raw_pages):
        kept = [
            line for line in lines
            if line.strip() not in repeated
            and not re.fullmatch(r"Page \d+", line.strip())   # footer such as "Page 3"
        ]
        text = " ".join(" ".join(kept).split())
        if text:
            pages.append({"page": i + 1, "text": text})
    return pages


def chunk_text(text, chunk_size=500, overlap=100):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):               # last chunk reached, stop
            break
        start += chunk_size - overlap      # move forward by chunk_size - overlap
    return chunks


def build_chunks(pdf_path):
    """Whole PDF -> list of chunks, each with its page number."""
    all_chunks = []
    for p in load_pdf(pdf_path):
        for c in chunk_text(p["text"]):
            all_chunks.append({
                "id": len(all_chunks),
                "page": p["page"],
                "source": Path(pdf_path).name,
                "text": c,
            })
    return all_chunks


if __name__ == "__main__":
    pdf = next(Path("data").glob("*.pdf"))   # first PDF in the data folder
    chunks = build_chunks(pdf)

    print(f"PDF: {pdf.name}")
    print(f"Total chunks: {len(chunks)}")
    print("\n--- Chunk 1 (end) ---")
    print("..." + chunks[1]["text"][-200:])
    print("\n--- Chunk 2 (start) ---")
    print(chunks[2]["text"][:200] + "...")
