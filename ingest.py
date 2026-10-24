import re
from collections import Counter
from pathlib import Path
from pypdf import PdfReader


def load_pdf(path):
    """PDF ke har page ka saaf text nikalo, page number ke saath."""
    reader = PdfReader(path)
    raw_pages = [(page.extract_text() or "").splitlines() for page in reader.pages]

    # Jo line aadhe se zyada pages par repeat ho, wo header/footer hai
    counts = Counter(line.strip() for lines in raw_pages for line in set(lines))
    repeated = {line for line, n in counts.items() if n > len(raw_pages) / 2}

    pages = []
    for i, lines in enumerate(raw_pages):
        kept = [
            line for line in lines
            if line.strip() not in repeated
            and not re.fullmatch(r"Page \d+", line.strip())   # "Page 3" jaisa footer
        ]
        text = " ".join(" ".join(kept).split())
        if text:
            pages.append({"page": i + 1, "text": text})
    return pages


def chunk_text(text, chunk_size=500, overlap=100):
    """Text ko overlapping tukdon mein todo."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):               # aakhri tukda ho gaya, ruk jao
            break
        start += chunk_size - overlap      # 650 aage barho
    return chunks


def build_chunks(pdf_path):
    """Poori PDF -> chunks ki list, har chunk ke saath uska page number."""
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
    pdf = next(Path("data").glob("*.pdf"))   # data folder ki pehli PDF
    chunks = build_chunks(pdf)

    print(f"PDF: {pdf.name}")
    print(f"Total chunks: {len(chunks)}")
    print("\n--- Chunk 1 (aakhir) ---")
    print("..." + chunks[1]["text"][-200:])
    print("\n--- Chunk 2 (shuru) ---")
    print(chunks[2]["text"][:200] + "...")