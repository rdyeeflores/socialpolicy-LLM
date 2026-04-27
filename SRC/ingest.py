from pathlib import Path
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "DATA" / "raw"
DB_DIR = str(BASE_DIR / "chroma_db")
COLLECTION = "policy_docs"

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection(COLLECTION)


def read_file(path):
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")

    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    return None


def chunk_text(text, size=800, overlap=200):
    chunks = []
    start = 0

    while start < len(text):
        chunk = text[start:start + size].strip()

        if chunk:
            chunks.append(chunk)

        start += size - overlap

    return chunks


def main():
    print(f"Looking for files in: {RAW_DIR}\n")

    files = list(RAW_DIR.glob("*"))

    if not files:
        print(f"No files found in {RAW_DIR}")
        return

    for path in files:
        print(f"Processing: {path.name}")

        text = read_file(path)

        if not text or not text.strip():
            print(f"Skipped or unreadable: {path.name}\n")
            continue

        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            collection.upsert(
                ids=[f"{path.name}-{i}"],
                documents=[chunk],
                embeddings=[embedder.encode(chunk).tolist()],
                metadatas=[{"source": path.name, "chunk": i}],
            )

        print(f"Ingested {path.name}: {len(chunks)} chunks\n")

    print("Ingestion complete.")


if __name__ == "__main__":
    main()