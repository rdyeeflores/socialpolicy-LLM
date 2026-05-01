from pathlib import Path
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer
import hashlib

# NOTE:
# This file does NOT require any API key.
# It builds a local vector database from your documents.

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


def chunk_text(text, size=2000, overlap=300):
    """
    Approx ~500–1000 tokens depending on text density
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += size - overlap

    return chunks


def make_doc_id(path, chunk_index):
    """
    Create stable unique ID using file path + chunk index
    """
    base = f"{path.resolve()}-{chunk_index}"
    return hashlib.md5(base.encode()).hexdigest()


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

        # Batch embeddings (much faster)
        embeddings = embedder.encode(chunks, show_progress_bar=False)

        ids = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            ids.append(make_doc_id(path, i))
            metadatas.append({
                "source": path.name,
                "chunk": i,
                "path": str(path.resolve())
            })

        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

        print(f"Ingested {path.name}: {len(chunks)} chunks\n")

    print("Ingestion complete.")


if __name__ == "__main__":
    main()