import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = str(BASE_DIR / "chroma_db")
COLLECTION = "policy_docs"

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "mistralai/mistral-small-3.1-24b-instruct"

if not API_KEY:
    raise ValueError("Missing OPENROUTER_API_KEY in .env")

llm = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
db = chromadb.PersistentClient(path=DB_DIR)
collection = db.get_or_create_collection(COLLECTION)


def retrieve(question, n=5):
    query_embedding = embedder.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    context_parts = []

    for doc, meta in zip(docs, metas):
        context_parts.append(
            f"Source: {meta['source']}, chunk {meta['chunk']}\n{doc}"
        )

    return "\n\n---\n\n".join(context_parts)


def ask(question, debug=True):
    context = retrieve(question)

    if debug:
        print("\n--- RETRIEVED CONTEXT PREVIEW ---")
        print(context[:2000] if context else "[No context retrieved]")
        print("--- END CONTEXT PREVIEW ---\n")

    prompt = f"""
You are a careful research assistant.

Answer the question using only the provided context.

Question:
{question}

Context:
{context}

Write a clear answer with:
1. A direct answer
2. Key evidence from the context
3. Any limitations or uncertainty

If the context does not contain enough information, say exactly:
"The provided context is not sufficient to answer this."
"""

    response = llm.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=800,
        temperature=0.3,
    )

    return response.choices[0].message.content


def main():
    print("Policy LLM")
    print("Type 'exit' or 'quit' to stop.")
    print("Type 'debug off' to hide retrieved context previews.")
    print("Type 'debug on' to show retrieved context previews.")
    print()

    debug = True

    while True:
        question = input("You: ")

        if question.lower() in ["exit", "quit"]:
            break

        if question.lower() == "debug off":
            debug = False
            print("Debug context preview disabled.\n")
            continue

        if question.lower() == "debug on":
            debug = True
            print("Debug context preview enabled.\n")
            continue

        print()
        print(ask(question, debug=debug))
        print()


if __name__ == "__main__":
    main()