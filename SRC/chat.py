import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = str(BASE_DIR / "chroma_db")
COLLECTION = "policy_docs"

# ============================================
# 🔑 USER SETUP REQUIRED
# ============================================
# 1. Create a .env file in the project root
# 2. Add your API key like this:
#
#    OPENROUTER_API_KEY=your_key_here
#
# 3. (Optional) You may switch providers:
#    - OpenRouter (default below)
#    - OpenAI, Anthropic, etc.
#    If you switch, update base_url + model
# ============================================

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "mistralai/mistral-small-3.2-24b-instruct"

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
        include=["documents", "metadatas"],
    )

    docs_result = results.get("documents")
    metas_result = results.get("metadatas")
    distances_result = results.get("distances")

    docs = docs_result[0] if docs_result and len(docs_result) > 0 else []
    metas = metas_result[0] if metas_result and len(metas_result) > 0 else []
    distances = (
        distances_result[0]
        if distances_result and len(distances_result) > 0
        else []
    )

    if not docs:
        return ""

    context_parts = []

    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) and metas[i] else {}
        distance = distances[i] if i < len(distances) else None

        source = meta.get("source", "unknown source")
        chunk = meta.get("chunk", "unknown chunk")

        if distance is not None:
            header = f"Source: {source}, chunk {chunk}, distance {distance:.4f}"
        else:
            header = f"Source: {source}, chunk {chunk}"

        context_parts.append(f"{header}\n{doc}")

    return "\n\n---\n\n".join(context_parts)


def preview_context(context, max_chars=400, max_chunks=2):
    parts = context.split("\n\n---\n\n")[:max_chunks]
    short = "\n\n---\n\n".join(parts)

    if len(short) > max_chars:
        short = short[:max_chars].rstrip() + "..."

    return short


def ask(question, debug=True):
    context = retrieve(question)

    if debug:
        print("\n--- CONTEXT PREVIEW ---")
        print(preview_context(context) if context else "[No context retrieved]")
        print("--- END PREVIEW ---\n")

    if not context:
        return "The provided context is not sufficient to answer this."

    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful research assistant. "
                "Answer using only the provided context. "
                "If the context is insufficient, say exactly: "
                '"The provided context is not sufficient to answer this."'
            ),
        },
        {
            "role": "user",
            "content": f"""
Question:
{question}

Context:
{context}

Write a clear answer with:
1. A direct answer
2. Key evidence from the context
3. Any limitations or uncertainty
""",
        },
    ]

    try:
        response = llm.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=800,
            temperature=0.3,
        )

        if not response or not response.choices:
            print("\n--- RAW LLM RESPONSE ---")
            print(response)
            print("--- END RAW LLM RESPONSE ---\n")
            return "LLM response failed: no choices were returned."

        message = response.choices[0].message

        if not message or not message.content:
            print("\n--- RAW LLM RESPONSE ---")
            print(response)
            print("--- END RAW LLM RESPONSE ---\n")
            return "LLM response failed: no message content returned."

        return message.content

    except Exception as e:
        return f"LLM request failed: {e}"


def main():
    print("Policy LLM")
    print("Type 'exit' or 'quit' to stop.")
    print("Type 'debug off' to hide retrieved context previews.")
    print("Type 'debug on' to show retrieved context previews.")
    print()

    debug = True

    while True:
        question = input("You: ").strip()

        if not question:
            continue

        command = question.lower()

        if command in ["exit", "quit"]:
            break

        if command == "debug off":
            debug = False
            print("Debug context preview disabled.\n")
            continue

        if command == "debug on":
            debug = True
            print("Debug context preview enabled.\n")
            continue

        print()
        print(ask(question, debug=debug))
        print()


if __name__ == "__main__":
    main()