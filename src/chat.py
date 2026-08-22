
# Set-up
import os
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message=r".*clean_up_tokenization_spaces.*",
    category=FutureWarning,
    module=r"transformers\.tokenization_utils_base",
)

# Main tools (local vector db, API, openai, transformer)
import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = str(BASE_DIR / "chroma_db")
COLLECTION = "policy_docs"

# ==================================================
# 🔑 USER SETUP REQUIRED
# ==================================================
# 1. Create a .env file in the project root
# 2. Add your API key, provider, and model like this:
#
#    API_KEY=your_key_here
#    PROVIDER=https://openrouter.ai/api/v1
#    MODEL=openai/gpt-5.6-luna
# 
# ==================================================

# Loading API key and LLM set-up
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("API_KEY")
PROVIDER = os.getenv("PROVIDER", "https://openrouter.ai/api/v1")
MODEL = os.getenv("MODEL", "openai/gpt-5.6-luna")

if not API_KEY:
    raise ValueError("Missing API_KEY in .env")

llm = OpenAI(
    base_url=PROVIDER,
    api_key=API_KEY,
)

# Embedding and database set-up
embedder = SentenceTransformer("all-MiniLM-L6-v2")
db = chromadb.PersistentClient(path=DB_DIR)
collection = db.get_or_create_collection(COLLECTION)

# Retrieval specifications
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

    # Building the context
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

# Previewing context
def preview_context(context, max_chars=400, max_chunks=2):
    parts = context.split("\n\n---\n\n")[:max_chunks]
    short = "\n\n---\n\n".join(parts)

    if len(short) > max_chars:
        short = short[:max_chars].rstrip() + "..."

    return short

# MAIN ANSWERING FUNCTION
def ask(question, debug=True):
    context = retrieve(question)

    if debug:
        print("\n--- CONTEXT PREVIEW ---")
        print(preview_context(context) if context else "[No context retrieved]")
        print("--- END PREVIEW ---\n")

    if not context:
        return "The provided context is not sufficient to answer this."

    # RAG prompt sent to model
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

    # Calling the LLM
    try:
        response = llm.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=800,
            temperature=0.3,
        )

        # Response validation
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

# Creates terminal interface
def main():
    print("\n--- CHAT STARTED ---")
    print('Begin with an initial question. Responses will be grounded on the provided corpus of source documents. When done type "exit" or close the terminal.\n')
    
    debug = True

    while True:
        question = input("You (type question): ").strip()

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

# Script entry point
if __name__ == "__main__":
    main()