# policy-LLM

A lightweight **Retrieval-Augmented Generation (RAG)** system for exploring policy documents, surveys, and scientific articles using an LLM.

This project lets you:
- Load your own documents (PDF, text)
- Convert them into searchable embeddings
- Ask questions and get answers grounded in your data

---

## Features

- Local document ingestion (PDF + text)
- Vector search using ChromaDB
- LLM responses via OpenRouter (Mistral)
- Simple command-line interface
- Modular structure for future local LLM integration

---

## Project Structure

```
policy-LLM/
│
├── SRC/
│   ├── ingest.py
│   └── chat.py
│
├── DATA/
│   └── raw/
│
├── mistral_test.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

```
git clone https://github.com/YOUR_USERNAME/policy-LLM.git
cd policy-LLM
python -m pip install -r requirements.txt
```

---

## Environment Setup

Create a `.env` file in the root directory:

```
OPENROUTER_API_KEY=your_api_key_here
```

---

## Usage

### 1. Add Documents

Place your files in:

```
DATA/raw/
```

Supported formats:
- `.pdf`
- `.txt`

---

### 2. Build the Database

```
python SRC/ingest.py
```

---

### 3. Start Chat

```
python SRC/chat.py
```

Example questions:

```
What is ForecastBench?
What patterns appear in the survey data?
Summarize the main findings of the article.
```

---

## Example Workflow

```
# Add file
DATA/raw/ForecastBench.pdf

# Ingest
python SRC/ingest.py

# Query
python SRC/chat.py
```

---

## Notes

- `.env` is not tracked in Git (contains API key)
- `chroma_db/` is generated locally and not committed
- `DATA/raw/` is ignored by default (your private data)

---

## Limitations

- PDF parsing quality varies (especially scientific papers)
- Retrieval quality depends on chunking and embeddings
- Not a fine-tuned model — uses RAG instead

---

## Roadmap

- Local LLM backend (GGUF / llama.cpp)
- Improved chunking for scientific texts
- Metadata filtering
- Streaming responses

---

## License

MIT (or update as needed)