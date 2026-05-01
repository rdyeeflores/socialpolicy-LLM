# socialpolicy-LLM

A lightweight retrieval-augmented LLM system for answering social policy questions from document text. The system ingests documents, stores searchable embeddings locally, retrieves relevant passages, and uses an LLM to produce grounded responses.

This project demonstrates:

- Local vector search with ChromaDB
- Local text embeddings with SentenceTransformers
- LLM integration through OpenRouter using Mistral by default
- Safe API-key handling with a user-modified `.env.example` file

---

## Structure

```
socialpolicy-LLM/
│
├── DATA/
│   └── raw/          # Includes default files; more could be added here (.pdf or .txt) 
│
├── SRC/
│   ├── ingest.py     # Reads files and builds a local vector database
│   └── chat.py       # Retrieves context for chat and sends queries to an LLM
│
├── .env.example      # NOTW: Rename to .env and add your API key
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/socialpolicy-LLM.git
cd socialpolicy-LLM
python -m pip install -r requirements.txt
```

### API Key Setup

1. Open `.env.example`
2. Add your API key:

```env
OPENROUTER_API_KEY=your_api_key_here
```

3. Rename the file:

```
.env.example → .env
```

The project uses :contentReference[oaicite:0]{index=0} by default with a Mistral model. You can change the provider or model in `SRC/chat.py`.

---

## Usage

### 1. Add Documents

Place `.txt` or `.pdf` files in:

```
DATA/raw/
```

(Default files are already included.)

---

### 2. Build the Local Database

```bash
python SRC/ingest.py
```

This reads the documents, chunks the text, creates embeddings, and stores them in a local ChromaDB database.

---

### 3. Start Chat

```bash
python SRC/chat.py
```

Example questions:

```
How can social policy reduce income inequality?

What are the advantages and disadvantages of welfare programs?

How can social policy improve access to quality education?
```

---

## Notes

- `.env` contains private API credentials and should not be committed
- `chroma_db/` is generated locally and should not be committed
- `DATA/raw/` includes default text files and may also contain user-added documents
- PDF parsing quality may vary
- Retrieval quality depends on document quality, chunking, and embeddings
- This is not a fine-tuned model; it uses retrieval-augmented generation (RAG)