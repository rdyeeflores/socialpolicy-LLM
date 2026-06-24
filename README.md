# socialpolicy-LLM

A lightweight retrieval-augmented generation (RAG) LLM system for answering social policy questions using document-grounded responses. The project ingests text, stores searchable embeddings locally, retrieves relevant passages, and uses an LLM to generate answers based on retrieved evidence. Answers can then be evaluated across a collection of performance measures.

This project demonstrates:

- Local vector search with ChromaDB
- Local text embeddings with SentenceTransformers
- LLM integration through OpenRouter using Mistral by default
- Safe API-key handling with a user-modified `.env.example` file
- Evaluation of retrieval and response quality, including ethics/bias checks, NLP metrics (ROUGE-L, BLEU, BERTScore), response-quality heuristics, and retrieval relevancy scoring

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
│   ├── chat.py       # Retrieves context for chat and sends queries to an LLM
│   └── evaluate.py   # Runs evaluation metrics against the RAG pipeline
│
├── .env.template      # Copy to .env and add your API key
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

1. Open `.env.template`
2. Add your API key:

```env
OPENROUTER_API_KEY=your_api_key_here
```

3. Copy it to `.env`:

```bash
cp env.template .env
```

The project uses OpenRouter by default with a Mistral model. You can change the provider or model in `SRC/chat.py`.

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

What are the advantages and disadvantages of social welfare programs?

How can social policy improve access to quality healthcare?
```

### 4. Evaluate

```bash
python SRC/evaluate.py
```

Runs four evaluation sections in order from least to most RAG-dependent:

1. **Ethics / Bias** — demographic parity, ideological balance, harmful prompt refusal
2. **NLP Metrics** — ROUGE-L, BLEU, BERTScore against reference answers
3. **Response Quality** — heuristic scores for length, grounding, and refusal rate
4. **RAG Retrieval Relevancy** — cosine similarity between queries and retrieved chunks

---

## Notes

- `.env` contains private API credentials and should not be committed
- `env.template` is safe to commit because it only contains placeholders
- `chroma_db/` is generated locally and should not be committed
- `DATA/raw/` includes default text files and may also contain user-added documents
- PDF parsing quality may vary
- Retrieval quality depends on document quality, chunking, and embeddings
- This is not a fine-tuned model; it uses retrieval-augmented generation (RAG)
- Run `SRC/evaluate.py` only after `SRC/ingest.py` has built the local database
- `results.json` is generated locally if using `--out` and should not be committed