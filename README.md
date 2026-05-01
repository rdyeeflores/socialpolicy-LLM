# socialpolicy-LLM

A lightweight retrieval-augmented LLM system for answering social policy questions based on document text. The project includes some default text files in `DATA/raw/`, and users can add their own supported documents. Documents are converted into embeddings and retrieved at query time to produce grounded, lower-hallucination responses.

- Ask questions grounded in included or user-provided data
- Supports `.txt` and `.pdf` files by default
- LLM responses via OpenRouter using Mistral by default
- Local document search with ChromaDB
- Local embeddings with SentenceTransformers
- Modular design for future local LLM integration

---

## Structure

```
socialpolicy-LLM/
│
├── DATA/
│   └── raw/          # Includes default text files; users may add more .txt or .pdf files
│
├── SRC/
│   ├── ingest.py
│   └── chat.py
│
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Important: Bring Your Own API Key

This project does **not** include an API key.

Each user must provide their own API key from OpenRouter or another compatible provider.

By default, `SRC/chat.py` is configured for:

```python
MODEL = "mistralai/mistral-small-3.2-24b-instruct"
```

and:

```python
base_url="https://openrouter.ai/api/v1"
```

To use a different provider or model, update those lines in `SRC/chat.py`.

Never commit your real API key to GitHub.

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/socialpolicy-LLM.git
cd socialpolicy-LLM
python -m pip install -r requirements.txt
```

Create a `.env` file in the root directory:

```env
OPENROUTER_API_KEY=your_api_key_here
```

Recommended: include a public `.env.example` file with this placeholder:

```env
OPENROUTER_API_KEY=your_api_key_here
```

Your real `.env` file should stay private.

---

## Usage

### 1. Review or Add Documents

The project includes some default text files in:

```
DATA/raw/
```

You may also add your own supported files there.

Currently supported file types:

```
.txt
.pdf
```

Other formats such as `.docx`, `.csv`, `.xlsx`, `.html`, or `.json` are not currently ingested unless support is added to `SRC/ingest.py`.

---

### 2. Build the Local Database

```bash
python SRC/ingest.py
```

This reads the supported files in `DATA/raw/`, chunks the text, creates embeddings, and stores them in a local ChromaDB database.

---

### 3. Start Chat

```bash
python SRC/chat.py
```

Example questions:

```
How can social policy reduce income inequality?

Should healthcare be free for all citizens?

What are the advantages and disadvantages of welfare programs?

How do governments design effective social policies?

How can social policy improve access to quality education?
```

---

## Notes

- `.env` is not tracked because it contains private API keys
- `chroma_db/` is generated locally and should not be committed
- `DATA/raw/` includes default text files and may also contain user-added documents
- Be careful before committing copyrighted, private, or restricted documents
- PDF parsing quality may vary
- Retrieval quality depends on chunking, embeddings, and document quality
- This is not a fine-tuned model; it uses retrieval-augmented generation (RAG)