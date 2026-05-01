# socialpolicy-LLM

A lightweight retrieval-augmented LLM system for answering social policy questions based on user-provided  documents (PDF, TXT, etc.). Documents are converted into embeddings and retrieved at query time to produce grounded, lower-hallucination responses. 

- Ask questions grounded in your own data
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
│   └── raw/
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

```
OPENROUTER_API_KEY=your_api_key_here
```

---

## Usage

### 1. Add Documents

Place files in:

```
DATA/raw/
```

---

### 2. Build Database

```bash
python SRC/ingest.py
```

---

### 3. Start Chat

```bash
python SRC/chat.py
```

Example:

```
1. How can social policy reduce income inequality?

2. Should healthcare be free for all citizens?

3. What are the advantages and disadvantages of welfare programs?

4. How do governments design effective social policies? 

5. How can social policy improve access to quality education?
```

---

## Notes

- `.env` is not tracked (contains API keys)  
- `chroma_db/` is generated locally and not committed  
- PDF parsing quality may vary  
- Retrieval quality depends on chunking and embeddings  
- This is not a fine-tuned model; it uses RAG