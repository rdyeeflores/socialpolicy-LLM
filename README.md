# socialpolicy-LLM

A lightweight retrieval-augmented LLM system for answering social policy questions based on user-provided  documents (PDF, TXT, etc.). Documents are converted into embeddings and retrieved at query time to produce grounded, lower-hallucination responses.

- Ask questions grounded in your data  
- LLM responses via OpenRouter (Mistral)  
- Modular design for future local LLM integration  

---

## Structure

```
socialpolicy-LLM/
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
Why is social equality important?
```

---

## Notes

- `.env` is not tracked (contains API keys)  
- `chroma_db/` is generated locally and not committed  
- PDF parsing quality may vary  
- Retrieval quality depends on chunking and embeddings  
- This is not a fine-tuned model; it uses RAG