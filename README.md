# socialpolicy-LLM

A lightweight Large Language Model (LLM) system built to answer basic social policy questions, using Retrieval-Augmented Generation (RAG) to reduce hallucinations and produce grounded responses based on user-provided documents (PDF, txt, etc). This is done by converting documents into searchable embeddings, which must then be referenced wehn responding to a user query.  

- Ask questions and get answers grounded in your data
- LLM responses via OpenRouter (Mistral)
- Modular structure for future local LLM integration

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

```
git clone https://github.com/YOUR_USERNAME/policy-LLM.git
cd policy-LLM
python -m pip install -r requirements.txt
```

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
What is the value of social equality in creating policy?

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
- PDF parsing quality varies (especially scientific papers)
- Retrieval quality depends on chunking and embeddings
- Not a fine-tuned model — uses RAG instead

