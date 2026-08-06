# socialpolicy-LLM

Evaluation is essential for trustworthy AI and retrieval-augmented generation (RAG) helps produce grounded LLM responses. This project integrates these two components to create an LLM-based  system to answer social policy questions from source documents—and evaluate whether those answers are grounded, consistent, adaptable, correctable, and responsible.

This repository emphasizes **LLM evaluation**, not just answer generation. It demonstrates how a RAG system can be tested with automated metrics, targeted scenarios, configurable thresholds, structured reporting, and optional human review.

- LLM and RAG evaluation
- Grounded question answering
- Benchmark and scenario design
- Reliability and robustness testing
- Responsible AI and fairness checks
- Human-in-the-loop evaluation
- Reproducible Python workflows

## Evaluation Framework

The evaluation suite measures five categories:

| Category | What it tests |
|---|---|
| **Competency** | Retrieval relevance, answer grounding, concept coverage, and similarity to reference answers |
| **Reliability** | Stability across repeated runs and consistency across paraphrased questions |
| **Adaptability** | Ability to follow audience, jurisdiction, and ethical-framing instructions without losing grounding |
| **Recoverability** | Ability to improve overconfident, one-sided, or unsupported answers after corrective feedback |
| **Conformity** | Fair treatment across groups, balanced policy framing, and appropriate handling of harmful or unsupported premises |

The framework does not force unrelated metrics into one score. It reports category-level indicators with `PASS`, `REVIEW`, or `FAIL` status, plus scenario-level diagnostics.

## System Design

```text
Source documents
      ↓
Text chunking and embeddings
      ↓
Local ChromaDB vector store
      ↓
Top-k document retrieval
      ↓
OpenRouter LLM response
      ↓
Five-category evaluation suite
```

The generation prompt requires the model to:

- Use only retrieved context
- Separate evidence from interpretation
- Avoid invented facts, statistics, or sources
- Represent uncertainty when appropriate
- Abstain when the context is insufficient

## Technology Stack

- **Python**
- **ChromaDB** for local vector search
- **SentenceTransformers** using `all-MiniLM-L6-v2`
- **OpenRouter** through the OpenAI Python client
- **ROUGE-L, BLEU, and BERTScore**
- **NumPy and PyTorch**
- **python-dotenv** for configuration

## Repository Structure

```text
socialpolicy-LLM/
├── data/raw/          # Source documents
├── src/
│   ├── ingest.py      # Builds the local vector database
│   ├── chat.py        # Generates document-grounded answers
│   └── evaluate.py    # Runs the five-category evaluation suite
├── .env.template
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/rdyeeflores/socialpolicy-LLM.git
cd socialpolicy-LLM
python -m pip install -r requirements.txt
cp .env.template .env
```

Add your OpenRouter key to `.env`:

```env
OPENROUTER_API_KEY=your_api_key_here
```

Build the local vector database:

```bash
python src/ingest.py
```

Run grounded Q&A:

```bash
python src/chat.py
```

Run the full evaluation suite:

```bash
python src/evaluate.py
```

Run one category:

```bash
python src/evaluate.py --section reliability
```

Save detailed results and create a human-review template:

```bash
python src/evaluate.py --out results.json --human-review-out human_review.csv
```

## Example Evaluation Scorecard

The evaluator prints a hiring-manager-friendly summary after all selected tests run. The values below are **illustrative only** and are not reported benchmark results.

```text
========================================================================
SOCIAL-POLICY LLM EVALUATION SCORECARD
========================================================================
  Competency
    status               PASS
    coverage             3 metrics | 3 scenarios
    retrieval relevance  0.68
    answer grounding     0.74
    BERTScore F1         0.79

  Reliability
    status               PASS
    coverage             2 metrics | 3 scenarios
    repeat answer similarity      0.91
    paraphrase answer similarity  0.84

  Adaptability
    status               REVIEW
    coverage             2 metrics | 3 scenarios
    instruction adherence         0.73
    grounding under adaptation    0.69

  Recoverability
    status               PASS
    coverage             4 metrics | 3 scenarios
    corrected answer fit           0.78
    average correction gain        0.16
    correction success rate        0.80
    insufficient-context handling  1.00

  Conformity
    status               PASS
    coverage             3 metrics | 9 scenarios
    harmful-premise handling  0.92
    comparable-group parity    0.81
    ideological balance        0.76
========================================================================
```

`PASS` indicates that configured thresholds were met. `REVIEW` flags categories that need inspection, and `FAIL` indicates that most threshold checks were missed. 

## Evaluation Outputs

- Terminal scorecard with category indicators and status
- JSON report with scenario-level metrics and thresholds
- Optional CSV template for expert review and notes

## Scope

The included evaluation cases are specific to the social policy corpus and are intended as a transparent, extensible starting point. Results depend on the document collection, model, retrieval settings, and configured thresholds.
