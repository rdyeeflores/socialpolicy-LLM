# socialpolicy-LLM

Evaluation is essential for building trustworthy AI systems, while retrieval-augmented generation (RAG) helps ground responses in source material. This Python project combines these two components to create a **customizable system built around a frontier LLM** for answering social policy questions and evaluating whether those answers are grounded, consistent, adaptable, correctable, and responsible. 

This project **emphasizes frontier LLM evaluation**, not just answer generation. It demonstrates how AI responses can be assessed using automated metrics, targeted benchmarks and scenarios, configurable thresholds, structured reporting, and optional human review.


## Evaluation Framework

The evaluation suite measures five categories:

| Category | What it tests | Method |
|---|---|---|
| **Competency** | Retrieval relevance, answer grounding, concept coverage, and similarity to reference answers | Cosine similarity between query and retrieved-document embeddings (retrieval relevance); stopword-filtered lexical overlap between the answer and retrieved context (grounding); BERTScore F1 against a reference answer (semantic accuracy) |
| **Reliability** | Stability across repeated runs and consistency across paraphrased questions | Embedding similarity between answers generated from repeated, identical, zero-temperature runs (answer stability), and embedding similarity between the original answer and the answer to a paraphrased version of the question (paraphrase invariance) |
| **Adaptability** | Ability to follow audience, jurisdiction, and ethical-framing instructions without losing grounding | Weighted composite of required-instruction term coverage, preservation of the original question's core terms, and grounding in retrieved context (adherence); grounding under adaptation is also reported as its own indicator |
| **Recoverability** | Ability to improve overly confident, one-sided, or unsupported answers after corrective feedback | A weighted fit score (desired language, avoidance of forbidden claims, qualification, groundedness) computed before and after correction; success requires corrected fit to clear a threshold without a meaningful drop, except for one scenario where omitted information produces abstention instead |
| **Conformity** | Fair treatment across groups, balanced policy framing, and appropriate handling of harmful or unsupported premises | Comparable group pairs are scored on relevance and grounding (parity); pro/con framing pairs are scored on response-length, grounding, and hedging-language (balance); harmful or leading premises are scored separately on premise-challenging language |

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
OpenRouter LLM response (API)
      ↓
Five-category evaluation suite
      ↓
Scorecard + optional JSON / human-review CSV export
```

The generation prompt requires the model to:

- Use only retrieved context
- Abstain when the context is insufficient
- Separate evidence from interpretation
- Avoid invented facts, statistics, or sources
- Represent uncertainty when appropriate


## Repository Structure

```text
socialpolicy-LLM/
├── corpus/          # Source document corpus
|
├── src/
│   ├── ingest.py      # Builds the local vector database
│   ├── chat.py        # Generates document-grounded answers
│   └── evaluate.py    # Runs the 5-category evaluation suite
|
├── .env_template      # Copied and used for API key
├── requirements.txt   
└── README.md
```

## Installation

Before running the project, install the required Python dependencies and create a local environment file for your API credentials. The .env_template file shows the required settings and the cp command copies it to a new .env file, where you can add your credentials securely.

```bash
git clone https://github.com/rdyeeflores/socialpolicy-LLM.git
cd socialpolicy-LLM
python -m pip install -r requirements.txt
cp .env_template .env
```

This system is configured by default to use **OpenAI GPT-5.6 Luna** through **OpenRouter**, providing fast and cost-effective access to a frontier LLM. Add your own API key to the `.env` file:

```env
OPENROUTER_API_KEY=your_api_key_here
```

Build a local vector database based on the source documents:

```bash
python src/ingest.py
```

Begin a grounded LLM Q&A session:

```bash
python src/chat.py
```

Run a full 5-category evaluation suite:

```bash
python src/evaluate.py
```

Save detailed results and create a human-review template:

```bash
python src/evaluate.py --out results.json --human-review-out human_review.csv
```

## Example Evaluation Scorecard

The evaluator prints a summary after all tests are completed. The output below shows results from a completed evaluation run using the current social-policy corpus, model configuration, scenarios, and configured thresholds.

```text
========================================================================
SOCIAL-POLICY LLM EVALUATION SCORECARD
========================================================================

  Competency
    status               PASS
    coverage             3 metrics | 3 scenarios
    retrieval relevance            0.59
    answer grounding               0.55
    BERTScore F1                   0.84

  Reliability
    status               PASS
    coverage             2 metrics | 3 scenarios
    repeat answer similarity       0.91
    paraphrase answer similarity   0.84

  Adaptability
    status               PASS
    coverage             2 metrics | 3 scenarios
    instruction adherence          0.92
    grounding under adaptation     0.73

  Recoverability
    status               PASS
    coverage             4 metrics | 3 scenarios
    corrected answer fit           0.95
    average correction gain        0.38
    correction success rate        1.00
    insufficient-context handling  1.00

  Conformity
    status               PASS
    coverage             3 metrics | 9 scenarios
    harmful-premise handling       1.00
    comparable-group parity        0.75
    ideological balance            0.75

========================================================================
```

`PASS` means the configured thresholds were met. `REVIEW` means inspect the scenario-level output. `FAIL` means the category missed most of its threshold checks. 

**Interpretation**: Overall, the system satisfied the specified thresholds across all five evaluation categories. Recoverability was particularly strong, with successful correction across all tested scenarios and perfect insufficient-context handling, while reliability and adaptability also showed high consistency and instruction adherence. Competency produced a strong BERTScore F1 alongside more moderate retrieval-relevance and grounding scores. Conformity also passed across the larger nine-scenario set, including perfect harmful-premise handling, with comparable-group parity and ideological balance showing more moderate performance, and therefore useful areas for continued evaluation.


## Future Steps

A future extension could incorporate **multi-human assessment** to evaluate how well the automated metrics align with expert judgment. Multiple assessors could independently rate model responses on dimensions already in the framework, such as grounding, adaptability, recoverability, and balanced treatment of policy questions. Agreement among reviewers could then be quantified, while human ratings could be compared with the automated evaluation scores. This would provide additional evidence about whether the current metrics and specified thresholds capture the response qualities they are meant to measure, while also identifying categories where automated evaluation may require human assessment.

Another extension could apply the same evaluation scenarios to **multiple frontier LLMs**. Each model would receive the same queries, retrieved context, generation instructions, and evaluation criteria, allowing differences in competency, reliability, adaptability, recoverability, and conformity to be compared systematically. Rather than treating a single model's score as an isolated result, this design would make it possible to examine whether the evaluation framework detects meaningful performance differences across models and whether those differences remain consistent across evaluation categories.



## Scope

The included evaluation cases are specific to the social policy corpus and are intended as a transparent, extensible starting point. Results depend on the document collection, model, retrieval settings, and configured thresholds.
