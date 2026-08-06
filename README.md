# socialpolicy-LLM

Evaluation is essential for trustworthy AI and retrieval-augmented generation (RAG) helps produce grounded responses. This Python project integrates these two components to create a **customizable LLM-based system** to answer social policy questions from source documents—and evaluate whether those answers are grounded, consistent, adaptable, correctable, and responsible.

This project **emphasizes LLM evaluation**, not just answer generation. It demonstrates how a system can be tested with automated metrics, targeted benchmarks/scenarios, configurable thresholds, structured reporting, and optional human review.


## Evaluation Framework

The evaluation suite measures five categories:

| Category | What it tests | Method |
|---|---|---|
| **Competency** | Retrieval relevance, answer grounding, concept coverage, and similarity to reference answers | Cosine similarity between query and retrieved-document embeddings (retrieval relevance); lexical overlap between the answer and retrieved context, stopword-filtered (grounding — a proxy for evidence use, not an entailment measure); ROUGE-L, BLEU, and BERTScore F1 computed jointly against reference answers rather than any single metric alone |
| **Reliability** | Stability across repeated runs and consistency across paraphrased questions | Two independent stability checks: retrieval stability (Jaccard overlap of retrieved document IDs across repeated runs) and answer stability (embedding similarity of repeated, temperature-0 answers), plus a separate paraphrase-invariance check and an answer-length coefficient of variation |
| **Adaptability** | Ability to follow audience, jurisdiction, and ethical-framing instructions without losing grounding | Weighted composite of required-instruction term coverage, preservation of the original question's core terms, and context grounding, so surface compliance can't substitute for an on-topic, grounded answer |
| **Recoverability** | Ability to improve overconfident, one-sided, or unsupported answers after corrective feedback | Before/after correction compared with a weighted fit score (desired language, avoidance of forbidden overclaims, appropriate qualification, grounding); a correction only counts as successful if fit clears a threshold, improvement is positive, *and* abstention is correct on cases where the fact is unsupported |
| **Conformity** | Fair treatment across groups, balanced policy framing, and appropriate handling of harmful or unsupported premises | Matched-pair prompts (e.g. urban vs. rural, employed vs. unemployed; pro vs. con framings) scored on response-length parity, semantic similarity, and grounding balance between the pair; harmful or leading premises are separately checked for premise-challenging language rather than compliance |

The framework does not force unrelated metrics into one score. It reports category-level indicators with `PASS`, `REVIEW`, or `FAIL` status, plus scenario-level diagnostics.

### Design notes

- **Reliability tests two independent failure modes, not one.** A RAG system can be unstable because retrieval keeps surfacing different documents, because generation phrases the same evidence differently, or both. This suite scores retrieval stability and answer stability separately instead of collapsing them into a single "consistency" number.
- **Every category runs on matched or paired scenarios**, not isolated prompts: paraphrase pairs (reliability), instruction variants of the same base question (adaptability), before/after correction pairs (recoverability), and demographic/ideological pairs (conformity). Holding the underlying question constant while varying one condition is what makes the comparison interpretable.
- **Abstention is a first-class tested behavior**, not an incidental side effect. One recoverability case asks about a fact that cannot exist in the corpus (a fictional country and year) specifically to check whether the system invents a statistic rather than declining to answer.
- **Correction success requires more than a higher score.** A recoverability case only passes if the corrected answer clears a minimum fit threshold, shows a positive improvement over the initial answer, *and* abstains correctly when the underlying fact is unsupported — so a more fluent but still-overconfident correction won't pass.
- **Scoring weights are explicit and stated, not hidden.** Composite scores (e.g. adaptability's instruction adherence, recoverability's fit score, conformity's fairness and balance scores) are documented weighted sums of interpretable sub-scores, so the basis for a PASS/REVIEW/FAIL can be inspected and re-weighted rather than trusted as a black box.
- **Grounding is stated as a proxy, deliberately.** Lexical overlap between an answer and its retrieved context is cheap and interpretable, but it is not a factual-entailment check. It's reported alongside reference-based metrics (ROUGE-L/BLEU/BERTScore) rather than presented as a standalone correctness score.

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
- Separate evidence from interpretation
- Avoid invented facts, statistics, or sources
- Represent uncertainty when appropriate
- Abstain when the context is insufficient


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

This system is configured to a **Mistral** LLM with API calls serviced by **OpenRouter** (both of which can be modified). Add you own API key to `.env`:

```env
OPENROUTER_API_KEY=your_api_key_here
```

Build a local vector database based on the source documents:

```bash
python src/ingest.py
```

Run grounded Q&A with LLM:

```bash
python src/chat.py
```

Run the full 5-category evaluation suite:

```bash
python src/evaluate.py
```

Save detailed results and create a human-review template:

```bash
python src/evaluate.py --out results.json --human-review-out human_review.csv
```

## Example Evaluation Scorecard

The evaluator prints a summary after all tests are completed. The values below are **illustrative only** and are not reported benchmark results.

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

## Scope

The included evaluation cases are specific to the social policy corpus and are intended as a transparent, extensible starting point. Results depend on the document collection, model, retrieval settings, and configured thresholds.
