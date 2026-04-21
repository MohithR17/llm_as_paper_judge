# Novelty Classification Prototype

This prototype supports two scoring modes.

## 1. Embedding Mode

This implements the scoring rule:

`novelty(s_i) = 1 - max_{d in C} cosine(e(s_i), e(d))`

where:

- `s_i` is a claim sentence extracted from the target PDF
- `C` is the grounding corpus built from literature survey abstracts
- `e(.)` is an embedding model used as the dense encoder

## 2. LLM Judge Mode

If your gateway does not expose embedding models, the script can instead:

1. shortlist candidate abstracts with lexical retrieval
2. ask the LLM whether each abstract contains the same core proposition as the claim
3. set `novelty = 1 - max groundedness_score`

This is not exactly the same as the embedding formula, but it is often the more practical option when embeddings are unavailable.

## What It Does

1. Extracts text from a PDF.
2. Uses an LLM to extract claim sentences.
3. Scores each claim against literature abstracts using either embeddings or LLM judging.
4. Computes the novelty score for each claim.
5. Flags claims with `novelty > threshold` as novel.

## Important Limitation

Using only abstracts is a useful first baseline, but it is only an approximation of true novelty. A claim may look novel against abstracts even if the full paper body already contains the same proposition.

## Install

```bash
pip install openai pypdf
```

## Expected Literature Survey JSON

The script accepts either:

- a JSON list of papers
- an object with `paper_pool`, `papers`, or `data`

Each paper should include a title-like field and an abstract-like field, for example:

```json
[
  {
    "title": "Paper A",
    "abstract": "We propose a method for..."
  },
  {
    "paper_title": "Paper B",
    "paper_abstract": "This work shows..."
  }
]
```

## Usage

```bash
python3 /Users/mohithrajesh/Documents/Playground/novelty_classifier/run_novelty_pipeline.py \
  --pdf /absolute/path/to/paper.pdf \
  --survey-json /absolute/path/to/literature_survey.json \
  --api-key "$OPENAI_API_KEY" \
  --base-url "https://ai-gateway.andrew.cmu.edu" \
  --llm-model "gpt-4.1" \
  --scoring-mode "llm_judge" \
  --novelty-threshold 0.35 \
  --output-json /absolute/path/to/novelty_report.json
```

Embedding mode is still available if your environment supports it:

```bash
python3 /Users/mohithrajesh/Documents/Playground/novelty_classifier/run_novelty_pipeline.py \
  --pdf /absolute/path/to/paper.pdf \
  --survey-json /absolute/path/to/literature_survey.json \
  --api-key "$OPENAI_API_KEY" \
  --base-url "https://ai-gateway.andrew.cmu.edu" \
  --llm-model "gpt-4.1" \
  --scoring-mode "embeddings" \
  --embedding-model "text-embedding-3-large"
```

```bash
python novelty_classifier/run_novelty_pipeline.py \
  --pdf literature_survey_agent/attention_is_all_you_need.pdf \
  --survey-json literature_survey_agent/output/survey_result_saved.json \
  --api-key "sk-9X8sGo8E6JylU94mB-lkPw" \
  --base-url "https://ai-gateway.andrew.cmu.edu" \
  --llm-model "gpt-5-mini" \
  --embedding-model "text-embedding-3-large" \
  --novelty-threshold 0.35 \
  --output-json output/novelty_report.json
```

## OpenAI Client Note

The official Python client is initialized like this:

```python
from openai import OpenAI

client = OpenAI(
    api_key=args.api_key,
    base_url="https://ai-gateway.andrew.cmu.edu",
)
```

Then pass the model per request:

```python
client.chat.completions.create(model=args.llm_model, ...)
client.embeddings.create(model=args.embedding_model, ...)
```

If your project wraps `OpenAI(...)` differently, you can keep that wrapper, but the official client itself does not take `model=` in the constructor.
