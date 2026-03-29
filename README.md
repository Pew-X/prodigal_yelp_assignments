# Advanced AI Systems for Yelp Reviews

An end-to-end exploration of prompt engineering, chain-of-thought reasoning, multi-objective LLM generation, and supervised fine-tuning for 5-class sentiment classification on Yelp reviews.

**Model:** llama-3.1-8b-instant (Groq) · **Classifier:** DistilBERT-base-uncased · **Dataset:** Yelp Review 

---

## What This Is

This project was built as an assignment to go beyond basic sentiment classification. Every design decision is documented and reported.

Four tasks, each building on the last:

1. **Zero-shot and few-shot prompting** with strict JSON output enforcement
2. **Chain-of-thought vs. direct prompting** with two-approach mismatch detection
3. **Multi-objective business assistant** rating + key point extraction + business response, evaluated via LLM-as-judge
4. **DistilBERT fine-tuning + domain shift** Yelp → Amazon → IMDB, plus 20 handcrafted adversarial examples

---

## Project Structure

```
├── src/
│   ├── llm_client.py        # Groq wrapper with retry, backoff, 3-tier JSON parsing
│   ├── prompts.py           # Versioned prompt templates for all tasks
│   ├── data_loader.py       # Stratified sampling from HuggingFace datasets
│   └── evaluator.py         # Accuracy, Macro-F1, error taxonomy, mismatch detection
│
├── experiments/
│   ├── task1_prompting.py   # Zero shot and few-shot classification
│   ├── task2_cot.py         # Direct vs chain of thought comparison
│   ├── task2_judge.py       # LLM as judge mismatch detection pass
│   ├── task3_assistant.py   # multi objective generation + judge evaluation
│   ├── task4_finetune.py    # DistilBERT training on Yelp
│   ├── task4_domain_shift.py # OOD evaluation on Amazon and IMDB
│   └── task4_adversarial.py  # Adversarial examples + ensemble mitigation
│
├── tests/
│   ├── all pytest test cases for software best practices in modular codebase.
│   
│   
│
├── notebooks/
│   ├── task1_analysis.ipynb
│   ├── task2_analysis.ipynb
│   └── task3_analysis.ipynb
│
├── results/                 # Auto-generated result artefacts
├── models/                  # Fine-tuned checkpoint, gitignored (uploaded on googledrive)
├── .env.example
├── requirements.txt

```

---

## Setup

```bash
git clone <repo-url>
cd yelp-ai-systems

python -m venv venv            # or you can also use uv package manager
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Add your Groq API key to .env   get one free at groq.com
```

---

## Running Experiments

```bash
# Tasks 1–3 (LLM-based, ~30–40 min total)
python experiments/task1_prompting.py
python experiments/task2_cot.py
python experiments/task2_judge.py
python experiments/task3_assistant.py

# Task 4 fine-tuning (~46 min on 4GB GPU)
python experiments/task4_finetune.py
python experiments/task4_domain_shift.py
python experiments/task4_adversarial.py
```

Results are written to `results/` as JSON and CSV after each run.

---

## Running Tests

```bash
pytest tests/ -v   OR uv run pytest
```


---

## Key Results

| Approach | Accuracy | Macro-F1 |
|---|---|---|
| Zero-Shot LLM | 0.620 | 0.601 |
| Few-Shot LLM | 0.650 | 0.639 |
| Direct LLM | **0.665** | **0.655** |
| CoT LLM | 0.618 | 0.609 |
| DistilBERT (fine-tuned) | 0.616 | 0.617 |

---

## Software Practices

- **Reproducibility:** `seed` fixed across all sampling, training, and evaluation. `temperature=0.0` on all LLM calls.
- **Separation of concerns:** generation and evaluation are separate phases and separate files. You can re-run the judge without re-running predictions. Modular first approach
- **Checkpointing:** Task 3 saves generated outputs to CSV before the judge pass begins. Long inference runs don't restart from zero on failure.
- **Git branching**: task wise version control with precise branching strategies per task and merged PRs with  git log.
- **Logging and Tests**: Ensured Production Grade setup and devlopment practices by following logging and test driven devlopment approach.

---

## Hardware

All LLM experiments ran on CPU via the Groq API. Fine-tuning ran on an NVIDIA GPU with 4GB VRAM.

---

## Dependencies

See `requirements.txt`. Core: `groq`, `transformers`, `datasets`, `scikit-learn`, `torch`, `pandas`, `pytest`.
