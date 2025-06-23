# ERGO: Entropy-guided Resetting for Generation Optimization

This repository accompanies the paper **"ERGO: Entropy-guided Resetting for Generation Optimization in Multi-turn Language Models"** (under review at INTERPLAY 25' @ COLM 2025). It contains all code necessary to replicate our experiments and evaluate ERGO’s performance across a suite of multi-turn generation tasks.

## Overview

ERGO is a model-agnostic inference-time framework that monitors token-level predictive entropy in multi-turn conversations to detect spikes in uncertainty and trigger automatic prompt resets. This helps large language models (LLMs) recover from context degradation and maintain high performance across tasks.

We evaluate ERGO on five generation tasks:
- Math word problems (GSM8K)
- Code generation (LiveCodeBench)
- Text-to-SQL (Spider)
- Action/API call generation (Berkeley Function Calling Leaderboard)
- Data-to-text generation (ToTTo)

## Repository Structure

```
├── Evaluation
│   └── Combined_ContextResets.ipynb
|
└── Running
      └── GSM8K_ContextReset.ipynb
      └── Coding_ContextReset.ipynb
      └── DB_ContextReset.ipynb
      └── D2T_ContextReset.ipynb
      └── API_ContextReset.ipynb

````

- `Evaluation/Combined_ContextResets.ipynb`: Aggregates all experimental logs and computes performance metrics: average performance (P̄), aptitude (A90), and unreliability (U90–10).
- `Running/`: Contains one notebook per dataset. Each notebook simulates multi-turn conversations with ERGO, using the OpenAI API. Results are stored in `.json` format for evaluation.

## Usage Instructions

### Prerequisites
- Python 3.8+
- OpenAI API access
- Required packages listed in each notebook (typically includes `openai`, `tqdm`, `numpy`, etc.)

### Step 1: Run Multi-Turn Simulations

Navigate to the `Running/` directory and execute the notebook corresponding to your target dataset. Each notebook:
- Loads input prompts sharded across conversation turns.
- Simulates multi-turn interactions using a selected LLM.
- Applies entropy-based resets when uncertainty exceeds a calibrated threshold.
- Stores the conversation logs and outputs in JSON format.

### Step 2: Evaluate Results

Use the `Combined_ContextResets.ipynb` notebook in the `Evaluation/` directory. This notebook:

* Parses the `.json` output files.
* Computes per-run scores and aggregates them into P̄, A90, and U90.

## License

Released under an anonymous open-source license for review purposes.

---

For any questions related to experimental design or evaluation metrics, please refer to the methodology described in the paper.
