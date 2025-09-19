## NOTE: CODEBASE IS CURRENTLY BEING REFACTORED TO MAKE IT EASIER TO FORK AND RECREATE OUR EXPERIMENTS (Work In Progress)



# ERGO: Entropy-guided Resetting for Generation Optimization

<p align="center">
  <img src="READMEimg/Representative_Diagram.png" alt="ERGO Diagram" width="600"/>
</p>

This repository accompanies the paper **"ERGO: Entropy-guided Resetting for Generation Optimization in Multi-turn Language Models"**. It contains all code necessary to replicate our experiments and evaluate ERGO’s performance across a suite of multi-turn generation tasks.

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

## Experimental Results

Compared to standard multi-turn conversations, ERGO significantly improves multi-turn generation across five tasks and several LLMs (GPT-4o, GPT-4.1, Phi-4, LLaMA 3.1–8B):

• +56.6% improvement in average performance (𝑃̄).

• +24.7% increase in best-case aptitude (𝐴₉₀).

• −35.3% reduction in unreliability (𝑈₉₀–₁₀).

• Outperforms repetition-based methods (SNOWBALL, RECAP) and naive resets (random, fixed).

Please refer to Section 5 of the paper for more info.

## License

Released under an anonymous open-source license for review purposes.

---

For any questions related to experimental design or evaluation metrics, please refer to the methodology described in the paper or feel free to email haziqkhalid04@gmail.com
