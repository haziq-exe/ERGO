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

TBD

## Usage Instructions

TBD


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
