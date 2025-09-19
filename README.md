# ERGO: Entropy-guided Resetting for Generation Optimization

<div align="center">

![ERGO Banner](READMEimg/Representative_Diagram.png)

[![Paper](https://img.shields.io/badge/📄_Read_Paper-8A2BE2?style=for-the-badge)](https://github.com/haziq-exe/ERGO)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Beta_Testing-orange?style=for-the-badge)](https://github.com/haziq-exe/ERGO/issues)

**Transforming Multi-turn Conversations with Uncertainty-Aware Intelligence**

[📖 Paper](https://github.com/haziq-exe/ERGO) • [🚀 Quick Start](#-quick-start) • [📊 Results](#-key-results) • [📧 Contact](mailto:haziqkhalid04@gmail.com)

</div>

---

## Overview

**ERGO** introduces a paradigm shift in handling multi-turn LLM conversations by treating uncertainty as a first-class signal. When large language models get "lost" in extended conversations, ERGO detects these moments through entropy spikes and strategically resets the context, recovering both accuracy and reliability. This repository contains all code necessary to replicate our experiments and evaluate ERGO’s performance across a suite of models and multi-turn generation tasks.

### Key Innovation

Unlike traditional approaches that fight against model uncertainty, ERGO *embraces* it, using Shannon entropy over next-token distributions as an internal behavioral signal to detect and correct conversational drift in real-time.

## Core Features

<table>
<tr>
<td width="33%" align="center">

### 56.6%
**Performance Gain**  
Over standard baselines

</td>
<td width="33%" align="center">

### 24.7%
**Aptitude Increase**  
Peak performance capability

</td>
<td width="33%" align="center">

### 35.3%
**Unreliability Reduction**  
Improved consistency

</td>
</tr>
</table>

## 🚀 Quick Start

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/haziq-exe/ERGO.git
cd ERGO


pip install -r requirements.txt
```

### Basic Usage

```python
from experiments.runExperiment import RunExperiment
from experiments.runEvaluation import RunEvaluation

# Initialize experiment with your chosen model
experiment = RunExperiment(
    model_name="HuggingFaceTB/SmolLM-135M-Instruct",
    device="cpu",
    device_map=None,
    max_new_tokens=1000
)

# Run ERGO on GSM8K dataset
experiment.run_GSM8K(
    dataset_path="sharded_dataset.json",
    num_Qs=20,
    num_runs=1,
    threshold=0.5,
    output_path="outputs/gsm8k_example.json"
)

# Evaluate results
evaluation = RunEvaluation(dataset_path="sharded_dataset.json")
evaluation.GSM8K_evaluate(
    numQ=20,
    num_runs=1,
    input_path="outputs/gsm8k_example.json"
)
```

Run from root directory:
```bash
python -m main.example_main
```

## 📁 Repository Structure

```
ERGO/
│
├── 📊 evaluation/          # Evaluation metrics and scoring
│   └── evaluator.py
│
├── 🧠 core/               # Core ERGO implementation
│   ├── dataset.py         # Dataset handlers
│   ├── model.py          # Model interfaces
│   └── utils.py          # Utility functions
│
├── 🔬 experiments/        # Experiment runners
│   ├── runEvaluation.py  # Evaluation pipeline
│   └── runExperiment.py  # Main experiment logic
│
├── 🤖 generation/         # Generation strategies
│   └── generator.py      # ERGO generation logic
│
└── ✍️ main/              # Example scripts
    └── example_main.py   # Quick start example
```

## 🧪 Evaluated Tasks

ERGO has been rigorously tested across five diverse generation tasks:

| Task | Dataset | Description | Metric |
|------|---------|-------------|--------|
| 🧮 **Math** | GSM8K | Elementary math word problems | Exact Match |
| 💻 **Code** | LiveCodeBench | Python function generation | Test Suite Pass |
| 🗃️ **SQL** | Spider | Text-to-SQL query generation | Query Accuracy |
| 🔧 **API Calls** | Berkeley FCL | Function calling from instructions | Call Validity |
| 📝 **Data-to-Text** | ToTTo | Table caption generation | BLEU Score |

## 📊 Key Results

<div align="center">

### Performance Across Models

| Model | FULL | SHARDED | **ERGO** | Improvement |
|-------|------|---------|----------|-------------|
| GPT-4o | 79.2 | 51.4 | **75.6** | +47% |
| GPT-4.1 | 86.5 | 46.0 | **77.2** | +68% |
| GPT-4o-mini | 73.8 | 44.3 | **71.8** | +62% | 
| Phi-4 | 62.0 | 35.1 | **59.2** | +69% |
| LLaMA-3.1-8B | 35.7 | 29.4 | **50.9** | +73% |

</div>

## Important Notes

> **Beta Status**: While the codebase is complete and functional, It is still in its early stages. You may encounter bugs – please report them via [Issues](https://github.com/haziq-exe/ERGO/issues).

> **Documentation**: Comprehensive documentation is in development. For now, please refer to the paper for detailed methodology and theoretical foundations.


## 📄 Citation

If you use ERGO in your research, please cite our paper:

```bibtex
@inproceedings{khalid2025ergo,
  title={ERGO: Entropy-guided Resetting for Generation Optimization in Multi-turn Language Models},
  author={Khalid, Haziq Mohammad and Jeyaganthan, Athikash and Do, Timothy and 
          Fu, Yicheng and O'Brien, Sean and Sharma, Vasu and Zhu, Kevin},
  booktitle={Proceedings of the Conference on Uncertainty in Natural Language Processing (UncertaiNLP)},
  year={2025},
  organization={Algoverse AI Research}
}
```

## 📬 Contact

**Lead Author**: Haziq Mohammad Khalid  
📧 haziqkhalid04@gmail.com

---

[⬆ Back to Top](#-ergo-entropy-guided-resetting-for-generation-optimization)

</div>