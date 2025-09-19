import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.runExperiment import RunExperiment
from experiments.runEvaluation import RunEvaluation

# If model name has 'gpt' in it, it will use OpenAI API, 
# otherwise it will try to load the model locally with HuggingFace

# Example_Experiment = RunExperiment("HuggingFaceTB/SmolLM-135M-Instruct", device="cpu", device_map=None, max_new_tokens=100)
# Example_Experiment.run_GSM8K(dataset_path="sharded_dataset.json", num_Qs=3, threshold=0.5, output_path="outputs/gsm8k_example.json", num_runs=1)

Example_Evaluation = RunEvaluation(dataset_path="sharded_dataset.json")
Example_Evaluation.GSM8K_evaluate(numQ=3, num_runs=1, input_path="outputs/gsm8k_example.json")
