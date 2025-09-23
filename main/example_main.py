from experiments.runExperiment import RunExperiment

# "OPENAI_KEY" key must be set in environment variable
# If using OpenAI API, set openai = True,
# otherwise it will try to load the model locally with HuggingFace

Example_Experiment = RunExperiment(
    model_name="HuggingFaceTB/SmolLM-135M-Instruct", 
    device="cpu", 
    device_map=None, 
    max_new_tokens=1000, 
    openai = False
)

Example_Experiment.run_GSM8K(
    dataset_path="sharded_dataset.json", # Path to sharded dataset from Laban et al.
    num_Qs=5, 
    num_runs=1, 
    threshold=0.3, 
    output_path="outputs/gsm8k_example.json"
)