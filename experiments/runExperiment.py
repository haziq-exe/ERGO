from core.model import OpenAIModel, LocalLLMModel
from core.dataset import GSM8K, Database, Code, Actions, DataToText
from core.ergo import Ergo
from core.utils import Logger
from generation.generator import RunERGO

class RunExperiment():
    def __init__(self, model_name, api_key = None, device="cuda", device_map="auto", max_new_tokens=1024, dtype="float16", temperature=1.0, do_sample=True, openai=False):
        self.model_name = model_name
        if 'gpt' in model_name or openai:
            self.model = OpenAIModel(
                model_name=model_name,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_new_tokens
            )
        else:
            self.model = LocalLLMModel(
                model_name=model_name,
                device=device,
                max_new_tokens=max_new_tokens,
                dtype=dtype,
                temperature=temperature,
                do_sample=do_sample,
                device_map=device_map
            )
            self.tokenizer = self.model.tokenizer

    def run_GSM8K(self, dataset_path, num_Qs = None, threshold=0.5, output_path=None, num_runs=1):
        dataset = GSM8K(dataset_path=dataset_path)
        ergo = Ergo(model=self.model, threshold=threshold)
        logger = Logger(model=self.model, dataset=dataset, output_path=output_path)
        runner = RunERGO(model=self.model, dataset=dataset, ergo=ergo, logger=logger, num_Qs=num_Qs, num_runs=num_runs)
        runner.execute()

    def run_Database(self, dataset_path, num_Qs = None, threshold=0.5, output_path=None, num_runs=1):
        dataset = Database(dataset_path=dataset_path)
        ergo = Ergo(model=self.model, threshold=threshold)
        logger = Logger(model=self.model, dataset=dataset, output_path=output_path)
        runner = RunERGO(model=self.model, dataset=dataset, ergo=ergo, logger=logger, num_Qs=num_Qs, num_runs=num_runs)
        runner.execute()

    def run_Code(self, dataset_path, num_Qs = None, threshold=0.5, output_path=None, num_runs=1):
        dataset = Code(dataset_path=dataset_path)
        ergo = Ergo(model=self.model, threshold=threshold)
        logger = Logger(model=self.model, dataset=dataset, output_path=output_path)
        runner = RunERGO(model=self.model, dataset=dataset, ergo=ergo, logger=logger, num_Qs=num_Qs, num_runs=num_runs)
        runner.execute()

    def run_Actions(self, dataset_path, num_Qs = None, threshold=0.5, output_path=None, num_runs=1):
        dataset = Actions(dataset_path=dataset_path)
        ergo = Ergo(model=self.model, threshold=threshold)
        logger = Logger(model=self.model, dataset=dataset, output_path=output_path)
        runner = RunERGO(model=self.model, dataset=dataset, ergo=ergo, logger=logger, num_Qs=num_Qs, num_runs=num_runs)
        runner.execute()

    def run_DataToText(self, dataset_path, num_Qs = None, threshold=0.5, output_path=None, num_runs=1):
        dataset = DataToText(dataset_path=dataset_path)
        ergo = Ergo(model=self.model, threshold=threshold)
        logger = Logger(model=self.model, dataset=dataset, output_path=output_path)
        runner = RunERGO(model=self.model, dataset=dataset, ergo=ergo, logger=logger, num_Qs=num_Qs, num_runs=num_runs)
        runner.execute()