from core.model import OpenAIModel, LocalLLMModel
from core.dataset import GSM8K, Database, Code, Actions, DataToText
from core.ergo import Ergo
from core.utils import Logger
from generation.generator import RunERGO

class RunExperiment():
    def __init__(self, model_name, api_key = None, device="cuda", max_length=1024, torch_dtype="float16", temperature=1.0, do_sample=True):
        self.model_name = model_name
        if 'gpt' in model_name:
            self.model = OpenAIModel(
                model_name=model_name,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_length
            )
        else:
            self.model = LocalLLMModel(
                model_name=model_name,
                device=device,
                max_length=max_length,
                torch_dtype=torch_dtype,
                temperature=temperature,
                do_sample=do_sample
            )
            self.tokenizer = self.model.tokenizer

    def run_GSM8K(self, num_Qs = None, threshold=0.5, output_path=None):
        dataset = GSM8K()
        ergo = Ergo(model=self.model, threshold=threshold)
        logger = Logger(model=self.model, dataset=dataset, output_path=output_path)
        runner = RunERGO(model=self.model, dataset=dataset, ergo=ergo, logger=logger, num_Qs=num_Qs)
        runner.execute()

    def run_Database(self, num_Qs = None, threshold=0.5, output_path=None):
        dataset = Database()
        ergo = Ergo(model=self.model, threshold=threshold)
        logger = Logger(model=self.model, dataset=dataset, output_path=output_path)
        runner = RunERGO(model=self.model, dataset=dataset, ergo=ergo, logger=logger, num_Qs=num_Qs)
        runner.execute()
    
    def run_Code(self, num_Qs = None, threshold=0.5, output_path=None):
        dataset = Code()
        ergo = Ergo(model=self.model, threshold=threshold)
        logger = Logger(model=self.model, dataset=dataset, output_path=output_path)
        runner = RunERGO(model=self.model, dataset=dataset, ergo=ergo, logger=logger, num_Qs=num_Qs)
        runner.execute()
    
    def run_Actions(self, num_Qs = None, threshold=0.5, output_path=None):
        dataset = Actions()
        ergo = Ergo(model=self.model, threshold=threshold)
        logger = Logger(model=self.model, dataset=dataset, output_path=output_path)
        runner = RunERGO(model=self.model, dataset=dataset, ergo=ergo, logger=logger, num_Qs=num_Qs)
        runner.execute()