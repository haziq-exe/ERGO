import json
from core.dataset import GSM8K, Code, Database, DataToText, Actions
class Evaluator:
    def __init__(self, output_file):
        """
        Initialize the Evaluator with an output file path.
        """
        self.output_file = output_file
    
    def load_outputs(self):
        """
        Load model outputs from the specified JSON file.
        """
        with open(self.output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data