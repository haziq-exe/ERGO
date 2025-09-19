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
    
class CodeEvaluator(Evaluator):

    def __init__(self, output_file):
        super().__init__(output_file)

    def evaluate(self):
        """
        Evaluate code generation outputs.
        """
        outputs = self.load_outputs()
        correct = 0
        total = len(outputs)
        
        for item in outputs:
            if item.get('is_correct'):
                correct += 1
        
        accuracy = correct / total if total > 0 else 0
        return {
            'total': total,
            'correct': correct,
            'accuracy': accuracy
        }