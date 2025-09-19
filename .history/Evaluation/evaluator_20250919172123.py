import json
from core.dataset import GSM8K, Code, Database, DataToText, Actions
from utils import CodeEvalUtils
import ast
import numpy as np

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

    def evaluate(self, output_file, numruns, numQ):
        finalcorr = []
        utils = CodeEvalUtils()
        dataset = Code().load_data()

        for i in range(numruns):
            correct = []
            res = []
            run_path = self.output_file + f"_run{i}.json"
            with open(run_path, "r", encoding="utf-8") as f:
                output = json.load(f)

            for x in range(numQ):
                final_output = output[x].get("final_output", "")
                function, func_name = utils.extract_first_function_block_and_name(final_output)
                test_cases = ast.literal_eval(dataset[x]["public_test_cases"])

                passed = False
                if function and func_name:
                    passed = utils.run_function_and_check(func_name, function, test_cases)
                    if passed:
                        correct.append(1)
                        output[x]["correct"] = True

                if not passed:
                    if output[x].get("resets", 0) > 0:
                        for func_block, fname in utils.extract_all_function_blocks_and_names(output[x].get("chat_history", [])):
                            passed = utils.run_function_and_check(fname, func_block, test_cases)
                            if passed:
                                correct.append(1)
                                output[x]["correct"] = True
                                break
                        if not passed:
                            correct.append(0)
                            output[x]["correct"] = False
                    else:
                        for entry in output[x].get("chat_history", []):
                            if entry.get("role") == "assistant":
                                function, func_name = utils.extract_first_function_block_and_name(entry.get("content", ""))
                                if function and func_name:
                                    passed = utils.run_function_and_check(func_name, function, test_cases)
                                    if passed:
                                        correct.append(1)
                                        output[x]["correct"] = True
                                        break
                        if not passed:
                            correct.append(0)
                            output[x]["correct"] = False

                res.append(output[x].get("resets", 0))

            corrected_path = self.output_file + f"_run{i}_CORRECTED.json"
            with open(corrected_path, "w", encoding="utf-8") as f_out:
                json.dump(output, f_out, indent=2)

            finalcorr.append(correct)

        sums = 0.0
        for i, corr in enumerate(finalcorr):
            score = np.sum(corr) / numQ
            print(f"Run {i} Score: {score}")
            sums += score

        print(f"Average {sums / len(finalcorr)}")

        return finalcorr
