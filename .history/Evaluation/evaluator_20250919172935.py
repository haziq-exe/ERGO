import json
from core.dataset import GSM8K, Code, Database, DataToText, Actions
from utils import CodeEvalUtils, DatabaseEvalUtils, DataToTextEvalUtils, ActionsEvalUtils
import ast
import numpy as np
import re

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

    def evaluate(self, numruns, numQ):
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

class GSM8KEvaluator(Evaluator):


    
    def __init__(self, output_file):
        super().__init__(output_file)

    def evaluate(self, numruns, numQ):
        dataset = GSM8K().load_data()
        all_runs = []

        for run in range(numruns):
            run_path = self.output_file + f"_run{run}.json"

            orig_output_file = self.output_file
            self.output_file = run_path
            NRoutput = self.load_outputs()
            self.output_file = orig_output_file

            corr = []
            for i in range(numQ):
                gold_raw = dataset[i].get("answer", "")
                m = re.search(r'####\s*(.*)', gold_raw)
                gold_ans = m.group(1).strip() if m else gold_raw.strip()
                gold_ans = gold_ans.replace(",", "")

                final_out = NRoutput[i].get("final_output", "") or ""
                model_answers = re.findall(r'<Answer>\s*(.*?)(?:</Answer>|$)', final_out, flags=re.DOTALL)
                is_correct = False

                if model_answers:
                    cand = model_answers[0].strip().replace(",", "")
                    if gold_ans in cand:
                        is_correct = True

                if not is_correct and NRoutput[i].get("resets", 0) == 0:
                    for msg in NRoutput[i].get("chat_history", []):
                        if msg.get("role") == "assistant":
                            matches = re.findall(r'<Answer>\s*(.*?)(?:</Answer>|$)', msg.get("content", ""), flags=re.DOTALL)
                            for m in matches:
                                ans = m.strip().replace(",", "")
                                if gold_ans in ans:
                                    is_correct = True
                                    break
                    if is_correct:
                        break

                corr.append(1 if is_correct else 0)
                NRoutput[i]["correct"] = bool(is_correct)

                corrected_path = orig_output_file + f"_run{run}_CORRECTED.json"
                with open(corrected_path, "w", encoding="utf-8") as f_out:
                    json.dump(NRoutput, f_out, indent=2)

                all_runs.append(corr)

        sums = 0.0
        scores = []
        for i, run_corr in enumerate(all_runs):
            score = np.sum(run_corr) / numQ
            print(f"Run {i} Score: {score}")
            scores.append(score)

        if all_runs:
            Average = {sums / len(all_runs)}

        return all_runs, scores, Average
    
class ActionsEvaluator(Evaluator):
    def __init__(self, output_file):
        super().__init__(output_file)
        import json

    def evaluate(self, numruns, numQ):

        avg = []
        for i in range(numruns):
            correct_count = 0
            with open(data_path + f"_run{i}.json") as f:
                output = json.load(f)

            for x in range(numQ):
                raw_model_output = output[x]['final_output']
                model_answer = re.findall(r'<Answer>\s*(.*?)(?:</Answer>|$)', raw_model_output, re.DOTALL)
                if model_answer:
                raw_model_output = model_answer[0]
                mod_ans = extract_function_block(raw_model_output)
                mod_ans = clean_function_block(mod_ans)
                corr = evaluator_function(predicted_answer=mod_ans, sample=dataset[x])

                if "Failing to parse the predicted answer as an AST" in corr["error"]:
                mod_ans = extract_function_block(f"[{raw_model_output}]")
                mod_ans = clean_function_block(mod_ans)
                corr = evaluator_function(predicted_answer=mod_ans, sample=dataset[x])

                is_correct = False
                if corr["is_correct"]:
                    correct_count += 1
                    is_correct = True
                else:
                    if output[x]['resets'] == 0:
                    for entry in output[x]['chat_history']:
                        if entry["role"] == "assistant":
                        for block in extract_all_function_blocks(entry['content']):
                            corr = evaluator_function(predicted_answer=block, sample=dataset[x])
                            if corr["is_correct"]:
                                correct_count += 1
                                is_correct = True
                                break
                        if is_correct:
                            break
                    else:
                    for block in extract_all_function_blocks(output[x]['chat_history']):
                        corr = evaluator_function(predicted_answer=block, sample=dataset[x])
                        if corr["is_correct"]:
                            correct_count += 1
                            is_correct = True
                            break

                output[x]['correct'] = is_correct


            with open(data_path + f"_run{i}_CORRECTED.json", "w") as f_out:
                json.dump(output, f_out, indent=2)

            avg.append(correct_count / numQ)
            print(f"Run {i}: {correct_count/numQ}")

        print(f"Average Accuracy: {np.mean(avg)}")