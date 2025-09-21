import json
from core.dataset import GSM8K, Code, Database, DataToText, Actions
from .utils import CodeEvalUtils, DatabaseEvalUtils, DataToTextEvalUtils, ActionsEvalUtils
import ast
import numpy as np
import re
import os

class Evaluator:
    def __init__(self, output_file, dataset_path):
        """
        Initialize the Evaluator with an output file path to save eval to.
        Adds a "correct" boolean field to each entry in the output file 
        (float "score" for DataToText dataset)

        To do: Fix the mess that is the code of its subclasses.
        """
        self.output_file = output_file
        self.dataset_path = dataset_path
        
    
class CodeEvaluator(Evaluator):

    def __init__(self, output_file, dataset_path):
        super().__init__(output_file, dataset_path)

    def evaluate(self, numruns, numQ):
        finalcorr = []
        utils = CodeEvalUtils()
        dataset = Code(self.dataset_path).load_data()

        for i in range(numruns):
            correct = []
            res = []
            run_path = self.output_file.replace(".json", f"_run{i}.json")
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
        scores = []
        for i, corr in enumerate(finalcorr):
            score = np.sum(corr) / numQ
            # print(f"Run {i} Score: {score}")
            scores.append(score)

        Average = {sums / len(finalcorr)}

        return scores, Average

class GSM8KEvaluator(Evaluator):

    def __init__(self, output_file, dataset_path):
        super().__init__(output_file, dataset_path)

    def evaluate(self, numruns, numQ):
        dataset = GSM8K(self.dataset_path).load_data()
        all_runs = []

        for run in range(numruns):
            run_path = self.output_file.replace(".json", f"_run{run}.json")

            with open(run_path, "r", encoding="utf-8") as f:
                NRoutput = json.load(f)

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

                corrected_path = run_path + f"_run{run}_CORRECTED.json"
                with open(corrected_path, "w", encoding="utf-8") as f_out:
                    json.dump(NRoutput, f_out, indent=2)

                all_runs.append(corr)

        sums = 0.0
        scores = []
        for i, run_corr in enumerate(all_runs):
            score = np.sum(run_corr) / numQ
            # print(f"Run {i} Score: {score}")
            scores.append(score)

        if all_runs:
            Average = {sums / len(all_runs)}

        return scores, Average
    
class ActionsEvaluator(Evaluator):

    def __init__(self, output_file, dataset_path):
        super().__init__(output_file, dataset_path)

    def evaluate(self, numruns, numQ):

        utils = ActionsEvalUtils()
        dataset = Actions(self.dataset_path).load_data()

        runs = []
        scores = []

        for i in range(numruns):
            run_path = self.output_file.replace(".json", f"_run{i}.json")
            with open(run_path, "r", encoding="utf-8") as f:
                output = json.load(f)

            run_results = []
            correct_count = 0

            for x in range(numQ):
                sample = dataset[x]
                raw_model_output = output[x].get('final_output', "") or ""

                model_answers = re.findall(r'<Answer>\s*(.*?)(?:</Answer>|$)', raw_model_output, flags=re.DOTALL)
                if model_answers:
                    raw_model_output = model_answers[0]


                mod_ans = utils.extract_function_block(raw_model_output)
                mod_ans = utils.clean_function_block(mod_ans)
                corr = utils.evaluator_function(predicted_answer=mod_ans, sample=sample)


                if "Failing to parse the predicted answer as an AST" in (corr.get("error") or ""):
                    mod_ans = utils.extract_function_block(f"[{raw_model_output}]")
                    mod_ans = utils.clean_function_block(mod_ans)
                    corr = utils.evaluator_function(predicted_answer=mod_ans, sample=sample)

                is_correct = False
                if corr.get("is_correct"):
                    correct_count += 1
                    is_correct = True
                else:
                    if output[x].get('resets', 0) == 0:
                        for entry in output[x].get('chat_history', []):
                            if entry.get("role") == "assistant":
                                for block in utils.extract_all_function_blocks(entry.get('content', "")):
                                    corr = utils.evaluator_function(predicted_answer=block, sample=sample)
                                    if corr.get("is_correct"):
                                        correct_count += 1
                                        is_correct = True
                                        break
                            if is_correct:
                                break
                    else:
                        for block in utils.extract_all_function_blocks(output[x].get('chat_history', [])):
                            corr = utils.evaluator_function(predicted_answer=block, sample=sample)
                            if corr.get("is_correct"):
                                correct_count += 1
                                is_correct = True
                                break

                output[x]['correct'] = bool(is_correct)
                run_results.append(1 if is_correct else 0)

            corrected_path = self.output_file + f"_run{i}_CORRECTED.json"
            with open(corrected_path, "w", encoding="utf-8") as f_out:
                json.dump(output, f_out, indent=2)

            runs.append(run_results)
            score = correct_count / numQ if numQ else 0.0
            scores.append(score)

        average_accuracy = float(np.mean(scores)) if scores else 0.0
        return scores, average_accuracy
    
class DatabaseEvaluator(Evaluator):
    def __init__(self, output_file, dataset_path):
        super().__init__(output_file, dataset_path)
    
    def evaluate(self, numruns, numQ, spider_DB_path):
        utils = DatabaseEvalUtils()
        dataset = Database(self.dataset_path).load_data()

        data_path = self.output_file
        DB_FOLDER = spider_DB_path

        avg = []

        for run_idx in range(numruns):
            run_path = self.output_file.replace(".json", f"_run{run_idx}.json")
            with open(run_path, "r", encoding="utf-8") as f:
                entries = json.load(f)

            correct = 0
            res = []

            for e in range(numQ):
                db_id = dataset[e]['db_id']
                ref_sql = dataset[e]['reference_sql']
                db_path = os.path.join(DB_FOLDER, f"{db_id}/{db_id}.sqlite")

            try:
                ref_res = utils.run_query(db_path, ref_sql)
            except Exception:
                entries[e]['correct'] = False
                res.append(entries[e].get('resets', 0))
                continue

            matched = False

            for llm_sql in utils.extract_sql_queries(entries[e].get('final_output', "")):
                try:
                    llm_res = utils.run_query(db_path, llm_sql)
                except Exception:
                    continue
                if ref_res == llm_res:
                    matched = True
                    break


            if not matched:
                if entries[e].get("resets", 0) == 0:
                    for msg in entries[e].get('chat_history', []):
                        if msg.get("role") != "assistant":
                            continue
                        for llm_sql in utils.extract_sql_queries(msg.get('content', "")):
                            try:
                                llm_res = utils.run_query(db_path, llm_sql)
                            except Exception:
                                continue
                            if ref_res == llm_res:
                                matched = True
                                break
                    if matched:
                        break
                else:
                    for llm_sql in utils.extract_sql_queries(entries[e].get('chat_history', [])):
                        try:
                            llm_res = utils.run_query(db_path, llm_sql)
                        except Exception:
                            continue
                        if ref_res == llm_res:
                            matched = True
                            break

            if matched:
                correct += 1
                entries[e]['correct'] = True
            else:
                entries[e]['correct'] = False

            res.append(entries[e].get('resets', 0))

            accuracy = correct / numQ if numQ else 0.0
            avg.append(accuracy)
            # print(f"Accuracy = {accuracy}")

            corrected_path = data_path + f"_run{run_idx}_CORRECTED.json"
            with open(corrected_path, "w", encoding="utf-8") as f_out:
                json.dump(entries, f_out, indent=2)

        overall = float(np.mean(avg)) if avg else 0.0
        # print(f"Average = {overall}")
        return avg, overall
    
class DataToTextEvaluator(Evaluator):
    def __init__(self, output_file, dataset_path):
        super().__init__(output_file, dataset_path)

    def evaluate(self, numruns, numQ):
        scores = []
        D2TEval = DataToTextEvalUtils()
        D2T = DataToText(self.dataset_path).load_data()
        for x in range(numruns):
            run_path = self.output_file.replace(".json", f"_run{x}.json")
            with open(run_path, "r", encoding="utf-8") as f:
                output = json.load(f)

            run_scores = []
            for y in range(numQ):
                score = D2TEval.D2T_evaluator_function(output[y]['final_output'], D2T[y])
                run_scores.append(score)
                output[y]["score"] = score

            avg_score = np.mean(run_scores) if run_scores else 0.0
            scores.append(avg_score)

        average = np.mean(scores)
        
        return scores, average