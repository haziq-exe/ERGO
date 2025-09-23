import json
from core.dataset import GSM8K, Code, Database, DataToText, Actions, Dataset
from .utils import CodeEvalUtils, DatabaseEvalUtils, DataToTextEvalUtils, ActionsEvalUtils
import ast
import numpy as np
import re
import os

class Evaluator:
    def __init__(self, output_file=None, dataset_path=None):
        """
        Initialize the Evaluator with an output file path to save eval to.
        Adds a "score" to each entry in the output file.
        """
        self.output_file = output_file
        self.dataset_path = dataset_path
    
    def evaluate(self):
        """
        Evaluate the extracted answer against the ground truth.
        Returns: {"score": float, "error": str or None}
        """
        raise NotImplementedError
    
    def identifier(self):
        """
        Returns a string identifier for the evaluator.
        """
        raise NotImplementedError
        

        
class CodeEvaluator(Evaluator):

    def __init__(self, output_file=None, dataset_path=None):
        super().__init__(output_file, dataset_path)

    def evaluate(self, dataset: Dataset, extracted_answer, question_id):
        sample = dataset.data[question_id]
        utils = CodeEvalUtils()
        function, func_name = utils.extract_first_function_block_and_name(extracted_answer)

        test_cases = ast.literal_eval(sample["public_test_cases"])

        if function and func_name:
            passed = utils.run_function_and_check(func_name=func_name, user_code=function, test_cases=test_cases)
            if passed:
                return {"score": 1.0, "error": None}
            else:
                return {"score": 0.0, "error": "Function did not pass public test cases."}
        else:
            return {"score": 0.0, "error": "No function found in model output."}

    def identifier(self):
        return "Code"

class GSM8KEvaluator(Evaluator):

    def __init__(self, output_file=None, dataset_path=None):
        super().__init__(output_file, dataset_path)
    
    def evaluate(self, dataset: Dataset, extracted_answer, question_id):

        # Following is taken directly from https://github.com/microsoft/lost_in_conversation/blob/main/tasks/math/task_math.py
        # Only slightly modified to fit our code structure.

        regexes_to_ignore = [",", "\\$", "(?s).*#### ", "\\.$"]
        sample = dataset.data[question_id]

        # ground truth
        gold = sample["answer"].split("####")[1].strip().lower()

        try:
            # https://github.com/EleutherAI/lm-evaluation-harness/blob/bb098f13b05e361f01a5afe7b612779ce362b3f2/lm_eval/tasks/gsm8k/gsm8k.yaml#L42
            extracted_answer = extracted_answer.strip()
            # strict
            # extracted_answer = re.findall(r"(\-?[0-9\.\,]+)", extracted_answer)[0]
            # flexible
            extracted_answer = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", extracted_answer)[-1]
            extracted_answer = [m for m in extracted_answer if m][0]
        except:
            return {"score": 0.0, "error": f"Answer could not be extracted: {repr(extracted_answer)}"}

        # custom formatting fix
        # if dollar mark is in the answer, check for the cents and trim if necessary
        if re.search(r'\$', extracted_answer) and extracted_answer.endswith(".00"):
            extracted_answer = extracted_answer.rstrip(".00")                

        # ref: https://github.com/EleutherAI/lm-evaluation-harness/blob/52df63b7b30da53c481ed9090598d9189fab1d91/lm_eval/api/metrics.py#L198
        # further normalize $ and , for both extracted_answer and gold
        for regex in regexes_to_ignore:
            extracted_answer = re.sub(regex, "", extracted_answer)
            gold = re.sub(regex, "", gold)
        score = 1.0 if extracted_answer == gold else 0.0
        return {"score": score, "error": None}
    
    def identifier(self):
        return "GSM8K"
    
class ActionsEvaluator(Evaluator):

    def __init__(self, output_file=None, dataset_path=None):
        super().__init__(output_file, dataset_path)

    def evaluate(self, dataset: Dataset, extracted_answer, question_id):
        sample = dataset.data[question_id]
        utils = ActionsEvalUtils()

        try:
            mod_ans = utils.extract_function_block(extracted_answer)
            mod_ans = utils.clean_function_block(mod_ans)
            corr = utils.evaluator_function(predicted_answer=mod_ans, sample=sample)
        except:
            return {"score": 0.0, "error": "Exception during evaluation"}

        if corr.get("is_correct"):
            return {"score": 1.0, "error": None}
        else:
            return {"score": 0.0, "error": corr.get("error")}

    def identifier(self):
        return "GSM8K"
    
class DatabaseEvaluator(Evaluator):
    def __init__(self, output_file=None, dataset_path=None):
        super().__init__(output_file, dataset_path)
    

    def evaluate(self, dataset: Dataset, extracted_answer, spider_DB_path, question_id):
        sample = dataset.data[question_id]
        utils = DatabaseEvalUtils()
        db_id = sample['db_id']
        ref_sql = sample['reference_sql']
        db_path = os.path.join(spider_DB_path, f"{db_id}/{db_id}.sqlite")
        try:
            ref_res = utils.run_query(db_path, ref_sql)
        except Exception as e:
            return {"score": 0.0, "error": f"Error running reference SQL (issue with spiderDB not with model answer): {str(e)}"}


        for llm_sql in utils.extract_sql_queries(extracted_answer):
            try:
                llm_res = utils.run_query(db_path, llm_sql)
            except Exception as e:
                continue
            if ref_res == llm_res:
                return {"score": 1.0, "error": None}


        return {"score": 1.0, "error": None}
    

    def identifier(self):
        return "Database"
    
class DataToTextEvaluator(Evaluator):
    def __init__(self, output_file=None, dataset_path=None):
        super().__init__(output_file, dataset_path)

    def evaluate(self, dataset: Dataset, extracted_answer, question_id):
        D2TEval = DataToTextEvalUtils()
        reference_answer = dataset.data[question_id]

        score = D2TEval.D2T_evaluator_function(extracted_answer, reference_answer)
        
        return {"score": np.round(score, 4), "error": None}
    
    def identifier(self):
        return "DataToText"