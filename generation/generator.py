from core.model import BaseModel
from core.dataset import Dataset  
from core.ergo import Ergo
from core.utils import Logger
from evaluation.evaluator import GSM8KEvaluator, ActionsEvaluator, CodeEvaluator, DatabaseEvaluator, DataToTextEvaluator, Evaluator
import random, gc, torch

class RunERGO():
    def __init__(self, model: BaseModel, dataset: Dataset, evaluator: Evaluator,  ergo: Ergo, logger: Logger, num_Qs : int, num_runs : int = 1):
        """
        Executes the ERGO process on a given dataset and model, logging results.
        :param model: Instance of BaseModel (OpenAIModel, LocalModel)
        :param dataset: Instance of Dataset (GSM8K, Database, etc.)
        :param ergo: Instance of Ergo for entropy checking and rewriting
        :param logger: Instance of Logger to save results
        :param num_Qs: Number of questions/items to process from the dataset
        :param num_runs: Number of times to run the entire process (default 1)
        """
        self.model = model
        self.dataset = dataset
        self.ergo = ergo
        self.logger = logger
        self.num_runs = num_runs
        self.evaluator = evaluator

        if not num_Qs:
            self.num_Qs = len(dataset)
        else:
            self.num_Qs = num_Qs

    def execute(self, spider_DB_path=None, clear_cache=False):
        for run in range(self.num_runs):
            for question in range(self.num_Qs):
                item = self.dataset.data[question]
                messages = [self.dataset.get_base_system(question)]
                message_history = []
                prev_entropy = float("inf")
                resets = []
                entropies = {}

                for shard in item["shards"]:
                    user_content = shard['shard']
                    if shard['shard_id'] != 1:
                        connector = random.choice(self.dataset.connectors)
                        user_content = connector + user_content
                    if shard["shard_id"] == len(item["shards"]):
                        user_content += self.dataset.final_shard_instruct
                    messages.append({"role": "user", "content": user_content})

                    entropy, new_message, reset, messages, prev_prompts = self.ergo.run(messages, self.dataset, prev_entropy)
                    messages.append({"role": "assistant", "content": new_message})
                    prev_entropy = float("inf")
                    for key in entropy.keys():
                        if key not in entropies.keys():
                            entropies[key] = []
                        entropies[key].append(entropy[key])

                    if reset:
                        resets.append(1)
                        # message_history.append(prev_prompts)
                    else:
                        resets.append(0)

                    if self.evaluator.identifier() == "Database":
                        result = self.evaluator.evaluate(dataset=self.dataset, extracted_answer=new_message, spider_DB_path=spider_DB_path, question_id=question)
                    elif self.evaluator.identifier() == "DataToText":
                        continue
                    else:
                        result = self.evaluator.evaluate(dataset=self.dataset, extracted_answer=new_message, question_id=question)

                    if shard['shard_id'] == len(item["shards"]) or result["score"] == 1.0:
                        if self.evaluator.identifier() == "DataToText":
                            result = self.evaluator.evaluate(dataset=self.dataset, extracted_answer=new_message, question_id=question)

                        self.logger.log_entry(question, messages, new_message, entropies, resets, result, message_history)
                        self.logger.save(run)
                    
                        if clear_cache:
                            gc.collect()
                            torch.cuda.empty_cache()
                        
                        break

    def execute_full(self, spider_DB_path=None, clear_cache=False):
        for run in range(self.num_runs):
            for question in range(self.num_Qs):
                item = self.dataset.data[question]
                messages = [self.dataset.get_base_system(question)]
                prompt = "Task: \n"

                for shard in item["shards"]:
                    prompt += f"- {shard['shard']}\n"
                
                print("Full Prompt: " + prompt)

                messages.append({"role": "user", "content": prompt})
                response, attention_scores = self.ergo.run_FULL(messages)

                if self.evaluator.identifier() == "Database":
                    result = self.evaluator.evaluate(dataset=self.dataset, extracted_answer=response, spider_DB_path=spider_DB_path, question_id=question)
                elif self.evaluator.identifier() == "DataToText":
                    continue
                else:
                    result = self.evaluator.evaluate(dataset=self.dataset, extracted_answer=response, question_id=question)

                if shard['shard_id'] == len(item["shards"]) or result["score"] == 1.0:
                    if self.evaluator.identifier() == "DataToText":
                        result = self.evaluator.evaluate(dataset=self.dataset, extracted_answer=response, question_id=question)

                messages.append({"role": "assistant", "content": response})

                self.logger.log_entry_full(question, messages, attention_scores, result)
                self.logger.save(run)
                
                if clear_cache:
                    gc.collect()
                    torch.cuda.empty_cache()
                    
# class RunERGO_FULL(RunERGO):
#     def __init__(self, model: BaseModel, dataset: Dataset, evaluator: Evaluator,  ergo: Ergo, logger: Logger, num_Qs : int = None, num_runs : int = 1):
#         super().__init__(model, dataset, evaluator, ergo, logger, num_Qs, num_runs)

#     def execute(self, spider_DB_path=None, clear_cache=False):
#         for run in range(self.num_runs):
#             for question in range(self.num_Qs):
#                 item = self.dataset.data[question]
#                 messages = [self.dataset.get_base_system(question)]
#                 message_history = []
#                 entropies = {"avg_entropy": [], "rds_entail": [], "rds_embed": [], "margin": [], "norm_entropy": [], "perplexity": []}
#                 prev_entropy = float("inf")
#                 resets = []

#                 user_content = item['full_input']
#                 messages.append({"role": "user", "content": user_content})

#                 entropy, new_message, reset, messages, prev_prompts, r0, rh = self.ergo.run(messages, self.dataset, prev_entropy)
#                 messages.append({"role": "assistant", "content": new_message})
#                 prev_entropy = entropy
#                 message_history.append({"r0": r0, "rh": rh})
                
#                 for k in entropies.keys():
#                     entropies[k].append(entropy[k])

#                 if reset:
#                     resets.append(1)
#                 else:
#                     resets.append(0)

#                 if self.evaluator.identifier() == "Database":
#                     result = self.evaluator.evaluate(dataset=self.dataset, extracted_answer=new_message, spider_DB_path=spider_DB_path, question_id=question)
#                 else:
#                     result = self.evaluator.evaluate(dataset=self.dataset, extracted_answer=new_message, question_id=question)

#                 self.logger.log_entry(question, messages, new_message, entropies, resets, result, message_history)
#                 self.logger.save(run)
                
#                 if clear_cache:
#                     gc.collect()
#                     torch.cuda.empty_cache()
            

