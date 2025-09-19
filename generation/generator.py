from core.model import BaseModel
from core.dataset import Dataset  
from core.ergo import Ergo
from core.utils import Logger
import random

class RunERGO():
    def __init__(self, model: BaseModel, dataset: Dataset, ergo: Ergo, logger: Logger, num_Qs : int, num_runs : int = 1):
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

        if not num_Qs:
            self.num_Qs = len(dataset)
        else:
            self.num_Qs = num_Qs

    def execute(self):
        for x in range(self.num_runs):
            for i in range(self.num_Qs):
                item = self.dataset.data[i]
                messages = [self.dataset.get_base_system(i)]
                entropies = []
                prev_entropy = float("inf")
                resets = 0

                for shard in item["shards"]:
                    user_content = shard['shard']
                    if shard['shard_id'] != 1:
                        connector = random.choice(self.dataset.connectors)
                        user_content = connector + user_content
                    if shard["shard_id"] == len(item["shards"]):
                        user_content += self.dataset.final_shard_instruct
                    messages.append({"role": "user", "content": user_content})

                    entropy, new_message, reset = self.ergo.run(messages, self.dataset, prev_entropy)
                    messages.append({"role": "assistant", "content": new_message})
                    prev_entropy = entropy
                    entropies.append(entropy)
                    if reset:
                        resets += 1
                    
                    if shard['shard_id'] == len(item["shards"]):
                        self.logger.log_entry(i, messages, entropies)
                        self.logger.save(x)
                
            

