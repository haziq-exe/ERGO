# core/utils.py
import json
from pathlib import Path
from .model import BaseModel
from .dataset import Dataset


class Logger:
    def __init__(self, model: BaseModel, dataset: Dataset, output_path):
        """
        Logger for saving model responses and entropy values to a JSON file.
        If output_path = None, defaults to ModelName_DatasetName.json
        """

        self.model_name = model.model_name
        self.dataset_name = dataset.dataset_name
        self.output_file = output_path

        self.logs = []

    def log_entry(self, item_id, chat_history, final_output, entropy, resets, result, message_history):
        """
        Log a single entry consisting of:
        - item_id: unique identifier for the data item
        - chat_history: list of (user_message, model_response) tuples
        - final_output: final response from the model
        - entropy: list of entropy values for each model response
        - resets: array 0 and 1 indicating which shard reset was triggered
        - message_history: list of previous message histories before resets
        """
        entry = {
            "item_id": item_id,
            "chat_history": chat_history,
            "final_output": final_output,
            "entropies": entropy,
            "resets": resets,
            "result": result,
            "message_history": message_history
        }
        self.logs.append(entry)

    def save(self, i):
        """
        Save all logs to the output JSON file.
        """

        out_path = self.output_file.replace(".json", f"_run{i}.json")

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.logs, f, indent=2, ensure_ascii=False)

    def __len__(self):
        return len(self.logs)
