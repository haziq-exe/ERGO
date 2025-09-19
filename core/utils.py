# core/utils.py
import json
from pathlib import Path
from .model import BaseModel
from .dataset import Dataset


class Logger:
    def __init__(self, model: BaseModel = None, dataset: Dataset = None, output_path: str = None):
        """
        Logger for saving model responses and entropy values to a JSON file.

        Args:
            model: instance of BaseModel (has model_name)
            dataset: instance of ShardedDataset (has dataset_name)
            output_path: optional, custom output file path. If None, defaults to ModelName_DatasetName.json
        """

        self.model_name = model.model_name
        self.dataset_name = dataset.dataset_name

        if output_path is None and self.model_name and self.dataset_name:
            self.output_file = f"{self.model_name}_{self.dataset_name}.json"
        else:
            self.output_file = output_path

        self.logs = []

    def log_entry(self, item_id, chat_history, entropy):
        """
        Add a log entry to memory.
        """
        entry = {
            "item_id": item_id,
            "chat_history": chat_history,
            "entropies": entropy
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
