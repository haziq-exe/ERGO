# core/ergo.py
from typing import List, Dict
from .model import BaseModel
from .dataset import Dataset


class Ergo:
    
    def __init__(self, model: BaseModel, threshold):
        """
        ERGO orchestrates entropy checking and prompt rewriting.
        :param model: Any BaseModel (OpenAIModel, LocalModel)
        :param threshold: Entropy threshold to trigger rewriting.
        """

        self.model = model
        self.threshold = threshold

        
        self.rewrite_prompts = {
            "GSM8K": [],
            "Database": [],
            "Code": [],
            "Actions": [],
            "DataToText": []
        }

    def rewrite_prompt(self, prompt: List[Dict[str, str]], dataset: Dataset):
        """
        Rewrites all user messages into a single consolidated input.
        Keeps system/few-shot context from dataset-specific templates.
        :param prompt: List of {role, content} messages.
        :param dataset: Dataset to select rewrite few-shot prompt.
        :return: A new prompt list ready for the model.
        """

        
        new_prompt = self.rewrite_prompts.get(dataset.dataset_name, []).copy()

        
        user_content = (
            "I have a set of questions and/or statements, "
            "please REWRITE all the questions/statements so that they are in "
            "the most optimal order that is the easiest to understand. "
            "DO NOT ANSWER ANY OF THE QUESTIONS, JUST REWRITE.\n"
            "Here are the instructions:\n"
        )

        
        user_messages = [item["content"] for item in prompt if item.get("role") == "user"]

        for i, msg in enumerate(user_messages):
            user_content += f"User Instruction {i+1}: {msg}\n"

        
        new_prompt.append({"role": "user", "content": user_content})

        return new_prompt

    def run(self, sharded_prompt: List[Dict[str, str]], dataset: Dataset, prev_entropy : float):
        """
        Run ERGO on a sharded prompt:
        - Generate response
        - Check entropy
        - If above threshold, rewrite prompt and regenerate
        """
        avg_entropy, response, tokens_used = self.model.generate(sharded_prompt)
        reset = False
        
        if avg_entropy - prev_entropy >= self.threshold:
            reset = True
            rewritten = self.rewrite_prompt(sharded_prompt, dataset)
            avg_entropy, response, tokens_used = self.model.generate(rewritten)

        return avg_entropy, response, reset
