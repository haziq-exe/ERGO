# core/ergo.py
from typing import List, Dict
from .model import BaseModel
from .dataset import Dataset
from .prompts import GSM8K_prompt, Code_prompt, D2T_prompt, DB_prompt


class Ergo:
    
    def __init__(self, model: BaseModel, threshold):
        """
        Initialize ERGO with a model and entropy threshold.
        model: Any BaseModel (OpenAIModel, LocalModel)
        threshold: Threshold 𝚫Entropy must exceed to trigger rewriting.
        rewrite_prompts: Dataset specific few-shot prompts for rewriting.
        """

        self.model = model
        self.threshold = threshold

        
        self.rewrite_prompts = {
            "GSM8K": GSM8K_prompt,
            "Database": DB_prompt,
            "Code": Code_prompt,
            "Actions": GSM8K_prompt, # We reused GSM8K prompt for Actions
            "DataToText": D2T_prompt
        }

    def rewrite_prompt(self, prompt, dataset: Dataset):
        """
        Rewrites all prior user messages into a single consolidated input.
        Keeps system/few-shot context from dataset-specific templates.

        returns: Prompt for model to rewrite along with few-shot examples.
        """

        
        new_prompt = self.rewrite_prompts.get(dataset.dataset_name, []).copy()

        
        user_content = (
            "I have a set of questions and/or statements, "
            "please REWRITE all the questions/statements so that they are in "
            "the most optimal order that is the easiest to understand. "
            "DO NOT ANSWER ANY OF THE QUESTIONS, JUST REWRITE. JUST RETURN THE REWRITTEN PROMPT\n"
            "Here are the instructions:\n"
        )

        
        user_messages = [item["content"] for item in prompt if item.get("role") == "user"]

        for i, msg in enumerate(user_messages):
            user_content += f"User Instruction {i+1}: {msg}\n"

        
        new_prompt.append({"role": "user", "content": user_content})

        return new_prompt

    def run(self, sharded_prompt, dataset: Dataset, prev_entropy):
        """
        Run ERGO on a prompt:
        - Generate response
        - Check entropy
        - If above threshold, rewrite prompt, start new context with just rewritten prompt and regenerate
        """
        avg_entropy, response = self.model.generate(sharded_prompt)
        reset = False
        
        if avg_entropy - prev_entropy >= self.threshold:
            reset = True
            rewritten = self.rewrite_prompt(sharded_prompt, dataset)
            _, rewritten_context = self.model.generate(rewritten)

            sharded_prompt = [msg for msg in sharded_prompt if msg["role"] == "system"]

            sharded_prompt.append({"role": "user", "content": rewritten_context})
            avg_entropy, response = self.model.generate(sharded_prompt)

        return avg_entropy, response, reset
