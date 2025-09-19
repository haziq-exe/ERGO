# core/model.py
from openai import OpenAI
import math
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class BaseModel:
    def __init__(self, model_name: str, api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key

    def generate(self, prompt: List[Dict[str, str]]) -> str:
        """
        Generate output based on the given prompt.
        `prompt` is expected to be a list of dicts with 'role' and 'content'.
        """
        raise NotImplementedError

    def compute_entropy(self, output: Any) -> float:
        """
        Compute the average token-level entropy of the model's output.
        """
        raise NotImplementedError
    

class LocalLLMModel(BaseModel):

    def __init__(self, model_name, device, max_length, torch_dtype=torch.float16, temperature=1.0, do_sample=True):
        super().__init__(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto"
        )
        self.device = device
        self.temperature = temperature
        self.max_length = max_length
        self.do_sample = do_sample

    def generate(self, messages):
        """
        Generate text using a local Hugging Face model.
        Returns: (avg_entropy, generated_text, tokens_used)
        """

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # ensures assistant tag is included
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=inputs.input_ids.shape[1] + self.max_length,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                temperature=self.temperature,
            )

            logits = torch.stack(outputs.scores)
            probs = torch.softmax(logits, dim=-1)
            avg_entropy = self.compute_entropy(probs)
        
        response_ids = outputs.sequences
        input_len = inputs.input_ids.shape[1]
        new_tokens = response_ids[:, input_len:]
        response_only = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

        tokens_used = response_ids.shape[1] 

        return avg_entropy, response_only, tokens_used

    def compute_entropy(self, probs: torch.Tensor) -> float:
        """
        Compute Shannon entropy across generated tokens.
        probs: Tensor of shape [new_tokens, batch, vocab]
        """
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        avg_entropy = entropy.mean().item()
        return avg_entropy


class OpenAIModel(BaseModel):

    def __init__(self, model_name, api_key, temperature, max_tokens, top_logprobs: int = 20):
        super().__init__(model_name, api_key)
        self.client = OpenAI(api_key=self.api_key)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_logprobs = top_logprobs

    def generate(self, prompt: List[Dict[str, str]]):
        """
        Sends a prompt to OpenAI Chat API and returns:
        (average_entropy, generated_text, tokens_used)
        """
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=prompt,
            max_completion_tokens=self.max_tokens,
            temperature=self.temperature,
            logprobs=True,
            top_logprobs=self.top_logprobs,
        )

        generated_text = resp.choices[0].message.content
        token_logprobs = resp.choices[0].logprobs.content

        avg_entropy = self.compute_entropy(token_logprobs)
        tokens_used = resp.usage.completion_tokens + resp.usage.prompt_tokens

        return avg_entropy, generated_text, tokens_used

    def compute_entropy(self, token_logprobs: List[Dict[str, Any]]) -> float:
        """
        Compute average Shannon entropy across tokens using top_logprobs info.
        """
        entropies = []
        for token_info in token_logprobs:
            entropy = 0.0
            for logprob in token_info.top_logprobs.values():
                p = math.exp(logprob)
                entropy += -p * logprob
            entropies.append(entropy)

        avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0
        return avg_entropy
