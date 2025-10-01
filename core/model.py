# core/model.py
from openai import OpenAI
import math
import os
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
import torch
import sys

# logging.set_verbosity_info()

class BaseModel:
    def __init__(self, model_name):
        """
        Base class for language models. Either local huggingface model or OpenAI API.
        if openai=True it uses OpenAI API.
        """
        self.model_name = model_name

    def generate(self, prompt):
        """
        Generate text using a local Hugging Face model or OpenAI API.
        Returns: (avg_entropy, generated_text)
        """
        raise NotImplementedError

    def compute_entropy(self, output: Any) -> float:
        """
        Compute the average token-level entropy of the generated text.
        Returns: avg_entropy
        """
        raise NotImplementedError
    

class LocalLLMModel(BaseModel):

    def __init__(self, model_name, device, max_new_tokens, dtype=torch.float16, temperature=1.0, do_sample=True, device_map="auto"):
        super().__init__(model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name,  trust_remote_code=True)

        if device:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
                device_map=device_map,
                trust_remote_code=True
            ).to(device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
                trust_remote_code=True
            )

        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample

    def generate(self, messages):
        """
        Device-aware generation for single-device and device_map='auto' sharded models.
        Returns: (avg_entropy, generated_text)
        """

        # Build prompt
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize (returns BatchEncoding)
        raw_inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)

        # Convert BatchEncoding -> plain dict (use .data if available)
        if hasattr(raw_inputs, "data"):
            inputs_dict = dict(raw_inputs.data)
        else:
            inputs_dict = dict(raw_inputs)

        # Keep only torch.Tensor values (drop None and non-tensor fields)
        tensor_inputs = {k: v for k, v in inputs_dict.items() if isinstance(v, torch.Tensor)}

        if "input_ids" not in tensor_inputs:
            raise RuntimeError("tokenizer did not return 'input_ids' as a tensor")

        # Determine device of input embeddings (most reliable)
        embed_device = None
        try:
            emb = self.model.get_input_embeddings()
            embed_param = next(emb.parameters())  # will raise if no params
            embed_device = embed_param.device
        except Exception:
            # Fallback to hf_device_map lookup for standard key
            if hasattr(self.model, "hf_device_map"):
                dev_id = self.model.hf_device_map.get("model.embed_tokens", None)
                if dev_id is not None:
                    if isinstance(dev_id, str):
                        embed_device = torch.device(dev_id)
                    else:
                        embed_device = torch.device(f"cuda:{dev_id}")

        # Final fallback to model.device or cpu
        if embed_device is None:
            embed_device = getattr(self.model, "device", torch.device("cpu"))

        # Move tensors to embed device (only tensors)
        tensor_inputs = {k: v.to(embed_device) for k, v in tensor_inputs.items()}

        # DEBUG: show devices (remove or comment out later)
        # print("Input tensor devices:", {k: v.device for k, v in tensor_inputs.items()})

        # Generation
        with torch.no_grad():
            outputs = self.model.generate(
                **tensor_inputs,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                temperature=self.temperature,
            )

            # Stack scores safely on the device they're on
            scores = outputs.scores  # list of tensors
            if scores:
                base_dev = scores[0].device
                logits = torch.stack([s.to(base_dev) for s in scores])  # (seq_len, batch, vocab)
                probs = torch.softmax(logits, dim=-1)
                avg_entropy = self.compute_entropy(probs)
            else:
                avg_entropy = None

        # Extract generated tokens (move to CPU before decoding)
        response_ids = outputs.sequences  # (batch, seq_len_total)
        input_len = tensor_inputs["input_ids"].shape[1]
        new_tokens = response_ids[:, input_len:].cpu()
        response_only = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

        return avg_entropy, response_only

    def compute_entropy(self, probs):
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        avg_entropy = entropy.mean().item()
        return avg_entropy


class OpenAIModel(BaseModel):

    def __init__(self, model_name, temperature, max_tokens, top_logprobs: int = 20):
        super().__init__(model_name)
        self.client = OpenAI(api_key = os.environ["OPENAI_KEY"])
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_logprobs = top_logprobs

    def generate(self, prompt: List[Dict[str, str]]):
        """
        Sends a prompt to OpenAI Chat API and returns:
        (average_entropy, generated_text)
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
        try: 
            token_logprobs = resp.choices[0].logprobs.content
        except:
            sys.exit("ERROR: INPUTTED OPENAI MODEL DOESNT RETURN LOGPROBS")

        avg_entropy = self.compute_entropy(token_logprobs)
        # tokens_used = resp.usage.completion_tokens + resp.usage.prompt_tokens

        return avg_entropy, generated_text#, tokens_used

    def compute_entropy(self, token_logprobs):
        entropies = []
        for token_info in token_logprobs:
            entropy = 0.0
            for logprob in token_info.top_logprobs.values():
                p = math.exp(logprob)
                entropy += -p * logprob
            entropies.append(entropy)

        avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0
        return avg_entropy
