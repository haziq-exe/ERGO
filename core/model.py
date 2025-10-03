# core/model.py
from openai import OpenAI
import math
import os
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from torch.nn import functional as F
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
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
                trust_remote_code=True
            ).to(device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
                device_map=device_map,
                trust_remote_code=True
            )

        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.entailment_model = pipeline("text-classification", model="roberta-large-mnli")


    def generate_with_temperature(self, messages, temperature, perturb_first_n=0):
        """
        Generate with given temperature, 
        with optional hybrid perturbation (first N tokens at temp).
        """

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # custom loop if hybrid perturbation needed
        if perturb_first_n > 0:
            generated = inputs["input_ids"]
            past_key_values = None
            for i in range(self.max_new_tokens):
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=generated,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

                if i < perturb_first_n:
                    t = 1.5  # perturb temp
                else:
                    t = temperature

                probs = F.softmax(logits / t, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=-1)

            new_tokens = generated[:, inputs["input_ids"].shape[1]:]

            return self.tokenizer.decode(new_tokens[0], skip_special_tokens=True)

        # simple case: normal generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens[0], skip_special_tokens=True)
    
    def semantic_similarity_embeddings(self, text1, text2):
        emb1 = self.embedding_model.encode(text1, convert_to_tensor=True)
        emb2 = self.embedding_model.encode(text2, convert_to_tensor=True)
        return util.pytorch_cos_sim(emb1, emb2).item()

    def semantic_similarity_entailment(self, text1, text2):
        # check entailment both ways
        res1 = self.entailment_model({"text": text1, "text_pair": text2})[0]
        res2 = self.entailment_model({"text": text2, "text_pair": text1})[0]
        score1 = res1["score"] if res1["label"] == "ENTAILMENT" else 0.0
        score2 = res2["score"] if res2["label"] == "ENTAILMENT" else 0.0
        return (score1 + score2) / 2.0

    def generate(self, messages):
        """
        Device-aware generation for single-device and device_map='auto' sharded models.
        Returns: (avg_entropy, generated_text)
        """


        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )



        print(f"Model's reported device: {self.model.device}")
        print(f"First parameter device: {next(self.model.parameters()).device}")
        print(f"Embedding layer device: {self.model.get_input_embeddings().weight.device}")

        # 2. Check model's device map (shows which layers are on which GPU)
        if hasattr(self.model, 'hf_device_map'):
            print(f"Model device map: {self.model.hf_device_map}")

        # 3. Check input tensors
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        print("\nInput tensor devices:")
        for key, value in inputs.items():
            print(f"  {key}: device={value.device}, shape={value.shape}")

        # 4. After moving inputs (if you do)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        print(f"\nAfter moving to {self.model.device}:")

        for key, value in inputs.items():
            print(f"  {key}: {value.device}")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                temperature=self.temperature,
            )

            scores = outputs.scores
            if scores:
                base_dev = scores[0].device
                logits = torch.stack([s.to(base_dev) for s in scores])
                probs = torch.softmax(logits, dim=-1)
                avg_entropy = self.compute_entropy(probs)
                avg_margin = self.compute_logit_margin(probs)
                norm_entropy = self.compute_normalized_entropy(probs)
                ppl = self.compute_perplexity(avg_entropy)
            else:
                avg_entropy = None

        response_ids = outputs.sequences
        input_len = inputs["input_ids"].shape[1]
        new_tokens = response_ids[:, input_len:].cpu()
        response_only = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]


                # ----- Other runs for robustness divergence -----
        r0 = self.generate_with_temperature(messages, temperature=0.2)
        r1 = response_only
        # perturb: 1% of length of r1, at least 1 token
        perturb_n = max(1, int(0.01 * len(new_tokens[0])))
        rh = self.generate_with_temperature(messages, temperature=1.0, perturb_first_n=perturb_n)

        # Pairwise semantic distances
        sims_embed = [
            self.semantic_similarity_embeddings(r0, r1),
            self.semantic_similarity_embeddings(r0, rh),
            self.semantic_similarity_embeddings(r1, rh)
        ]
        sims_entail = [
            self.semantic_similarity_entailment(r0, r1),
            self.semantic_similarity_entailment(r0, rh),
            self.semantic_similarity_entailment(r1, rh)
        ]

        rds_embed = 1 - sum(sims_embed) / 3.0
        rds_entail = 1 - sum(sims_entail) / 3.0

        return {
            "avg_entropy": avg_entropy,
            "margin": avg_margin,
            "norm_entropy": norm_entropy,
            "perplexity": ppl,
            "rds_embed": rds_embed,
            "rds_entail": rds_entail
        }, response_only

    def compute_entropy(self, probs):
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        avg_entropy = entropy.mean().item()
        return avg_entropy

    def compute_normalized_entropy(self, probs):
        vocab_size = probs.size(-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  # [steps, batch]
        norm_entropy = entropy / torch.log(torch.tensor(vocab_size, device=probs.device))
        return norm_entropy.mean().item()

    def compute_logit_margin(self, probs):
        top2 = torch.topk(probs, k=2, dim=-1).values  # [steps, batch, 2]
        margin = top2[..., 0] - top2[..., 1]
        return margin.mean().item()

    def compute_perplexity(self, avg_entropy):
        if avg_entropy is None:
            return None
        return float(torch.exp(torch.tensor(avg_entropy)))



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
