# core/model.py
import gc
from openai import OpenAI
import math
import os
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from torch.nn import functional as F
import re
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import sys
from sklearn.cluster import AgglomerativeClustering
import numpy as np


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

    def generate_FULL(self, prompt):
        """
        Generate text using a local Hugging Face model or OpenAI API.
        Used in FULL mode where the entire question is given at once.
        Returns: generated_text
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

    def generate_with_temperature(self, messages, temperature, clear_cache=True):
        """
        Generate with given temperature.
        """
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        
        # Handle multi-GPU device placement correctly
        if hasattr(self.model, 'hf_device_map'):
            device = next(self.model.parameters()).device
        else:
            device = self.model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        input_length = inputs["input_ids"].shape[1]

        # Simple case: use built-in generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        new_tokens = outputs[:, input_length:]
        generated_text = self.tokenizer.decode(new_tokens[0], skip_special_tokens=True)
        print(f"Generated (temp={temperature}): {generated_text}")
        if clear_cache:
            gc.collect()    
            torch.cuda.empty_cache()
        return generated_text
    
    def semantic_similarity_embeddings(self, text1, text2):
        emb1 = self.embedding_model.encode(text1, convert_to_tensor=True)
        emb2 = self.embedding_model.encode(text2, convert_to_tensor=True)
        return util.pytorch_cos_sim(emb1, emb2).item()

    def semantic_similarity_entailment(self, text1, text2):
        res1 = self.entailment_model(
            f"{text1}</s></s>{text2}",
            truncation=True
        )[0]
        res2 = self.entailment_model(
            f"{text2}</s></s>{text1}",
            truncation=True
        )[0]

        return (res1['score'] + res2['score']) / 2

    def run_semantic_entropy(self, messages, runs=5):
        responses = [self.generate_with_temperature(messages, 1.0) for _ in range(runs)]
        embeddings = self.embedding_model.encode(responses)
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.15, metric="cosine", linkage='average').fit(embeddings)
        labels = clustering.labels_
        counts = np.bincount(labels)
        p = counts / counts.sum()
        return -(p * np.log(p + 1e-8)).sum()

    def generate(self, messages, clear_cache=True):
        """
        Device-aware generation for single-device and device_map='auto' sharded models.
        Returns: (avg_entropy, generated_text)
        """

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        # input_length = inputs["input_ids"].shape[1]

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
                output_hidden_states=True,
            )

            scores = outputs.scores
            hidden_states = outputs.hidden_states

            # print(f"FINAL UPDATED Type: {type(hidden_states)}")
            # print(f"Num steps: {len(hidden_states)}")
            # if len(hidden_states) > 0:
            #     print(f"Type of first step: {type(hidden_states[0])}")
            #     print(f"Num layers: {len(hidden_states[0])}")
            #     if len(hidden_states[0]) > 0:
            #         print(f"First layer, first step shape: {hidden_states[0][0].shape}")
            

            if scores:
                base_dev = scores[0].device
                logits = torch.stack([s.to(base_dev) for s in scores])
                probs = torch.softmax(logits, dim=-1)
                avg_entropy = self.compute_entropy(probs)
                norm_entropy = self.compute_normalized_entropy(probs)
            else:
                avg_entropy = None


        response_ids = outputs.sequences
        input_len = inputs["input_ids"].shape[1]
        new_tokens = response_ids[:, input_len:].cpu()
        response_only = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

        print(f"Generated (temp={self.temperature}): {response_only}")
        if clear_cache:
            gc.collect()
            torch.cuda.empty_cache()

        # ----- Hidden State Entropy -----
        # cov_entropy = self.get_covariance_entropy(hidden_states)
        # pca_entropy = self.get_pca_entropy(hidden_states)
        transition_entropy = self.get_transition_entropy(hidden_states)
        perturbation_entropy_dimwise = self.get_perturbation_entropy_dimwise(hidden_states)

        # ----- Semantic Entropy Runs -----
        semantic_entropy = self.run_semantic_entropy(messages, runs=5)

        return {
            "avg_entropy": avg_entropy,
            "norm_entropy": norm_entropy,
            "semantic_entropy": semantic_entropy,
            # "cov_entropy": cov_entropy,
            # "pca_entropy": pca_entropy,
            "transition_entropy": transition_entropy,
            "perturbation_entropy_dimwise": perturbation_entropy_dimwise,
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
    
    def _extract_hidden_states(self, hidden_states):
        """
        Helper to extract tensors from various hidden_states formats.
        Handles the nested tuple structure from model.generate() with output_hidden_states=True.
        Keeps tensors on their original device (GPU).
        """
        if not hidden_states or len(hidden_states) == 0:
            return []
        
        num_steps = len(hidden_states)
        num_layers = len(hidden_states[0]) if hidden_states[0] else 0
        
        if num_layers == 0:
            return []
        
        # Transpose: collect all steps for each layer
        layer_tensors = []
        for layer_idx in range(num_layers):
            try:
                layer_steps = []
                for step_idx in range(num_steps):
                    step_hidden = hidden_states[step_idx][layer_idx]
                    if isinstance(step_hidden, torch.Tensor):
                        layer_steps.append(step_hidden)
                
                if layer_steps:
                    # Concatenate along sequence dimension - STAYS ON GPU
                    layer_tensor = torch.cat(layer_steps, dim=1)
                    layer_tensors.append(layer_tensor)
            except (IndexError, TypeError) as e:
                print(f"Warning: Could not extract layer {layer_idx}: {e}")
                continue
        
        return layer_tensors

    def get_covariance_entropy(self, hidden_states):
        """
        Estimate differential entropy (Gaussian approx) per-layer.
        """
        hidden_states = self._extract_hidden_states(hidden_states)
        layer_entropies = []
        for h in hidden_states:
            N, L, D = h.shape
            h_flat = h.reshape(N * L, D).float()
            h_centered = h_flat - h_flat.mean(dim=0, keepdim=True)
            cov = (h_centered.T @ h_centered) / max(h_flat.shape[0] - 1, 1)
            cov = cov + torch.eye(D, device=cov.device) * 1e-6
            sign, logdet = torch.slogdet(cov)
            if sign <= 0:
                eigvals = torch.linalg.eigvals(cov).real.clamp_min(1e-12)
                logdet = torch.log(eigvals).sum()
            d = float(D)
            h_entropy = 0.5 * (d * (1.0 + np.log(2.0 * np.pi)) + logdet.item())
            layer_entropies.append(float(h_entropy))
        return layer_entropies

    def get_pca_entropy(self, hidden_states):
        """
        Spectral entropy: Shannon entropy of normalized eigenvalues from covariance matrix.
        """
        hidden_states = self._extract_hidden_states(hidden_states)
        layer_entropies = []
        for h in hidden_states:
            N, L, D = h.shape
            h_flat = h.reshape(N * L, D).float()
            
            # Center the data
            h_centered = h_flat - h_flat.mean(dim=0, keepdim=True)
            
            try:
                # Compute covariance matrix: [D, D]
                n_samples = h_centered.shape[0]
                cov = (h_centered.T @ h_centered) / max(n_samples - 1, 1)
                
                # Add small regularization for numerical stability
                cov = cov + torch.eye(D, device=cov.device) * 1e-8
                
                # Extract eigenvalues (variance explained by each PC)
                eigvals = torch.linalg.eigvalsh(cov)
                eigvals = eigvals.clamp_min(1e-12)
                
                # Normalize to create probability distribution
                p = eigvals / eigvals.sum()
                
                # Compute Shannon entropy (in nats) - KEEP ON GPU
                spec_entropy = -(p * torch.log(p + 1e-12)).sum().item()
                
                layer_entropies.append(float(spec_entropy))
                
            except Exception as e:
                # Fallback: use SVD on centered data directly
                try:
                    _, s, _ = torch.svd(h_centered)
                    eigvals = (s ** 2) / max(h_centered.shape[0] - 1, 1)
                    eigvals = eigvals.clamp_min(1e-12)
                    
                    p = eigvals / eigvals.sum()
                    spec_entropy = -(p * torch.log(p + 1e-12)).sum().item()
                    
                    layer_entropies.append(float(spec_entropy))
                except Exception:
                    layer_entropies.append(float(np.log(D)))
        
        return layer_entropies

    def get_transition_entropy(self, hidden_states):
        """
        Measure entropy of temporal transitions between consecutive timesteps.
        """
        hidden_states = self._extract_hidden_states(hidden_states)
        layer_entropies = []
        for h in hidden_states:
            N, L, D = h.shape
            
            if L < 2:
                layer_entropies.append(0.0)
                continue
            
            # Convert to float32 for numerical operations - STAYS ON GPU
            h = h.float()
            
            # Compute transitions
            transitions = h[:, 1:, :] - h[:, :-1, :]
            transition_norms = torch.norm(transitions, dim=-1, p=2)
            all_transitions = transition_norms.reshape(-1)
            all_transitions = all_transitions[all_transitions > 1e-8]
            
            if len(all_transitions) < 2:
                layer_entropies.append(0.0)
                continue
            
            n_bins = min(50, len(all_transitions) // 10)
            n_bins = max(n_bins, 10)
            
            # ALL GPU OPERATIONS
            all_transitions = all_transitions.float()
            hist = torch.histc(all_transitions, bins=n_bins, 
                            min=all_transitions.min().item(), 
                            max=all_transitions.max().item())
            
            hist = hist + 1e-12
            probs = hist / hist.sum()
            
            # Compute entropy on GPU, only transfer final scalar
            ent = -(probs * torch.log(probs)).sum().item()
            
            layer_entropies.append(float(ent))
        
        return layer_entropies

    def get_perturbation_entropy_dimwise(self, hidden_states, noise_scales=(1e-3, 1e-2, 1e-1), n_samples=10):

        """
        Measure which dimensions are most sensitive to perturbations.
        """
        hidden_states = self._extract_hidden_states(hidden_states)
        layer_entropies = []
        for h in hidden_states:
            N, L, D = h.shape
            h_flat = h.reshape(N * L, D).float()
            
            ent_scales = []
            for scale in noise_scales:
                dim_variances = torch.zeros(D, device=h_flat.device)
                
                for _ in range(n_samples):
                    noise = torch.randn_like(h_flat) * float(scale)
                    h_pert = h_flat + noise
                    
                    dim_changes = (h_pert - h_flat).abs().mean(dim=0)
                    dim_variances += dim_changes
                
                dim_variances = dim_variances / n_samples
                dim_variances = dim_variances.clamp_min(1e-12)
                probs = dim_variances / dim_variances.sum()
                
                ent = -(probs * torch.log(probs + 1e-12)).sum().item()
                ent_scales.append(ent)
            
            layer_entropies.append(float(np.mean(ent_scales)))
        
        return layer_entropies

    def generate_FULL(self, messages, clear_cache=True):
        """
        Original generate function that returns avg_entropy only.
        Device-aware generation for single-device and device_map='auto' sharded models.
        Returns: (avg_entropy, generated_text)
        """

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                temperature=self.temperature,
            )


        response_ids = outputs.sequences
        new_tokens = response_ids[:, input_length:].cpu()
        response_only = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

        if clear_cache:
            gc.collect()
            torch.cuda.empty_cache()

        return response_only
    
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
