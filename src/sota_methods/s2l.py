from src.sft import PEFT_CONFIG, generate_sft_trainer_args
from src.schemas.suite import SuiteConfig
from src.prune import get_formatting_function
import torch
import numpy as np
import math
import faiss  
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.preprocessing import StandardScaler
from collections import Counter
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from src.utils.ensure_pad_token import ensure_pad_token
from trl.trainer.sft_trainer import SFTTrainer
from huggingface_hub import HfApi
from datasets import get_dataset_config_names


class TrajectorySFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.num_samples = len(self.train_dataset)
        self.num_epochs = int(self.args.num_train_epochs)
        
        # 2. Initialize trajectory matrix directly on the single training GPU
        self.trajectories = torch.zeros(
            (self.num_samples, self.num_epochs),
            device=self.args.device, 
            dtype=torch.float32
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sample_indices = inputs.pop("sample_idx", None)
        labels = inputs.pop("labels", None)
        
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        token_losses = token_losses.view(shift_logits.size(0), shift_logits.size(1))
        
        mask = (shift_labels != -100).float()
        per_sample_loss = (token_losses * mask).sum(dim=1) / mask.sum(dim=1)
        
        # 3. Update tracking matrix (100% On-Device)
        if sample_indices is not None:
            current_epoch = math.floor(self.state.epoch) 
            
            if current_epoch < self.num_epochs:
                # Direct GPU-to-GPU assignment
                self.trajectories[sample_indices, current_epoch] = per_sample_loss.detach().float()

        loss = per_sample_loss.mean()
        
        return (loss, outputs) if return_outputs else loss
    

class S2LConsensusEngine:
    def __init__(
        self, 
        suite: SuiteConfig,
        dataset_name: str, 
        hub_repo_id: str, 
        proxy_models: list[str], 
        proxy_tokenizers: list[str],
        ratio: float, 
        batch_size: int, 
        split: str,
        epochs: int =3, 
        max_length: int =256,
        sample_limit: int=None
    ):
        """
        Initializes the Consensus Engine following the true S2L methodology.
        """
        self.dataset_name = dataset_name
        self.hub_repo_id = hub_repo_id
        self.proxies = proxy_models
        self.proxy_tokenizers = proxy_tokenizers
        self.ratio = ratio
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_length = max_length
        self.suite = suite
        self.split = split
        
        self.raw_dataset = load_dataset(dataset_name, split=self.split)
        self.raw_dataset = self.raw_dataset.select(range(suite.prune.cfg.num_samples))
        
        if sample_limit is not None:
            limit = min(sample_limit, len(self.raw_dataset))
            self.raw_dataset = self.raw_dataset.select(range(limit))
            print(f"🐛 Debug mode enabled: Dataset truncated to {limit} samples.")
            
        self.all_proxy_indices = {}

    def _tokenize_with_assistant_labels(self, tokenizer, messages):
        """
        Masks everything except the final assistant response.
        Works best when each example is a chat transcript ending with an assistant turn.
        """
        # Full conversation
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Everything up to the assistant response
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )

        # Prompt length before padding
        prompt_ids = tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
            padding=False,
        )["input_ids"]

        # Full sequence with padding
        full = tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
            padding="max_length",
        )

        input_ids = full["input_ids"]
        attention_mask = full["attention_mask"]

        labels = input_ids.copy()
        prompt_len = min(len(prompt_ids), self.max_length)
        labels[:prompt_len] = [-100] * prompt_len

        # Mask padding too
        for i, m in enumerate(attention_mask):
            if m == 0:
                labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


    def _per_sample_loss(self, logits, labels):
        """
        Computes mean loss per sequence, ignoring labels == -100.
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = CrossEntropyLoss(reduction="none", ignore_index=-100)
        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        token_loss = token_loss.view(shift_labels.size(0), -1)

        valid_mask = (shift_labels != -100).float()
        denom = valid_mask.sum(dim=1).clamp(min=1.0)

        return (token_loss * valid_mask).sum(dim=1) / denom

    def _get_trajectories(self, model_name, tokenizer_name):
        """Records authentic training loss trajectories by actively training the proxy."""

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model, tokenizer = ensure_pad_token(model, tokenizer)

        # if tokenizer.pad_token is None:
        #     tokenizer.pad_token = tokenizer.eos_token

        if tokenizer.chat_template is None:
            raise ValueError(
                f"Tokenizer for '{model_name}' does not have a chat_template. "
                f"Please use an instruction-tuned proxy model or manually set a template."
            )

        # def tokenize_batch(batch):
        #     examples = [
        #         self._tokenize_with_assistant_labels(tokenizer, messages)
        #         for messages in batch["messages"]
        #     ]
        #     return {
        #         "input_ids": [ex["input_ids"] for ex in examples],
        #         "attention_mask": [ex["attention_mask"] for ex in examples],
        #         "labels": [ex["labels"] for ex in examples],
        #     }

        # ds = self.raw_dataset.map(tokenize_batch, batched=True, remove_columns=self.raw_dataset.column_names)
        # ds.set_format("torch")
        formatter_func = get_formatting_function(self.dataset_name)

        str_dataset = self.raw_dataset.map(
            lambda examples: formatter_func(tokenizer, examples, is_inference=False),
            batched=True,
            batch_size=1000,  # batch_size,
            num_proc=12,
        )
        ds = str_dataset.map(
            lambda example, idx: {"sample_idx": idx}, 
            with_indices=True
        )
        ds = ds.remove_columns(
            [c for c in ds.column_names if c != "messages"]
        )

        # tokenized_dataset = str_dataset.map(
        #     lambda examples: tokenizer(
        #         examples["text"],
        #         padding="max_length",
        #         truncation=True,
        #         max_length=self.max_length,
        #         return_tensors=None 
        #     ),
        #     batched=True,
        #     batch_size=1000,
        #     num_proc=12,
        #     remove_columns=str_dataset.column_names 
        # )
        # tokenized_dataset.set_format(type="torch", columns=["input_ids"])


        # loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        # optimizer = AdamW(model.parameters(), lr=5e-5)
        # trajectories = np.zeros((len(ds), self.epochs), dtype=np.float32)

        # device = next(model.parameters()).device
        trainer_args = generate_sft_trainer_args(suite=self.suite, output_dir=None)
        trainer_args.remove_unused_columns=False

        trainer_kwargs = {
            "model": model,
            "args": trainer_args,
            "train_dataset": ds,
            "processing_class": tokenizer,
            "peft_config": PEFT_CONFIG,
        }

        trainer = TrajectorySFTTrainer(**trainer_kwargs)

        trainer.train()

        # for e in range(self.epochs):
        #     curr = 0
        #     model.train()

        #     for batch in tqdm(loader, desc=f"📉 {model_name} | Epoch {e}"):
        #         batch = {k: v.to(device) for k, v in batch.items()}

        #         outputs = model(
        #             input_ids=batch["input_ids"],
        #             attention_mask=batch["attention_mask"],
        #         )
        #         logits = outputs.logits

        #         per_sample_loss = self._per_sample_loss(logits, batch["labels"])

        #         batch_size_actual = per_sample_loss.size(0)
        #         trajectories[curr:curr + batch_size_actual, e] = per_sample_loss.detach().cpu().numpy()

        #         batch_loss = per_sample_loss.mean()
        #         batch_loss.backward()
        #         optimizer.step()
        #         optimizer.zero_grad()

        #         curr += batch_size_actual

        del model
        # del optimizer
        torch.cuda.empty_cache()

        return trainer.trajectories.cpu()

    def _select_from_trajectories(self, traj):
        """Authentic S2L logic: Cluster trajectories and sample uniformly from within them."""
        scaled = StandardScaler().fit_transform(traj).astype(np.float32)
        
        # The S2L paper clusters trajectories into a fixed number of groups (e.g., 10)
        n_clusters = 10 
        d = scaled.shape[1] 
        
        kmeans = faiss.Kmeans(d=d, k=n_clusters, niter=20, nredo=10, verbose=False, gpu=True)
        kmeans.train(scaled)
        
        # Get cluster assignments for every sample in the dataset
        _, I = kmeans.index.search(scaled, 1)
        cluster_labels = I.flatten()
        
        target_size = max(1, int(len(traj) * self.ratio))
        samples_per_cluster = target_size // n_clusters
        
        selected_indices = set()
        
        # Sample uniformly from each cluster
        for i in range(n_clusters):
            cluster_idx = np.where(cluster_labels == i)[0]
            if len(cluster_idx) > 0:
                replace = len(cluster_idx) < samples_per_cluster
                sampled = np.random.choice(cluster_idx, min(samples_per_cluster, len(cluster_idx)), replace=replace)
                selected_indices.update(sampled.tolist())
                
        # Pad randomly if integer division left us slightly short
        remaining = target_size - len(selected_indices)
        if remaining > 0:
            available = list(set(range(len(traj))) - selected_indices)
            if available:
                pad_samples = np.random.choice(available, remaining, replace=False)
                selected_indices.update(pad_samples.tolist())
                
        return selected_indices

    def run(self):
        """Executes the multi-proxy S2L selection and pushes consensus to Hub."""
        api = HfApi()
        
        # 1. Check if the repository exists on the Hub
        repo_exists = api.repo_exists(repo_id=self.hub_repo_id, repo_type="dataset")
        existing_configs = []
        
        if repo_exists:
            try:
                # 2. Fetch all existing configurations (subsets) in the repo
                existing_configs = get_dataset_config_names(self.hub_repo_id)
            except Exception:
                # Catch exceptions if the repo was just created but is completely empty
                existing_configs = []

        for p, t in zip(self.proxies, self.proxy_tokenizers):
            config_name = p.split("/")[-1].replace("-", "_")
            
            # 3. Skip if the repo and config already exist
            if repo_exists and config_name in existing_configs:
                print(f"Skipping {p}: Config '{config_name}' already exists in '{self.hub_repo_id}'.")
                
                # NOTE: Because we are skipping the computation, self.all_proxy_indices[p] 
                # won't be populated. If your pipeline relies on this dictionary later, 
                # you will need to load the indices from the Hub here.
                continue
                
            print(f"Processing {p} into config '{config_name}'...")
            traj = self._get_trajectories(p, t)
            selected_set = self._select_from_trajectories(traj)
            self.all_proxy_indices[p] = selected_set
            
            self.raw_dataset.select(list(selected_set)).push_to_hub(
                self.hub_repo_id, 
                config_name=config_name
            )
            
            # 4. Update local state so subsequent loop iterations know the repo/config now exists
            repo_exists = True
            existing_configs.append(config_name)

        # --- Consensus Logic ---
        # all_votes = []
        # for idx_set in self.all_proxy_indices.values():
        #     all_votes.extend(list(idx_set))
        
        # vote_counts = Counter(all_votes)
        # target_size = max(1, int(len(self.raw_dataset) * self.ratio))
        
        # consensus_indices = [idx for idx, count in vote_counts.most_common(target_size)]
        
        # print(f"👑 Consensus reached with {len(self.proxies)} proxies!")
        # consensus_ds = self.raw_dataset.select(consensus_indices)
        # consensus_ds.push_to_hub(self.hub_repo_id, config_name="s2l_consensus")

# # --- Usage ---
# if __name__ == "__main__":
#     engine = S2LConsensusEngine(
#         dataset_name="allenai/tulu-3-sft-mixture", 
#         hub_repo_id="your-hf-username/s2l-tulu-3-sft",
#         proxies=[
#             "Qwen/Qwen1.5-0.5B-Chat", # Swapped to a model with a chat template
#             "HuggingFaceTB/SmolLM-135M-Instruct" # Swapped to a model with a chat template
#         ],
#         ratio=0.1,
#         batch_size=8,
#         epochs=2
#     )
    
#     engine.run()