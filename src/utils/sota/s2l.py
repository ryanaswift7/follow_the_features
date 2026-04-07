import torch
import numpy as np
import faiss  
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.preprocessing import StandardScaler
from collections import Counter
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

class S2LConsensusEngine:
    def __init__(
        self, 
        dataset_name, 
        hub_repo_id, 
        proxies, 
        ratio, 
        batch_size, 
        epochs=3, 
        max_length=512,
        debug_max_samples=None
    ):
        """
        Initializes the Consensus Engine following the true S2L methodology.
        """
        self.dataset_name = dataset_name
        self.hub_repo_id = hub_repo_id
        self.proxies = proxies
        self.ratio = ratio
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_length = max_length
        
        self.raw_dataset = load_dataset(dataset_name, split="train")
        
        if debug_max_samples is not None:
            limit = min(debug_max_samples, len(self.raw_dataset))
            self.raw_dataset = self.raw_dataset.select(range(limit))
            print(f"🐛 Debug mode enabled: Dataset truncated to {limit} samples.")
            
        self.all_proxy_indices = {}

    def _get_trajectories(self, model_name):
        """Records authentic training loss trajectories by actively training the proxy."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if tokenizer.pad_token is None: 
            tokenizer.pad_token = tokenizer.eos_token
            
        if tokenizer.chat_template is None:
            raise ValueError(
                f"🚨 Tokenizer for '{model_name}' does not have a chat_template. "
                f"Please use an instruction-tuned proxy model or manually set a template."
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        
        def tokenize(b): 
            texts = [
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                for messages in b["messages"]
            ]
            return tokenizer(texts, truncation=True, padding="max_length", max_length=self.max_length)
        
        ds = self.raw_dataset.map(tokenize, batched=True, remove_columns=self.raw_dataset.column_names)
        ds.set_format("torch")
        
        loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        
        # S2L requires actual training to generate trajectories
        optimizer = AdamW(model.parameters(), lr=5e-5)
        loss_fct = CrossEntropyLoss(reduction='none')
        trajectories = np.zeros((len(ds), self.epochs))

        for e in range(self.epochs):
            curr = 0
            model.train() # Ensure the model is in training mode
            
            for batch in tqdm(loader, desc=f"📉 {model_name} | Epoch {e}"):
                batch = {k: v.to(model.device) for k, v in batch.items()}
                
                # --- Forward Pass ---
                outputs = model(**batch)
                logits = outputs.logits
                
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch["input_ids"][..., 1:].contiguous()
                
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = loss.view(shift_labels.size(0), -1)
                
                pad_mask = (shift_labels != tokenizer.pad_token_id).float()
                per_sample_loss = (loss * pad_mask).sum(dim=1) / pad_mask.sum(dim=1)
                
                # Record trajectory BEFORE the optimizer updates the weights for this batch
                batch_size_actual = shift_labels.size(0)
                trajectories[curr : curr + batch_size_actual, e] = per_sample_loss.detach().cpu().numpy()
                
                # --- Backward Pass (Crucial for S2L) ---
                batch_loss = per_sample_loss.mean()
                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                curr += batch_size_actual
                
        del model
        del optimizer
        torch.cuda.empty_cache()
        
        return trajectories

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
        for p in self.proxies:
            traj = self._get_trajectories(p)
            selected_set = self._select_from_trajectories(traj)
            self.all_proxy_indices[p] = selected_set
            
            config_name = p.split("/")[-1].replace("-", "_")
            self.raw_dataset.select(list(selected_set)).push_to_hub(
                self.hub_repo_id, 
                config_name=config_name
            )

        # --- Consensus Logic ---
        all_votes = []
        for idx_set in self.all_proxy_indices.values():
            all_votes.extend(list(idx_set))
        
        vote_counts = Counter(all_votes)
        target_size = max(1, int(len(self.raw_dataset) * self.ratio))
        
        consensus_indices = [idx for idx, count in vote_counts.most_common(target_size)]
        
        print(f"👑 Consensus reached with {len(self.proxies)} proxies!")
        consensus_ds = self.raw_dataset.select(consensus_indices)
        consensus_ds.push_to_hub(self.hub_repo_id, config_name="s2l_consensus")

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