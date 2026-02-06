import requests
import torch
from rouge_score import rouge_scorer


class Evaluator:
    def __init__(self, model, tokenizer, ollama_judge):
        self.tokenizer = tokenizer
        self.model = model
        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        self.ollama_judge = ollama_judge

    def compute_rouge_l(self, generated, reference):
        """Computes lexical similarity (0 to 1)."""
        scores = self.scorer.score(reference, generated)
        return scores["rougeL"].fmeasure

    def compute_perplexity(self, text):
        """Measures model confidence. Lower is better."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
        return torch.exp(outputs.loss).item()

    def llm_judge_score(self, prompt, generated, reference, port: str):
        # assert self.verify_model_loaded(port=port), (
        #     f"Ollama model {self.ollama_judge} did not load correctly"
        # )
        """Uses a local Ollama instance to score instruction following (0 to 1)."""
        eval_prompt = f"""
        [Instruction]
        {prompt}
        [Reference Answer]
        {reference}
        [Model Output]
        {generated}
        
        Task: Rate how well the Model Output follows the Instruction compared to the Reference.
        Score from 0.0 to 1.0 (1.0 being perfect). 
        Output ONLY the numerical float score.
        """
        url = f"http://127.0.0.1:{port}/api/generate"

        payload = {
            "model": self.ollama_judge,
            "prompt": eval_prompt,
            "stream": False,
            "options": {"temperature": 0},
        }

        # 1. Post the request
        response = requests.post(url, json=payload)

        # 2. Raise an error if the HTTP request failed (4xx or 5xx)
        # This will stop your script immediately so you don't get junk data
        response.raise_for_status()

        # 3. Parse the score
        try:
            score_text = response.json().get("response", "0").strip()
            return float(score_text)
        except (ValueError, KeyError) as e:
            # Raise a custom error if the LLM output isn't a valid number
            raise ValueError(f"Judge returned non-numeric output: {score_text}") from e

    def run_full_eval(self, test_data, port="11434"):
        """
        test_data: list of dicts with {'prompt', 'reference', 'generation'}
        Returns: dictionary containing the mean of each metric across all test cases.
        """
        if not test_data:
            return {"rouge_l": 0, "perplexity": 0, "judge_score": 0}

        results = []
        for item in test_data:
            results.append(
                {
                    "rouge_l": self.compute_rouge_l(
                        item["generation"], item["reference"]
                    ),
                    "perplexity": self.compute_perplexity(item["generation"]),
                    "judge_score": self.llm_judge_score(
                        prompt=item["prompt"],
                        generated=item["generation"],
                        reference=item["reference"],
                        port=port,
                    ),
                }
            )

        # Calculate means
        num_samples = len(results)
        means = {
            "rouge_l": sum(r["rouge_l"] for r in results) / num_samples,
            "perplexity": sum(r["perplexity"] for r in results) / num_samples,
            "judge_score": sum(r["judge_score"] for r in results) / num_samples,
        }

        return means

    def verify_model_loaded(self, port):
        """Returns True if the correct model is currently in memory."""
        try:
            response = requests.get(f"http://127.0.0.1:{port}/api/ps")
            if response.status_code == 200:
                running_models = response.json().get("models", [])
                # Check if our model name is in the list of running models
                return any(self.ollama_judge in m["name"] for m in running_models)
        except Exception:
            return False
        return False
