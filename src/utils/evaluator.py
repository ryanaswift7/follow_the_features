import datetime
import re

import requests
import torch
from rouge_score import rouge_scorer


def extract_score(text):
    """
    Extracts the first valid comparative score (-1, 0, or 1) found in 'text'.
    Raises a ValueError if no valid number is found.
    """
    if text is None:
        raise ValueError("Input text is None.")

    matches = re.findall(r"(-?\d+(?:\.\d+)?)", str(text))

    for match in matches:
        try:
            val = float(match)
            if val in [-1.0, 0.0, 1.0]:
                return int(val)
        except ValueError:
            continue

    raise ValueError(
        f"Could not extract a valid score (-1, 0, or 1) from response: {repr(text)}"
    )


class Evaluator:
    # def __init__(self, model, tokenizer, ollama_judge):
    def __init__(self, ollama_judge):
        # self.tokenizer = tokenizer
        # self.model = model
        # self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        self.ollama_judge = ollama_judge

    # def compute_rouge_l(self, generated, reference):
    #     scores = self.scorer.score(reference, generated)
    #     return scores["rougeL"].fmeasure

    # def compute_perplexity(self, text):
    #     inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
    #     with torch.no_grad():
    #         # A causal LM needs at least 2 tokens to calculate next-token loss.
    #         # If it's too short, we return a neutral 0.0 to avoid NaN/Inf errors.
    #         if inputs["input_ids"].size(1) < 2:
    #             return 0.0

    #         outputs = self.model(**inputs, labels=inputs["input_ids"])
    #         loss = outputs.loss

    #     # Catch FP16 overflow that results in NaN or Inf loss
    #     if torch.isnan(loss) or torch.isinf(loss):
    #         print("Warning: Model returned NaN or Inf loss. Defaulting loss to 100.0.")
    #         # loss = 100.0
    #         loss = torch.tensor(100.0, device=self.model.device)

    #     # Clamp the loss to prevent torch.exp() from overflowing into 'inf'
    #     clamped_loss = torch.clamp(loss, max=100.0)

    #     return torch.exp(clamped_loss).item()

    def _query_ollama(self, eval_prompt, port, log_file):
        """Helper to send a single prompt to Ollama, extract the score, and log failures."""
        url = f"http://127.0.0.1:{port}/api/generate"
        payload = {
            "model": self.ollama_judge,
            "prompt": eval_prompt,
            "stream": False,
            "options": {"temperature": 0},
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()

        raw_response = response.json().get("response", "").strip()

        try:
            return extract_score(raw_response)
        except (ValueError, KeyError) as e:
            # Log the exact prompt and invalid output before raising the error up the chain
            with open(log_file, "a", encoding="utf-8") as f:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n[{timestamp}] --- INVALID JUDGE RESPONSE ---\n")
                f.write(f"PROMPT SENT:\n{eval_prompt}\n\n")
                f.write(f"RAW OLLAMA OUTPUT:\n{raw_response}\n")
                f.write(f"ERROR: {e}\n")
                f.write("-" * 50 + "\n")

            raise ValueError(
                f"Judge returned invalid output: {repr(raw_response)}"
            ) from e

    def _pairwise_compare(self, prompt, generated, reference, port, log_file):
        """
        Runs the comparison twice (A vs B, B vs A) to cancel positional bias.
        If either run fails, the ValueError bubbles up and the sample is discarded.
        """
        base_instruction = f"""
        [Instruction]
        {prompt}

        Task: Compare Response A and Response B.
        - Return 1 if Response B is strictly better.
        - Return -1 if Response A is strictly better.
        - Return 0 if equal.
        Output ONLY the integer score (-1, 0, or 1). DO NOT include any explanations or other text.
        
        [BEGIN TASK]
        """

        score_reminder = "\n[END TASK]\nRemember: Output ONLY the integer score (-1, 0, or 1). DO NOT include any explanations or other text."

        # Run 1: Ref=A, Gen=B
        prompt_1 = f"{base_instruction}\n[Response A]\n{reference}\n[Response B]\n{generated}\n{score_reminder}"
        score_1 = self._query_ollama(prompt_1, port, log_file)

        # Run 2: Gen=A, Ref=B (Invert result)
        prompt_2 = f"{base_instruction}\n[Response A]\n{generated}\n[Response B]\n{reference}\n{score_reminder}"
        score_2 = -self._query_ollama(prompt_2, port, log_file)

        return (score_1 + score_2) / 2.0

    def llm_judge_score(self, prompt, generated, reference_judges, port, log_file):
        """
        reference_judges: A LIST of strings (one response from each ref model).
        Returns the average win-rate against all reference models.
        """
        scores = []
        for ref_resp in reference_judges:
            scores.append(
                self._pairwise_compare(prompt, generated, ref_resp, port, log_file)
            )

        # Return average score across all reference models
        return (sum(scores) / len(scores)) + 1.0 if scores else 1.0

    def run_full_eval(self, test_data, log_file, port="11434"):
        if not test_data:
            return {
                "rouge_l": 0.0,
                "perplexity": 0.0,
                "judge_score": 0.0,
                "invalid_judge_responses": 0,
            }

        results = []
        invalid_judge_responses = 0

        for item in test_data:
            try:
                judge_score = self.llm_judge_score(
                    prompt=item["prompt"],
                    generated=item["generation"],
                    reference_judges=item["reference_judges"],
                    port=port,
                    log_file=log_file,
                )
            except ValueError:
                print(
                    f"Warning: Skipping sample due to invalid judge response. Logged to {log_file}"
                )
                invalid_judge_responses += 1
                continue

            results.append(
                {
                    # "rouge_l": self.compute_rouge_l(
                    #     item["generation"], item["reference_metrics"]
                    # ),
                    # "perplexity": self.compute_perplexity(item["reference_metrics"]),
                    "judge_score": judge_score,
                }
            )

        num_samples = len(results)

        # Handle the edge case where every single sample failed
        if num_samples == 0:
            return {
                # "rouge_l": 0.0,
                # "perplexity": 0.0,
                "judge_score": 0.0,
                "invalid_judge_responses": invalid_judge_responses,
            }

        return {
            # "rouge_l": sum(r["rouge_l"] for r in results) / num_samples,
            # "perplexity": sum(r["perplexity"] for r in results) / num_samples,
            "judge_score": sum(r["judge_score"] for r in results) / num_samples,
            "invalid_judge_responses": invalid_judge_responses,
        }

    def verify_model_loaded(self, port):
        try:
            response = requests.get(f"http://127.0.0.1:{port}/api/ps")
            if response.status_code == 200:
                running_models = response.json().get("models", [])
                return any(self.ollama_judge in m["name"] for m in running_models)
        except Exception:
            return False
        return False
