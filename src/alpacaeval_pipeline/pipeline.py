import os
import yaml
import multiprocessing as mp
from pathlib import Path

# Import the standalone worker function
from vllm_worker import _vllm_generate_worker

class AlpacaEvalPipeline:
    def __init__(self, output_dir="alpaca_eval_workspace"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.local_judge_dir = self.output_dir / "local_llama_judge"
        
    def setup_local_judge_config(self, base_url="http://localhost:8000/v1", model_name="meta-llama/Llama-3.3-70B-Instruct", max_tokens=256):
        self.local_judge_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.local_judge_dir / "configs.yaml"
        
        config_data = {
            "vllm_llama3_3_70b": {
                "prompt_template": "alpaca_eval_gpt4_turbo/prompt.txt",
                "fn_completions": "openai_chat_completions",
                "completions_kwargs": {
                    "model_name": model_name,
                    "max_tokens": max_tokens, 
                    "temperature": 0.0,
                    "client_kwargs": {
                        "base_url": base_url
                    }
                },
                "fn_completion_parser": "regex_parser",
                "completion_parser_kwargs": {
                    "outputs_to_match": {1: "(?:^|\n) ?1", 2: "(?:^|\n) ?2"}
                }
            }
        }
        
        with open(config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)
        return str(self.local_judge_dir)

    def generate_outputs(self, model_path, generator_name, max_instances=None, apply_chat_template=True, max_tokens=256):
        output_file = self.output_dir / f"{generator_name}_outputs.json"
        
        ctx = mp.get_context("spawn")
        p = ctx.Process(
            target=_vllm_generate_worker,
            args=(model_path, generator_name, str(output_file), max_instances, apply_chat_template, max_tokens)
        )
        
        p.start()
        p.join()
        
        if p.exitcode != 0:
            raise RuntimeError(f"Generation subprocess failed with exit code {p.exitcode}.")
            
        print("Generation complete. VRAM fully reclaimed by the OS.")
        return str(output_file)

    def evaluate(self, model_outputs_path, judge_type="local"):
        import alpaca_eval
        
        print(f"\nStarting evaluation on {model_outputs_path} using [{judge_type.upper()}] judge...")
        
        if judge_type == "local":
            if not (self.local_judge_dir / "configs.yaml").exists():
                self.setup_local_judge_config()
            annotators_config = str(self.local_judge_dir)
            
        elif judge_type == "gpt4":
            if "OPENAI_API_KEY" not in os.environ:
                raise ValueError("OPENAI_API_KEY environment variable is not set.")
            annotators_config = "alpaca_eval_gpt4_turbo_fn"
            
        else:
            raise ValueError("judge_type must be either 'local' or 'gpt4'.")
            
        df_results, _ = alpaca_eval.evaluate(
            model_outputs=model_outputs_path,
            annotators_config=annotators_config,
            is_return_instead_of_print=True
        )
        
        print("\n--- Evaluation Complete ---")
        print(df_results[['win_rate', 'length_controlled_winrate']])
        return df_results
    

"""
from pipeline import AlpacaEvalPipeline
from vllm_server import LocalVllmServer

if __name__ == "__main__":
    pipeline = AlpacaEvalPipeline(output_dir="./eval_runs")

    # 1. Spawn a process to generate outputs 
    qwen_outputs = pipeline.generate_outputs(
        model_path="Qwen/Qwen2.5-9B-Instruct", 
        generator_name="qwen_9b",
        # max_instances=50  # Uncomment for a quick test run
    )

    # 2. Spin up the judge server, run eval, and tear it down automatically
    with LocalVllmServer(model_name="meta-llama/Llama-3.3-70B-Instruct", tp=2):
        pipeline.evaluate(model_outputs_path=qwen_outputs, judge_type="local")
"""