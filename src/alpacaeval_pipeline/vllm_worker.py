import json
from datasets import load_dataset

def _vllm_generate_worker(model_path, generator_name, output_file, max_instances, apply_chat_template, max_tokens):
    """
    Runs in an isolated process. When this function returns, 
    the process dies and the OS reclaims 100% of the CUDA memory.
    """
    from vllm import LLM, SamplingParams
    
    print(f"[Subprocess] Loading model '{model_path}'...")
    llm = LLM(model=model_path, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.7, max_tokens=max_tokens)
    
    dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval")
    if max_instances:
        dataset = dataset.select(range(max_instances))
        
    instructions = dataset["instruction"]
    tokenizer = llm.get_tokenizer()
    
    if apply_chat_template and hasattr(tokenizer, "apply_chat_template"):
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": inst}], 
                tokenize=False, 
                add_generation_prompt=True
            ) for inst in instructions
        ]
    else:
        prompts = instructions
        
    print(f"[Subprocess] Generating {len(prompts)} responses...")
    outputs = llm.generate(prompts, sampling_params)
    
    results = [
        {
            "instruction": instructions[i],
            "output": out.outputs[0].text.strip(),
            "generator": generator_name
        }
        for i, out in enumerate(outputs)
    ]
        
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Subprocess] Finished saving to {output_file}. Shutting down...")