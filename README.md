# Feature Data Pruning

A pruning and retraining pipeline for selecting training examples with SAE activations, then fine-tuning and evaluating the resulting model variants.

The main entrypoint is `fdp` from `src.main:main`.

## What the pipeline does

1. Load a suite config from YAML.
2. Prune a dataset by scoring examples with SAE activations.
3. Fine-tune a base model on the pruned dataset.
4. Run LM eval and AlpacaEval-style evaluation.
5. Aggregate experiment results.

## Quick Start

Install dependencies with `uv`:

```bash
uv sync
```

Run an experiment suite:

```bash
uv run fdp --config configs/debug.yaml
```

Or use a larger config such as:

```bash
uv run fdp --config configs/base_full_gemma3.yaml
```

## SOTA Comparisons

Run the checked-in Ultrachat and Alpaca GRAPE/S2L comparisons:

```bash
bash scripts/sota/main_results_sota.sh
```

Run one comparison directly:

```bash
uv run grape-s2l \
  -c configs/filtering/llama_ultrachat_100_1k.yaml \
  -s configs/sota/llama_ultrachat_100_1k.yaml
```

The main suite config supplies the target model, SFT/evaluation settings, and
comparison budgets from `prune.params.dataset_sizes`. The SOTA config supplies
the dataset, methods, proxies, and output directory. Its main fields are:

```yaml
dataset_id: HuggingFaceH4/ultrachat_200k
split: train_sft
seeds: [6, 7, 67]
output_dir: sota_results/llama_ultrachat_100_1k
grape:
  teacher_ids: []
  max_tokens: 256
  temperature: 0.7
  batch_size: 32
s2l:
  proxy_model_id: meta-llama/Llama-3.2-1B
  proxy_tokenizer_id: meta-llama/Llama-3.2-1B-Instruct
  trajectory_checkpoints: [1000, 2000, 3000]
  n_clusters: 100
```

If `seeds` is omitted, all seeds from the main suite config are used. If it is
provided, every seed must be present in the main config. GRAPE candidate
generation and scoring, S2L proxy training and trajectory inference, final SFT,
and evaluation all run independently for each requested seed.

GRAPE always includes original dataset responses. An empty `teacher_ids` list
runs original-response-only GRAPE. `grape.batch_size` is the response-level
scoring minibatch size. Prompt and response scoring segments are each capped at
the main config's `sft.cfg.max_seq_length`.

SOTA runs are stored in a hash-specific subdirectory of `output_dir`. The run
directory contains copies of both input configs and a `run_manifest.json`.
Completed stages can be reused after a crash. Results are organized as:

```text
<output_dir>/config-<hash>/
  suite_config.yaml
  sota_config.yaml
  run_manifest.json
  seeds/seed_6/
    comparison.json
    datasets/
    models/
  seeds/seed_7/
    ...
  comparison_by_seed.json
  comparison.json
```

`comparison_by_seed.json` and the per-seed `comparison.json` files retain raw
seed results. The top-level `comparison.json` contains averaged numeric metrics
across the requested seeds. Changing either config creates a new hash-specific
run directory instead of invalidating an existing run.

Create and push a reproducible 100,000-example subset of
`allenai/Dolci-Instruct-SFT-No-Tools`:

```bash
hf auth login
uv run gen-dolci-subset
```

The default destination is
`ryanaswift7/dolci-instruct-sft-no-tools-100k`, using seed `1234`. Override
the destination, sample count, or seed as needed:

```bash
uv run gen-dolci-subset \
  --repo-id YOUR_USERNAME/dolci-subset \
  --num-samples 100000 \
  --seed 1234
```

Pass `--private` to create a private destination dataset.

The comparison budgets are taken from `prune.params.dataset_sizes`. GRAPE
selects at most one response per unique instruction and fails if a requested
budget is larger than that instruction count. The SOTA configuration specifies
the dataset, GRAPE teachers, one canonical S2L proxy, trajectory checkpoints,
and the S2L cluster count.

## Repository Layout

- `src/main.py` - suite runner and experiment orchestration
- `src/prune.py` - pruning logic and sample ranking
- `src/sft.py` - supervised fine-tuning
- `src/eval.py` - evaluation entrypoints
- `src/schemas/` - Pydantic config schemas
- `configs/` - example YAML configs

## Main Config Schema

The main pipeline uses a nested YAML config that maps to `src.schemas.suite.SuiteConfig`.

### Top-level fields

| Field | Type | Notes |
| --- | --- | --- |
| `title` | string | Suite name used in output paths. |
| `output_dir` | path/string | Root directory for generated artifacts. |
| `seeds` | list[int] | Reproducibility seeds. Every model is trained and evaluated for each seed, then numeric scores are averaged across seeds. |
| `debug_mode` | bool | Convenience flag for small/debug runs. |
| `full_data_reference` | bool | If `true`, also trains and evaluates a reference model on all `prune.cfg.num_samples` samples. Defaults to `false`. |
| `prune` | object | Pruning config and parameters. |
| `sft` | object | Fine-tuning config and parameters. |
| `eval` | object | Evaluation config and parameters. |
| `timestamp` | string | Auto-generated if omitted. |

### `prune.cfg`

| Field | Type | Options / Notes |
| --- | --- | --- |
| `prune_model_id` | string | Hugging Face model ID used for pruning activations. |
| `sae_release` | string | SAE release name. |
| `sae_id` | string | SAE identifier within the release. |
| `dataset_id` | string | Hugging Face dataset ID. |
| `num_samples` | int | Optional cap on samples used for pruning. |
| `seq_agg_method` | string | One of `sum`, `mean`, `max`, `last_token`. |
| `feat_agg_method` | string | One of `sum`, `mean`. |
| `divide_by_entropy` | bool | Enables entropy normalization in feature scoring. |
| `diversity_param` | float or null | Fraction in `[0, 1]` of the selected top-k features to replace with a uniform sample from the remaining features. Set it to `null` or `0.0` to disable diversity sampling. |

### `prune.params`

| Field | Type | Notes |
| --- | --- | --- |
| `batch_size` | int | Pruning dataloader batch size. |
| `dataset_sizes` | list[int] | Target pruned dataset sizes to generate. |
| `topk_features` | list[int] | Top-k feature counts to evaluate. |

### `sft.cfg`

| Field | Type | Options / Notes |
| --- | --- | --- |
| `max_seq_length` | int | Max token length used during fine-tuning. |
| `learning_rate` | float | SFT learning rate. |
| `lr_scheduler_type` | string | Required learning-rate scheduler. Supported values match Transformers' scheduler types. |
| `tokenizer_id` | string | Hugging Face tokenizer ID. |
| `base_model_id` | string | Hugging Face base model ID for SFT. |
| `full_ft` | bool | Required. `true` for full fine-tuning, `false` for LoRA/PEFT. |

### `sft.params`

| Field | Type | Notes |
| --- | --- | --- |
| `batch_size` | int | SFT dataloader batch size. |

### `eval.cfg`

| Field | Type | Options / Notes |
| --- | --- | --- |
| `seeds` | list[int] | Seeds used for pruning, SFT, reference models, and evaluation. Results for each model are averaged across these seeds. |
| `lm_eval` | object | LM-eval configuration. |
| `alpacaeval` | object | AlpacaEval configuration. |

### `eval.cfg.lm_eval`

| Field | Type | Options / Notes |
| --- | --- | --- |
| `run` | bool | Enables LM eval. |
| `max_new_tokens` | int | Generation length for LM eval tasks. |
| `tasks` | list[string] or null | Optional task list. If null, default tasks are used by the runner. |
| `task_limit` | int or null | Optional cap on examples per task. |

### `eval.cfg.alpacaeval`

| Field | Type | Options / Notes |
| --- | --- | --- |
| `run` | bool | Enables AlpacaEval-style judging. |
| `candidate_backend` | string | `huggingface` or `vllm`; selects the candidate-generation backend. |
| `judge_backend` | string | `api` or `vllm`; selects the judge backend. |
| `model_name` | string or null | Model name passed to the judge backend. |
| `candidate_dtype` | string | Candidate model dtype; defaults to `bfloat16`. |
| `judge_dtype` | string | Local judge dtype; defaults to `auto` so quantized checkpoints use their native format. |
| `dataset` | string | One of `alpaca_eval`, `alpaca_eval_gpt4_baseline`. |
| `max_instances` | int or null | Optional cap on evaluated instances. |
| `max_new_tokens` | int | Generation length. |
| `judge_max_tokens` | int | Maximum judge response length; defaults to `512`. |
| `judge_enable_thinking` | bool | Explicitly enables or disables chat-template reasoning for a local vLLM judge; defaults to `false`. |
| `temperature` | float | Sampling temperature. |
| `top_p` | float | Nucleus sampling parameter. |
| `model_kwargs` | map | Compatibility alias for local judge model kwargs. |
| `candidate_model_kwargs` | map | Keyword arguments passed to the local candidate model. |
| `judge_model_kwargs` | map | Keyword arguments passed to the local vLLM judge. |

Local vLLM mode uses an in-process decoder and does not start an OpenAI-compatible HTTP server. Set `candidate_backend: vllm` and `judge_backend: vllm` for fully local evaluation, or set only `candidate_backend: vllm` with `judge_backend: api` to use a local candidate and an API judge. vLLM tensor parallelism is set automatically to the number of CUDA GPUs visible to the process, including the effect of `CUDA_VISIBLE_DEVICES`; any configured `tensor_parallel_size` is overridden. Local judge thinking is explicitly disabled by default through the tokenizer's `enable_thinking=False` setting and a zero thinking-token budget.

### `eval.params`

| Field | Type | Notes |
| --- | --- | --- |
| `batch_size` | int | Batch size used by LM eval; AlpacaEval judge annotations run with batch size 1 to match the single-example prompt template. |

## Supported `seq_agg_method` Values

The pruning scorer supports these sequence aggregation methods:

- `sum` - sum activations across the sequence dimension
- `mean` - average activations across the sequence dimension
- `max` - take the maximum activation over sequence positions
- `last_token` - use only the activation at the final sequence position

## Example Config

```yaml
title: debug
output_dir: debugging
seeds: [3, 4]
debug_mode: true
full_data_reference: false
prune:
  cfg:
    prune_model_id: google/gemma-3-270m-it
    sae_release: gemma-scope-2-270m-it-res
    sae_id: layer_9_width_65k_l0_medium
    dataset_id: yahma/alpaca-cleaned
    num_samples: 2
    seq_agg_method: last_token
    feat_agg_method: sum
    divide_by_entropy: true
    diversity_param: 0.2
  params:
    batch_size: 2
    dataset_sizes: [2]
    topk_features: [1]
sft:
  cfg:
    tokenizer_id: google/gemma-3-270m-it
    base_model_id: google/gemma-3-1b-pt
    max_seq_length: 10
    learning_rate: 0.0002
    lr_scheduler_type: cosine
    full_ft: false
  params:
    batch_size: 2
eval:
  cfg:
    lm_eval:
      run: true
      max_new_tokens: 2
      tasks: [leaderboard]
      task_limit: 1
    alpacaeval:
      run: true
       model_name: Qwen/Qwen3.8-27B-FP8
       candidate_backend: vllm
       judge_backend: vllm
       candidate_dtype: bfloat16
       judge_dtype: auto
      dataset: alpaca_eval_gpt4_baseline
      max_instances: 2
      max_new_tokens: 2
      judge_max_tokens: 512
      temperature: 0.0
      top_p: 1.0
      candidate_model_kwargs:
        gpu_memory_utilization: 0.9
      judge_model_kwargs:
        gpu_memory_utilization: 0.9
  params:
    batch_size: 2
```

## Output Layout

Generated models and datasets are written under `output_dir/models/...` and `output_dir/datasets/...`. Experiment results are written under `output_dir/results/title/<experiment_id>/`, including the aggregate `manifest.json`, tables, and plots.

The pruning stage also writes dataset metadata files so later runs can verify config hashes before reusing existing artifacts.

## Notes

- The project expects Python 3.13 or newer.
- The repo defines CLI scripts in `pyproject.toml`, including `fdp`, `gen-ref-models`, `gen-sni-data`, and `sota-comps`.
