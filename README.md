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
| `seed` | int | Reproducibility seed. |
| `debug_mode` | bool | Convenience flag for small/debug runs. |
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
| `max_original_samples` | int | Upper bound on source training samples. |
| `tokenizer_id` | string | Hugging Face tokenizer ID. |
| `base_model_id` | string | Hugging Face base model ID for SFT. |
| `full_ft` | bool | `true` for full fine-tuning, otherwise adapter-based training. Default: `false`. |

### `sft.params`

| Field | Type | Notes |
| --- | --- | --- |
| `batch_size` | int | SFT dataloader batch size. |

### `eval.cfg`

| Field | Type | Options / Notes |
| --- | --- | --- |
| `num_ref_models` | int | Number of reference models to evaluate. |
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
| `backend` | string | One of `api`, `vllm`. |
| `model_name` | string or null | Model name passed to the judge backend. |
| `dataset` | string | One of `alpaca_eval`, `alpaca_eval_gpt4_baseline`. |
| `max_instances` | int or null | Optional cap on evaluated instances. |
| `max_new_tokens` | int | Generation length. |
| `judge_max_tokens` | int | Maximum judge response length. |
| `temperature` | float | Sampling temperature. |
| `top_p` | float | Nucleus sampling parameter. |
| `batch_size` | int | AlpacaEval batch size. |
| `local_server_host` | string | Host for local vLLM server. |
| `local_server_port` | int or null | Port for local vLLM server. |
| `model_kwargs` | map | Backend-specific model kwargs. |
| `client_kwargs` | map | Backend-specific client kwargs. |

### `eval.params`

| Field | Type | Notes |
| --- | --- | --- |
| `batch_size` | int | Batch size used by evaluation orchestration. |

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
seed: 3
debug_mode: true
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
    max_original_samples: 2
    full_ft: false
  params:
    batch_size: 2
eval:
  cfg:
    num_ref_models: 2
    lm_eval:
      run: true
      max_new_tokens: 2
      tasks: [leaderboard]
      task_limit: 1
    alpacaeval:
      run: true
      backend: vllm
      model_name: gemma3:27b-it-qat
      dataset: alpaca_eval_gpt4_baseline
      max_instances: 2
      max_new_tokens: 2
      judge_max_tokens: 100
      temperature: 0.0
      top_p: 1.0
      batch_size: 1
      local_server_host: 127.0.0.1
      local_server_port: 8000
      model_kwargs:
        tensor_parallel_size: 1
        gpu_memory_utilization: 0.9
      client_kwargs: {}
  params:
    batch_size: 2
```

## Output Layout

Generated artifacts are written under `output_dir/title/...` and experiment result summaries are stored under `results/<experiment_id>/...` within the suite root.

The pruning stage also writes dataset metadata files so later runs can verify config hashes before reusing existing artifacts.

## Notes

- The project expects Python 3.13 or newer.
- The repo defines CLI scripts in `pyproject.toml`, including `fdp`, `gen-ref-models`, `gen-sni-data`, and `sota-comps`.
