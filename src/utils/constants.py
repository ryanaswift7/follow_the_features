# Evaluation/Experiment file names and keys
EVAL_RESULTS_FILENAME = "eval_results.json"
EVAL_CONFIG_FILENAME = "eval_config.json"
EVAL_RESULTS_KEY = "reference_results"
EVAL_ID_KEY = "eval_id"
PRUNE_ID_KEY = "prune_id"
SFT_ID_KEY = "sft_id"
MODEL_EXISTENCE_FILENAME = "config.json"
MODEL_CONFIG_FILENAME = "sft_model_config.json"
DATASET_CONFIG_FILENAME = "prune_dataset_config.json"
INVALID_JUDGE_LOG_FILENAME = "logs/invalid_judge_logs.txt"

FINAL_LM_EVAL_KEYS = frozenset(
    {"leaderboard", "ifeval", "bbh", "math_hard", "gpqa", "musr", "mmlu_pro"}
)


def has_current_eval_metrics(
    results: dict, run_lm_eval: bool, run_judge_eval: bool
) -> bool:
    if run_lm_eval:
        lm_results = results.get("lm_eval_results")
        if not isinstance(lm_results, dict) or not FINAL_LM_EVAL_KEYS.issubset(lm_results):
            return False
    if run_judge_eval:
        judge_results = results.get("reference_results")
        if not isinstance(judge_results, dict) or "alpacaeval" not in judge_results:
            return False
    return True

# Analysis/Plotting
NAN_RESULT = -1
