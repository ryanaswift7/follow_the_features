"""AlpacaEval entrypoint with the project's explicit-thinking vLLM decoder."""

from src.utils.alpaca_eval_vllm import vllm_local_completions


def _install_decoder() -> None:
    from alpaca_eval import decoders
    import alpaca_eval.main as alpaca_eval_main
    from alpaca_eval.decoders import vllm_local

    original_get_fn_completions = decoders.get_fn_completions

    # Use AlpacaEval's built-in decoder name in generated configs, while
    # retaining the project's chat-template and thinking controls.
    vllm_local.vllm_local_completions = vllm_local_completions

    def get_fn_completions(name):
        return original_get_fn_completions(name)

    decoders.get_fn_completions = get_fn_completions
    # Keep the reference used by AlpacaEval's main module in sync as well.
    alpaca_eval_main.decoders.get_fn_completions = get_fn_completions


_install_decoder()


if __name__ == "__main__":
    from alpaca_eval.main import main

    main()
