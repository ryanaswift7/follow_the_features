import argparse

from datasets import load_dataset


SOURCE_DATASET = "allenai/Dolci-Instruct-SFT-No-Tools"
DEFAULT_REPO_ID = "ryanaswift7/dolci-instruct-sft-no-tools-100k"
DEFAULT_NUM_SAMPLES = 100_000
DEFAULT_SEED = 1234


def create_and_push_subset(
    repo_id: str = DEFAULT_REPO_ID,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    seed: int = DEFAULT_SEED,
    private: bool = False,
) -> None:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    dataset = load_dataset(SOURCE_DATASET, split="train")
    if num_samples > len(dataset):
        raise ValueError(
            f"Requested {num_samples:,} rows, but the source dataset has "
            f"{len(dataset):,} rows"
        )

    subset = dataset.shuffle(seed=seed).select(range(num_samples))
    subset.push_to_hub(
        repo_id,
        split="train",
        private=private,
        commit_message=(
            f"Create {num_samples:,}-example random subset of "
            f"{SOURCE_DATASET} with seed {seed}"
        ),
    )
    print(f"Pushed {len(subset):,} rows to https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create and push a reproducible Dolci dataset subset"
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    create_and_push_subset(
        repo_id=args.repo_id,
        num_samples=args.num_samples,
        seed=args.seed,
        private=args.private,
    )


if __name__ == "__main__":
    main()
