import random

from datasets import Dataset, DatasetDict, load_dataset


def create_and_push_sni_samples(
    repo_id=None, push_to_hub=False, num_versions=4, base_seed=3, sample_size=10000
):
    if push_to_hub and not repo_id:
        raise ValueError("You must provide a 'repo_id' to push to the Hub.")

    print("Loading the base dataset from Hugging Face in streaming mode...")
    base_dataset = load_dataset(
        "Muennighoff/natural-instructions", split="train", streaming=True
    )

    sampled_versions = {}

    for i in range(1, num_versions + 1):
        version_name = f"v{i}"
        current_seed = base_seed + i

        random.seed(current_seed)

        # Cap the max skip to a safe number (e.g., 500,000) so we never jump past the end.
        # This prevents the stream from returning 0 items and causing the `{}` mismatch.
        safe_max_skip = 500000
        skip_amount = random.randint(0, safe_max_skip)

        print(f"\nCreating {version_name} with seed {current_seed}...")
        print(f"  -> Skipping {skip_amount:,} rows...")

        # Skip -> Filter for word count -> Shuffle the valid buffer
        # Note: The correct column name for this dataset is "targets" (plural)
        shuffled_iterable = (
            base_dataset.skip(skip_amount)
            .filter(
                lambda x: (
                    x.get("targets") is not None
                    and 10 <= len(x["targets"].split()) <= 1000
                )
            )
            .shuffle(seed=current_seed, buffer_size=10000)
        )

        sampled_iterable = shuffled_iterable.take(sample_size)

        # Realize the stream into a standard Python list first
        sampled_list = list(sampled_iterable)

        # Safety check to ensure we actually grabbed data
        if len(sampled_list) == 0:
            raise RuntimeError(
                f"Stream exhausted for {version_name}! The skip_amount ({skip_amount}) "
                "jumped past the end of the dataset or the filter was too strict."
            )

        print(f"  -> Fetching {len(sampled_list):,} filtered samples into memory...")
        sampled_dataset = Dataset.from_list(sampled_list)

        sampled_versions[version_name] = sampled_dataset

    final_dataset_dict = DatasetDict(sampled_versions)
    print(
        "\nDataset Dictionary successfully created with splits:",
        list(final_dataset_dict.keys()),
    )

    if push_to_hub:
        print(f"\nPushing the DatasetDict to the Hub at '{repo_id}'...")
        final_dataset_dict.push_to_hub(repo_id)
        print("Successfully pushed to the Hub!")

    return final_dataset_dict


def main():
    """Entry point for the command line script."""
    MY_REPO_ID = "ryanaswift7/natural-instructions-10k"

    _ = create_and_push_sni_samples(
        repo_id=MY_REPO_ID, push_to_hub=True, num_versions=4
    )


if __name__ == "__main__":
    main()
