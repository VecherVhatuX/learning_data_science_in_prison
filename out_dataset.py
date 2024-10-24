import json
import random
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer


def create_triplet_dataset(
    snippet_folder_path: Path,
    swebench_dataset,
    instance_id_field: str = 'instance_id',
    num_negatives_per_positive: int = 3
):
    """
    Creates a triplet dataset (anchor, positive, negative).

    Args:
        snippet_folder_path (Path): Path to the folder containing commit data.
        swebench_dataset (Dataset): SWE-bench dataset containing 'instance_id' and 'problem_statement' fields.
        instance_id_field (str): Name of the field for matching instances.
        num_negatives_per_positive (int): Number of negative examples per positive example.

    Returns:
        List[Dict]: List of triplets.
    """
    all_dataset = []
    all_test_folders = list(snippet_folder_path.iterdir())

    # Create an index instance_id -> problem_statement for faster lookup
    # Assumes 'instance_id' is unique in swebench_dataset
    swebench_dict = {item['instance_id']: item['problem_statement'] for item in swebench_dataset}

    for folder in tqdm(all_test_folders, desc="Processing folders"):
        instance_id = folder.name
        problem_statement = swebench_dict.get(instance_id)
        if not problem_statement:
            print(f"Instance ID {instance_id} not found in SWE-bench dataset.")
            continue

        snippet_file = folder / 'snippet.json'
        if not snippet_file.exists():
            print(f"File {snippet_file} does not exist.")
            continue

        with open(snippet_file, 'r', encoding='utf-8') as f:
            try:
                snippets = json.load(f)
            except json.JSONDecodeError:
                print(f"JSON decode error in file {snippet_file}.")
                continue

        # Separate positive and negative examples
        positive_snippets = [
            item['snippet'] for item in snippets
            if item.get('is_bug', False) and item.get('snippet')
        ]
        negative_snippets = [
            item['snippet'] for item in snippets
            if not item.get('is_bug', False) and item.get('snippet')
        ]

        if not positive_snippets or not negative_snippets:
            # Require at least one positive and one negative example
            continue

        # For each positive example, select several negative examples
        for positive_doc in positive_snippets:
            # If fewer negatives than required, use all
            if len(negative_snippets) <= num_negatives_per_positive:
                selected_negatives = negative_snippets
            else:
                selected_negatives = random.sample(negative_snippets, num_negatives_per_positive)

            for negative_doc in selected_negatives:
                triplet = {
                    "anchor": problem_statement,
                    "positive": positive_doc,
                    "negative": negative_doc
                }
                all_dataset.append(triplet)

    return all_dataset


def main():
    swebench_dataset_path = 'datasets/SWE-bench_oracle'
    swebench_dataset = load_dataset(swebench_dataset_path)['test']

    dataset_path = Path('datasets/10_10_after_fix_pytest')

    # Step 3: Create the triplet list
    print("Creating triplet dataset...")
    triplet_data = create_triplet_dataset(
        dataset_path,
        swebench_dataset,
        instance_id_field='instance_id',
        num_negatives_per_positive=3
    )
    print(f"Number of triplets: {len(triplet_data)}")

    if not triplet_data:
        print("No available triplets to create the dataset.")
        return

    # Step 4: Create a HuggingFace dataset
    triplet_dataset = Dataset.from_list(triplet_data)

    # Step 5: Split into train and test
    split_dataset = triplet_dataset.train_test_split(test_size=0.1, seed=42)
    dataset_dict = DatasetDict({
        'train': split_dataset['train'],
        'test': split_dataset['test']
    })

    # Step 6: Save the dataset (optional)
    dataset_save_path = 'datasets/triplet_dataset'
    dataset_dict.save_to_disk(dataset_save_path)
    print(f"Dataset saved at: {dataset_save_path}")

    # Step 7: Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('unsloth/Llama-3.2-1B')

    # Ensure the tokenizer has a pad token
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 8: Define the tokenization function
    def tokenize_triplet(examples, max_length=512):
        """
        Tokenizes triplets separately for anchor, positive, and negative.

        Args:
            examples (Dict): Dictionary with 'anchor', 'positive', 'negative' fields.
            max_length (int): Maximum sequence length.

        Returns:
            Dict: Dictionary with tokenized fields.
        """
        # Tokenize each part separately
        anchor_enc = tokenizer(
            examples['anchor'],
            truncation=True,
            padding='max_length',
            max_length=max_length
        )
        positive_enc = tokenizer(
            examples['positive'],
            truncation=True,
            padding='max_length',
            max_length=max_length
        )
        negative_enc = tokenizer(
            examples['negative'],
            truncation=True,
            padding='max_length',
            max_length=max_length
        )

        # Create separate keys for each component
        tokenized_examples = {
            'anchor_input_ids': anchor_enc['input_ids'],
            'anchor_attention_mask': anchor_enc['attention_mask'],
            'positive_input_ids': positive_enc['input_ids'],
            'positive_attention_mask': positive_enc['attention_mask'],
            'negative_input_ids': negative_enc['input_ids'],
            'negative_attention_mask': negative_enc['attention_mask'],
        }
        return tokenized_examples

    # Step 9: Apply tokenization to the dataset
    print("Tokenizing dataset...")
    tokenized_dataset = dataset_dict.map(
        tokenize_triplet,
        batched=True,
        remove_columns=['anchor', 'positive', 'negative'],
        desc="Tokenizing triplet dataset"
    )
    print("Tokenization completed.")

    # Step 10: Verify the structure of the tokenized dataset
    print("Verifying the structure of the tokenized dataset:")
    print(tokenized_dataset['train'].column_names)
    print(tokenized_dataset['train'][0])

    # Step 11: Save the tokenized dataset (optional)
    tokenized_save_path = 'datasets/tokenized_triplet_dataset'
    tokenized_dataset.save_to_disk(tokenized_save_path)
    print(f"Tokenized dataset saved at: {tokenized_save_path}")


if __name__ == "__main__":
    main()
