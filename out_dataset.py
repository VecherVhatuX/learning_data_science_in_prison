import json
import random
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer


def load_swebench_dataset(dataset_path: str) -> Dataset:
    return load_dataset(dataset_path)['test']


def load_triplet_data(snippet_folder_path: Path) -> list:
    return list(snippet_folder_path.iterdir())


def create_swebench_dict(swebench_dataset: Dataset) -> dict:
    return {item['instance_id']: item['problem_statement'] for item in swebench_dataset}


def load_snippet_file(snippet_file: Path) -> list:
    try:
        with open(snippet_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"JSON decode error in file {snippet_file}.")
        return []


def separate_snippets(snippets: list) -> (list, list):
    positive_snippets = [
        item['snippet'] for item in snippets
        if item.get('is_bug', False) and item.get('snippet')
    ]
    negative_snippets = [
        item['snippet'] for item in snippets
        if not item.get('is_bug', False) and item.get('snippet')
    ]
    return positive_snippets, negative_snippets


def create_triplets(problem_statement: str, positive_snippets: list, negative_snippets: list, num_negatives_per_positive: int) -> list:
    triplets = []
    for positive_doc in positive_snippets:
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
            triplets.append(triplet)
    return triplets


def create_triplet_dataset(snippet_folder_path: Path, swebench_dataset: Dataset, instance_id_field: str = 'instance_id', num_negatives_per_positive: int = 3) -> list:
    all_dataset = []
    all_test_folders = load_triplet_data(snippet_folder_path)
    swebench_dict = create_swebench_dict(swebench_dataset)
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
        snippets = load_snippet_file(snippet_file)
        positive_snippets, negative_snippets = separate_snippets(snippets)
        if not positive_snippets or not negative_snippets:
            continue
        triplets = create_triplets(problem_statement, positive_snippets, negative_snippets, num_negatives_per_positive)
        all_dataset.extend(triplets)
    return all_dataset


def create_huggingface_dataset(triplet_data: list) -> DatasetDict:
    triplet_dataset = Dataset.from_list(triplet_data)
    split_dataset = triplet_dataset.train_test_split(test_size=0.1, seed=42)
    return DatasetDict({
        'train': split_dataset['train'],
        'test': split_dataset['test']
    })


def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_triplet(examples, max_length=512):
    tokenizer = load_tokenizer('unsloth/Llama-3.2-1B')
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
    tokenized_examples = {
        'anchor_input_ids': anchor_enc['input_ids'],
        'anchor_attention_mask': anchor_enc['attention_mask'],
        'positive_input_ids': positive_enc['input_ids'],
        'positive_attention_mask': positive_enc['attention_mask'],
        'negative_input_ids': negative_enc['input_ids'],
        'negative_attention_mask': negative_enc['attention_mask'],
    }
    return tokenized_examples


def tokenize_dataset(dataset_dict: DatasetDict) -> DatasetDict:
    return dataset_dict.map(
        tokenize_triplet,
        batched=True,
        remove_columns=['anchor', 'positive', 'negative'],
        desc="Tokenizing triplet dataset"
    )


def main():
    swebench_dataset_path = 'datasets/SWE-bench_oracle'
    swebench_dataset = load_swebench_dataset(swebench_dataset_path)
    dataset_path = Path('datasets/10_10_after_fix_pytest')
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
    dataset_dict = create_huggingface_dataset(triplet_data)
    dataset_save_path = 'datasets/triplet_dataset'
    dataset_dict.save_to_disk(dataset_save_path)
    print(f"Dataset saved at: {dataset_save_path}")
    print("Tokenizing dataset...")
    tokenized_dataset = tokenize_dataset(dataset_dict)
    print("Tokenization completed.")
    print("Verifying the structure of the tokenized dataset:")
    print(tokenized_dataset['train'].column_names)
    print(tokenized_dataset['train'][0])
    tokenized_save_path = 'datasets/tokenized_triplet_dataset'
    tokenized_dataset.save_to_disk(tokenized_save_path)
    print(f"Tokenized dataset saved at: {tokenized_save_path}")


if __name__ == "__main__":
    main()