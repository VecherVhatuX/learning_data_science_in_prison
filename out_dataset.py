import json
import random
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer


def load_swebench_dataset(dataset_path: str) -> Dataset:
    """Loads the SWE-bench dataset."""
    return load_dataset(dataset_path)['test']


def load_triplet_data(snippet_folder_path: Path) -> list:
    """Loads the triplet data from the given folder path."""
    return list(snippet_folder_path.iterdir())


def create_swebench_dict(swebench_dataset: Dataset) -> dict:
    """Creates a dictionary from the SWE-bench dataset."""
    return {item['instance_id']: item['problem_statement'] for item in swebench_dataset}


def load_snippet_file(snippet_file: Path) -> list:
    """Loads the snippet file."""
    try:
        with open(snippet_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"JSON decode error in file {snippet_file}.")
        return []


def separate_snippets(snippets: list) -> (list, list):
    """Separates the snippets into positive and negative snippets."""
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
    """Creates triplets from the problem statement, positive snippets, negative snippets, and the number of negatives per positive."""
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
    """Creates the triplet dataset from the snippet folder path and the SWE-bench dataset."""
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
    """Creates the Hugging Face dataset from the triplet data."""
    triplet_dataset = Dataset.from_list(triplet_data)
    split_dataset = triplet_dataset.train_test_split(test_size=0.1, seed=42)
    return DatasetDict({
        'train': split_dataset['train'],
        'test': split_dataset['test']
    })


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Loads the tokenizer from the given model name."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_triplet(examples, max_length=512):
    """Tokenizes the triplet."""
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
    """Tokenizes the dataset."""
    return dataset_dict.map(
        tokenize_triplet,
        batched=True,
        remove_columns=['anchor', 'positive', 'negative'],
        desc="Tokenizing triplet dataset"
    )


def calculate_triplet_loss(model, batch):
    """Calculates the triplet loss."""
    anchor_input_ids = batch['anchor_input_ids'].to(model.device)
    anchor_attention_mask = batch['anchor_attention_mask'].to(model.device)
    positive_input_ids = batch['positive_input_ids'].to(model.device)
    positive_attention_mask = batch['positive_attention_mask'].to(model.device)
    negative_input_ids = batch['negative_input_ids'].to(model.device)
    negative_attention_mask = batch['negative_attention_mask'].to(model.device)

    anchor_outputs = model(anchor_input_ids, attention_mask=anchor_attention_mask)
    positive_outputs = model(positive_input_ids, attention_mask=positive_attention_mask)
    negative_outputs = model(negative_input_ids, attention_mask=negative_attention_mask)

    anchor_embeddings = anchor_outputs.last_hidden_state[:, 0, :]
    positive_embeddings = positive_outputs.last_hidden_state[:, 0, :]
    negative_embeddings = negative_outputs.last_hidden_state[:, 0, :]

    triplet_loss = (
        anchor_embeddings - positive_embeddings
    ).norm(2, dim=1) - (
        anchor_embeddings - negative_embeddings
    ).norm(2, dim=1) + 1
    return triplet_loss.mean()


class DatasetCreator:
    def __init__(self, swebench_dataset_path, snippet_folder_path, instance_id_field='instance_id', num_negatives_per_positive=3):
        self.swebench_dataset_path = swebench_dataset_path
        self.snippet_folder_path = Path(snippet_folder_path)
        self.instance_id_field = instance_id_field
        self.num_negatives_per_positive = num_negatives_per_positive
        self.swebench_dataset = load_swebench_dataset(swebench_dataset_path)

    def create_triplet_dataset(self):
        print("Creating triplet dataset...")
        triplet_data = create_triplet_dataset(
            self.snippet_folder_path,
            self.swebench_dataset,
            self.instance_id_field,
            self.num_negatives_per_positive
        )
        print(f"Number of triplets: {len(triplet_data)}")
        return triplet_data

    def create_huggingface_dataset(self, triplet_data):
        dataset_dict = create_huggingface_dataset(triplet_data)
        return dataset_dict

    def tokenize_dataset(self, dataset_dict):
        tokenized_dataset = tokenize_dataset(dataset_dict)
        return tokenized_dataset

    def save_dataset(self, dataset, path):
        dataset.save_to_disk(path)
        print(f"Dataset saved at: {path}")


class TripletModelTrainer:
    def __init__(self, model, dataset_creator, batch_size=16, epochs=5):
        self.model = model
        self.dataset_creator = dataset_creator
        self.batch_size = batch_size
        self.epochs = epochs

    def train(self):
        triplet_data = self.dataset_creator.create_triplet_dataset()
        dataset_dict = self.dataset_creator.create_huggingface_dataset(triplet_data)
        tokenized_dataset = self.dataset_creator.tokenize_dataset(dataset_dict)

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch in tokenized_dataset['train'].batch(self.batch_size):
                loss = calculate_triplet_loss(self.model, batch)
                loss.backward()
                self.model.optimizer.step()
                self.model.optimizer.zero_grad()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(tokenized_dataset['train'])}")

        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            for batch in tokenized_dataset['test'].batch(self.batch_size):
                loss = calculate_triplet_loss(self.model, batch)
                total_loss += loss.item()
            print(f"Test Loss: {total_loss / len(tokenized_dataset['test'])}")


def main():
    swebench_dataset_path = 'datasets/SWE-bench_oracle'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'

    dataset_creator = DatasetCreator(swebench_dataset_path, snippet_folder_path)
    triplet_data = dataset_creator.create_triplet_dataset()
    if not triplet_data:
        print("No available triplets to create the dataset.")
        return
    dataset_dict = dataset_creator.create_huggingface_dataset(triplet_data)
    dataset_creator.save_dataset(dataset_dict, 'datasets/triplet_dataset')

    print("Tokenizing dataset...")
    tokenized_dataset = dataset_creator.tokenize_dataset(dataset_dict)
    print("Tokenization completed.")
    print("Verifying the structure of the tokenized dataset:")
    print(tokenized_dataset['train'].column_names)
    print(tokenized_dataset['train'][0])
    tokenized_save_path = 'datasets/tokenized_triplet_dataset'
    dataset_creator.save_dataset(tokenized_dataset, tokenized_save_path)

    # Initialize the model and trainer
    model = YourModel()  # Replace with your model
    trainer = TripletModelTrainer(model, dataset_creator)
    trainer.train()


if __name__ == "__main__":
    import torch
    main()