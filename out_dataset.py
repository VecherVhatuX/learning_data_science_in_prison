import json
import random
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
import torch

# Function to load SWE-bench dataset
def load_swebench_dataset(dataset_path):
    return load_dataset(dataset_path)['test']

# Function to load triplet data
def load_triplet_data(snippet_folder_path):
    return list(snippet_folder_path.iterdir())

# Function to create SWE-bench dictionary
def create_swebench_dict(swebench_dataset):
    return {item['instance_id']: item['problem_statement'] for item in swebench_dataset}

# Function to load snippet file
def load_snippet_file(snippet_file):
    return json.load(open(snippet_file, 'r', encoding='utf-8')) if open(snippet_file, 'r', encoding='utf-8').readable() else []

# Function to separate snippets
def separate_snippets(snippets):
    return ([item['snippet'] for item in snippets if item.get('is_bug', False) and item.get('snippet')], 
            [item['snippet'] for item in snippets if not item.get('is_bug', False) and item.get('snippet')])

# Function to create triplets
def create_triplets(problem_statement, positive_snippets, negative_snippets, num_negatives_per_positive):
    return [{'anchor': problem_statement, 'positive': positive_doc, 'negative': negative_doc} 
            for positive_doc in positive_snippets 
            for negative_doc in (negative_snippets if len(negative_snippets) <= num_negatives_per_positive else random.sample(negative_snippets, num_negatives_per_positive))]

# Function to create triplet dataset
def create_triplet_dataset(snippet_folder_path, swebench_dataset, instance_id_field='instance_id', num_negatives_per_positive=3):
    return [triplet 
            for folder in tqdm(load_triplet_data(snippet_folder_path), desc="Processing folders") 
            for triplet in create_triplets(
                create_swebench_dict(swebench_dataset).get(folder.name), 
                *separate_snippets(load_snippet_file(folder / 'snippet.json')), 
                num_negatives_per_positive) if create_swebench_dict(swebench_dataset).get(folder.name) and load_snippet_file(folder / 'snippet.json')]

# Function to create Huggingface dataset
def create_huggingface_dataset(triplet_data):
    return DatasetDict({
        'train': Dataset.from_list(triplet_data).train_test_split(test_size=0.1, seed=42)['train'],
        'test': Dataset.from_list(triplet_data).train_test_split(test_size=0.1, seed=42)['test']
    })

# Function to load tokenizer
def load_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)

# Function to tokenize triplet
def tokenize_triplet(examples, max_length=512, model_name='unsloth/Llama-3.2-1B'):
    tokenizer = load_tokenizer(model_name)
    return {
        'anchor_input_ids': tokenizer(examples['anchor'], truncation=True, padding='max_length', max_length=max_length)['input_ids'],
        'anchor_attention_mask': tokenizer(examples['anchor'], truncation=True, padding='max_length', max_length=max_length)['attention_mask'],
        'positive_input_ids': tokenizer(examples['positive'], truncation=True, padding='max_length', max_length=max_length)['input_ids'],
        'positive_attention_mask': tokenizer(examples['positive'], truncation=True, padding='max_length', max_length=max_length)['attention_mask'],
        'negative_input_ids': tokenizer(examples['negative'], truncation=True, padding='max_length', max_length=max_length)['input_ids'],
        'negative_attention_mask': tokenizer(examples['negative'], truncation=True, padding='max_length', max_length=max_length)['attention_mask'],
    }

# Function to tokenize dataset
def tokenize_dataset(dataset_dict, model_name='unsloth/Llama-3.2-1B'):
    return dataset_dict.map(
        lambda examples: tokenize_triplet(examples, model_name=model_name),
        batched=True,
        remove_columns=['anchor', 'positive', 'negative'],
        desc="Tokenizing triplet dataset"
    )

# Function to calculate triplet loss
def calculate_triplet_loss(model, batch):
    return ((model(batch['anchor_input_ids'].to(model.device), attention_mask=batch['anchor_attention_mask'].to(model.device)).last_hidden_state[:, 0, :] - 
             model(batch['positive_input_ids'].to(model.device), attention_mask=batch['positive_attention_mask'].to(model.device)).last_hidden_state[:, 0, :]).norm(2, dim=1) - 
            (model(batch['anchor_input_ids'].to(model.device), attention_mask=batch['anchor_attention_mask'].to(model.device)).last_hidden_state[:, 0, :] - 
             model(batch['negative_input_ids'].to(model.device), attention_mask=batch['negative_attention_mask'].to(model.device)).last_hidden_state[:, 0, :]).norm(2, dim=1) + 1
    ).mean()

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
    model = None  # Replace with your model
    trainer = TripletModelTrainer(model, dataset_creator)
    trainer.train()

if __name__ == "__main__":
    main()