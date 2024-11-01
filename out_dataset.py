import json
import random
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
import torch

class Dataset:
    def __init__(self, data, batch_size=16, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batched_data = self._batch_data()

    def _batch_data(self):
        if self.shuffle:
            random.shuffle(self.data)
        return [self.data[i:i + self.batch_size] for i in range(0, len(self.data), self.batch_size)]

    def __iter__(self):
        for batch in self.batched_data:
            yield batch

    def __len__(self):
        return len(self.batched_data)

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.data)
        self.batched_data = self._batch_data()

class DatasetCreator:
    def __init__(self, swebench_dataset_path, snippet_folder_path, instance_id_field='instance_id', num_negatives_per_positive=3):
        self.swebench_dataset_path = swebench_dataset_path
        self.snippet_folder_path = Path(snippet_folder_path)
        self.instance_id_field = instance_id_field
        self.num_negatives_per_positive = num_negatives_per_positive
        self.swebench_dataset = load_swebench_dataset(swebench_dataset_path)

    def load_swebench_dataset(self, dataset_path):
        return load_dataset(dataset_path)['test']

    def load_triplet_data(self, snippet_folder_path):
        return list(snippet_folder_path.iterdir())

    def create_swebench_dict(self, swebench_dataset):
        return {item['instance_id']: item['problem_statement'] for item in swebench_dataset}

    def load_snippet_file(self, snippet_file):
        return json.load(open(snippet_file, 'r', encoding='utf-8')) if open(snippet_file, 'r', encoding='utf-8').readable() else []

    def separate_snippets(self, snippets):
        return ([item['snippet'] for item in snippets if item.get('is_bug', False) and item.get('snippet')], 
                [item['snippet'] for item in snippets if not item.get('is_bug', False) and item.get('snippet')])

    def create_triplets(self, problem_statement, positive_snippets, negative_snippets, num_negatives_per_positive):
        return [{'anchor': problem_statement, 'positive': positive_doc, 'negative': negative_doc} 
                for positive_doc in positive_snippets 
                for negative_doc in (negative_snippets if len(negative_snippets) <= num_negatives_per_positive else random.sample(negative_snippets, num_negatives_per_positive))]

    def create_triplet_dataset(self):
        print("Creating triplet dataset...")
        triplet_data = [triplet 
                        for folder in tqdm(self.load_triplet_data(self.snippet_folder_path), desc="Processing folders") 
                        for triplet in self.create_triplets(
                            self.create_swebench_dict(self.swebench_dataset).get(folder.name), 
                            *self.separate_snippets(self.load_snippet_file(folder / 'snippet.json')), 
                            self.num_negatives_per_positive) if self.create_swebench_dict(self.swebench_dataset).get(folder.name) and self.load_snippet_file(folder / 'snippet.json')]
        print(f"Number of triplets: {len(triplet_data)}")
        return Dataset(triplet_data)

    def load_tokenizer(self, model_name):
        return AutoTokenizer.from_pretrained(model_name)

    def tokenize_triplet(self, examples, max_length=512, model_name='unsloth/Llama-3.2-1B'):
        tokenizer = self.load_tokenizer(model_name)
        return {
            'anchor_input_ids': tokenizer(examples['anchor'], truncation=True, padding='max_length', max_length=max_length)['input_ids'],
            'anchor_attention_mask': tokenizer(examples['anchor'], truncation=True, padding='max_length', max_length=max_length)['attention_mask'],
            'positive_input_ids': tokenizer(examples['positive'], truncation=True, padding='max_length', max_length=max_length)['input_ids'],
            'positive_attention_mask': tokenizer(examples['positive'], truncation=True, padding='max_length', max_length=max_length)['attention_mask'],
            'negative_input_ids': tokenizer(examples['negative'], truncation=True, padding='max_length', max_length=max_length)['input_ids'],
            'negative_attention_mask': tokenizer(examples['negative'], truncation=True, padding='max_length', max_length=max_length)['attention_mask'],
        }

    def tokenize_dataset(self, dataset, model_name='unsloth/Llama-3.2-1B'):
        tokenized_data = []
        for batch in dataset:
            tokenized_batch = []
            for example in batch:
                tokenized_batch.append(self.tokenize_triplet(example, model_name=model_name))
            tokenized_data.append(tokenized_batch)
        return Dataset(tokenized_data)

class TripletModelTrainer:
    def __init__(self, model, dataset_creator, batch_size=16, epochs=5):
        self.model = model
        self.dataset_creator = dataset_creator
        self.batch_size = batch_size
        self.epochs = epochs

    def calculate_triplet_loss(self, model, batch):
        return ((model(batch[0]['anchor_input_ids'].to(model.device), attention_mask=batch[0]['anchor_attention_mask'].to(model.device)).last_hidden_state[:, 0, :] - 
                 model(batch[0]['positive_input_ids'].to(model.device), attention_mask=batch[0]['positive_attention_mask'].to(model.device)).last_hidden_state[:, 0, :]).norm(2, dim=1) - 
                (model(batch[0]['anchor_input_ids'].to(model.device), attention_mask=batch[0]['anchor_attention_mask'].to(model.device)).last_hidden_state[:, 0, :] - 
                 model(batch[0]['negative_input_ids'].to(model.device), attention_mask=batch[0]['negative_attention_mask'].to(model.device)).last_hidden_state[:, 0, :]).norm(2, dim=1) + 1
        ).mean()

    def train(self):
        dataset = self.dataset_creator.create_triplet_dataset()
        tokenized_dataset = self.dataset_creator.tokenize_dataset(dataset)

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch in tokenized_dataset:
                loss = self.calculate_triplet_loss(self.model, batch)
                loss.backward()
                self.model.optimizer.step()
                self.model.optimizer.zero_grad()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(tokenized_dataset)}")
            dataset.on_epoch_end()
            tokenized_dataset = self.dataset_creator.tokenize_dataset(dataset)

        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            for batch in tokenized_dataset:
                loss = self.calculate_triplet_loss(self.model, batch)
                total_loss += loss.item()
            print(f"Test Loss: {total_loss / len(tokenized_dataset)}")

def main():
    swebench_dataset_path = 'datasets/SWE-bench_oracle'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'

    dataset_creator = DatasetCreator(swebench_dataset_path, snippet_folder_path)
    dataset = dataset_creator.create_triplet_dataset()
    if not dataset:
        print("No available triplets to create the dataset.")
        return
    tokenized_dataset = dataset_creator.tokenize_dataset(dataset)
    print("Tokenization completed.")
    print("Verifying the structure of the tokenized dataset:")
    print(tokenized_dataset)

    # Initialize the model and trainer
    model = None  # Replace with your model
    trainer = TripletModelTrainer(model, dataset_creator)
    trainer.train()

if __name__ == "__main__":
    main()