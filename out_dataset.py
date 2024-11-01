import json
import random
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
import torch

def load_swebench_dataset(dataset_path):
    return load_dataset(dataset_path)['test']

def load_triplet_data(snippet_folder_path):
    return list(snippet_folder_path.iterdir())

def load_snippet_file(snippet_file):
    return json.load(open(snippet_file, 'r', encoding='utf-8')) if open(snippet_file, 'r', encoding='utf-8').readable() else []

def separate_snippets(snippets):
    return ([item['snippet'] for item in snippets if item.get('is_bug', False) and item.get('snippet')], 
            [item['snippet'] for item in snippets if not item.get('is_bug', False) and item.get('snippet')])

def create_triplets(problem_statement, positive_snippets, negative_snippets, num_negatives_per_positive):
    return [{'anchor': problem_statement, 'positive': positive_doc, 'negative': negative_doc} 
            for positive_doc in positive_snippets 
            for negative_doc in (negative_snippets if len(negative_snippets) <= num_negatives_per_positive else random.sample(negative_snippets, num_negatives_per_positive))]

def create_swebench_dict(swebench_dataset):
    return {item['instance_id']: item['problem_statement'] for item in swebench_dataset}

def batch_data(data, batch_size=16, shuffle=True):
    if shuffle:
        random.shuffle(data)
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

def create_triplet_dataset(swebench_dataset_path, snippet_folder_path, instance_id_field='instance_id', num_negatives_per_positive=3):
    swebench_dataset = load_swebench_dataset(swebench_dataset_path)
    swebench_dict = create_swebench_dict(swebench_dataset)
    snippet_folder_path = Path(snippet_folder_path)
    
    print("Creating triplet dataset...")
    triplet_data = [triplet 
                    for folder in tqdm(load_triplet_data(snippet_folder_path), desc="Processing folders") 
                    for triplet in create_triplets(
                        swebench_dict.get(folder.name), 
                        *separate_snippets(load_snippet_file(folder / 'snippet.json')), 
                        num_negatives_per_positive) if swebench_dict.get(folder.name) and load_snippet_file(folder / 'snippet.json')]
    print(f"Number of triplets: {len(triplet_data)}")
    return batch_data(triplet_data)

def load_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)

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

def tokenize_dataset(dataset, model_name='unsloth/Llama-3.2-1B'):
    tokenized_data = []
    for batch in dataset:
        tokenized_batch = []
        for example in batch:
            tokenized_batch.append(tokenize_triplet(example, model_name=model_name))
        tokenized_data.append(tokenized_batch)
    return tokenized_data

def calculate_triplet_loss(model, batch):
    return ((model(batch[0]['anchor_input_ids'].to(model.device), attention_mask=batch[0]['anchor_attention_mask'].to(model.device)).last_hidden_state[:, 0, :] - 
             model(batch[0]['positive_input_ids'].to(model.device), attention_mask=batch[0]['positive_attention_mask'].to(model.device)).last_hidden_state[:, 0, :]).norm(2, dim=1) - 
            (model(batch[0]['anchor_input_ids'].to(model.device), attention_mask=batch[0]['anchor_attention_mask'].to(model.device)).last_hidden_state[:, 0, :] - 
             model(batch[0]['negative_input_ids'].to(model.device), attention_mask=batch[0]['negative_attention_mask'].to(model.device)).last_hidden_state[:, 0, :]).norm(2, dim=1) + 1
    ).mean()

def train(model, dataset, batch_size=16, epochs=5):
    tokenized_dataset = tokenize_dataset(dataset)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tokenized_dataset:
            loss = calculate_triplet_loss(model, batch)
            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(tokenized_dataset)}")
        
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for batch in tokenized_dataset:
            loss = calculate_triplet_loss(model, batch)
            total_loss += loss.item()
        print(f"Test Loss: {total_loss / len(tokenized_dataset)}")

def main():
    swebench_dataset_path = 'datasets/SWE-bench_oracle'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'

    dataset = create_triplet_dataset(swebench_dataset_path, snippet_folder_path)
    if not dataset:
        print("No available triplets to create the dataset.")
        return
    tokenized_dataset = tokenize_dataset(dataset)
    print("Tokenization completed.")
    print("Verifying the structure of the tokenized dataset:")
    print(tokenized_dataset)

    # Initialize the model and trainer
    model = None  # Replace with your model
    train(model, dataset)

if __name__ == "__main__":
    main()