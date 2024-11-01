import json
import random
import os
from multiprocessing import Pool
import numpy as np
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def load_swebench_dataset(dataset_path):
    return np.load(dataset_path, allow_pickle=True)

def load_triplet_data(snippet_folder_path):
    return [os.path.join(snippet_folder_path, f) for f in os.listdir(snippet_folder_path) if os.path.isdir(os.path.join(snippet_folder_path, f))]

def load_snippet_file(snippet_file):
    try:
        with open(snippet_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load snippet file: {snippet_file}, error: {str(e)}")
        return []

def separate_snippets(snippets):
    bug_snippets = [item['snippet'] for item in snippets if item.get('is_bug', False) and item.get('snippet')]
    non_bug_snippets = [item['snippet'] for item in snippets if not item.get('is_bug', False) and item.get('snippet')]
    return bug_snippets, non_bug_snippets

def create_triplets(problem_statement, positive_snippets, negative_snippets, num_negatives_per_positive):
    triplets = []
    for positive_doc in positive_snippets:
        for negative_doc in (negative_snippets if len(negative_snippets) <= num_negatives_per_positive else random.sample(negative_snippets, num_negatives_per_positive)):
            triplets.append({'anchor': problem_statement, 'positive': positive_doc, 'negative': negative_doc})
    return triplets

def create_swebench_dict(swebench_dataset):
    return {item['instance_id']: item['problem_statement'] for item in swebench_dataset}

def batch_data(data, batch_size=16, shuffle=True):
    if shuffle:
        random.shuffle(data)
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

def create_triplet_dataset(swebench_dataset_path, snippet_folder_path, instance_id_field='instance_id', num_negatives_per_positive=3):
    swebench_dataset = load_swebench_dataset(swebench_dataset_path)
    swebench_dict = create_swebench_dict(swebench_dataset)
    snippet_folder_path = snippet_folder_path
    
    print("Creating triplet dataset...")
    triplet_data = []
    for folder in os.listdir(snippet_folder_path):
        folder_path = os.path.join(snippet_folder_path, folder)
        if os.path.isdir(folder_path):
            snippet_file = os.path.join(folder_path, 'snippet.json')
            snippets = load_snippet_file(snippet_file)
            if snippets:
                bug_snippets, non_bug_snippets = separate_snippets(snippets)
                problem_statement = swebench_dict.get(folder)
                if problem_statement:
                    triplets = create_triplets(problem_statement, bug_snippets, non_bug_snippets, num_negatives_per_positive)
                    triplet_data.extend(triplets)
    print(f"Number of triplets: {len(triplet_data)}")
    return batch_data(triplet_data)

def tokenize_data(data, max_length=512):
    tokenizer = word_tokenize
    input_ids = []
    attention_masks = []
    for example in data:
        anchor = tokenizer(example['anchor'])
        positive = tokenizer(example['positive'])
        negative = tokenizer(example['negative'])
        
        anchor_input_id = pad_sequences([anchor], maxlen=max_length, padding='post', truncating='post')[0]
        positive_input_id = pad_sequences([positive], maxlen=max_length, padding='post', truncating='post')[0]
        negative_input_id = pad_sequences([negative], maxlen=max_length, padding='post', truncating='post')[0]
        
        anchor_attention_mask = [1] * len(anchor_input_id)
        positive_attention_mask = [1] * len(positive_input_id)
        negative_attention_mask = [1] * len(negative_input_id)
        
        input_ids.append([anchor_input_id, positive_input_id, negative_input_id])
        attention_masks.append([anchor_attention_mask, positive_attention_mask, negative_attention_mask])
    
    return np.array(input_ids), np.array(attention_masks)

def calculate_triplet_loss(model, batch):
    anchor_input_ids, positive_input_ids, negative_input_ids = batch[0]
    anchor_attention_masks, positive_attention_masks, negative_attention_masks = batch[1]
    
    anchor_embedding = model.predict([anchor_input_ids, anchor_attention_masks])
    positive_embedding = model.predict([positive_input_ids, positive_attention_masks])
    negative_embedding = model.predict([negative_input_ids, negative_attention_masks])
    
    loss = np.mean(np.linalg.norm(anchor_embedding - positive_embedding, axis=1) - np.linalg.norm(anchor_embedding - negative_embedding, axis=1) + 1)
    return loss

def train(model, dataset, batch_size=16, epochs=5):
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataset:
            input_ids, attention_masks = tokenize_data(batch)
            loss = calculate_triplet_loss(model, [input_ids, attention_masks])
            total_loss += loss
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataset)}")

def main():
    swebench_dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'

    dataset = create_triplet_dataset(swebench_dataset_path, snippet_folder_path)
    if not dataset:
        print("No available triplets to create the dataset.")
        return
    
    # Initialize the model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense
    model = Sequential()
    model.add(Embedding(input_dim=10000, output_dim=128, input_length=512))
    model.add(LSTM(64, dropout=0.2))
    model.add(Dense(64, activation='relu'))
    
    train(model, dataset)

if __name__ == "__main__":
    main()