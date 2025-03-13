import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

def collect_all_texts(triplet_data):
    return [text for item in triplet_data for text in (item['anchor'], item['positive'], item['negative'])]

def convert_to_sequences(tokenizer, data_item):
    return {
        'anchor_seq': torch.tensor(tokenizer.transform([data_item['anchor']])[0], dtype=torch.long),
        'positive_seq': torch.tensor(tokenizer.transform([data_item['positive']])[0], dtype=torch.long),
        'negative_seq': torch.tensor(tokenizer.transform([data_item['negative']])[0], dtype=torch.long)
    }

def randomize_data(data_samples):
    random.shuffle(data_samples)
    return data_samples

def create_triplets(instance_dict, bug_samples, non_bug_samples):
    return [
        {
            'anchor': instance_dict[os.path.basename(folder)],
            'positive': bug_sample,
            'negative': random.choice(non_bug_samples)
        }
        for folder, _ in snippet_files
        for bug_sample in bug_samples
    ]

def load_json_file(file_path, root_dir):
    with open(file_path, 'r') as f:
        json_content = json.load(f)
    instance_dict = {entry['instance_id']: entry['problem_statement'] for entry in json_content}
    snippet_files = [
        (folder, os.path.join(root_dir, 'snippet.json'))
        for folder in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, folder))
    ]
    return instance_dict, snippet_files

def prepare_dataset(instance_dict, snippet_files):
    bug_samples, non_bug_samples = zip(*(map(lambda path: json.load(open(path)), snippet_files)))
    bug_samples = [s['snippet'] for s in bug_samples if s.get('is_bug') and s['snippet']]
    non_bug_samples = [s['snippet'] for s in non_bug_samples if not s.get('is_bug') and s['snippet']]
    return create_triplets(instance_dict, bug_samples, non_bug_samples)

class TripletData:
    def __init__(self, triplet_data):
        self.triplet_data = triplet_data
        self.tokenizer = LabelEncoder()
        self.tokenizer.fit(collect_all_texts(triplet_data))

    def get_samples(self):
        return self.triplet_data

class TripletDataset(Dataset):
    def __init__(self, triplet_data):
        self.data = TripletData(triplet_data)

    def __len__(self):
        return len(self.data.get_samples())

    def __getitem__(self, index):
        data_item = self.data.get_samples()[index]
        return convert_to_sequences(self.data.tokenizer, data_item)

class TripletModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(TripletModel, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.dense_network = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 128)
        )

    def forward(self, anchor, positive, negative):
        anchor_embed = self.dense_network(self.embedding_layer(anchor))
        positive_embed = self.dense_network(self.embedding_layer(positive))
        negative_embed = self.dense_network(self.embedding_layer(negative))
        return anchor_embed, positive_embed, negative_embed

def compute_loss(anchor_embeds, positive_embeds, negative_embeds):
    return torch.mean(torch.clamp(0.2 + torch.norm(anchor_embeds - positive_embeds, dim=1) -
                          torch.norm(anchor_embeds - negative_embeds, dim=1), min=0))

def train_network(model, train_loader, valid_loader, epochs):
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    history = []
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            anchor, positive, negative = batch['anchor_seq'], batch['positive_seq'], batch['negative_seq']
            anchor_embeds, positive_embeds, negative_embeds = model(anchor, positive, negative)
            loss = compute_loss(anchor_embeds, positive_embeds, negative_embeds)
            loss.backward()
            optimizer.step()
        
        train_loss = loss.item()
        validation_loss, accuracy = evaluate_network(model, valid_loader)
        history.append((train_loss, validation_loss, accuracy))
    
    return history

def evaluate_network(model, valid_loader):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for batch in valid_loader:
            anchor, positive, negative = batch['anchor_seq'], batch['positive_seq'], batch['negative_seq']
            anchor_embeds, positive_embeds, negative_embeds = model(anchor, positive, negative)
            loss += compute_loss(anchor_embeds, positive_embeds, negative_embeds).item()
            correct += count_matches(anchor_embeds, positive_embeds, negative_embeds)
    accuracy = correct / len(valid_loader.dataset)
    return loss / len(valid_loader), accuracy

def count_matches(anchor_output, positive_output, negative_output):
    positive_similarity = torch.sum(anchor_output * positive_output, dim=1)
    negative_similarity = torch.sum(anchor_output * negative_output, dim=1)
    return torch.sum((positive_similarity > negative_similarity).int()).item()

def visualize_results(history):
    train_losses, val_losses, train_accuracies = zip(*history)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def store_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f'Model saved at {filepath}')

def retrieve_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
    print(f'Model loaded from {filepath}')
    return model

def main():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippets_directory = 'datasets/10_10_after_fix_pytest'
    
    instance_dict, snippet_paths = load_json_file(dataset_path, snippets_directory)
    triplet_data = prepare_dataset(instance_dict, snippet_paths)
    train_data, valid_data = np.array_split(np.array(triplet_data), 2)
    
    train_loader = DataLoader(TripletDataset(train_data.tolist()), batch_size=32, shuffle=True)
    valid_loader = DataLoader(TripletDataset(valid_data.tolist()), batch_size=32, shuffle=False)
    
    model = TripletModel(vocab_size=len(train_loader.dataset.data.tokenizer.classes_) + 1, embedding_dim=128)
    
    history = train_network(model, train_loader, valid_loader, epochs=5)
    visualize_results(history)

    store_model(model, 'triplet_model.pth')

if __name__ == "__main__":
    main()