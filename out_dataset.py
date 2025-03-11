import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

class TripletDataset(Dataset):
    def __init__(self, triplet_data, max_length):
        self.triplet_data = triplet_data
        self.max_length = max_length
        self.tokenizer = LabelEncoder()
        self.tokenizer.fit(self._gather_texts())
        self.vocab_size = len(self.tokenizer.classes_) + 1

    def _gather_texts(self):
        return [item[key] for item in self.triplet_data for key in ['anchor', 'positive', 'negative']]

    def _convert_to_sequences(self, item):
        return {
            'anchor_seq': self.tokenizer.transform([item['anchor']])[0],
            'positive_seq': self.tokenizer.transform([item['positive']])[0],
            'negative_seq': self.tokenizer.transform([item['negative']])[0]
        }

    def __len__(self):
        return len(self.triplet_data)

    def __getitem__(self, idx):
        item = self.triplet_data[idx]
        return self._convert_to_sequences(item)

    def shuffle_data(self):
        random.shuffle(self.triplet_data)

    def next_epoch(self):
        self.shuffle_data()

def load_json_data(data_path, folder_path):
    with open(data_path, 'r') as file:
        dataset = json.load(file)
    instance_map = {item['instance_id']: item['problem_statement'] for item in dataset}
    snippets = [
        (folder, os.path.join(folder, 'snippet.json'))
        for folder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, folder))
    ]
    return instance_map, snippets

def generate_triplets(instance_map, snippets):
    bug_snippets, non_bug_snippets = zip(*(map(lambda snippet_file: json.load(open(snippet_file)), snippets)))
    bug_snippets = [s['snippet'] for s in bug_snippets if s.get('is_bug') and s['snippet']]
    non_bug_snippets = [s['snippet'] for s in non_bug_snippets if not s.get('is_bug') and s['snippet']]
    return create_triplet_structure(instance_map, snippets, bug_snippets, non_bug_snippets)

def create_triplet_structure(instance_map, snippets, bug_snippets, non_bug_snippets):
    return [
        {
            'anchor': instance_map[os.path.basename(folder)],
            'positive': pos_doc,
            'negative': random.choice(non_bug_snippets)
        }
        for folder, _ in snippets
        for pos_doc in bug_snippets
    ]

class TripletModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(TripletModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 128)
        )

    def forward(self, anchor, positive, negative):
        anchor_out = self.fc(self.embedding(anchor))
        positive_out = self.fc(self.embedding(positive))
        negative_out = self.fc(self.embedding(negative))
        return anchor_out, positive_out, negative_out

def triplet_loss(anchor_embeds, positive_embeds, negative_embeds):
    return torch.mean(torch.clamp(0.2 + torch.norm(anchor_embeds - positive_embeds, dim=1) - 
                                   torch.norm(anchor_embeds - negative_embeds, dim=1), min=0))

def train_model(model, train_loader, test_loader, num_epochs, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    train_losses, test_losses, train_accs = [], [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            anchor_seq = batch['anchor_seq'].to(device)
            positive_seq = batch['positive_seq'].to(device)
            negative_seq = batch['negative_seq'].to(device)

            optimizer.zero_grad()
            anchor_out, positive_out, negative_out = model(anchor_seq, positive_seq, negative_seq)
            loss = triplet_loss(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))
        print(f'Epoch {epoch + 1}, Train Loss: {train_losses[-1]}')

        test_loss, acc = evaluate_model(model, test_loader, device)
        test_losses.append(test_loss)
        test_accs.append(acc)
        print(f'Test Loss: {test_loss}, Test Accuracy: {acc}')

    return train_losses, test_losses, train_accs

def evaluate_model(model, test_loader, device):
    model.eval()
    test_loss, correct_preds = 0, 0

    with torch.no_grad():
        for batch in test_loader:
            anchor_seq = batch['anchor_seq'].to(device)
            positive_seq = batch['positive_seq'].to(device)
            negative_seq = batch['negative_seq'].to(device)

            anchor_out, positive_out, negative_out = model(anchor_seq, positive_seq, negative_seq)
            loss = triplet_loss(anchor_out, positive_out, negative_out)
            test_loss += loss.item()

            pos_similarity = torch.sum(anchor_out * positive_out, dim=1)
            neg_similarity = torch.sum(anchor_out * negative_out, dim=1)
            correct_preds += torch.sum(pos_similarity > neg_similarity).item()

    acc = correct_preds / len(test_loader.dataset)
    return test_loss / len(test_loader), acc

def plot_results(train_losses, test_losses, train_accs, test_accs):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.title('Accuracy Progression')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy Rate')
    plt.legend()
    plt.show()

def main():
    data_path = 'datasets/SWE-bench_oracle.npy'
    snippet_path = 'datasets/10_10_after_fix_pytest'
    
    instance_map, snippets = load_json_data(data_path, snippet_path)
    triplets = generate_triplets(instance_map, snippets)
    train_triplets, valid_triplets = np.array_split(np.array(triplets), 2)
    
    train_loader = DataLoader(TripletDataset(train_triplets.tolist(), max_length=512), batch_size=32, shuffle=True)
    test_loader = DataLoader(TripletDataset(valid_triplets.tolist(), max_length=512), batch_size=32, shuffle=False)
    
    model = TripletModel(vocab_size=train_loader.dataset.vocab_size, embed_dim=128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_losses, test_losses, train_accs = train_model(model, train_loader, test_loader, num_epochs=5, device=device)
    plot_results(train_losses, test_losses, train_accs, test_accs)

if __name__ == "__main__":
    main()