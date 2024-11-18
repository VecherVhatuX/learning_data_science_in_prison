import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TripletNetwork(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, margin):
        super(TripletNetwork, self).__init__()
        self.margin = margin
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.dense = nn.Linear(embedding_dim, embedding_dim)
        self.normalize = nn.BatchNorm1d(embedding_dim)

    def forward(self, inputs):
        x = self.embedding(inputs)
        x = x.permute(0, 2, 1)
        x = self.pooling(x).squeeze(2)
        x = self.dense(x)
        x = self.normalize(x)
        x = x / x.norm(dim=-1, keepdim=True)
        return x

def triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, margin):
    anchor_positive_distance = (anchor_embeddings - positive_embeddings).norm(dim=-1)
    anchor_negative_distance = (anchor_embeddings.unsqueeze(1) - negative_embeddings).norm(dim=-1)
    min_anchor_negative_distance = anchor_negative_distance.min(dim=-1).values
    return (anchor_positive_distance - min_anchor_negative_distance + margin).clamp_min(0).mean()

class TripletDataset(Dataset):
    def __init__(self, samples, labels, batch_size, num_negatives):
        self.samples = torch.tensor(samples, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.batch_size = batch_size
        self.num_negatives = num_negatives

    def __len__(self):
        return -(-len(self.samples) // self.batch_size)

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.samples))
        anchor_idx = torch.arange(start_idx, end_idx)
        anchor_labels = self.labels[anchor_idx]

        positive_idx = torch.tensor([np.random.choice(np.where(self.labels.numpy() == label)[0], size=1)[0] for label in anchor_labels.numpy()])
        negative_idx = torch.tensor([np.random.choice(np.where(self.labels.numpy() != label)[0], size=self.num_negatives, replace=False) for label in anchor_labels.numpy()])

        return {
            'anchor_input_ids': self.samples[anchor_idx],
            'positive_input_ids': self.samples[positive_idx],
            'negative_input_ids': self.samples[negative_idx]
        }

class TripletModel:
    def __init__(self, num_embeddings, embedding_dim, margin, lr, device):
        self.network = TripletNetwork(num_embeddings, embedding_dim, margin)
        self.margin = margin
        self.lr = lr
        self.device = device
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def train(self, dataset, epochs):
        self.network.to(self.device)
        data_loader = DataLoader(dataset, batch_size=1)
        for epoch in range(epochs):
            total_loss = 0.0
            for i, data in enumerate(data_loader):
                anchor_inputs = data['anchor_input_ids'].to(self.device)
                positive_inputs = data['positive_input_ids'].to(self.device)
                negative_inputs = data['negative_input_ids'].to(self.device)

                self.optimizer.zero_grad()
                anchor_embeddings = self.network(anchor_inputs)
                positive_embeddings = self.network(positive_inputs)
                negative_embeddings = self.network(negative_inputs)
                loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, self.margin)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f'Epoch: {epoch+1}, Loss: {total_loss/(i+1):.3f}')

    def evaluate(self, dataset):
        self.network.to(self.device)
        self.network.eval()
        total_loss = 0.0
        data_loader = DataLoader(dataset, batch_size=1)
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                anchor_inputs = data['anchor_input_ids'].to(self.device)
                positive_inputs = data['positive_input_ids'].to(self.device)
                negative_inputs = data['negative_input_ids'].to(self.device)

                anchor_embeddings = self.network(anchor_inputs)
                positive_embeddings = self.network(positive_inputs)
                negative_embeddings = self.network(negative_inputs)

                loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, self.margin)
                total_loss += loss.item()
        print(f'Validation Loss: {total_loss / (i+1):.3f}')
        self.network.train()

    def predict(self, input_ids):
        self.network.to(self.device)
        self.network.eval()
        with torch.no_grad():
            return self.network(input_ids.to(self.device))

    def save_model(self, path):
        torch.save(self.network.state_dict(), path)

    def load_model(self, path):
        self.network.load_state_dict(torch.load(path))

def main():
    np.random.seed(42)
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    samples = np.random.randint(0, 100, (100, 10))
    labels = np.random.randint(0, 2, (100,))
    batch_size = 32
    num_negatives = 5
    epochs = 10
    num_embeddings = 101
    embedding_dim = 10
    margin = 1.0
    lr = 1e-4

    dataset = TripletDataset(samples, labels, batch_size, num_negatives)
    model = TripletModel(num_embeddings, embedding_dim, margin, lr, device)
    model.train(dataset, epochs)
    input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)[None, :]
    output = model.predict(input_ids)
    print(output)
    model.save_model("triplet_model.pth")
    model.load_model("triplet_model.pth")
    print("Model saved and loaded successfully.")

if __name__ == "__main__":
    main()