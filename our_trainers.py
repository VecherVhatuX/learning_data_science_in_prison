import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TripletDataset(Dataset):
    def __init__(self, samples, labels, batch_size, num_negatives):
        self.samples = samples
        self.labels = labels
        self.batch_size = batch_size
        self.num_negatives = num_negatives

    def __len__(self):
        return -(-len(self.samples) // self.batch_size)

    def __getitem__(self, idx):
        anchor_idx = np.arange(idx * self.batch_size, min((idx + 1) * self.batch_size, len(self.samples)))
        anchor_labels = self.labels[anchor_idx]
        positive_idx = np.array([np.random.choice(np.where(self.labels == label)[0], size=1)[0] for label in anchor_labels])
        negative_idx = np.array([np.random.choice(np.where(self.labels != label)[0], size=self.num_negatives, replace=False) for label in anchor_labels])
        return {
            'anchor_input_ids': torch.tensor(self.samples[anchor_idx], dtype=torch.long),
            'positive_input_ids': torch.tensor(self.samples[positive_idx], dtype=torch.long),
            'negative_input_ids': torch.tensor(self.samples[negative_idx], dtype=torch.long)
        }

    def get_samples(self):
        return self.samples

    def get_labels(self):
        return self.labels

    def get_batch_size(self):
        return self.batch_size

    def get_num_negatives(self):
        return self.num_negatives

    def get_info(self):
        print("Dataset info:")
        print(f"  Samples: {self.samples.shape}")
        print(f"  Labels: {self.labels.shape}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Number of negatives: {self.num_negatives}")


class TripletNetwork(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(TripletNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Embedding(num_embeddings, embedding_dim),
            nn.LazyLinear(embedding_dim),
            nn.Lambda(lambda x: x / torch.norm(x, dim=-1, keepdim=True))
        )

    def forward(self, inputs):
        return self.model(inputs)

    def get_embedding_dim(self):
        return self.model[1].out_features

    def get_model_summary(self):
        print("Model summary:")
        print(self.model)


class TripletModel:
    def __init__(self, num_embeddings, embedding_dim, margin, learning_rate, device):
        self.device = device
        self.model = TripletNetwork(num_embeddings, embedding_dim).to(self.device)
        self.loss_fn = nn.MarginRankingLoss(margin=margin, reduction='mean')
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, dataloader, epochs):
        for epoch in range(epochs):
            total_loss = 0.0
            self.model.train()
            for i, data in enumerate(dataloader):
                anchor_inputs = data['anchor_input_ids'].to(self.device)
                positive_inputs = data['positive_input_ids'].to(self.device)
                negative_inputs = data['negative_input_ids'].to(self.device)
                anchor_embeddings = self.model(anchor_inputs)
                positive_embeddings = self.model(positive_inputs)
                negative_embeddings = self.model(negative_inputs)
                anchor_positive_distance = torch.norm(anchor_embeddings - positive_embeddings, dim=-1)
                anchor_negative_distance = torch.norm(anchor_embeddings[:, None] - negative_embeddings, dim=-1)
                min_anchor_negative_distance = torch.min(anchor_negative_distance, dim=-1)[0]
                loss = self.loss_fn(min_anchor_negative_distance, anchor_positive_distance, torch.ones_like(min_anchor_negative_distance))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f'Epoch: {epoch+1}, Loss: {total_loss/(i+1):.3f}')

    def evaluate(self, dataloader):
        total_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                anchor_inputs = data['anchor_input_ids'].to(self.device)
                positive_inputs = data['positive_input_ids'].to(self.device)
                negative_inputs = data['negative_input_ids'].to(self.device)
                anchor_embeddings = self.model(anchor_inputs)
                positive_embeddings = self.model(positive_inputs)
                negative_embeddings = self.model(negative_inputs)
                anchor_positive_distance = torch.norm(anchor_embeddings - positive_embeddings, dim=-1)
                anchor_negative_distance = torch.norm(anchor_embeddings[:, None] - negative_embeddings, dim=-1)
                min_anchor_negative_distance = torch.min(anchor_negative_distance, dim=-1)[0]
                loss = self.loss_fn(min_anchor_negative_distance, anchor_positive_distance, torch.ones_like(min_anchor_negative_distance))
                total_loss += loss.item()
        print(f'Validation Loss: {total_loss / (i+1):.3f}')

    def predict(self, input_ids):
        self.model.eval()
        with torch.no_grad():
            return self.model(input_ids.to(self.device))

    def get_device(self):
        return self.device

    def get_model(self):
        return self.model

    def get_optimizer(self):
        return self.optimizer

    def get_model_info(self):
        print("Model info:")
        print(f"  Device: {self.device}")
        print(f"  Model: {self.model}")
        print(f"  Optimizer: {self.optimizer}")


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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    validation_dataset = TripletDataset(samples, labels, batch_size, num_negatives)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    model = TripletModel(num_embeddings, embedding_dim, margin, lr, device)
    model.train(dataloader, epochs)
    model.evaluate(validation_dataloader)
    input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    output = model.predict(input_ids)
    print(output)

    # Example usage of new methods
    dataset.get_info()
    model.get_model_info()
    print(model.get_device())
    print(model.get_model())
    print(model.get_optimizer())

if __name__ == "__main__":
    main()