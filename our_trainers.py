import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class TripletDataset(Dataset):
    def __init__(self, samples, labels, batch_size, num_negatives):
        self.samples = torch.tensor(samples, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.batch_size = batch_size
        self.num_negatives = num_negatives

    def __len__(self):
        return (len(self.samples) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        indices = torch.randperm(len(self.samples))
        batch = indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        anchor_idx = batch

        positive_idx = []
        negative_indices = []
        for anchor in anchor_idx:
            idx = torch.where(self.labels == self.labels[anchor])[0]
            positive_idx.append(torch.randint(0, len(idx[idx != anchor]), (1,)).item())
            idx = torch.where(self.labels != self.labels[anchor])[0]
            negative_idx = torch.randperm(len(idx))[:self.num_negatives]
            negative_indices.extend(idx[negative_idx])

        anchor_input_ids = self.samples[anchor_idx]
        positive_input_ids = self.samples[torch.tensor(positive_idx, dtype=torch.long)]
        negative_input_ids = self.samples[torch.tensor(negative_indices, dtype=torch.long)].view(self.batch_size, self.num_negatives, -1)

        return {
            'anchor_input_ids': anchor_input_ids,
            'positive_input_ids': positive_input_ids,
            'negative_input_ids': negative_input_ids
        }

class TripletModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_negatives):
        super(TripletModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.num_negatives = num_negatives
        self.criterion = nn.TripletMarginLoss(margin=1.0, reduction='mean')

    def embed(self, input_ids):
        embeddings = self.embedding(input_ids)
        embeddings = self.pooling(embeddings).squeeze(1)
        embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
        return embeddings

    def forward(self, inputs):
        anchor_input_ids = inputs['anchor_input_ids']
        positive_input_ids = inputs['positive_input_ids']
        negative_input_ids = inputs['negative_input_ids']

        anchor_embeddings = self.embed(anchor_input_ids)
        positive_embeddings = self.embed(positive_input_ids)
        negative_embeddings = self.embed(negative_input_ids.view(-1, input_ids.shape[-1])).view(*negative_input_ids.shape[:-1], -1)

        return anchor_embeddings, positive_embeddings, negative_embeddings

def train(model, data_loader, optimizer, epochs, device):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(data_loader):
            data = {k: v.to(device) for k, v in data.items()}
            optimizer.zero_grad()
            anchor_embeddings, positive_embeddings, negative_embeddings = model(data)
            if len(positive_embeddings) > 0:
                loss = model.criterion(anchor_embeddings, positive_embeddings, negative_embeddings.view(-1, anchor_embeddings.shape[1]))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
        print(f'Epoch: {epoch+1}, Loss: {running_loss/(i+1):.3f}')

def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            data = {k: v.to(device) for k, v in data.items()}
            anchor_embeddings, positive_embeddings, negative_embeddings = model(data)
            if len(positive_embeddings) > 0:
                loss = model.criterion(anchor_embeddings, positive_embeddings, negative_embeddings.view(-1, anchor_embeddings.shape[1]))
                total_loss += loss.item()
    print(f'Validation Loss: {total_loss / (i+1):.3f}')

def predict(model, input_ids, device):
    model.eval()
    with torch.no_grad():
        input_ids = input_ids.to(device)
        return model.embed(input_ids)

def main():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    samples = torch.randint(0, 100, (100, 10))
    labels = torch.randint(0, 2, (100,))
    batch_size = 32
    num_negatives = 5
    epochs = 10

    model = TripletModel(101, 10, num_negatives).to(device)
    dataset = TripletDataset(samples, labels, batch_size, num_negatives)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    optimizer = optim.SGD(model.parameters(), lr=1e-4)
    train(model, data_loader, optimizer, epochs, device)

    torch.save(model.state_dict(), 'model.pth')

if __name__ == "__main__":
    main()