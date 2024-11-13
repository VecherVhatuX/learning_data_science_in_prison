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
        positive_idx = torch.tensor([], dtype=torch.long)
        negative_indices = torch.tensor([], dtype=torch.long)

        for anchor in anchor_idx:
            idx = torch.where(self.labels == self.labels[anchor])[0]
            positive_idx = torch.cat((positive_idx, torch.randint(0, len(idx[idx != anchor]), (1,))))
            idx = torch.where(self.labels != self.labels[anchor])[0]
            negative_idx = torch.randperm(len(idx))[:self.num_negatives]
            negative_indices = torch.cat((negative_indices, idx[negative_idx]))

        anchor_input_ids = self.samples[anchor_idx]
        if len(positive_idx) > 0:
            positive_input_ids = self.samples[positive_idx]
        else:
            positive_input_ids = torch.tensor([], dtype=torch.long)
        negative_input_ids = self.samples[negative_indices].view(self.batch_size, self.num_negatives, -1)

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

    def forward(self, inputs):
        anchor_input_ids = inputs['anchor_input_ids']
        positive_input_ids = inputs['positive_input_ids']
        negative_input_ids = inputs['negative_input_ids']

        anchor_embeddings = self.embedding(anchor_input_ids)
        anchor_embeddings = self.pooling(anchor_embeddings).squeeze(1)
        anchor_embeddings = anchor_embeddings / torch.norm(anchor_embeddings, dim=1, keepdim=True)

        positive_embeddings = self.embedding(positive_input_ids)
        positive_embeddings = self.pooling(positive_embeddings).squeeze(1)
        positive_embeddings = positive_embeddings / torch.norm(positive_embeddings, dim=1, keepdim=True)

        negative_embeddings = self.embedding(negative_input_ids)
        negative_embeddings = self.pooling(negative_embeddings).squeeze(1)
        negative_embeddings = negative_embeddings / torch.norm(negative_embeddings, dim=1, keepdim=True)

        return anchor_embeddings, positive_embeddings, negative_embeddings[:, 0, :]

def train(model, data_loader, optimizer, epochs):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(data_loader):
            optimizer.zero_grad()
            anchor_embeddings, positive_embeddings, negative_embeddings = model(data)
            if len(positive_embeddings) > 0:
                loss = model.criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
        print('Epoch: %d, Loss: %.3f' % (epoch+1, running_loss/(i+1)))

def main():
    torch.manual_seed(42)
    samples = torch.randint(0, 100, (100, 10))
    labels = torch.randint(0, 2, (100,))
    batch_size = 32
    num_negatives = 5
    epochs = 10

    model = TripletModel(101, 10, num_negatives)
    dataset = TripletDataset(samples, labels, batch_size, num_negatives)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    optimizer = optim.SGD(model.parameters(), lr=1e-4)
    train(model, data_loader, optimizer, epochs)

    torch.save(model.state_dict(), 'model.pth')

def evaluate(model, data_loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            anchor_embeddings, positive_embeddings, negative_embeddings = model(data)
            if len(positive_embeddings) > 0:
                loss = model.criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
                total_loss += loss.item()
    print('Validation Loss: %.3f' % (total_loss / (i+1)))

def predict(model, input_ids):
    model.eval()
    with torch.no_grad():
        embeddings = model.embedding(input_ids)
        embeddings = model.pooling(embeddings).squeeze(1)
        embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
        return embeddings

if __name__ == "__main__":
    main()