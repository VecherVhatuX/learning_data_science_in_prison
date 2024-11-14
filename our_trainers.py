import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TripletDataset:
    def __init__(self, samples, labels, batch_size, num_negatives):
        self.samples = samples
        self.labels = labels
        self.batch_size = batch_size
        self.num_negatives = num_negatives

    def __call__(self, idx):
        anchor_idx = np.arange(idx * self.batch_size, (idx + 1) * self.batch_size)
        positive_idx = np.concatenate([np.random.choice(np.where(self.labels == self.labels[anchor])[0], size=1) for anchor in anchor_idx], axis=0)
        negative_idx = np.random.choice(np.where(self.labels != self.labels[anchor_idx])[0], size=self.batch_size * self.num_negatives, replace=False)
        return {
            'anchor_input_ids': torch.from_numpy(self.samples[anchor_idx]),
            'positive_input_ids': torch.from_numpy(self.samples[positive_idx]),
            'negative_input_ids': torch.from_numpy(self.samples[negative_idx]).view(self.batch_size, self.num_negatives, -1)
        }

    def on_epoch_end(self):
        indices = np.random.permutation(len(self.samples))
        return self.samples[indices], self.labels[indices]


class TripletModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(TripletModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def embed(self, input_ids):
        embeddings = self.embedding(input_ids)
        embeddings = self.pooling(embeddings.permute(0, 2, 1)).squeeze()
        return F.normalize(embeddings, p=2, dim=1)

    def forward(self, inputs):
        anchor_embeddings = self.embed(inputs['anchor_input_ids'])
        positive_embeddings = self.embed(inputs['positive_input_ids'])
        negative_embeddings = self.embed(inputs['negative_input_ids'].view(-1, inputs['negative_input_ids'].shape[2]))
        return anchor_embeddings, positive_embeddings, negative_embeddings


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        return torch.mean(torch.clamp(torch.norm(anchor - positive, p=2, dim=1) - torch.norm(anchor.unsqueeze(1) - negative, p=2, dim=2) + self.margin, min=0))


class TripletTrainer:
    def __init__(self, model, loss_fn, epochs, lr, dataset):
        self.model = model
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.lr = lr
        self.dataset = dataset
        self.optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)

    def train_step(self, data):
        self.optimizer.zero_grad()
        anchor_embeddings, positive_embeddings, negative_embeddings = self.model(data)
        if len(positive_embeddings) > 0:
            loss = self.loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
            loss.backward()
            self.optimizer.step()
        return loss.item()

    def train(self):
        for epoch in range(self.epochs):
            total_loss = 0
            for i in range(len(self.dataset.samples) // self.dataset.batch_size):
                data = self.dataset(i)
                loss = self.train_step(data)
                total_loss += loss
            print(f'Epoch: {epoch+1}, Loss: {total_loss/(i+1):.3f}')


class TripletEvaluator:
    def __init__(self, model, loss_fn, dataset):
        self.model = model
        self.loss_fn = loss_fn
        self.dataset = dataset

    def evaluate_step(self, data):
        anchor_embeddings, positive_embeddings, negative_embeddings = self.model(data)
        if len(positive_embeddings) > 0:
            loss = self.loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
            return loss.item()
        return 0.0

    def evaluate(self):
        total_loss = 0.0
        for i in range(len(self.dataset.samples) // self.dataset.batch_size):
            data = self.dataset(i)
            loss = self.evaluate_step(data)
            total_loss += loss
        print(f'Validation Loss: {total_loss / (i+1):.3f}')


class TripletPredictor:
    def __init__(self, model):
        self.model = model

    def predict(self, input_ids):
        return self.model({'anchor_input_ids': input_ids})[0]


def main():
    np.random.seed(42)
    torch.manual_seed(42)
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
    model = TripletModel(num_embeddings, embedding_dim)
    loss_fn = TripletLoss(margin)
    trainer = TripletTrainer(model, loss_fn, epochs, lr, dataset)
    trainer.train()

    evaluator = TripletEvaluator(model, loss_fn, dataset)
    evaluator.evaluate()

    predictor = TripletPredictor(model)
    input_ids = torch.tensor([1, 2, 3, 4, 5])
    output = predictor.predict(input_ids)
    print(output)


if __name__ == "__main__":
    main()