import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim

class TripletDataset(Dataset):
    """
    Generates triplets for training a neural network using the triplet loss function.
    """
    def __init__(self, samples, labels, batch_size, num_negatives):
        """
        Initializes the TripletDataset.

        Args:
        samples (numpy array): Input samples.
        labels (numpy array): Labels corresponding to the input samples.
        batch_size (int): Batch size for generating triplets.
        num_negatives (int): Number of negative samples per anchor.
        """
        self.samples = torch.tensor(samples, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.batch_size = batch_size
        self.num_negatives = num_negatives

    def __len__(self):
        """
        Returns the number of batches in the dataset.
        """
        return (len(self.samples) + self.batch_size - 1) // self.batch_size

    def get_indices(self):
        """
        Returns a shuffled version of the indices of the input samples.
        """
        return np.random.permutation(len(self.samples))

    def get_anchor_idx(self, indices, idx):
        """
        Returns the indices of the anchor samples for a given batch.

        Args:
        indices (numpy array): Shuffled indices of the input samples.
        idx (int): Batch index.

        Returns:
        torch tensor: Indices of the anchor samples.
        """
        return indices[idx * self.batch_size:(idx + 1) * self.batch_size]

    def get_positive_idx(self, anchor_idx):
        """
        Returns the indices of the positive samples corresponding to the anchor samples.

        Args:
        anchor_idx (torch tensor): Indices of the anchor samples.

        Returns:
        list: Indices of the positive samples.
        """
        positive_idx = []
        for anchor in anchor_idx:
            idx = torch.where(self.labels == self.labels[anchor])[0]
            positive_idx.append(torch.randint(0, len(idx[idx != anchor]), (1,)).item())
        return positive_idx

    def get_negative_idx(self, anchor_idx):
        """
        Returns the indices of the negative samples corresponding to the anchor samples.

        Args:
        anchor_idx (torch tensor): Indices of the anchor samples.

        Returns:
        list: Indices of the negative samples.
        """
        negative_indices = []
        for anchor in anchor_idx:
            idx = torch.where(self.labels != self.labels[anchor])[0]
            negative_idx = torch.randperm(len(idx))[:self.num_negatives]
            negative_indices.extend(negative_idx)
        return negative_indices

    def get_data(self, anchor_idx, positive_idx, negative_idx):
        """
        Returns the input data for a given batch.

        Args:
        anchor_idx (torch tensor): Indices of the anchor samples.
        positive_idx (list): Indices of the positive samples.
        negative_idx (list): Indices of the negative samples.

        Returns:
        dict: Input data for the batch.
        """
        anchor_input_ids = self.samples[anchor_idx]
        positive_input_ids = self.samples[torch.tensor(positive_idx, dtype=torch.long)]
        negative_input_ids = self.samples[torch.tensor(negative_idx, dtype=torch.long)].numpy().reshape(self.batch_size, self.num_negatives, -1)
        return {
            'anchor_input_ids': anchor_input_ids,
            'positive_input_ids': positive_input_ids,
            'negative_input_ids': negative_input_ids
        }

    def __getitem__(self, idx):
        """
        Returns the input data for a given batch index.

        Args:
        idx (int): Batch index.

        Returns:
        dict: Input data for the batch.
        """
        indices = self.get_indices()
        anchor_idx = self.get_anchor_idx(indices, idx)
        positive_idx = self.get_positive_idx(anchor_idx)
        negative_idx = self.get_negative_idx(anchor_idx)
        return self.get_data(anchor_idx, positive_idx, negative_idx)

class TripletModel(nn.Module):
    """
    A neural network model for learning embeddings using the triplet loss function.
    """
    def __init__(self, num_embeddings, embedding_dim, num_negatives):
        """
        Initializes the TripletModel.

        Args:
        num_embeddings (int): Number of embeddings.
        embedding_dim (int): Dimension of the embeddings.
        num_negatives (int): Number of negative samples per anchor.
        """
        super(TripletModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.num_negatives = num_negatives

    def normalize_embeddings(self, embeddings):
        """
        Normalizes the embeddings to have unit length.

        Args:
        embeddings (torch tensor): Embeddings to normalize.

        Returns:
        torch tensor: Normalized embeddings.
        """
        return embeddings / torch.norm(embeddings, dim=1, keepdim=True)

    def embed(self, input_ids):
        """
        Embeds the input IDs into a dense representation.

        Args:
        input_ids (torch tensor): Input IDs to embed.

        Returns:
        torch tensor: Embedded input IDs.
        """
        embeddings = self.embedding(input_ids)
        embeddings = self.pooling(embeddings.permute(0, 2, 1)).squeeze()
        return self.normalize_embeddings(embeddings)

    def embed_negative(self, negative_input_ids):
        """
        Embeds the negative input IDs into a dense representation.

        Args:
        negative_input_ids (numpy array): Negative input IDs to embed.

        Returns:
        torch tensor: Embedded negative input IDs.
        """
        return self.embed(torch.tensor(negative_input_ids, dtype=torch.long).view(-1, negative_input_ids.shape[2]))

    def forward(self, inputs):
        """
        Defines the forward pass through the network.

        Args:
        inputs (dict): Input data for the batch.

        Returns:
        tuple: Embedded anchor, positive, and negative input IDs.
        """
        anchor_input_ids = inputs['anchor_input_ids']
        positive_input_ids = inputs['positive_input_ids']
        negative_input_ids = inputs['negative_input_ids']

        anchor_embeddings = self.embed(anchor_input_ids)
        positive_embeddings = self.embed(positive_input_ids)
        negative_embeddings = self.embed_negative(negative_input_ids)
        return anchor_embeddings, positive_embeddings, negative_embeddings

class TripletLoss:
    """
    A loss function for training a neural network using the triplet loss function.
    """
    def __init__(self, margin=1.0):
        """
        Initializes the TripletLoss.

        Args:
        margin (float, optional): Margin for the triplet loss function. Defaults to 1.0.
        """
        self.margin = margin

    def calculate_loss(self, anchor, positive, negative):
        """
        Calculates the triplet loss for a given batch.

        Args:
        anchor (torch tensor): Embedded anchor input IDs.
        positive (torch tensor): Embedded positive input IDs.
        negative (torch tensor): Embedded negative input IDs.

        Returns:
        torch tensor: Triplet loss for the batch.
        """
        return torch.clamp(torch.norm(anchor - positive, dim=1) - torch.norm(anchor.unsqueeze(1) - negative, dim=2) + self.margin, min=0.0)

    def __call__(self, anchor, positive, negative):
        """
        Calculates the mean triplet loss for a given batch.

        Args:
        anchor (torch tensor): Embedded anchor input IDs.
        positive (torch tensor): Embedded positive input IDs.
        negative (torch tensor): Embedded negative input IDs.

        Returns:
        torch tensor: Mean triplet loss for the batch.
        """
        return torch.mean(self.calculate_loss(anchor, positive, negative))

class Trainer:
    """
    A trainer for training a neural network using the triplet loss function.
    """
    def __init__(self, model, optimizer, loss_fn):
        """
        Initializes the Trainer.

        Args:
        model (nn.Module): Neural network model to train.
        optimizer (optim.Optimizer): Optimizer for training the model.
        loss_fn (TripletLoss): Triplet loss function for training the model.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        """
        Performs a single training step.

        Args:
        data (dict): Input data for the batch.

        Returns:
        float: Loss for the batch.
        """
        self.optimizer.zero_grad()
        anchor_embeddings, positive_embeddings, negative_embeddings = self.model(data)
        if len(positive_embeddings) > 0:
            loss = self.loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
            loss.backward()
            self.optimizer.step()
            return loss.item()
        return 0.0

    def train(self, dataset, epochs):
        """
        Trains the model for a given number of epochs.

        Args:
        dataset (DataLoader): Training dataset.
        epochs (int): Number of epochs to train.
        """
        for epoch in range(epochs):
            total_loss = 0
            for i, data in enumerate(dataset):
                loss = self.train_step(data)
                total_loss += loss
            print(f'Epoch: {epoch+1}, Loss: {total_loss/(i+1):.3f}')

class Evaluator:
    """
    An evaluator for evaluating a trained neural network.
    """
    def __init__(self, model, loss_fn):
        """
        Initializes the Evaluator.

        Args:
        model (nn.Module): Trained neural network model.
        loss_fn (TripletLoss): Triplet loss function for evaluating the model.
        """
        self.model = model
        self.loss_fn = loss_fn

    def evaluate_step(self, data):
        """
        Evaluates the model for a given batch.

        Args:
        data (dict): Input data for the batch.

        Returns:
        float: Loss for the batch.
        """
        with torch.no_grad():
            anchor_embeddings, positive_embeddings, negative_embeddings = self.model(data)
            if len(positive_embeddings) > 0:
                loss = self.loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
                return loss.item()
            return 0.0

    def evaluate(self, dataset):
        """
        Evaluates the model for a given dataset.

        Args:
        dataset (DataLoader): Dataset to evaluate.
        """
        total_loss = 0.0
        for i, data in enumerate(dataset):
            loss = self.evaluate_step(data)
            total_loss += loss
        print(f'Validation Loss: {total_loss / (i+1):.3f}')

class Predictor:
    """
    A predictor for making predictions with a trained neural network.
    """
    def __init__(self, model):
        """
        Initializes the Predictor.

        Args:
        model (nn.Module): Trained neural network model.
        """
        self.model = model

    def predict(self, input_ids):
        """
        Makes a prediction for a given input ID.

        Args:
        input_ids (torch tensor): Input ID to predict.

        Returns:
        torch tensor: Embedded input ID.
        """
        return self.model.embed(input_ids)

def main():
    """
    Main function for training and evaluating a neural network using the triplet loss function.
    """
    np.random.seed(42)
    torch.manual_seed(42)
    samples = np.random.randint(0, 100, (100, 10))
    labels = np.random.randint(0, 2, (100,))
    batch_size = 32
    num_negatives = 5
    epochs = 10

    model = TripletModel(101, 10, num_negatives)
    dataset = TripletDataset(samples, labels, batch_size, num_negatives)
    data_loader = DataLoader(dataset, batch_size=1)
    optimizer = optim.SGD(model.parameters(), lr=1e-4)
    loss_fn = TripletLoss()

    trainer = Trainer(model, optimizer, loss_fn)
    evaluator = Evaluator(model, loss_fn)

    trainer.train(data_loader, epochs)
    evaluator.evaluate(data_loader)

    torch.save(model.state_dict(), 'model.pth')

    predictor = Predictor(model)
    input_ids = torch.tensor([1, 2, 3, 4, 5])
    output = predictor.predict(input_ids)
    print(output)

if __name__ == "__main__":
    main()