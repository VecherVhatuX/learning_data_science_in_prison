import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class TripletDataset(Dataset):
    """
    A custom dataset class for creating triplet datasets.
    """
    def __init__(self, samples, labels, batch_size, num_negatives):
        # Initialize the dataset with samples, labels, batch size, and number of negatives.
        self.samples = torch.tensor(samples, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.batch_size = batch_size
        self.num_negatives = num_negatives

    def __len__(self):
        # Calculate the length of the dataset.
        return (len(self.samples) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        # Get a batch of samples.
        indices = torch.randperm(len(self.samples))
        batch = indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        anchor_idx = batch

        positive_idx = []
        negative_indices = []
        for anchor in anchor_idx:
            # Find the indices of the samples with the same label as the anchor.
            idx = torch.where(self.labels == self.labels[anchor])[0]
            # Randomly select a positive sample from the indices.
            positive_idx.append(torch.randint(0, len(idx[idx != anchor]), (1,)).item())
            # Find the indices of the samples with different labels than the anchor.
            idx = torch.where(self.labels != self.labels[anchor])[0]
            # Randomly select a specified number of negative samples from the indices.
            negative_idx = torch.randperm(len(idx))[:self.num_negatives]
            negative_indices.extend(idx[negative_idx])

        # Get the input IDs for the anchor, positive, and negative samples.
        anchor_input_ids = self.samples[anchor_idx]
        positive_input_ids = self.samples[torch.tensor(positive_idx, dtype=torch.long)]
        negative_input_ids = self.samples[torch.tensor(negative_indices, dtype=torch.long)].view(self.batch_size, self.num_negatives, -1)

        # Return the input IDs as a dictionary.
        return {
            'anchor_input_ids': anchor_input_ids,
            'positive_input_ids': positive_input_ids,
            'negative_input_ids': negative_input_ids
        }

class TripletModel(nn.Module):
    """
    A custom model class for training a triplet network.
    """
    def __init__(self, num_embeddings, embedding_dim, num_negatives):
        # Initialize the model with the number of embeddings, embedding dimension, and number of negatives.
        super(TripletModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.num_negatives = num_negatives
        self.criterion = nn.TripletMarginLoss(margin=1.0, reduction='mean')

    def embed(self, input_ids):
        # Embed the input IDs and normalize the embeddings.
        embeddings = self.embedding(input_ids)
        embeddings = self.pooling(embeddings).squeeze(1)
        embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
        return embeddings

    def forward(self, inputs):
        # Get the anchor, positive, and negative input IDs.
        anchor_input_ids = inputs['anchor_input_ids']
        positive_input_ids = inputs['positive_input_ids']
        negative_input_ids = inputs['negative_input_ids']

        # Embed the input IDs.
        anchor_embeddings = self.embed(anchor_input_ids)
        positive_embeddings = self.embed(positive_input_ids)
        negative_embeddings = self.embed(negative_input_ids.view(-1, input_ids.shape[-1])).view(*negative_input_ids.shape[:-1], -1)

        # Return the embeddings.
        return anchor_embeddings, positive_embeddings, negative_embeddings

def train(model, data_loader, optimizer, epochs, device):
    # Train the model for a specified number of epochs.
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(data_loader):
            # Move the data to the device.
            data = {k: v.to(device) for k, v in data.items()}
            # Zero the gradients.
            optimizer.zero_grad()
            # Get the embeddings.
            anchor_embeddings, positive_embeddings, negative_embeddings = model(data)
            if len(positive_embeddings) > 0:
                # Calculate the loss.
                loss = model.criterion(anchor_embeddings, positive_embeddings, negative_embeddings.view(-1, anchor_embeddings.shape[1]))
                # Backpropagate the loss.
                loss.backward()
                # Update the model parameters.
                optimizer.step()
                # Add the loss to the running loss.
                running_loss += loss.item()
        # Print the epoch and loss.
        print(f'Epoch: {epoch+1}, Loss: {running_loss/(i+1):.3f}')

def evaluate(model, data_loader, device):
    # Evaluate the model on the validation set.
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            # Move the data to the device.
            data = {k: v.to(device) for k, v in data.items()}
            # Get the embeddings.
            anchor_embeddings, positive_embeddings, negative_embeddings = model(data)
            if len(positive_embeddings) > 0:
                # Calculate the loss.
                loss = model.criterion(anchor_embeddings, positive_embeddings, negative_embeddings.view(-1, anchor_embeddings.shape[1]))
                # Add the loss to the total loss.
                total_loss += loss.item()
    # Print the validation loss.
    print(f'Validation Loss: {total_loss / (i+1):.3f}')

def predict(model, input_ids, device):
    # Make predictions on the input IDs.
    model.eval()
    with torch.no_grad():
        # Move the input IDs to the device.
        input_ids = input_ids.to(device)
        # Get the embeddings.
        return model.embed(input_ids)

def main():
    # Set the random seed.
    torch.manual_seed(42)
    # Get the device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Generate some random samples and labels.
    samples = torch.randint(0, 100, (100, 10))
    labels = torch.randint(0, 2, (100,))
    # Set the batch size and number of negatives.
    batch_size = 32
    num_negatives = 5
    # Set the number of epochs.
    epochs = 10

    # Create the model.
    model = TripletModel(101, 10, num_negatives).to(device)
    # Create the dataset.
    dataset = TripletDataset(samples, labels, batch_size, num_negatives)
    # Create the data loader.
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Create the optimizer.
    optimizer = optim.SGD(model.parameters(), lr=1e-4)
    # Train the model.
    train(model, data_loader, optimizer, epochs, device)

    # Save the model.
    torch.save(model.state_dict(), 'model.pth')

if __name__ == "__main__":
    main()