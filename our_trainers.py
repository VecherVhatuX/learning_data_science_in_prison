import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

def create_triplet_dataset(samples, labels, batch_size, num_negatives):
    def __len__():
        return -(-len(samples) // batch_size)

    def __getitem__(idx):
        anchor_idx = np.arange(idx * batch_size, min((idx + 1) * batch_size, len(samples)))
        anchor_labels = labels[anchor_idx]
        positive_idx = np.array([np.random.choice(np.where(labels == label)[0], size=1)[0] for label in anchor_labels])
        negative_idx = np.array([np.random.choice(np.where(labels != label)[0], size=num_negatives, replace=False) for label in anchor_labels])
        return {
            'anchor_input_ids': torch.tensor(samples[anchor_idx], dtype=torch.long),
            'positive_input_ids': torch.tensor(samples[positive_idx], dtype=torch.long),
            'negative_input_ids': torch.tensor(samples[negative_idx], dtype=torch.long)
        }

    def get_samples():
        return samples

    def get_labels():
        return labels

    def get_batch_size():
        return batch_size

    def get_num_negatives():
        return num_negatives

    def get_info():
        print("Dataset info:")
        print(f"  Samples: {samples.shape}")
        print(f"  Labels: {labels.shape}")
        print(f"  Batch size: {batch_size}")
        print(f"  Number of negatives: {num_negatives}")

    return type('TripletDataset', (), {
        '__len__': __len__,
        '__getitem__': __getitem__,
        'get_samples': get_samples,
        'get_labels': get_labels,
        'get_batch_size': get_batch_size,
        'get_num_negatives': get_num_negatives,
        'get_info': get_info
    })

def create_triplet_network(num_embeddings, embedding_dim):
    model = nn.Sequential(
        nn.Embedding(num_embeddings, embedding_dim),
        nn.LazyLinear(embedding_dim),
        nn.Lambda(lambda x: x / torch.norm(x, dim=-1, keepdim=True))
    )

    def forward(inputs):
        return model(inputs)

    def get_embedding_dim():
        return model[1].out_features

    def get_model_summary():
        print("Model summary:")
        print(model)

    return type('TripletNetwork', (), {
        'forward': forward,
        'get_embedding_dim': get_embedding_dim,
        'get_model_summary': get_model_summary
    })

def create_triplet_model(num_embeddings, embedding_dim, margin, learning_rate, device):
    model = create_triplet_network(num_embeddings, embedding_dim)()
    loss_fn = nn.MarginRankingLoss(margin=margin, reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def train(dataloader, epochs):
        for epoch in range(epochs):
            total_loss = 0.0
            for i, data in enumerate(dataloader):
                anchor_inputs = data['anchor_input_ids'].to(device)
                positive_inputs = data['positive_input_ids'].to(device)
                negative_inputs = data['negative_input_ids'].to(device)
                anchor_embeddings = model.forward(anchor_inputs)
                positive_embeddings = model.forward(positive_inputs)
                negative_embeddings = model.forward(negative_inputs)
                anchor_positive_distance = torch.norm(anchor_embeddings - positive_embeddings, dim=-1)
                anchor_negative_distance = torch.norm(anchor_embeddings[:, None] - negative_embeddings, dim=-1)
                min_anchor_negative_distance = torch.min(anchor_negative_distance, dim=-1)[0]
                loss = loss_fn(min_anchor_negative_distance, anchor_positive_distance, torch.ones_like(min_anchor_negative_distance))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f'Epoch: {epoch+1}, Loss: {total_loss/(i+1):.3f}')

    def evaluate(dataloader):
        total_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                anchor_inputs = data['anchor_input_ids'].to(device)
                positive_inputs = data['positive_input_ids'].to(device)
                negative_inputs = data['negative_input_ids'].to(device)
                anchor_embeddings = model.forward(anchor_inputs)
                positive_embeddings = model.forward(positive_inputs)
                negative_embeddings = model.forward(negative_inputs)
                anchor_positive_distance = torch.norm(anchor_embeddings - positive_embeddings, dim=-1)
                anchor_negative_distance = torch.norm(anchor_embeddings[:, None] - negative_embeddings, dim=-1)
                min_anchor_negative_distance = torch.min(anchor_negative_distance, dim=-1)[0]
                loss = loss_fn(min_anchor_negative_distance, anchor_positive_distance, torch.ones_like(min_anchor_negative_distance))
                total_loss += loss.item()
        print(f'Validation Loss: {total_loss / (i+1):.3f}')

    def predict(input_ids):
        with torch.no_grad():
            return model.forward(input_ids.to(device))

    def get_device():
        return device

    def get_model():
        return model

    def get_optimizer():
        return optimizer

    def get_model_info():
        print("Model info:")
        print(f"  Device: {device}")
        print(f"  Model: {model}")
        print(f"  Optimizer: {optimizer}")

    return type('TripletModel', (), {
        'train': train,
        'evaluate': evaluate,
        'predict': predict,
        'get_device': get_device,
        'get_model': get_model,
        'get_optimizer': get_optimizer,
        'get_model_info': get_model_info
    })()

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

    dataset = create_triplet_dataset(samples, labels, batch_size, num_negatives)()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    validation_dataset = create_triplet_dataset(samples, labels, batch_size, num_negatives)()
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    model = create_triplet_model(num_embeddings, embedding_dim, margin, lr, device)()
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