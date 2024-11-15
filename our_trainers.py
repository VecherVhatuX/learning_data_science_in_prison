import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

def create_triplet_dataset(samples, labels, batch_size, num_negatives):
    def calculate_length():
        return -(-len(samples) // batch_size)

    def retrieve_item(idx):
        start_idx = idx * batch_size
        end_idx = min((idx + 1) * batch_size, len(samples))
        anchor_idx = np.arange(start_idx, end_idx)
        anchor_labels = labels[anchor_idx]
        
        positive_idx = np.array([np.random.choice(np.where(labels == label)[0], size=1)[0] for label in anchor_labels])
        negative_idx = np.array([np.random.choice(np.where(labels != label)[0], size=num_negatives, replace=False) for label in anchor_labels])
        
        return {
            'anchor_input_ids': torch.tensor(samples[anchor_idx], dtype=torch.long),
            'positive_input_ids': torch.tensor(samples[positive_idx], dtype=torch.long),
            'negative_input_ids': torch.tensor(samples[negative_idx], dtype=torch.long)
        }

    def fetch_samples():
        return samples

    def fetch_labels():
        return labels

    def fetch_batch_size():
        return batch_size

    def fetch_num_negatives():
        return num_negatives

    def print_dataset_info():
        print("Dataset Information:")
        print(f"  Number of Samples: {samples.shape}")
        print(f"  Number of Labels: {labels.shape}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Number of Negatives: {num_negatives}")

    return type('TripletDataset', (), {
        '__len__': calculate_length,
        '__getitem__': retrieve_item,
        'fetch_samples': fetch_samples,
        'fetch_labels': fetch_labels,
        'fetch_batch_size': fetch_batch_size,
        'fetch_num_negatives': fetch_num_negatives,
        'print_dataset_info': print_dataset_info
    })

def create_triplet_network(num_embeddings, embedding_dim):
    model = nn.Sequential(
        nn.Embedding(num_embeddings, embedding_dim),
        nn.LazyLinear(embedding_dim),
        nn.Lambda(lambda x: x / torch.norm(x, dim=-1, keepdim=True))
    )

    def forward_pass(inputs):
        return model(inputs)

    def fetch_embedding_dim():
        return model[1].out_features

    def print_model_summary():
        print("Model Architecture:")
        print(model)

    return type('TripletNetwork', (), {
        'forward_pass': forward_pass,
        'fetch_embedding_dim': fetch_embedding_dim,
        'print_model_summary': print_model_summary
    })

def create_triplet_model(num_embeddings, embedding_dim, margin, learning_rate, device):
    model = create_triplet_network(num_embeddings, embedding_dim)()
    loss_fn = nn.MarginRankingLoss(margin=margin, reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def train_model(dataloader, epochs):
        for epoch in range(epochs):
            total_loss = 0.0
            for i, data in enumerate(dataloader):
                anchor_inputs = data['anchor_input_ids'].to(device)
                positive_inputs = data['positive_input_ids'].to(device)
                negative_inputs = data['negative_input_ids'].to(device)
                
                anchor_embeddings = model.forward_pass(anchor_inputs)
                positive_embeddings = model.forward_pass(positive_inputs)
                negative_embeddings = model.forward_pass(negative_inputs)
                
                anchor_positive_distance = torch.norm(anchor_embeddings - positive_embeddings, dim=-1)
                anchor_negative_distance = torch.norm(anchor_embeddings[:, None] - negative_embeddings, dim=-1)
                
                min_anchor_negative_distance = torch.min(anchor_negative_distance, dim=-1)[0]
                loss = loss_fn(min_anchor_negative_distance, anchor_positive_distance, torch.ones_like(min_anchor_negative_distance))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f'Epoch: {epoch+1}, Loss: {total_loss/(i+1):.3f}')

    def evaluate_model(dataloader):
        total_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                anchor_inputs = data['anchor_input_ids'].to(device)
                positive_inputs = data['positive_input_ids'].to(device)
                negative_inputs = data['negative_input_ids'].to(device)
                
                anchor_embeddings = model.forward_pass(anchor_inputs)
                positive_embeddings = model.forward_pass(positive_inputs)
                negative_embeddings = model.forward_pass(negative_inputs)
                
                anchor_positive_distance = torch.norm(anchor_embeddings - positive_embeddings, dim=-1)
                anchor_negative_distance = torch.norm(anchor_embeddings[:, None] - negative_embeddings, dim=-1)
                
                min_anchor_negative_distance = torch.min(anchor_negative_distance, dim=-1)[0]
                loss = loss_fn(min_anchor_negative_distance, anchor_positive_distance, torch.ones_like(min_anchor_negative_distance))
                
                total_loss += loss.item()
        print(f'Validation Loss: {total_loss / (i+1):.3f}')

    def make_prediction(input_ids):
        with torch.no_grad():
            return model.forward_pass(input_ids.to(device))

    def fetch_device():
        return device

    def fetch_model():
        return model

    def fetch_optimizer():
        return optimizer

    def print_model_info():
        print("Model Information:")
        print(f"  Device: {device}")
        print(f"  Model: {model}")
        print(f"  Optimizer: {optimizer}")

    return type('TripletModel', (), {
        'train_model': train_model,
        'evaluate_model': evaluate_model,
        'make_prediction': make_prediction,
        'fetch_device': fetch_device,
        'fetch_model': fetch_model,
        'fetch_optimizer': fetch_optimizer,
        'print_model_info': print_model_info
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
    model.train_model(dataloader, epochs)
    model.evaluate_model(validation_dataloader)
    input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    output = model.make_prediction(input_ids)
    print(output)

    dataset.print_dataset_info()
    model.print_model_info()
    print(model.fetch_device())
    print(model.fetch_model())
    print(model.fetch_optimizer())

if __name__ == "__main__":
    main()