import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from functools import reduce
from operator import add

# Gather all texts from the data (anchor, positive, negative) into a single list
gather_texts = lambda data: [text for item in data for text in (item['anchor'], item['positive'], item['negative'])]

# Encode the sequences (anchor, positive, negative) using the provided encoder
encode_sequences = lambda encoder, item: {
    'anchor_seq': torch.tensor(encoder.transform([item['anchor']])[0],
    'positive_seq': torch.tensor(encoder.transform([item['positive']])[0]),
    'negative_seq': torch.tensor(encoder.transform([item['negative']])[0])
}

# Shuffle the data and return it
shuffle_data = lambda data: random.shuffle(data) or data

# Generate triplets (anchor, positive, negative) from the given mapping, bug samples, and non-bug samples
generate_triplets = lambda mapping, bug_samples, non_bug_samples: [
    {'anchor': mapping[os.path.basename(dir)], 'positive': bug_sample, 'negative': random.choice(non_bug_samples)}
    for dir, _ in snippet_files for bug_sample in bug_samples
]

# Fetch data from the given file path and root directory, returning a mapping and snippet files
fetch_data = lambda file_path, root_dir: (
    lambda data: ({item['instance_id']: item['problem_statement'] for item in data}, 
                  [(dir, os.path.join(root_dir, 'snippet.json')) for dir in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, dir))])
)(json.load(open(file_path)))

# Process the data by generating triplets from the mapping and snippet files
process_data = lambda mapping, snippet_files: generate_triplets(mapping, *zip(*[json.load(open(path)) for path in snippet_files]))

# DataManager class to manage and encode the data
class DataManager:
    def __init__(self, data):
        self.data = data
        self.encoder = LabelEncoder().fit(gather_texts(data))
    retrieve_data = lambda self: self.data

# TripletDataset class to handle the dataset for training and validation
class TripletDataset(Dataset):
    def __init__(self, data):
        self.data = DataManager(data)
        self.samples = self.data.retrieve_data()
    __len__ = lambda self: len(self.samples)
    __getitem__ = lambda self, idx: encode_sequences(self.data.encoder, self.samples[idx])

# EmbeddingModel class to define the neural network model for embeddings
class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(EmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.network = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 128)
        )
    forward = lambda self, anchor, positive, negative: (
        self.network(self.embedding(anchor)),
        self.network(self.embedding(positive)),
        self.network(self.embedding(negative))
    )

# Calculate the triplet loss
calculate_loss = lambda anchor, positive, negative: torch.mean(torch.clamp(0.2 + torch.norm(anchor - positive, dim=1) - torch.norm(anchor - negative, dim=1), min=0))

# Train the model using the provided training and validation data
train_model = lambda model, train_data, valid_data, epochs: reduce(
    lambda history, _: history + [(lambda loss, eval_result: (loss.item(), *eval_result))(
        (lambda anchor, positive, negative: (lambda loss: (loss.backward(), optimizer.step(), loss))(
            calculate_loss(anchor, positive, negative)
        )(model(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq'])),
        evaluate_model(model, valid_data)
    ) for batch in train_data],
    range(epochs),
    []
)

# Evaluate the model on the provided data
evaluate_model = lambda model, data: (
    sum(calculate_loss(*model(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq'])).item() for batch in data) / len(data),
    sum(count_correct(*model(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq'])) for batch in data) / len(data.dataset)
)

# Count the number of correct predictions
count_correct = lambda anchor, positive, negative: torch.sum((torch.sum(anchor * positive, dim=1) > torch.sum(anchor * negative, dim=1)).item()

# Plot the training and validation history
plot_history = lambda history: (
    plt.figure(figsize=(10, 5)),
    plt.subplot(1, 2, 1),
    plt.plot(*zip(*history)[:2], label=['Training Loss', 'Validation Loss']),
    plt.title('Loss Over Epochs'),
    plt.xlabel('Epochs'),
    plt.ylabel('Loss'),
    plt.legend(),
    plt.subplot(1, 2, 2),
    plt.plot(*zip(*history)[2:], label='Training Accuracy'),
    plt.title('Accuracy Over Epochs'),
    plt.xlabel('Epochs'),
    plt.ylabel('Accuracy'),
    plt.legend(),
    plt.show()
)

# Save the model to the specified path
save_model = lambda model, path: (torch.save(model.state_dict(), path), print(f'Model saved at {path}'))

# Load the model from the specified path
load_model = lambda model, path: (model.load_state_dict(torch.load(path)), print(f'Model loaded from {path}'), model)

# Visualize the embeddings in a 2D scatter plot
visualize_embeddings = lambda model, data: (
    plt.figure(figsize=(10, 10)),
    plt.scatter(*zip(*[(anchor.detach().numpy(), batch['anchor_seq'].numpy()) for batch in data for anchor, _, _ in [model(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq'])])), c='Spectral'),
    plt.colorbar(),
    plt.title('2D Embedding Visualization'),
    plt.show()
)

# Main function to run the entire pipeline
run = lambda: (
    lambda dataset_path, snippets_dir: (
        lambda mapping, snippet_files: (
            lambda data: (
                lambda train_data, valid_data: (
                    lambda train_loader, valid_loader: (
                        lambda model: (
                            lambda history: (
                                plot_history(history),
                                save_model(model, 'model.pth'),
                                visualize_embeddings(model, valid_loader)
                            )
                        )(train_model(model, train_loader, valid_loader, epochs=5))
                    )(EmbeddingModel(vocab_size=len(train_loader.dataset.data.encoder.classes_) + 1, embed_dim=128))
                )(DataLoader(TripletDataset(train_data.tolist()), batch_size=32, shuffle=True), DataLoader(TripletDataset(valid_data.tolist()), batch_size=32))
            )(np.array_split(np.array(data), 2))
        )(process_data(mapping, snippet_files))
    )(fetch_data(dataset_path, snippets_dir))
)('datasets/SWE-bench_oracle.npy', 'datasets/10_10_after_fix_pytest')

if __name__ == "__main__":
    run()