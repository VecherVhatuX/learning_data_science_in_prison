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

collect_all_texts = lambda triplet_data: [text for item in triplet_data for text in (item['anchor'], item['positive'], item['negative'])]

convert_to_sequences = lambda tokenizer, data_item: {
    'anchor_seq': torch.tensor(tokenizer.transform([data_item['anchor']])[0], dtype=torch.long),
    'positive_seq': torch.tensor(tokenizer.transform([data_item['positive']])[0], dtype=torch.long),
    'negative_seq': torch.tensor(tokenizer.transform([data_item['negative']])[0], dtype=torch.long)
}

randomize_data = lambda data_samples: random.shuffle(data_samples) or data_samples

create_triplets = lambda instance_dict, bug_samples, non_bug_samples: [
    {
        'anchor': instance_dict[os.path.basename(folder)],
        'positive': bug_sample,
        'negative': random.choice(non_bug_samples)
    }
    for folder, _ in snippet_files
    for bug_sample in bug_samples
]

load_json_file = lambda file_path, root_dir: (
    lambda json_content: (
        {entry['instance_id']: entry['problem_statement'] for entry in json_content},
        [(folder, os.path.join(root_dir, 'snippet.json')) for folder in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, folder))]
    )
)(json.load(open(file_path, 'r')))

prepare_dataset = lambda instance_dict, snippet_files: (
    lambda bug_samples, non_bug_samples: create_triplets(instance_dict, bug_samples, non_bug_samples)
)(*zip(*map(lambda path: json.load(open(path)), snippet_files)))

class TripletData:
    def __init__(self, triplet_data):
        self.triplet_data = triplet_data
        self.tokenizer = LabelEncoder().fit(collect_all_texts(triplet_data))

    def get_samples(self): return self.triplet_data

class TripletDataset(Dataset):
    def __init__(self, triplet_data):
        self.data = TripletData(triplet_data)

    def __len__(self): return len(self.data.get_samples())

    def __getitem__(self, index): return convert_to_sequences(self.data.tokenizer, self.data.get_samples()[index])

class TripletModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(TripletModel, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.dense_network = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 128)
        )

    def forward(self, anchor, positive, negative):
        return tuple(map(self.dense_network, map(self.embedding_layer, (anchor, positive, negative))))

compute_loss = lambda anchor_embeds, positive_embeds, negative_embeds: torch.mean(torch.clamp(0.2 + torch.norm(anchor_embeds - positive_embeds, dim=1) - torch.norm(anchor_embeds - negative_embeds, dim=1), min=0))

train_network = lambda model, train_loader, valid_loader, epochs: (
    lambda optimizer, history: [
        (lambda: [
            (lambda: [
                optimizer.zero_grad(),
                (lambda anchor, positive, negative: (
                    lambda anchor_embeds, positive_embeds, negative_embeds: (
                        compute_loss(anchor_embeds, positive_embeds, negative_embeds).backward(),
                        optimizer.step()
                    )
                )(model(anchor, positive, negative))
            ))(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq'])
            for batch in train_loader
        ])(),
        history.append((loss.item(), *evaluate_network(model, valid_loader)))
        for _ in range(epochs)
    ] and history
)(optim.Adam(model.parameters(), lr=1e-5), [])

evaluate_network = lambda model, valid_loader: (
    lambda loss, correct: (
        (lambda: [
            (lambda anchor, positive, negative: (
                lambda anchor_embeds, positive_embeds, negative_embeds: (
                    (lambda: loss.__iadd__(compute_loss(anchor_embeds, positive_embeds, negative_embeds).item()),
                    (lambda: correct.__iadd__(count_matches(anchor_embeds, positive_embeds, negative_embeds)))
                )()
            )(model(anchor, positive, negative))
            )(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq'])
            for batch in valid_loader
        ])(),
        (loss / len(valid_loader), correct / len(valid_loader.dataset))
    )
)(0, 0)

count_matches = lambda anchor_output, positive_output, negative_output: torch.sum((torch.sum(anchor_output * positive_output, dim=1) > torch.sum(anchor_output * negative_output, dim=1)).int()).item()

visualize_results = lambda history: (
    lambda train_losses, val_losses, train_accuracies: (
        plt.figure(figsize=(10, 5)),
        plt.subplot(1, 2, 1),
        plt.plot(train_losses, label='Training Loss'),
        plt.plot(val_losses, label='Validation Loss'),
        plt.title('Loss Over Epochs'),
        plt.xlabel('Epochs'),
        plt.ylabel('Loss'),
        plt.legend(),
        plt.subplot(1, 2, 2),
        plt.plot(train_accuracies, label='Training Accuracy'),
        plt.title('Accuracy Over Epochs'),
        plt.xlabel('Epochs'),
        plt.ylabel('Accuracy'),
        plt.legend(),
        plt.show()
    )
)(*zip(*history))

store_model = lambda model, filepath: (torch.save(model.state_dict(), filepath), print(f'Model saved at {filepath}'))

retrieve_model = lambda model, filepath: (model.load_state_dict(torch.load(filepath)), print(f'Model loaded from {filepath}'), model)

main = lambda: (
    lambda dataset_path, snippets_directory: (
        lambda instance_dict, snippet_paths: (
            lambda triplet_data: (
                lambda train_data, valid_data: (
                    lambda train_loader, valid_loader: (
                        lambda model: (
                            lambda history: (
                                visualize_results(history),
                                store_model(model, 'triplet_model.pth')
                            )
                        )(train_network(model, train_loader, valid_loader, epochs=5))
                    )(TripletModel(vocab_size=len(train_loader.dataset.data.tokenizer.classes_) + 1, embedding_dim=128))
                )(DataLoader(TripletDataset(train_data.tolist()), batch_size=32, shuffle=True), DataLoader(TripletDataset(valid_data.tolist()), batch_size=32, shuffle=False))
            )(np.array_split(np.array(triplet_data), 2))
        )(prepare_dataset(instance_dict, snippet_paths))
    )(load_json_file(dataset_path, snippets_directory))
)('datasets/SWE-bench_oracle.npy', 'datasets/10_10_after_fix_pytest')

if __name__ == "__main__": main()