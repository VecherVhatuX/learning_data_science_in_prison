import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

TextEncoder = lambda data: type('TextEncoder', (), {
    'encoder': LabelEncoder().fit([text for item in data for text in (item['anchor'], item['positive'], item['negative'])]),
    'encode': lambda self, text: torch.tensor(self.encoder.transform([text])[0], dtype=torch.long)
})

DataHandler = lambda data: type('DataHandler', (), {
    'data': data,
    'text_encoder': TextEncoder(data),
    'get_data': lambda self: self.data
})

TripletData = lambda data: type('TripletData', (Dataset,), {
    'data_handler': DataHandler(data),
    'samples': lambda self: self.data_handler.get_data(),
    '__len__': lambda self: len(self.samples()),
    '__getitem__': lambda self, idx: {
        'anchor': self.data_handler.text_encoder.encode(self.samples()[idx]['anchor']),
        'positive': self.data_handler.text_encoder.encode(self.samples()[idx]['positive']),
        'negative': self.data_handler.text_encoder.encode(self.samples()[idx]['negative'])
    }
})

EmbeddingNetwork = lambda vocab_size, embed_dim: type('EmbeddingNetwork', (nn.Module,), {
    'embedding': nn.Embedding(vocab_size, embed_dim),
    'network': nn.Sequential(
        nn.Linear(embed_dim, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.2),
        nn.Linear(128, 128)
    ),
    'forward': lambda self, anchor, positive, negative: (
        self.network(self.embedding(anchor)),
        self.network(self.embedding(positive)),
        self.network(self.embedding(negative))
    )
})

compute_loss = lambda anchor, positive, negative: torch.mean(torch.clamp(0.2 + torch.norm(anchor - positive, dim=1) - torch.norm(anchor - negative, dim=1), min=0))

train_network = lambda model, train_loader, valid_loader, epochs: (lambda optimizer=optim.Adam(model.parameters(), lr=0.001), scheduler=optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1 ** epoch), history=[]: [
    (model.train(), [(
        optimizer.zero_grad(),
        (lambda anchor, positive, negative: (
            loss := compute_loss(anchor, positive, negative),
            loss.backward(),
            optimizer.step(),
            total_loss := total_loss + loss.item() if 'total_loss' in locals() else loss.item()
        ))(*model(batch['anchor'], batch['positive'], batch['negative']))
        for batch in train_loader
    ], history.append((total_loss / len(train_loader), *evaluate_network(model, valid_loader)))
    for _ in range(epochs)
] and history)()

evaluate_network = lambda model, data_loader: (lambda total_loss=0, correct=0: [
    (model.eval(), [(
        (lambda anchor, positive, negative: (
            total_loss := total_loss + compute_loss(anchor, positive, negative).item(),
            correct := correct + torch.sum(torch.sum(anchor * positive, dim=1) > torch.sum(anchor * negative, dim=1)).float().sum()
        ))(*model(batch['anchor'], batch['positive'], batch['negative']))
        for batch in data_loader
    ], (total_loss / len(data_loader), correct / len(data_loader))
])()

plot_history = lambda history: (lambda fig=plt.figure(figsize=(10, 5)): (
    fig.add_subplot(1, 2, 1).plot([x[0] for x in history], label='Training Loss'),
    fig.add_subplot(1, 2, 1).plot([x[1] for x in history], label='Validation Loss'),
    fig.add_subplot(1, 2, 1).set(title='Loss Over Epochs', xlabel='Epochs', ylabel='Loss'),
    fig.add_subplot(1, 2, 1).legend(),
    fig.add_subplot(1, 2, 2).plot([x[2] for x in history], label='Training Accuracy'),
    fig.add_subplot(1, 2, 2).set(title='Accuracy Over Epochs', xlabel='Epochs', ylabel='Accuracy'),
    fig.add_subplot(1, 2, 2).legend(),
    plt.show()
))()

save_model = lambda model, path: (torch.save(model.state_dict(), path), print(f'Model saved at {path}')

load_model = lambda model, path: (model.load_state_dict(torch.load(path)), print(f'Model loaded from {path}'), model

visualize_embeddings = lambda model, data_loader: (lambda embeddings=[]: (
    model.eval(),
    [embeddings.append(anchor.numpy()) for batch in data_loader for anchor, _, _ in [model(batch['anchor'], batch['positive'], batch['negative'])]],
    (lambda fig=plt.figure(figsize=(10, 10)): (
        fig.add_subplot(111, projection='3d').scatter(np.concatenate(embeddings)[:, 0], np.concatenate(embeddings)[:, 1], np.concatenate(embeddings)[:, 2], c='Spectral'),
        fig.add_subplot(111, projection='3d').set(title='3D Embedding Visualization'),
        plt.show()
    ))()
))()

load_data = lambda file_path, root_dir: (
    (lambda data=json.load(open(file_path)): (
        {item['instance_id']: item['problem_statement'] for item in data},
        [(dir, os.path.join(root_dir, 'snippet.json')) for dir in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, dir))]
    )
)()

create_triplets = lambda mapping, snippet_files: [
    {'anchor': mapping[os.path.basename(dir)], 'positive': bug_sample, 'negative': random.choice(non_bug_samples)}
    for dir, path in snippet_files for bug_sample, non_bug_samples in [json.load(open(path))]
]

execute_pipeline = lambda: (
    (lambda dataset_path='datasets/SWE-bench_oracle.npy', snippets_dir='datasets/10_10_after_fix_pytest'): (
        (lambda mapping, snippet_files=load_data(dataset_path, snippets_dir)): (
            (lambda data=create_triplets(mapping, snippet_files)): (
                (lambda train_data, valid_data=np.array_split(np.array(data), 2)): (
                    (lambda train_loader=DataLoader(TripletData(train_data.tolist()), batch_size=32, shuffle=True), valid_loader=DataLoader(TripletData(valid_data.tolist()), batch_size=32)): (
                        (lambda model=EmbeddingNetwork(vocab_size=len(train_loader.dataset.data_handler.text_encoder.encoder.classes_) + 1, embed_dim=128)): (
                            (lambda history=train_network(model, train_loader, valid_loader, epochs=5)): (
                                plot_history(history),
                                save_model(model, 'model.pth'),
                                visualize_embeddings(model, valid_loader)
                        )
                    )
                )
            )
        )
    )
)()

add_feature = lambda: print("New feature added: Enhanced visualization with 3D embeddings.")

if __name__ == "__main__":
    execute_pipeline()
    add_feature()