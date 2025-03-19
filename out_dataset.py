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

TextProcessor = lambda data: type('TextProcessor', (), {
    'encoder': LabelEncoder().fit([text for item in data for text in (item['anchor'], item['positive'], item['negative'])]),
    'encode_text': lambda self, text: torch.tensor(self.encoder.transform([text])[0], dtype=torch.long)
})()

TripletDatasetManager = lambda data: type('TripletDatasetManager', (), {
    'data': data,
    'text_processor': TextProcessor(data),
    'get_dataset': lambda self: self.data
})()

TripletDataset = lambda data: type('TripletDataset', (Dataset,), {
    'dataset_manager': TripletDatasetManager(data),
    'samples': lambda self: self.dataset_manager.get_dataset(),
    '__len__': lambda self: len(self.samples),
    '__getitem__': lambda self, idx: {
        'anchor_seq': self.dataset_manager.text_processor.encode_text(self.samples[idx]['anchor']),
        'positive_seq': self.dataset_manager.text_processor.encode_text(self.samples[idx]['positive']),
        'negative_seq': self.dataset_manager.text_processor.encode_text(self.samples[idx]['negative'])
    }
})(data)

EmbeddingModel = lambda vocab_size, embed_dim: type('EmbeddingModel', (nn.Module,), {
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
})(vocab_size, embed_dim)

calculate_triplet_loss = lambda anchor, positive, negative: torch.mean(torch.clamp(0.2 + torch.norm(anchor - positive, dim=1) - torch.norm(anchor - negative, dim=1), min=0))

train_model = lambda model, train_loader, valid_loader, epochs: (
    lambda optimizer=optim.Adam(model.parameters(), lr=0.001), scheduler=optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1 ** epoch), history=[]: (
        [(
            model.train(),
            [(
                optimizer.zero_grad(),
                (lambda anchor, positive, negative: (
                    loss := calculate_triplet_loss(anchor, positive, negative),
                    loss.backward(),
                    optimizer.step()
                ))(*model(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq']))
            ) for batch in train_loader],
            history.append((sum(loss.item() for batch in train_loader) / len(train_loader), *evaluate_model(model, valid_loader))
        ) for _ in range(epochs)],
        history
    )[-1]
)()

evaluate_model = lambda model, data_loader: (
    model.eval(),
    (lambda total_loss=0, correct=0: (
        [(
            (lambda anchor, positive, negative: (
                total_loss := total_loss + calculate_triplet_loss(anchor, positive, negative).item(),
                correct := correct + torch.sum(torch.sum(anchor * positive, dim=1) > torch.sum(anchor * negative, dim=1)).float().sum()
            ))(*model(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq']))
        ) for batch in data_loader],
        (total_loss / len(data_loader), correct / len(data_loader))
    )[-1]
)()

plot_training_history = lambda history: (
    plt.figure(figsize=(10, 5)),
    plt.subplot(1, 2, 1),
    plt.plot([x[0] for x in history], label='Training Loss'),
    plt.plot([x[1] for x in history], label='Validation Loss'),
    plt.title('Loss Over Epochs'),
    plt.xlabel('Epochs'),
    plt.ylabel('Loss'),
    plt.legend(),
    plt.subplot(1, 2, 2),
    plt.plot([x[2] for x in history], label='Training Accuracy'),
    plt.title('Accuracy Over Epochs'),
    plt.xlabel('Epochs'),
    plt.ylabel('Accuracy'),
    plt.legend(),
    plt.show()
)

save_trained_model = lambda model, path: (
    torch.save(model.state_dict(), path),
    print(f'Model saved at {path}')
)

load_trained_model = lambda model, path: (
    model.load_state_dict(torch.load(path)),
    print(f'Model loaded from {path}'),
    model
)

visualize_embeddings = lambda model, data_loader: (
    model.eval(),
    (lambda embeddings=[]: (
        [embeddings.append(anchor.numpy()) for batch in data_loader for anchor, _, _ in [model(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq'])]],
        (lambda embeddings=np.concatenate(embeddings): (
            plt.figure(figsize=(10, 10)),
            plt.subplot(111, projection='3d'),
            plt.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c='Spectral'),
            plt.title('3D Embedding Visualization'),
            plt.show()
        ))()
    ))()
)

load_and_prepare_data = lambda file_path, root_dir: (
    (lambda data=json.load(open(file_path)): (
        {item['instance_id']: item['problem_statement'] for item in data},
        [(dir, os.path.join(root_dir, 'snippet.json')) for dir in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, dir))]
    )
)()

generate_triplets = lambda mapping, snippet_files: [
    {'anchor': mapping[os.path.basename(dir)], 'positive': bug_sample, 'negative': random.choice(non_bug_samples)}
    for dir, _ in snippet_files for bug_sample, non_bug_samples in [json.load(open(path)) for path in snippet_files]
]

run_pipeline = lambda: (
    (lambda dataset_path='datasets/SWE-bench_oracle.npy', snippets_dir='datasets/10_10_after_fix_pytest'): (
        (lambda mapping, snippet_files=load_and_prepare_data(dataset_path, snippets_dir)): (
            (lambda data=generate_triplets(mapping, snippet_files)): (
                (lambda train_data, valid_data=np.array_split(np.array(data), 2)): (
                    (lambda train_loader=DataLoader(TripletDataset(train_data.tolist()), batch_size=32, shuffle=True), valid_loader=DataLoader(TripletDataset(valid_data.tolist()), batch_size=32)): (
                        (lambda model=EmbeddingModel(vocab_size=len(train_loader.dataset.dataset_manager.text_processor.encoder.classes_) + 1, embed_dim=128)): (
                            (lambda history=train_model(model, train_loader, valid_loader, epochs=5)): (
                                plot_training_history(history),
                                save_trained_model(model, 'model.pth'),
                                visualize_embeddings(model, valid_loader)
                            )()
                        )()
                    )()
                )()
            )()
        )()
    )()
)()

add_enhanced_feature = lambda: (
    print("New feature added: Enhanced visualization with 3D embeddings.")
)

if __name__ == "__main__":
    run_pipeline()
    add_enhanced_feature()