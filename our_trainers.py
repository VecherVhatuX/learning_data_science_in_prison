import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

EmbeddingGenerator = lambda vocab_size, embed_dim: nn.Sequential(
    nn.Embedding(vocab_size, embed_dim),
    lambda x: nn.AdaptiveAvgPool1d(1)(x.transpose(1, 2)).squeeze(2),
    nn.Linear(embed_dim, embed_dim),
    nn.BatchNorm1d(embed_dim),
    nn.LayerNorm(embed_dim)
)

TripletDataset = lambda data, labels, neg_samples: type('TripletDataset', (Dataset,), {
    '__init__': lambda self, data, labels, neg_samples: (setattr(self, 'data', data), setattr(self, 'labels', labels), setattr(self, 'neg_samples', neg_samples)),
    '__len__': lambda self: len(self.data),
    '__getitem__': lambda self, idx: (
        self.data[idx],
        random.choice(self.data[self.labels == self.labels[idx]]),
        random.sample(self.data[self.labels != self.labels[idx]].tolist(), self.neg_samples)
})(data, labels, neg_samples)

calculate_triplet_loss = lambda anchor, pos, neg, margin=1.0: torch.mean(torch.clamp(
    torch.norm(anchor - pos, dim=1) - torch.min(torch.norm(anchor.unsqueeze(1) - neg, dim=2), dim=1)[0] + margin, min=0.0))

train_embedding_model = lambda model, loader, epochs, lr: (
    lambda optimizer, scheduler: [
        (lambda epoch_loss: [
            (lambda loss: (loss.backward(), optimizer.step(), epoch_loss.append(loss.item()))(calculate_triplet_loss(model(anchor), model(pos), model(neg)) + 0.01 * sum(torch.norm(p, p=2) for p in model.parameters())
            for anchor, pos, neg in loader
        ], scheduler.step(), epoch_loss
        )([]) for _ in range(epochs)
    ])(optim.Adam(model.parameters(), lr=lr), optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1))

evaluate_model = lambda model, data, labels, k=5: (
    lambda embeddings, distances, neighbors, true_positives: (
        print(f"Accuracy: {np.mean(np.any(labels[neighbors] == labels[:, np.newaxis], axis=1)):.4f}"),
        print(f"Precision: {np.mean(true_positives / k):.4f}"),
        print(f"Recall: {np.mean(true_positives / np.sum(labels == labels[:, np.newaxis], axis=1)):.4f}"),
        print(f"F1-score: {2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0:.4f}"),
        plt.figure(figsize=(8, 8)),
        plt.scatter(TSNE(n_components=2).fit_transform(embeddings)[:, 0], TSNE(n_components=2).fit_transform(embeddings)[:, 1], c=labels, cmap='viridis'),
        plt.colorbar(),
        plt.show()
    ))(model(torch.tensor(data, dtype=torch.long)).detach().numpy(),
        np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2),
        np.argsort(distances, axis=1)[:, 1:k+1],
        np.sum(labels[neighbors] == labels[:, np.newaxis], axis=1)
    )

save_model = lambda model, path: torch.save(model.state_dict(), path)

load_model = lambda model_class, path, vocab_size, embed_dim: (lambda model: (model.load_state_dict(torch.load(path)), model)[1])(model_class(vocab_size, embed_dim))

plot_loss = lambda losses: (plt.figure(figsize=(10, 5)), plt.plot(losses, label='Loss', color='blue'), plt.title('Training Loss Over Epochs'), plt.xlabel('Epochs'), plt.ylabel('Loss'), plt.legend(), plt.show())

generate_random_data = lambda data_size: (np.random.randint(0, 100, (data_size, 10)), np.random.randint(0, 10, data_size))

visualize_embeddings = lambda model, data, labels: (
    lambda embeddings, tsne: (
        plt.figure(figsize=(8, 8)),
        plt.scatter(tsne[:, 0], tsne[:, 1], c=labels, cmap='viridis'),
        plt.colorbar(),
        plt.gcf().canvas.mpl_connect('button_press_event', lambda event: print(f"Clicked on point with label: {labels[np.argmin(np.linalg.norm(tsne - np.array([event.xdata, event.ydata]), axis=1))]}") if event.inaxes is not None else None),
        plt.show()
    ))(model(torch.tensor(data, dtype=torch.long)).detach().numpy(), TSNE(n_components=2).fit_transform(embeddings))

display_similarity_matrix = lambda model, data: (
    lambda embeddings, cosine_sim: (
        plt.figure(figsize=(8, 8)),
        plt.imshow(cosine_sim, cmap='viridis', vmin=0, vmax=1),
        plt.colorbar(),
        plt.title('Cosine Similarity Matrix'),
        plt.show()
    ))(model(torch.tensor(data, dtype=torch.long)).detach().numpy(), np.dot(embeddings, embeddings.T) / (np.linalg.norm(embeddings, axis=1)[:, np.newaxis] * np.linalg.norm(embeddings, axis=1)))

plot_embedding_distribution = lambda model, data: (
    lambda embeddings: (
        plt.figure(figsize=(8, 8)),
        plt.hist(embeddings.flatten(), bins=50, color='blue', alpha=0.7),
        plt.title('Embedding Value Distribution'),
        plt.xlabel('Embedding Value'),
        plt.ylabel('Frequency'),
        plt.show()
    ))(model(torch.tensor(data, dtype=torch.long)).detach().numpy())

if __name__ == "__main__":
    data, labels = generate_random_data(100)
    dataset = TripletDataset(data, labels, 5)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = EmbeddingGenerator(101, 10)
    loss_history = train_embedding_model(model, loader, 10, 1e-4)
    save_model(model, "embedding_model.pth")
    plot_loss(loss_history)
    evaluate_model(model, data, labels)
    visualize_embeddings(load_model(EmbeddingGenerator, "embedding_model.pth", 101, 10), *generate_random_data(100))
    display_similarity_matrix(model, data)
    plot_embedding_distribution(model, data)