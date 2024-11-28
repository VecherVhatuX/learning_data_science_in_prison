import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class TripletModel(nn.Module):
    def __init__(self, embedding_dim, num_features):
        super(TripletModel, self).__init__()
        self.model = nn.Sequential(
            nn.Embedding(embedding_dim, num_features),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(num_features, num_features),
            nn.BatchNorm1d(num_features),
            nn.LayerNorm(num_features)
        )

    def build_criterion(self, margin=1.0):
        def triplet_loss(anchor, positive, negative):
            d_ap = torch.norm(anchor - positive, dim=-1)
            d_an = torch.norm(anchor.unsqueeze(1) - negative, dim=-1)
            loss = torch.max(d_ap - torch.min(d_an, dim=-1)[0] + margin, torch.zeros_like(d_ap))
            return torch.mean(loss)
        return triplet_loss

    def train(self, dataset, epochs, optimizer, criterion):
        for epoch in range(epochs):
            for batch in dataset:
                anchor, positive, negative = batch
                optimizer.zero_grad()
                anchor_embeddings = self.model(anchor)
                positive_embeddings = self.model(positive)
                negative_embeddings = self.model(negative)
                loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
                loss.backward()
                optimizer.step()
                print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

    def validate(self, embeddings, labels, k=5):
        predicted_embeddings = self.model(embeddings)
        print("Validation KNN Accuracy:", self.knn_accuracy(predicted_embeddings, labels, k))
        print("Validation KNN Precision:", self.knn_precision(predicted_embeddings, labels, k))
        print("Validation KNN Recall:", self.knn_recall(predicted_embeddings, labels, k))
        print("Validation KNN F1-score:", self.knn_f1(predicted_embeddings, labels, k))
        self.embedding_visualization(predicted_embeddings, labels)

    def distance(self, embedding1, embedding2):
        return torch.norm(embedding1 - embedding2, dim=-1)

    def similarity(self, embedding1, embedding2):
        return torch.sum(embedding1 * embedding2, dim=-1) / (torch.norm(embedding1, dim=-1) * torch.norm(embedding2, dim=-1))

    def cosine_distance(self, embedding1, embedding2):
        return 1 - self.similarity(embedding1, embedding2)

    def nearest_neighbors(self, embeddings, target_embedding, k=5):
        distances = self.distance(embeddings, target_embedding)
        return torch.argsort(distances)[:k]

    def similar_embeddings(self, embeddings, target_embedding, k=5):
        similarities = self.similarity(embeddings, target_embedding)
        return torch.argsort(-similarities)[:k]

    def embedding_visualization(self, embeddings, labels):
        tsne = TSNE(n_components=2)
        reduced_embeddings = tsne.fit_transform(embeddings.detach().numpy())
        plt.figure(figsize=(8, 8))
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels)
        plt.show()

    def knn_accuracy(self, embeddings, labels, k=5):
        return torch.mean(torch.any(torch.eq(labels[torch.argsort(self.distance(embeddings, embeddings), dim=1)[:, 1:k+1]], labels[:, None]), dim=1).float())

    def knn_precision(self, embeddings, labels, k=5):
        return torch.mean(torch.sum(torch.eq(labels[torch.argsort(self.distance(embeddings, embeddings), dim=1)[:, 1:k+1]], labels[:, None]), dim=1).float() / k)

    def knn_recall(self, embeddings, labels, k=5):
        return torch.mean(torch.sum(torch.eq(labels[torch.argsort(self.distance(embeddings, embeddings), dim=1)[:, 1:k+1]], labels[:, None]), dim=1).float() / torch.sum(torch.eq(labels, labels[:, None]), dim=1).float())

    def knn_f1(self, embeddings, labels, k=5):
        precision = self.knn_precision(embeddings, labels, k)
        recall = self.knn_recall(embeddings, labels, k)
        return 2 * (precision * recall) / (precision + recall)

def build_dataset(samples, labels, num_negatives, batch_size, shuffle=True):
    dataset = []
    indices = np.arange(len(samples))
    if shuffle:
        np.random.shuffle(indices)
    for i in range(len(samples) // batch_size):
        batch_indices = indices[i * batch_size:(i + 1) * batch_size]
        anchor_idx = np.random.choice(batch_indices, size=batch_size, replace=False)
        anchor_label = labels[anchor_idx]
        positive_idx = np.array([np.random.choice(np.where(labels == label)[0], size=1)[0] for label in anchor_label])
        negative_idx = np.array([np.random.choice(np.where(labels != label)[0], size=num_negatives, replace=False) for label in anchor_label])
        dataset.append((samples[anchor_idx], samples[positive_idx], samples[negative_idx]))
    return dataset

def build_optimizer(model, learning_rate):
    return optim.Adam(model.parameters(), lr=learning_rate)

def pipeline(learning_rate, batch_size, epochs, num_negatives, embedding_dim, num_features):
    samples = np.random.randint(0, 100, (100, 10))
    labels = np.random.randint(0, 2, (100,))
    model = TripletModel(embedding_dim, num_features)
    criterion = model.build_criterion()
    optimizer = build_optimizer(model, learning_rate)
    dataset = build_dataset(samples, labels, num_negatives, batch_size)
    model.train(dataset, epochs, optimizer, criterion)
    input_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32).reshape((1, 10))
    output = model(torch.from_numpy(input_ids))
    torch.save(model.state_dict(), "triplet_model.pth")
    predicted_embeddings = model(torch.from_numpy(samples))
    print(model.distance(output, output))
    print(model.similarity(output, output))
    print(model.cosine_distance(output, output))
    print(model.nearest_neighbors(predicted_embeddings, output, k=5))
    print(model.similar_embeddings(predicted_embeddings, output, k=5))
    model.validate(predicted_embeddings, torch.from_numpy(labels), k=5)

def main():
    np.random.seed(42)
    pipeline(1e-4, 32, 10, 5, 101, 10)

if __name__ == "__main__":
    main()