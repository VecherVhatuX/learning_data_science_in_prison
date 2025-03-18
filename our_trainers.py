import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

WordVectorGenerator = lambda vocab_size, embedding_size: nn.Sequential(
    nn.Embedding(vocab_size, embedding_size),
    lambda x: nn.AdaptiveAvgPool1d(1)(x.transpose(1, 2)).squeeze(2),
    nn.Linear(embedding_size, embedding_size),
    nn.BatchNorm1d(embedding_size),
    nn.LayerNorm(embedding_size)
)

TripletDataset = lambda data, labels, negative_samples: Dataset.from_tensor_slices((
    data,
    [random.choice(data[labels == label]) for label in labels],
    [random.sample(data[labels != label].tolist(), negative_samples) for label in labels]
))

calculate_triplet_loss = lambda anchor, positive, negative, margin=1.0: torch.mean(
    torch.clamp(torch.norm(anchor - positive, dim=1) - torch.min(torch.norm(anchor.unsqueeze(1) - negative, dim=2), dim=1) + margin, min=0.0)
)

train_vector_generator = lambda model, dataloader, num_epochs, learning_rate: (
    lambda optimizer, scheduler: [
        (lambda epoch_loss: (
            [optimizer.zero_grad(), (lambda loss: (loss.backward(), optimizer.step(), epoch_loss.append(loss.item()))(
                calculate_triplet_loss(model(anchor), model(positive), model(negative)) + add_custom_regularization(model)
            ) for anchor, positive, negative in dataloader],
            scheduler.step()
        )([]) for _ in range(num_epochs)
    ](optim.Adam(model.parameters(), lr=learning_rate), add_learning_rate_scheduler(optimizer)
)

evaluate_model = lambda model, data, labels, k=5: (
    lambda embeddings, metrics: (
        show_metrics(metrics),
        plot_embeddings(embeddings, labels)
    )(model(torch.tensor(data, dtype=torch.long)).detach().numpy(), compute_performance_metrics(embeddings, labels, k)
)

show_metrics = lambda metrics: (
    print(f"Accuracy: {metrics[0]:.4f}"),
    print(f"Precision: {metrics[1]:.4f}"),
    print(f"Recall: {metrics[2]:.4f}"),
    print(f"F1-score: {metrics[3]:.4f}")
)

save_model = lambda model, file_path: torch.save(model.state_dict(), file_path)

load_model = lambda model_class, file_path, vocab_size, embedding_size: (
    lambda model: model.load_state_dict(torch.load(file_path)) or model
)(model_class(vocab_size, embedding_size))

generate_embeddings = lambda model, data: model(torch.tensor(data, dtype=torch.long)).detach().numpy()

plot_embeddings = lambda embeddings, labels: (
    plt.figure(figsize=(8, 8)),
    plt.scatter(TSNE(n_components=2).fit_transform(embeddings)[:, 0], TSNE(n_components=2).fit_transform(embeddings)[:, 1], c=labels, cmap='viridis'),
    plt.colorbar(),
    plt.show()
)

compute_performance_metrics = lambda embeddings, labels, k=5: (
    lambda distances, nearest_neighbors, true_positives: (
        np.mean(np.any(labels[nearest_neighbors] == labels[:, np.newaxis], axis=1)),
        np.mean(true_positives / k),
        np.mean(true_positives / np.sum(labels == labels[:, np.newaxis], axis=1)),
        2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    )(np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2), np.argsort(distances, axis=1)[:, 1:k + 1], np.sum(labels[nearest_neighbors] == labels[:, np.newaxis], axis=1)
)

plot_loss_history = lambda loss_history: (
    plt.figure(figsize=(10, 5)),
    plt.plot(loss_history, label='Loss', color='blue'),
    plt.title('Training Loss Over Epochs'),
    plt.xlabel('Epochs'),
    plt.ylabel('Loss'),
    plt.legend(),
    plt.show()
)

run_training = lambda learning_rate, batch_size, num_epochs, negative_samples, vocab_size, embedding_size, data_size: (
    lambda data, labels, model, dataset, dataloader: (
        save_model(model, "word_vector_generator.pth"),
        plot_loss_history(train_vector_generator(model, dataloader, num_epochs, learning_rate)),
        evaluate_model(model, data, labels)
    )(*(lambda: (np.random.randint(0, 100, (data_size, 10)), np.random.randint(0, 10, data_size)))(), WordVectorGenerator(vocab_size, embedding_size), TripletDataset(data, labels, negative_samples), DataLoader(dataset, batch_size=batch_size, shuffle=True)
)

display_model_architecture = lambda model: print(model)

train_with_early_termination = lambda model, dataloader, num_epochs, learning_rate, patience=5: (
    lambda optimizer, scheduler, best_loss, no_improvement, loss_history: [
        (lambda avg_loss: (
            optimizer.zero_grad(),
            avg_loss.backward(),
            optimizer.step(),
            scheduler.step(),
            loss_history.append(avg_loss.item()),
            (lambda: (best_loss := avg_loss, no_improvement := 0) if avg_loss < best_loss else (no_improvement := no_improvement + 1))(),
            (lambda: (print(f"Early stopping at epoch {epoch}"), break)() if no_improvement >= patience else None
        ))(sum(calculate_triplet_loss(model(anchor), model(positive), model(negative)) + add_custom_regularization(model) for anchor, positive, negative in dataloader) / len(dataloader)) for epoch in range(num_epochs)
    ](optim.Adam(model.parameters(), lr=learning_rate), add_learning_rate_scheduler(optimizer), float('inf'), 0, [])
)

generate_data = lambda data_size: (np.random.randint(0, 100, (data_size, 10)), np.random.randint(0, 10, data_size))

visualize_embeddings_interactive = lambda model, data, labels: (
    lambda embeddings, tsne_result: (
        plt.figure(figsize=(8, 8)),
        plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis'),
        plt.colorbar(),
        plt.gcf().canvas.mpl_connect('button_press_event', lambda event: print(f"Clicked on point with label: {labels[np.argmin(np.linalg.norm(tsne_result - np.array([event.xdata, event.ydata]), axis=1))]}") if event.inaxes is not None else None),
        plt.show()
    )(generate_embeddings(model, data), TSNE(n_components=2).fit_transform(embeddings))
)

add_custom_regularization = lambda model, lambda_reg=0.01: lambda_reg * sum(torch.norm(param, p=2) for param in model.parameters())

add_learning_rate_scheduler = lambda optimizer, step_size=30, gamma=0.1: optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

if __name__ == "__main__":
    run_training(1e-4, 32, 10, 5, 101, 10, 100)
    display_model_architecture(WordVectorGenerator(101, 10))
    visualize_embeddings_interactive(load_model(WordVectorGenerator, "word_vector_generator.pth", 101, 10), *generate_data(100))