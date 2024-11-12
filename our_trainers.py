import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import SGD
import numpy as np
import random

class EmbeddingModel(models.Model):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = layers.Embedding(num_embeddings, embedding_dim)
        self.pooling = layers.GlobalAveragePooling1D()

    def call(self, input_ids, attention_mask=None):
        embeddings = self.embedding(input_ids)
        pooled_embeddings = self.pooling(embeddings)
        return pooled_embeddings

def create_triplet_dataset(samples, labels, batch_size, num_negatives):
    class TripletDataset:
        def __init__(self, samples, labels, batch_size, num_negatives):
            self.samples = samples
            self.labels = labels
            self.batch_size = batch_size
            self.num_negatives = num_negatives
            self.indices = list(range(len(samples)))

        def __len__(self):
            return len(self.samples) // self.batch_size

        def __getitem__(self, idx):
            batch = np.random.choice(self.indices, self.batch_size, replace=False)
            anchor_idx = batch
            positive_idx = [np.random.choice([i for i, label in enumerate(self.labels) if label == self.labels[anchor]], 1)[0] for anchor in anchor_idx]
            while np.any(positive_idx == anchor_idx):
                positive_idx = [np.random.choice([i for i, label in enumerate(self.labels) if label == self.labels[anchor]], 1)[0] for anchor in anchor_idx]

            negative_indices = [np.random.choice([i for i, label in enumerate(self.labels) if label != self.labels[anchor]], self.num_negatives) for anchor in anchor_idx]
            negative_indices = [np.setdiff1d(negative_idx, [anchor]) for anchor, negative_idx in zip(anchor_idx, negative_indices)]

            return {
                'anchor_input_ids': self.samples[anchor_idx],
                'positive_input_ids': self.samples[positive_idx],
                'negative_input_ids': np.stack([self.samples[negative_idx] for negative_idx in negative_indices]),
            }

        def on_epoch_end(self):
            random.shuffle(self.indices)
    return TripletDataset(samples, labels, batch_size, num_negatives)

def normalize_embeddings(embeddings):
    return embeddings / tf.norm(embeddings, axis=1, keepdims=True)

def triplet_margin_loss(anchor_embeddings, positive_embeddings, negative_embeddings, margin):
    return tf.reduce_mean(tf.maximum(margin + tf.reduce_sum(tf.square(anchor_embeddings - positive_embeddings), axis=1) - tf.reduce_sum(tf.square(anchor_embeddings - negative_embeddings[:, 0, :]), axis=1), 0))

class TripletLossTrainer:
    def __init__(self, model, margin, learning_rate):
        self.model = model
        self.margin = margin
        self.optimizer = SGD(learning_rate=learning_rate)

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            anchor_input_ids = inputs["anchor_input_ids"]
            positive_input_ids = inputs["positive_input_ids"]
            negative_input_ids = inputs["negative_input_ids"]

            anchor_embeddings = self.model(anchor_input_ids)
            positive_embeddings = self.model(positive_input_ids)
            negative_embeddings = self.model(negative_input_ids)

            anchor_embeddings = normalize_embeddings(anchor_embeddings)
            positive_embeddings = normalize_embeddings(positive_embeddings)
            negative_embeddings = normalize_embeddings(negative_embeddings)

            loss = triplet_margin_loss(anchor_embeddings, positive_embeddings, negative_embeddings, self.margin)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    def train(self, dataset, epochs):
        for epoch in range(epochs):
            total_loss = 0
            for i, batch in enumerate(dataset):
                loss = self.train_step(batch)
                total_loss += loss
            print(f'Epoch {epoch+1}, loss: {total_loss / len(dataset)}')

    def save_model(self, path):
        self.model.save_weights(path)

    def load_model(self, path):
        self.model.load_weights(path)

def main():
    samples = np.random.randint(0, 100, (100, 10))
    labels = np.random.randint(0, 2, 100)
    batch_size = 32
    num_negatives = 5
    epochs = 10

    model = EmbeddingModel(100, 10)
    dataset = create_triplet_dataset(samples, labels, batch_size, num_negatives)
    trainer = TripletLossTrainer(model, 1.0, 1e-4)
    trainer.train(dataset, epochs)
    trainer.save_model("model.h5")

if __name__ == "__main__":
    main()