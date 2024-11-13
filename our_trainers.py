import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class TripletData:
    def __init__(self, samples, labels, batch_size, num_negatives):
        self.samples = samples
        self.labels = labels
        self.batch_size = batch_size
        self.num_negatives = num_negatives
        self.indices = np.arange(len(samples))

    def __len__(self):
        return (len(self.samples) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        np.random.shuffle(self.indices)
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.samples))
        batch = self.indices[start_idx:end_idx]
        anchor_idx = batch
        positive_idx = np.array([np.random.choice(np.where(self.labels == self.labels[anchor])[0]) for anchor in anchor_idx])
        while np.any(positive_idx == anchor_idx):
            positive_idx = np.array([np.random.choice(np.where(self.labels == self.labels[anchor])[0]) for anchor in anchor_idx])

        negative_indices = [np.random.choice(np.where(self.labels != self.labels[anchor])[0], self.num_negatives, replace=False) for anchor in anchor_idx]
        negative_indices = [np.setdiff1d(negative_idx, [anchor]) for anchor, negative_idx in zip(anchor_idx, negative_indices)]
        anchor_input_ids = tf.constant(self.samples[anchor_idx])
        positive_input_ids = tf.constant(self.samples[positive_idx])
        negative_input_ids = tf.stack([tf.constant(self.samples[negative_idx]) for negative_idx in negative_indices])
        return {
            'anchor_input_ids': anchor_input_ids,
            'positive_input_ids': positive_input_ids,
            'negative_input_ids': negative_input_ids
        }

class TripletLossModel(tf.keras.Model):
    def __init__(self, num_embeddings, embedding_dim):
        super(TripletLossModel, self).__init__()
        self.embedding = layers.Embedding(num_embeddings, embedding_dim)
        self.pooling = layers.GlobalAveragePooling1D()

    def call(self, x):
        x = self.embedding(x)
        x = self.pooling(x)
        return x

    def standardize_vectors(self, embeddings):
        return embeddings / tf.norm(embeddings, axis=1, keepdims=True)

    def triplet_loss(self, anchor_embeddings, positive_embeddings, negative_embeddings, margin):
        return tf.reduce_mean(tf.maximum(margin + 
                                         tf.reduce_sum((anchor_embeddings - positive_embeddings) ** 2, axis=1) - 
                                         tf.reduce_sum((anchor_embeddings - negative_embeddings[:, 0, :]) ** 2, axis=1), 
                                         0.0))

def train_model(model, dataset, optimizer, margin, epochs):
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataset:
            with tf.GradientTape() as tape:
                anchor_input_ids = batch["anchor_input_ids"]
                positive_input_ids = batch["positive_input_ids"]
                negative_input_ids = batch["negative_input_ids"]
                anchor_embeddings = model.standardize_vectors(model(anchor_input_ids))
                positive_embeddings = model.standardize_vectors(model(positive_input_ids))
                negative_embeddings = model.standardize_vectors(model(negative_input_ids))
                loss = model.triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, margin)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            total_loss += loss.numpy()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataset)}")

def persist_model(model, filename):
    model.save_weights(filename)

def main():
    samples = np.random.randint(0, 100, (100, 10))
    labels = np.random.randint(0, 2, 100)
    batch_size = 32
    num_negatives = 5
    epochs = 10

    model = TripletLossModel(101, 10)
    dataset = TripletData(samples, labels, batch_size, num_negatives)
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4)
    margin = 1.0

    train_model(model, dataset, optimizer, margin, epochs)
    persist_model(model, "model")

if __name__ == "__main__":
    main()