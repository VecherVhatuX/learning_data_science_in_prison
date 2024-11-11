import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import SGD
import numpy as np
import random

def create_embedding_model(num_embeddings, embedding_dim):
    class EmbeddingModel(models.Model):
        def __init__(self):
            super().__init__()
            self.embedding = layers.Embedding(num_embeddings, embedding_dim)

        def call(self, input_ids, attention_mask=None):
            return self.embedding(input_ids)
    return EmbeddingModel()

def create_triplet_dataset(samples, labels, batch_size, num_negatives):
    class TripletDataset:
        def __init__(self, samples, labels, batch_size, num_negatives):
            self.samples = samples
            self.labels = labels
            self.batch_size = batch_size
            self.num_negatives = num_negatives
            self.indices = list(range(len(samples)))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            anchor_idx = idx
            positive_idx = random.choice([i for i, label in enumerate(self.labels) if label == self.labels[anchor_idx]])
            while positive_idx == anchor_idx:
                positive_idx = random.choice([i for i, label in enumerate(self.labels) if label == self.labels[anchor_idx]])

            negative_indices = random.sample([i for i, label in enumerate(self.labels) if label != self.labels[anchor_idx]], self.num_negatives)
            negative_indices = [i for i in negative_indices if i != anchor_idx]

            return {
                'anchor_input_ids': self.samples[anchor_idx],
                'positive_input_ids': self.samples[positive_idx],
                'negative_input_ids': np.stack([self.samples[i] for i in negative_indices]),
            }

        def on_epoch_end(self):
            random.shuffle(self.indices)
    return TripletDataset(samples, labels, batch_size, num_negatives)

def mean_pooling(hidden_state, attention_mask):
    input_mask_expanded = tf.expand_dims(attention_mask, 1)
    input_mask_expanded = tf.broadcast_to(input_mask_expanded, tf.shape(hidden_state))
    sum_embeddings = tf.reduce_sum(hidden_state * tf.cast(input_mask_expanded, tf.float32), axis=1)
    sum_mask = tf.reduce_sum(tf.cast(input_mask_expanded, tf.float32), axis=1)
    sum_mask = tf.maximum(sum_mask, 1e-9)
    return sum_embeddings / sum_mask

def normalize_embeddings(embeddings):
    return embeddings / tf.norm(embeddings, axis=1, keepdims=True)

def triplet_margin_loss(anchor_embeddings, positive_embeddings, negative_embeddings, margin):
    return tf.reduce_mean(tf.maximum(margin + tf.reduce_sum(tf.square(anchor_embeddings - positive_embeddings), axis=1) - tf.reduce_sum(tf.square(anchor_embeddings - negative_embeddings), axis=1), 0))

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

            anchor_outputs = self.model(anchor_input_ids)
            positive_outputs = self.model(positive_input_ids)
            negative_outputs = self.model(negative_input_ids)

            anchor_embeddings = tf.reduce_mean(anchor_outputs, axis=1)
            positive_embeddings = tf.reduce_mean(positive_outputs, axis=1)
            negative_embeddings = tf.reduce_mean(negative_outputs, axis=1)

            anchor_embeddings = normalize_embeddings(anchor_embeddings)
            positive_embeddings = normalize_embeddings(positive_embeddings)
            negative_embeddings = normalize_embeddings(negative_embeddings)

            loss = triplet_margin_loss(anchor_embeddings, positive_embeddings, negative_embeddings, self.margin)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    def train(self, dataset, epochs, batch_size):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(dataset) // batch_size):
                batch = [dataset[j] for j in range(i * batch_size, (i + 1) * batch_size)]
                inputs = {
                    'anchor_input_ids': np.stack([x['anchor_input_ids'] for x in batch]),
                    'positive_input_ids': np.stack([x['positive_input_ids'] for x in batch]),
                    'negative_input_ids': np.stack([x['negative_input_ids'] for x in batch]),
                }
                loss = self.train_step(inputs)
                total_loss += loss
            print(f'Epoch {epoch+1}, loss: {total_loss / (len(dataset) // batch_size)}')

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

    model = create_embedding_model(100, 10)
    dataset = create_triplet_dataset(samples, labels, batch_size, num_negatives)
    trainer = TripletLossTrainer(model, 1.0, 1e-4)
    trainer.train(dataset, epochs, batch_size)
    trainer.save_model("model.h5")

if __name__ == "__main__":
    main()