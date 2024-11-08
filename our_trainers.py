import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.optimizers import SGD
import numpy as np
import random

class EmbeddingModel(models.Model):
    def __init__(self, num_embeddings, embedding_dim):
        super(EmbeddingModel, self).__init__()
        self.embedding = layers.Embedding(num_embeddings, embedding_dim)

    def call(self, input_ids, attention_mask):
        return self.embedding(input_ids)

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
            'anchor_attention_mask': np.ones_like(self.samples[anchor_idx], dtype=np.int32),
            'positive_input_ids': self.samples[positive_idx],
            'positive_attention_mask': np.ones_like(self.samples[positive_idx], dtype=np.int32),
            'negative_input_ids': np.stack([self.samples[i] for i in negative_indices]),
            'negative_attention_mask': np.ones_like(np.stack([self.samples[i] for i in negative_indices]), dtype=np.int32)
        }

    def on_epoch_end(self):
        random.shuffle(self.indices)

class TripletLossTrainer:
    def __init__(self, model, triplet_margin, learning_rate):
        self.model = model
        self.triplet_margin = triplet_margin
        self.optimizer = SGD(learning_rate=learning_rate)

    def mean_pooling(self, hidden_state, attention_mask):
        input_mask_expanded = tf.expand_dims(attention_mask, 1)
        input_mask_expanded = tf.broadcast_to(input_mask_expanded, tf.shape(hidden_state))
        sum_embeddings = tf.reduce_sum(hidden_state * tf.cast(input_mask_expanded, tf.float32), axis=1)
        sum_mask = tf.reduce_sum(tf.cast(input_mask_expanded, tf.float32), axis=1)
        sum_mask = tf.maximum(sum_mask, 1e-9)
        return sum_embeddings / sum_mask

    def normalize_embeddings(self, embeddings):
        return embeddings / tf.norm(embeddings, axis=1, keepdims=True)

    def triplet_margin_loss(self, anchor_embeddings, positive_embeddings, negative_embeddings):
        return tf.reduce_mean(tf.maximum(self.triplet_margin + tf.reduce_sum(tf.square(anchor_embeddings - positive_embeddings), axis=1) - tf.reduce_sum(tf.square(anchor_embeddings - negative_embeddings), axis=1), 0))

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            anchor_input_ids = inputs["anchor_input_ids"]
            anchor_attention_mask = inputs["anchor_attention_mask"]
            positive_input_ids = inputs["positive_input_ids"]
            positive_attention_mask = inputs["positive_attention_mask"]
            negative_input_ids = inputs["negative_input_ids"]
            negative_attention_mask = inputs["negative_attention_mask"]

            anchor_outputs = self.model(anchor_input_ids, attention_mask=anchor_attention_mask)
            positive_outputs = self.model(positive_input_ids, attention_mask=positive_attention_mask)
            negative_outputs = self.model(negative_input_ids, attention_mask=negative_attention_mask)

            anchor_embeddings = self.mean_pooling(anchor_outputs, anchor_attention_mask)
            positive_embeddings = self.mean_pooling(positive_outputs, positive_attention_mask)
            negative_embeddings = self.mean_pooling(negative_outputs, negative_attention_mask)

            anchor_embeddings = self.normalize_embeddings(anchor_embeddings)
            positive_embeddings = self.normalize_embeddings(positive_embeddings)
            negative_embeddings = self.normalize_embeddings(negative_embeddings)

            loss = self.triplet_margin_loss(anchor_embeddings, positive_embeddings, negative_embeddings)

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
                    'anchor_attention_mask': np.stack([x['anchor_attention_mask'] for x in batch]),
                    'positive_input_ids': np.stack([x['positive_input_ids'] for x in batch]),
                    'positive_attention_mask': np.stack([x['positive_attention_mask'] for x in batch]),
                    'negative_input_ids': np.stack([x['negative_input_ids'] for x in batch]),
                    'negative_attention_mask': np.stack([x['negative_attention_mask'] for x in batch]),
                }
                loss = self.train_step(inputs)
                total_loss += loss
            print(f'Epoch {epoch+1}, loss: {total_loss / (len(dataset) // batch_size)}')

    def save_model(self, path):
        self.model.save_weights(path)

    def load_model(self, path):
        self.model.load_weights(path)

def main():
    samples = np.random.rand(100, 10)
    labels = np.random.randint(0, 2, 100)
    batch_size = 32
    num_negatives = 5
    epochs = 10

    model = EmbeddingModel(100, 10)
    dataset = TripletDataset(samples, labels, batch_size, num_negatives)
    trainer = TripletLossTrainer(model, 1.0, 1e-4)
    trainer.train(dataset, epochs, batch_size)
    trainer.save_model("model.h5")

if __name__ == "__main__":
    main()