import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.utils import to_categorical
import numpy as np

class TripletNetwork(Model):
    def __init__(self, num_embeddings, embedding_dim, margin):
        super(TripletNetwork, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.embedding = layers.Embedding(num_embeddings, embedding_dim)
        self.pooling = layers.GlobalAveragePooling1D()
        self.dense = layers.Dense(embedding_dim)
        self.normalize = layers.BatchNormalization()

    def call(self, inputs, training=None, mask=None):
        embedding = self.embedding(inputs)
        pooling = self.pooling(embedding)
        dense = self.dense(pooling)
        normalize = self.normalize(dense, training=training)
        outputs = normalize / tf.norm(normalize, axis=-1, keepdims=True)
        return outputs

class TripletDataset:
    def __init__(self, samples, labels, batch_size, num_negatives):
        self.samples = samples
        self.labels = labels
        self.batch_size = batch_size
        self.num_negatives = num_negatives

    def __len__(self):
        return -(-len(self.samples) // self.batch_size)

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.samples))
        anchor_idx = np.arange(start_idx, end_idx)
        anchor_labels = self.labels[anchor_idx]

        positive_idx = np.array([np.random.choice(np.where(self.labels == label)[0], size=1)[0] for label in anchor_labels])
        negative_idx = np.array([np.random.choice(np.where(self.labels != label)[0], size=self.num_negatives, replace=False) for label in anchor_labels])

        return {
            'anchor_input_ids': self.samples[anchor_idx],
            'positive_input_ids': self.samples[positive_idx],
            'negative_input_ids': self.samples[negative_idx]
        }

def triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, margin):
    anchor_positive_distance = tf.norm(anchor_embeddings - positive_embeddings, axis=-1)
    anchor_negative_distance = tf.norm(anchor_embeddings[:, tf.newaxis] - negative_embeddings, axis=-1)
    min_anchor_negative_distance = tf.reduce_min(anchor_negative_distance, axis=-1)
    return tf.reduce_mean(tf.maximum(anchor_positive_distance - min_anchor_negative_distance + margin, 0))

class Trainer:
    def __init__(self, network, margin, lr):
        self.network = network
        self.margin = margin
        self.optimizer = optimizers.Adam(learning_rate=lr)

    def train(self, dataset, epochs):
        for epoch in range(epochs):
            total_loss = 0.0
            for i, data in enumerate(dataset):
                anchor_inputs = data['anchor_input_ids']
                positive_inputs = data['positive_input_ids']
                negative_inputs = data['negative_input_ids']

                with tf.GradientTape() as tape:
                    anchor_embeddings = self.network(anchor_inputs, training=True)
                    positive_embeddings = self.network(positive_inputs, training=True)
                    negative_embeddings = self.network(negative_inputs, training=True)
                    loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, self.margin)
                gradients = tape.gradient(loss, self.network.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
                total_loss += loss
            print(f'Epoch: {epoch+1}, Loss: {total_loss/(i+1):.3f}')

    def evaluate(self, dataset):
        total_loss = 0.0
        for i, data in enumerate(dataset):
            anchor_inputs = data['anchor_input_ids']
            positive_inputs = data['positive_input_ids']
            negative_inputs = data['negative_input_ids']

            anchor_embeddings = self.network(anchor_inputs, training=False)
            positive_embeddings = self.network(positive_inputs, training=False)
            negative_embeddings = self.network(negative_inputs, training=False)

            loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, self.margin)
            total_loss += loss
        print(f'Validation Loss: {total_loss / (i+1):.3f}')

    def predict(self, input_ids):
        return self.network(input_ids, training=False)

    def save_model(self, path):
        self.network.save(path)

    def load_model(self, path):
        return tf.keras.models.load_model(path)

def main():
    np.random.seed(42)
    samples = np.random.randint(0, 100, (100, 10))
    labels = np.random.randint(0, 2, (100,))
    batch_size = 32
    num_negatives = 5
    epochs = 10
    num_embeddings = 101
    embedding_dim = 10
    margin = 1.0
    lr = 1e-4

    network = TripletNetwork(num_embeddings, embedding_dim, margin)
    dataset = TripletDataset(samples, labels, batch_size, num_negatives)
    trainer = Trainer(network, margin, lr)
    trainer.train(dataset, epochs)
    input_ids = np.array([1, 2, 3, 4, 5])[None, :]
    output = trainer.predict(input_ids)
    print(output)
    trainer.save_model("triplet_model.h5")
    loaded_network = trainer.load_model("triplet_model.h5")
    print("Model saved and loaded successfully.")

if __name__ == "__main__":
    main()