import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import SGD
import numpy as np

class Dataset:
    def __init__(self, samples, labels, batch_size, num_negatives):
        """Initialize the dataset with samples, labels, batch size and number of negatives."""
        self.samples = samples
        self.labels = labels
        self.batch_size = batch_size
        self.num_negatives = num_negatives
        self._shuffle_samples()

    def _shuffle_samples(self):
        """Shuffle the samples at the end of each epoch."""
        np.random.shuffle(self.samples)

    def _get_triplet_indices(self, idx):
        """Get the indices for the anchor, positive and negative samples."""
        batch = np.random.choice(len(self.samples), self.batch_size, replace=False)
        anchor_idx = batch
        positive_idx = np.array([np.random.choice(np.where(self.labels == self.labels[anchor])[0]) for anchor in anchor_idx])
        while np.any(positive_idx == anchor_idx):
            positive_idx = np.array([np.random.choice(np.where(self.labels == self.labels[anchor])[0]) for anchor in anchor_idx])

        negative_indices = [np.random.choice(np.where(self.labels != self.labels[anchor])[0], self.num_negatives, replace=False) for anchor in anchor_idx]
        negative_indices = [np.setdiff1d(negative_idx, [anchor]) for anchor, negative_idx in zip(anchor_idx, negative_indices)]

        return anchor_idx, positive_idx, negative_indices

    def _get_triplet_batch(self, idx):
        """Get a batch of triplet samples."""
        anchor_idx, positive_idx, negative_indices = self._get_triplet_indices(idx)
        return {
            'anchor_input_ids': self.samples[anchor_idx],
            'positive_input_ids': self.samples[positive_idx],
            'negative_input_ids': np.stack([self.samples[negative_idx] for negative_idx in negative_indices]),
        }

    def create_dataset(self):
        """Create a TensorFlow dataset from the triplet batches."""
        return tf.data.Dataset.from_generator(
            lambda: (self._get_triplet_batch(idx) for idx in range(-(-len(self.samples) // self.batch_size))),
            output_types={'anchor_input_ids': tf.int32, 'positive_input_ids': tf.int32, 'negative_input_ids': tf.int32},
            output_shapes={'anchor_input_ids': (self.batch_size, self.samples.shape[1]), 'positive_input_ids': (self.batch_size, self.samples.shape[1]), 'negative_input_ids': (self.batch_size, self.num_negatives, self.samples.shape[1])}
        ).prefetch(tf.data.AUTOTUNE)


class TripletModelTrainer:
    def __init__(self, model, optimizer, margin):
        """Initialize the trainer with the model, optimizer and margin."""
        self.model = model
        self.optimizer = optimizer
        self.margin = margin

    def _normalize_embeddings(self, embeddings):
        """Normalize the embeddings to have unit length."""
        return embeddings / tf.norm(embeddings, axis=1, keepdims=True)

    def _calculate_triplet_loss(self, anchor_embeddings, positive_embeddings, negative_embeddings, margin):
        """Calculate the triplet loss for the given embeddings."""
        return tf.reduce_mean(tf.maximum(margin + 
                                         tf.reduce_sum(tf.square(anchor_embeddings - positive_embeddings), axis=1) - 
                                         tf.reduce_sum(tf.square(anchor_embeddings - negative_embeddings[:, 0, :]), axis=1), 
                                         0.0))

    def train_step(self, inputs):
        """Perform a single training step."""
        with tf.GradientTape() as tape:
            anchor_input_ids = inputs["anchor_input_ids"]
            positive_input_ids = inputs["positive_input_ids"]
            negative_input_ids = inputs["negative_input_ids"]

            anchor_embeddings = self._normalize_embeddings(self.model(anchor_input_ids))
            positive_embeddings = self._normalize_embeddings(self.model(positive_input_ids))
            negative_embeddings = self._normalize_embeddings(self.model(negative_input_ids))

            loss = self._calculate_triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, self.margin)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return {"loss": loss}

    def train(self, dataset, epochs):
        """Train the model on the given dataset for the specified number of epochs."""
        for epoch in range(epochs):
            total_loss = 0
            for inputs in dataset:
                loss = self.train_step(inputs)
                total_loss += loss["loss"]
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataset)}")

    def save_model(self, filename):
        """Save the model to the specified file."""
        self.model.save_weights(filename)


class EmbeddingModel(models.Model):
    def __init__(self, num_embeddings, embedding_dim):
        """Initialize the embedding model with the number of embeddings and the embedding dimension."""
        super(EmbeddingModel, self).__init__()
        self.embedding = layers.Embedding(input_dim=num_embeddings, output_dim=embedding_dim)
        self.global_average_pooling = layers.GlobalAveragePooling1D()

    def call(self, inputs):
        """Call the model on the given inputs."""
        x = self.embedding(inputs)
        x = self.global_average_pooling(x)
        return x


def main():
    """Main function."""
    samples = np.random.randint(0, 100, (100, 10))
    labels = np.random.randint(0, 2, 100)
    batch_size = 32
    num_negatives = 5
    epochs = 10

    model = EmbeddingModel(100, 10)
    dataset = Dataset(samples, labels, batch_size, num_negatives).create_dataset()
    optimizer = SGD(learning_rate=1e-4)
    margin = 1.0

    trainer = TripletModelTrainer(model, optimizer, margin)
    trainer.train(dataset, epochs)
    trainer.save_model("model.h5")


if __name__ == "__main__":
    main()