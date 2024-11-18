import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Lambda, Dense, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import numpy as np

class TripletNetwork(Model):
    """
    A neural network designed to learn embeddings by optimizing a triplet loss function.
    """
    def __init__(self, num_embeddings, embedding_dim, margin):
        """
        Initializes the TripletNetwork model.

        Args:
            num_embeddings (int): The number of unique embeddings to learn.
            embedding_dim (int): The dimensionality of each embedding.
            margin (float): The margin value used in the triplet loss function.
        """
        super(TripletNetwork, self).__init__()
        self.margin = margin
        self.embedding = Embedding(num_embeddings, embedding_dim, input_length=10)
        self.pooling = GlobalAveragePooling1D()
        self.dense = Dense(embedding_dim)
        self.normalize = Lambda(lambda x: x / K.linalg.norm(x, axis=-1, keepdims=True))

    def call(self, inputs, training=None):
        """
        Defines the forward pass of the model.

        Args:
            inputs: The input data to the model.
            training: A boolean indicating whether the model is in training mode.

        Returns:
            The learned embeddings for the input data.
        """
        x = self.embedding(inputs)
        x = self.pooling(x)
        x = self.dense(x)
        x = self.normalize(x)
        return x

def triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, margin):
    """
    Computes the triplet loss for the given embeddings.

    Args:
        anchor_embeddings: The embeddings of the anchor samples.
        positive_embeddings: The embeddings of the positive samples.
        negative_embeddings: The embeddings of the negative samples.
        margin: The margin value used in the triplet loss function.

    Returns:
        The triplet loss value.
    """
    anchor_positive_distance = tf.norm(anchor_embeddings - positive_embeddings, axis=-1)
    anchor_negative_distance = tf.norm(anchor_embeddings[:, None] - negative_embeddings, axis=-1)

    min_anchor_negative_distance = tf.reduce_min(anchor_negative_distance, axis=-1)
    return tf.reduce_mean(tf.maximum(0.0, anchor_positive_distance - min_anchor_negative_distance + margin))

class TripletDataset:
    """
    A dataset class designed to generate batches of triplets.
    """
    def __init__(self, samples, labels, batch_size, num_negatives):
        """
        Initializes the TripletDataset.

        Args:
            samples: The input data samples.
            labels: The corresponding labels for the samples.
            batch_size: The batch size to use when generating triplets.
            num_negatives: The number of negative samples to include in each triplet.
        """
        self.samples = samples
        self.labels = labels
        self.batch_size = batch_size
        self.num_negatives = num_negatives

    def __len__(self):
        """
        Returns the number of batches in the dataset.
        """
        return -(-len(self.samples) // self.batch_size)

    def __getitem__(self, idx):
        """
        Returns a batch of triplets.

        Args:
            idx: The index of the batch to retrieve.

        Returns:
            A batch of triplets.
        """
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

    def get_samples(self):
        """
        Returns the input data samples.
        """
        return self.samples

    def get_labels(self):
        """
        Returns the corresponding labels.
        """
        return self.labels

    def get_batch_size(self):
        """
        Returns the batch size.
        """
        return self.batch_size

    def get_num_negatives(self):
        """
        Returns the number of negative samples per anchor.
        """
        return self.num_negatives

    def print_info(self):
        """
        Prints information about the dataset.
        """
        print("Dataset Information:")
        print(f"  Number of Samples: {self.samples.shape}")
        print(f"  Number of Labels: {self.labels.shape}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Number of Negatives: {self.num_negatives}")

class TripletModel:
    """
    A model class designed to train and evaluate a triplet network.
    """
    def __init__(self, num_embeddings, embedding_dim, margin, lr, device):
        """
        Initializes the TripletModel.

        Args:
            num_embeddings (int): The number of unique embeddings to learn.
            embedding_dim (int): The dimensionality of each embedding.
            margin (float): The margin value used in the triplet loss function.
            lr (float): The learning rate to use for training.
            device (str): The device to use for training.
        """
        self.network = TripletNetwork(num_embeddings, embedding_dim, margin)
        self.margin = margin
        self.lr = lr
        self.device = device
        self.optimizer = Adam(learning_rate=lr)

    def train(self, dataset, epochs):
        """
        Trains the model on the given dataset.

        Args:
            dataset: The dataset to train on.
            epochs (int): The number of epochs to train for.
        """
        for epoch in range(epochs):
            total_loss = 0.0
            for i, data in enumerate(dataset):
                with tf.device(self.device):
                    anchor_inputs = data['anchor_input_ids']
                    positive_inputs = data['positive_input_ids']
                    negative_inputs = data['negative_input_ids']

                    with tf.GradientTape() as tape:
                        anchor_embeddings = self.network(anchor_inputs, training=True)
                        positive_embeddings = self.network(positive_inputs, training=True)
                        negative_embeddings = self.network(negative_inputs, training=True)
                        loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, self.margin)
                    grads = tape.gradient(loss, self.network.trainable_variables)
                    self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))
                    total_loss += loss.numpy()
            print(f'Epoch: {epoch+1}, Loss: {total_loss/(i+1):.3f}')

    def evaluate(self, dataset):
        """
        Evaluates the model on the given dataset.

        Args:
            dataset: The dataset to evaluate on.
        """
        total_loss = 0.0
        for i, data in enumerate(dataset):
            anchor_inputs = data['anchor_input_ids']
            positive_inputs = data['positive_input_ids']
            negative_inputs = data['negative_input_ids']

            anchor_embeddings = self.network(anchor_inputs)
            positive_embeddings = self.network(positive_inputs)
            negative_embeddings = self.network(negative_inputs)

            loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, self.margin)
            total_loss += loss.numpy()
        print(f'Validation Loss: {total_loss / (i+1):.3f}')

    def predict(self, input_ids):
        """
        Makes predictions on the given input data.

        Args:
            input_ids: The input data to make predictions on.

        Returns:
            The learned embeddings for the input data.
        """
        return self.network(input_ids)

    def save_model(self, path):
        """
        Saves the model to the given path.

        Args:
            path (str): The path to save the model to.
        """
        self.network.save(path)

    def load_model(self, path):
        """
        Loads the model from the given path.

        Args:
            path (str): The path to load the model from.
        """
        self.network = tf.keras.models.load_model(path)

def main():
    np.random.seed(42)
    tf.random.set_seed(42)
    device = '/gpu:0' if tf.config.list_physical_devices('GPU') else '/cpu:0'
    samples = np.random.randint(0, 100, (100, 10))
    labels = np.random.randint(0, 2, (100,))
    batch_size = 32
    num_negatives = 5
    epochs = 10
    num_embeddings = 101
    embedding_dim = 10
    margin = 1.0
    lr = 1e-4

    dataset = TripletDataset(samples, labels, batch_size, num_negatives)

    model = TripletModel(num_embeddings, embedding_dim, margin, lr, device)
    model.train(dataset, epochs)
    input_ids = tf.constant([1, 2, 3, 4, 5], dtype=tf.int32)[None, :]
    output = model.predict(input_ids)
    print(output)
    model.save_model("triplet_model.h5")
    model.load_model("triplet_model.h5")
    print("Model saved and loaded successfully.")

if __name__ == "__main__":
    main()