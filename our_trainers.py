import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np

class TripletDataset(tf.keras.utils.Sequence):
    def __init__(self, samples, labels, batch_size, num_negatives):
        """
        Initialize the TripletDataset with samples, labels, batch size and number of negatives.
        
        Args:
            samples (numpy array): Input data.
            labels (numpy array): Labels corresponding to the input data.
            batch_size (int): Batch size for the dataset.
            num_negatives (int): Number of negative samples for each anchor.
        """
        self.samples = tf.convert_to_tensor(samples, dtype=tf.int32)
        self.labels = tf.convert_to_tensor(labels, dtype=tf.int32)
        self.batch_size = batch_size
        self.num_negatives = num_negatives

    def __len__(self):
        """
        Calculate the length of the dataset in terms of batches.
        
        Returns:
            int: Length of the dataset.
        """
        return -(-len(self.samples) // self.batch_size)

    def on_epoch_end(self):
        """
        Shuffle the dataset at the end of each epoch.
        """
        indices = np.random.permutation(len(self.samples))
        self.samples = tf.gather(self.samples, indices)
        self.labels = tf.gather(self.labels, indices)

    def __getitem__(self, idx):
        """
        Get a batch of data from the dataset.
        
        Args:
            idx (int): Index of the batch.
        
        Returns:
            dict: A dictionary containing the anchor, positive and negative input ids.
        """
        anchor_idx = tf.range(idx * self.batch_size, (idx + 1) * self.batch_size)
        positive_idx = tf.concat([tf.random.uniform(shape=[1], minval=0, maxval=len(tf.where(self.labels == self.labels[anchor])[0]), dtype=tf.int32) for anchor in anchor_idx], axis=0)
        negative_idx = tf.random.shuffle(tf.where(self.labels != self.labels[anchor_idx])[0])[:self.batch_size * self.num_negatives]
        return {
            'anchor_input_ids': tf.gather(self.samples, anchor_idx),
            'positive_input_ids': tf.gather(self.samples, positive_idx),
            'negative_input_ids': tf.gather(self.samples, negative_idx).numpy().reshape(self.batch_size, self.num_negatives, -1)
        }

class TripletModel(models.Model):
    def __init__(self, num_embeddings, embedding_dim):
        """
        Initialize the TripletModel with number of embeddings and embedding dimension.
        
        Args:
            num_embeddings (int): Number of embeddings.
            embedding_dim (int): Dimension of each embedding.
        """
        super(TripletModel, self).__init__()
        self.embedding = layers.Embedding(input_dim=num_embeddings, output_dim=embedding_dim)
        self.pooling = layers.GlobalAveragePooling1D()

    def normalize_embeddings(self, embeddings):
        """
        Normalize the embeddings to have unit length.
        
        Args:
            embeddings (tensorflow tensor): Embeddings to be normalized.
        
        Returns:
            tensorflow tensor: Normalized embeddings.
        """
        return embeddings / tf.norm(embeddings, axis=1, keepdims=True)

    def embed(self, input_ids):
        """
        Embed the input ids into a dense vector space.
        
        Args:
            input_ids (tensorflow tensor): Input ids to be embedded.
        
        Returns:
            tensorflow tensor: Embedded input ids.
        """
        embeddings = self.embedding(input_ids)
        embeddings = self.pooling(embeddings)
        return self.normalize_embeddings(embeddings)

    def call(self, inputs):
        """
        Call the TripletModel with a batch of data.
        
        Args:
            inputs (dict): A dictionary containing the anchor, positive and negative input ids.
        
        Returns:
            tuple: A tuple containing the anchor, positive and negative embeddings.
        """
        anchor_input_ids = inputs['anchor_input_ids']
        positive_input_ids = inputs['positive_input_ids']
        negative_input_ids = inputs['negative_input_ids']

        anchor_embeddings = self.embed(anchor_input_ids)
        positive_embeddings = self.embed(positive_input_ids)
        negative_embeddings = self.embed(tf.convert_to_tensor(negative_input_ids, dtype=tf.int32).reshape(-1, negative_input_ids.shape[2]))
        return anchor_embeddings, positive_embeddings, negative_embeddings

class TripletLoss(tf.keras.losses.Loss):
    def __init__(self, margin=1.0):
        """
        Initialize the TripletLoss with a margin.
        
        Args:
            margin (float, optional): Margin for the triplet loss. Defaults to 1.0.
        """
        super(TripletLoss, self).__init__()
        self.margin = margin

    def call(self, anchor, positive, negative):
        """
        Calculate the triplet loss for a batch of data.
        
        Args:
            anchor (tensorflow tensor): Anchor embeddings.
            positive (tensorflow tensor): Positive embeddings.
            negative (tensorflow tensor): Negative embeddings.
        
        Returns:
            tensorflow tensor: Triplet loss for the batch.
        """
        return tf.reduce_mean(tf.maximum(0.0, tf.norm(anchor - positive, axis=1) - tf.norm(anchor[:, tf.newaxis] - negative, axis=2) + self.margin))

class TripletTrainer:
    def __init__(self, model, loss_fn, epochs, lr):
        """
        Initialize the TripletTrainer with a model, loss function, epochs and learning rate.
        
        Args:
            model (TripletModel): Model to be trained.
            loss_fn (TripletLoss): Loss function for the model.
            epochs (int): Number of epochs to train the model.
            lr (float): Learning rate for the model.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.optimizer = optimizers.SGD(learning_rate=lr)

    def train_step(self, data):
        """
        Train the model for a single step.
        
        Args:
            data (dict): A dictionary containing the anchor, positive and negative input ids.
        
        Returns:
            tensorflow tensor: Loss for the step.
        """
        with tf.GradientTape() as tape:
            anchor_embeddings, positive_embeddings, negative_embeddings = self.model(data)
            if len(positive_embeddings) > 0:
                loss = self.loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def train(self, dataset):
        """
        Train the model for the specified number of epochs.
        
        Args:
            dataset (TripletDataset): Dataset to train the model on.
        """
        for epoch in range(self.epochs):
            total_loss = 0
            for i, data in enumerate(dataset):
                loss = self.train_step(data)
                total_loss += loss
            print(f'Epoch: {epoch+1}, Loss: {total_loss/(i+1):.3f}')

class TripletEvaluator:
    def __init__(self, model, loss_fn):
        """
        Initialize the TripletEvaluator with a model and loss function.
        
        Args:
            model (TripletModel): Model to be evaluated.
            loss_fn (TripletLoss): Loss function for the model.
        """
        self.model = model
        self.loss_fn = loss_fn

    def evaluate_step(self, data):
        """
        Evaluate the model for a single step.
        
        Args:
            data (dict): A dictionary containing the anchor, positive and negative input ids.
        
        Returns:
            float: Loss for the step.
        """
        anchor_embeddings, positive_embeddings, negative_embeddings = self.model(data)
        if len(positive_embeddings) > 0:
            loss = self.loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
            return loss
        return 0.0

    def evaluate(self, dataset):
        """
        Evaluate the model on the dataset.
        
        Args:
            dataset (TripletDataset): Dataset to evaluate the model on.
        """
        total_loss = 0.0
        for i, data in enumerate(dataset):
            loss = self.evaluate_step(data)
            total_loss += loss
        print(f'Validation Loss: {total_loss / (i+1):.3f}')

class TripletPredictor:
    def __init__(self, model):
        """
        Initialize the TripletPredictor with a model.
        
        Args:
            model (TripletModel): Model to make predictions with.
        """
        self.model = model

    def predict(self, input_ids):
        """
        Make predictions with the model.
        
        Args:
            input_ids (tensorflow tensor): Input ids to make predictions for.
        
        Returns:
            tensorflow tensor: Predictions for the input ids.
        """
        return self.model.embed(tf.convert_to_tensor(input_ids, dtype=tf.int32))

def main():
    """
    Main function to train and evaluate the TripletModel.
    """
    np.random.seed(42)
    tf.random.set_seed(42)
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

    model = TripletModel(num_embeddings, embedding_dim)
    loss_fn = TripletLoss(margin)

    trainer = TripletTrainer(model, loss_fn, epochs, lr)
    trainer.train(dataset)

    evaluator = TripletEvaluator(model, loss_fn)
    evaluator.evaluate(dataset)

    predictor = TripletPredictor(model)
    input_ids = tf.convert_to_tensor([1, 2, 3, 4, 5])
    output = predictor.predict(input_ids)
    print(output)

if __name__ == "__main__":
    main()