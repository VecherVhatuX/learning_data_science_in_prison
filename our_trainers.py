import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np

class TripletDataset:
    def __init__(self, samples, labels, batch_size, num_negatives):
        self.samples = samples
        self.labels = labels
        self.batch_size = batch_size
        self.num_negatives = num_negatives

    def __call__(self, idx):
        anchor_idx = tf.range(idx * self.batch_size, (idx + 1) * self.batch_size)
        positive_idx = tf.concat([tf.random.uniform(shape=[1], minval=0, maxval=len(tf.where(self.labels == self.labels[anchor])[0]), dtype=tf.int32) for anchor in anchor_idx], axis=0)
        negative_idx = tf.random.shuffle(tf.where(self.labels != self.labels[anchor_idx])[0])[:self.batch_size * self.num_negatives]
        return {
            'anchor_input_ids': tf.gather(self.samples, anchor_idx),
            'positive_input_ids': tf.gather(self.samples, positive_idx),
            'negative_input_ids': tf.gather(self.samples, negative_idx).numpy().reshape(self.batch_size, self.num_negatives, -1)
        }

    def on_epoch_end(self):
        indices = np.random.permutation(len(self.samples))
        return tf.gather(self.samples, indices), tf.gather(self.labels, indices)


class TripletModel(models.Model):
    def __init__(self, num_embeddings, embedding_dim):
        super(TripletModel, self).__init__()
        self.embedding = layers.Embedding(input_dim=num_embeddings, output_dim=embedding_dim)
        self.pooling = layers.GlobalAveragePooling1D()

    def embed(self, input_ids):
        embeddings = self.embedding(input_ids)
        embeddings = self.pooling(embeddings)
        return embeddings / tf.norm(embeddings, axis=1, keepdims=True)

    def call(self, inputs):
        anchor_embeddings = self.embed(inputs['anchor_input_ids'])
        positive_embeddings = self.embed(inputs['positive_input_ids'])
        negative_embeddings = self.embed(tf.convert_to_tensor(inputs['negative_input_ids'], dtype=tf.int32).reshape(-1, inputs['negative_input_ids'].shape[2]))
        return anchor_embeddings, positive_embeddings, negative_embeddings


class TripletLoss:
    def __init__(self, margin=1.0):
        self.margin = margin

    def __call__(self, anchor, positive, negative):
        return tf.reduce_mean(tf.maximum(0.0, tf.norm(anchor - positive, axis=1) - tf.norm(anchor[:, tf.newaxis] - negative, axis=2) + self.margin))


class TripletTrainer:
    def __init__(self, model, loss_fn, epochs, lr, dataset):
        self.model = model
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.lr = lr
        self.dataset = dataset
        self.optimizer = optimizers.SGD(learning_rate=self.lr)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            anchor_embeddings, positive_embeddings, negative_embeddings = self.model(data)
            if len(positive_embeddings) > 0:
                loss = self.loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def train(self):
        for epoch in range(self.epochs):
            total_loss = 0
            for i in range(len(self.dataset.samples) // self.dataset.batch_size):
                data = self.dataset(i)
                loss = self.train_step(data)
                total_loss += loss
            print(f'Epoch: {epoch+1}, Loss: {total_loss/(i+1):.3f}')


class TripletEvaluator:
    def __init__(self, model, loss_fn, dataset):
        self.model = model
        self.loss_fn = loss_fn
        self.dataset = dataset

    def evaluate_step(self, data):
        anchor_embeddings, positive_embeddings, negative_embeddings = self.model(data)
        if len(positive_embeddings) > 0:
            loss = self.loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
            return loss
        return 0.0

    def evaluate(self):
        total_loss = 0.0
        for i in range(len(self.dataset.samples) // self.dataset.batch_size):
            data = self.dataset(i)
            loss = self.evaluate_step(data)
            total_loss += loss
        print(f'Validation Loss: {total_loss / (i+1):.3f}')


class TripletPredictor:
    def __init__(self, model):
        self.model = model

    def predict(self, input_ids):
        return self.model({'anchor_input_ids': input_ids})[0]


def main():
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
    trainer = TripletTrainer(model, loss_fn, epochs, lr, dataset)
    trainer.train()

    evaluator = TripletEvaluator(model, loss_fn, dataset)
    evaluator.evaluate()

    predictor = TripletPredictor(model)
    input_ids = tf.convert_to_tensor([1, 2, 3, 4, 5])
    output = predictor.predict(input_ids)
    print(output)


if __name__ == "__main__":
    main()