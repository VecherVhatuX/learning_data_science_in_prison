import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np

def create_triplet_dataset(samples, labels, batch_size, num_negatives):
    def dataset(idx):
        anchor_idx = tf.range(idx * batch_size, (idx + 1) * batch_size)
        positive_idx = tf.concat([tf.random.uniform(shape=[1], minval=0, maxval=len(tf.where(labels == labels[anchor])[0]), dtype=tf.int32) for anchor in anchor_idx], axis=0)
        negative_idx = tf.random.shuffle(tf.where(labels != labels[anchor_idx])[0])[:batch_size * num_negatives]
        return {
            'anchor_input_ids': tf.gather(samples, anchor_idx),
            'positive_input_ids': tf.gather(samples, positive_idx),
            'negative_input_ids': tf.gather(samples, negative_idx).numpy().reshape(batch_size, num_negatives, -1)
        }

    def on_epoch_end():
        indices = np.random.permutation(len(samples))
        return tf.gather(samples, indices), tf.gather(labels, indices)

    return dataset, on_epoch_end

def create_triplet_model(num_embeddings, embedding_dim):
    def model(inputs):
        embedding = layers.Embedding(input_dim=num_embeddings, output_dim=embedding_dim)
        pooling = layers.GlobalAveragePooling1D()
        def embed(input_ids):
            embeddings = embedding(input_ids)
            embeddings = pooling(embeddings)
            return embeddings / tf.norm(embeddings, axis=1, keepdims=True)
        anchor_embeddings = embed(inputs['anchor_input_ids'])
        positive_embeddings = embed(inputs['positive_input_ids'])
        negative_embeddings = embed(tf.convert_to_tensor(inputs['negative_input_ids'], dtype=tf.int32).reshape(-1, inputs['negative_input_ids'].shape[2]))
        return anchor_embeddings, positive_embeddings, negative_embeddings
    return model

def create_triplet_loss(margin=1.0):
    def loss(anchor, positive, negative):
        return tf.reduce_mean(tf.maximum(0.0, tf.norm(anchor - positive, axis=1) - tf.norm(anchor[:, tf.newaxis] - negative, axis=2) + margin))
    return loss

def create_triplet_trainer(model, loss_fn, epochs, lr):
    def train_step(data):
        with tf.GradientTape() as tape:
            anchor_embeddings, positive_embeddings, negative_embeddings = model(data)
            if len(positive_embeddings) > 0:
                loss = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer = optimizers.SGD(learning_rate=lr)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def train(dataset):
        for epoch in range(epochs):
            total_loss = 0
            for i, data in enumerate(dataset):
                loss = train_step(data)
                total_loss += loss
            print(f'Epoch: {epoch+1}, Loss: {total_loss/(i+1):.3f}')
    return train

def create_triplet_evaluator(model, loss_fn):
    def evaluate_step(data):
        anchor_embeddings, positive_embeddings, negative_embeddings = model(data)
        if len(positive_embeddings) > 0:
            loss = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
            return loss
        return 0.0

    def evaluate(dataset):
        total_loss = 0.0
        for i, data in enumerate(dataset):
            loss = evaluate_step(data)
            total_loss += loss
        print(f'Validation Loss: {total_loss / (i+1):.3f}')
    return evaluate

def create_triplet_predictor(model):
    def predict(input_ids):
        return model({'anchor_input_ids': input_ids})[0]
    return predict

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

    dataset, _ = create_triplet_dataset(samples, labels, batch_size, num_negatives)
    model = create_triplet_model(num_embeddings, embedding_dim)
    loss_fn = create_triplet_loss(margin)
    train = create_triplet_trainer(model, loss_fn, epochs, lr)
    train([dataset(i) for i in range(len(samples) // batch_size)])

    evaluate = create_triplet_evaluator(model, loss_fn)
    evaluate([dataset(i) for i in range(len(samples) // batch_size)])

    predictor = create_triplet_predictor(model)
    input_ids = tf.convert_to_tensor([1, 2, 3, 4, 5])
    output = predictor(input_ids)
    print(output)

if __name__ == "__main__":
    main()