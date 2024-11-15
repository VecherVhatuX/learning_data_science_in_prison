import os
import random
import json
import jax
import jax.numpy as jnp
from jax.experimental import stax
from jax.experimental import optimizers
from transformers import AutoTokenizer
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np

class Config:
    INSTANCE_ID_FIELD = 'instance_id'
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    NUM_NEGATIVES_PER_POSITIVE = 3
    EMBEDDING_DIM = 128
    FC_DIM = 64
    DROPOUT = 0.2
    LEARNING_RATE = 1e-5
    MAX_EPOCHS = 5

class TripletModel:
    def __init__(self, rng: jax.random.PRNGKey, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.optimizer_init, self.optimizer_update, self.optimizer_get_params = optimizers.adam(Config.LEARNING_RATE)
        self.params = None
        self.optimizer_state = None

        self.model_init, self.model_apply = self._create_model(rng)

    def _create_model(self, rng):
        init_fn, apply_fn = stax.serial(
            stax.Dense(Config.EMBEDDING_DIM, W_init=jax.nn.initializers.zeros),
            stax.Relu(),
            stax.Dropout(Config.DROPOUT),
            stax.Dense(Config.FC_DIM, W_init=jax.nn.initializers.zeros),
            stax.Relu(),
            stax.Dropout(Config.DROPOUT)
        )
        return init_fn, apply_fn

    def init(self, rng, input_shape):
        output_shape, self.params = self.model_init(rng, input_shape)
        self.optimizer_state = self.optimizer_init(self.params)
        return output_shape

    def encode(self, inputs):
        encoding = self.tokenizer.encode_plus(
            inputs, 
            max_length=Config.MAX_LENGTH, 
            padding='max_length', 
            truncation=True, 
            return_attention_mask=True, 
            return_tensors='jax'
        )
        return encoding['input_ids'][:, 0, :]

    def forward(self, anchor, positive, negative):
        anchor_embedding = self.model_apply(self.params, self.encode(anchor))
        positive_embedding = self.model_apply(self.params, self.encode(positive))
        negative_embedding = self.model_apply(self.params, self.encode(negative))
        return anchor_embedding, positive_embedding, negative_embedding

    def loss(self, anchor, positive, negative):
        return jnp.mean(jnp.clip(jnp.linalg.norm(anchor - positive, axis=1) - jnp.linalg.norm(anchor - negative, axis=1) + 1.0, a_min=0.0))

    def update(self, grads):
        self.optimizer_state = self.optimizer_update(0, grads, self.optimizer_state)
        self.params = self.optimizer_get_params(self.optimizer_state)

class TripletDataset:
    def __init__(self, triplets, tokenizer):
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.batch_size = Config.BATCH_SIZE
        self.indices = list(range(len(self.triplets)))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        triplet = self.triplets[index]
        anchor_encoding = self.tokenizer.encode_plus(
            triplet['anchor'],
            max_length=Config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='jax'
        )
        positive_encoding = self.tokenizer.encode_plus(
            triplet['positive'],
            max_length=Config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='jax'
        )
        negative_encoding = self.tokenizer.encode_plus(
            triplet['negative'],
            max_length=Config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='jax'
        )
        return {
            'anchor': anchor_encoding['input_ids'][:, 0, :],
            'positive': positive_encoding['input_ids'][:, 0, :],
            'negative': negative_encoding['input_ids'][:, 0, :]
        }

    def shuffle(self):
        random.shuffle(self.indices)

    def batch(self):
        self.shuffle()
        for i in range(0, len(self), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            batch = [self.__getitem__(index) for index in batch_indices]
            anchors = np.stack([item['anchor'] for item in batch])
            positives = np.stack([item['positive'] for item in batch])
            negatives = np.stack([item['negative'] for item in batch])
            yield anchors, positives, negatives

def load_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load JSON file: {file_path}, error: {str(e)}")
        return []

def separate_snippets(snippets):
    return (
        [item['snippet'] for item in snippets if item.get('is_bug', False) and item.get('snippet')],
        [item['snippet'] for item in snippets if not item.get('is_bug', False) and item.get('snippet')]
    )

def create_triplets(problem_statement, positive_snippets, negative_snippets, num_negatives_per_positive):
    return [{'anchor': problem_statement, 'positive': positive_doc, 'negative': random.choice(negative_snippets)}
            for positive_doc in positive_snippets
            for _ in range(min(num_negatives_per_positive, len(negative_snippets)))]

def create_triplet_dataset(dataset_path, snippet_folder_path):
    dataset = np.load(dataset_path, allow_pickle=True)
    instance_id_map = {item['instance_id']: item['problem_statement'] for item in dataset}
    folder_paths = [os.path.join(snippet_folder_path, f) for f in os.listdir(snippet_folder_path) if os.path.isdir(os.path.join(snippet_folder_path, f))]
    snippets = [load_json_file(os.path.join(folder_path, 'snippet.json')) for folder_path in folder_paths]
    bug_snippets, non_bug_snippets = zip(*[separate_snippets(snippet) for snippet in snippets])
    problem_statements = [instance_id_map.get(os.path.basename(folder_path)) for folder_path in folder_paths]
    triplets = [create_triplets(problem_statement, bug_snippets[i], non_bug_snippets[i], Config.NUM_NEGATIVES_PER_POSITIVE) 
                for i, problem_statement in enumerate(problem_statements)]
    return [item for sublist in triplets for item in sublist]

def load_data(dataset_path, snippet_folder_path, tokenizer):
    triplets = create_triplet_dataset(dataset_path, snippet_folder_path)
    random.shuffle(triplets)
    train_triplets, test_triplets = triplets[:int(0.8 * len(triplets))], triplets[int(0.8 * len(triplets)):]
    train_dataset = TripletDataset(train_triplets, tokenizer)
    test_dataset = TripletDataset(test_triplets, tokenizer)
    return train_dataset, test_dataset

def train(model, dataset):
    total_loss = 0
    for batch in dataset.batch():
        anchor, positive, negative = batch
        anchor_embedding, positive_embedding, negative_embedding = model.forward(anchor, positive, negative)
        loss = model.loss(anchor_embedding, positive_embedding, negative_embedding)
        grads = jax.grad(lambda params: model.loss(model.model_apply(params, anchor), model.model_apply(params, positive), model.model_apply(params, negative)))(model.params)
        model.update(grads)
        total_loss += loss
    return total_loss / len(dataset)

def evaluate(model, dataset):
    total_loss = 0
    for batch in dataset.batch():
        anchor, positive, negative = batch
        anchor_embedding, positive_embedding, negative_embedding = model.forward(anchor, positive, negative)
        total_loss += model.loss(anchor_embedding, positive_embedding, negative_embedding)
    return total_loss / len(dataset)

def save_model(model, path):
    np.save(path, model.params)

def load_model(path, tokenizer):
    model = TripletModel(jax.random.PRNGKey(0), tokenizer)
    model.params = np.load(path)
    return model

def plot_history(history):
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()

def main():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    train_dataset, test_dataset = load_data(dataset_path, snippet_folder_path, tokenizer)
    model = TripletModel(jax.random.PRNGKey(0), tokenizer)
    model.init(jax.random.PRNGKey(0), (-1, 768))
    model_path = 'triplet_model.npy'
    history = {'loss': [], 'val_loss': []}
    for epoch in range(Config.MAX_EPOCHS):
        loss = train(model, train_dataset)
        val_loss = evaluate(model, test_dataset)
        history['loss'].append(loss)
        history['val_loss'].append(val_loss)
        print(f'Epoch {epoch+1}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}')
        save_model(model, model_path)
        print(f'Model saved at {model_path}')
    plot_history(history)

if __name__ == "__main__":
    main()