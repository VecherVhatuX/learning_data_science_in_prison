import os
import random
import json
import jax
import jax.numpy as jnp
from jax.experimental import stax
from jax.experimental import optimizers
from transformers import AutoTokenizer
from typing import List, Tuple

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
    def __init__(self, rng):
        self.distilbert = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        init_fn, apply_fn = stax.serial(
            stax.Dense(Config.EMBEDDING_DIM, W_init=jax.nn.initializers.zeros),
            stax.Relu(),
            stax.Dropout(Config.DROPOUT),
            stax.Dense(Config.FC_DIM, W_init=jax.nn.initializers.zeros),
            stax.Relu(),
            stax.Dropout(Config.DROPOUT)
        )
        self.params = init_fn(rng, (-1, 768))
        self.apply_fn = apply_fn

    def forward(self, params, inputs):
        anchor, positive, negative = inputs
        anchor_output = jax.jit(self.distilbert.encode_plus)(
            anchor['input_ids'], 
            max_length=Config.MAX_LENGTH, 
            padding='max_length', 
            truncation=True, 
            return_attention_mask=True, 
            return_tensors='jax'
        )
        positive_output = jax.jit(self.distilbert.encode_plus)(
            positive['input_ids'], 
            max_length=Config.MAX_LENGTH, 
            padding='max_length', 
            truncation=True, 
            return_attention_mask=True, 
            return_tensors='jax'
        )
        negative_output = jax.jit(self.distilbert.encode_plus)(
            negative['input_ids'], 
            max_length=Config.MAX_LENGTH, 
            padding='max_length', 
            truncation=True, 
            return_attention_mask=True, 
            return_tensors='jax'
        )
        anchor_embedding = self.apply_fn(params, anchor_output['input_ids'][:, 0, :])
        positive_embedding = self.apply_fn(params, positive_output['input_ids'][:, 0, :])
        negative_embedding = self.apply_fn(params, negative_output['input_ids'][:, 0, :])
        return anchor_embedding, positive_embedding, negative_embedding

    def triplet_loss(self, params, anchor, positive, negative):
        return jnp.mean(jnp.clip(jnp.linalg.norm(anchor - positive, axis=1) - jnp.linalg.norm(anchor - negative, axis=1) + 1.0, a_min=0.0))

class TripletDataset:
    def __init__(self, triplets, tokenizer):
        self.triplets = triplets
        self.tokenizer = tokenizer

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
            'anchor': anchor_encoding,
            'positive': positive_encoding,
            'negative': negative_encoding
        }

def load_json_file(file_path: str) -> List:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load JSON file: {file_path}, error: {str(e)}")
        return []

def separate_snippets(snippets: List) -> Tuple[List, List]:
    return (
        [item['snippet'] for item in snippets if item.get('is_bug', False) and item.get('snippet')],
        [item['snippet'] for item in snippets if not item.get('is_bug', False) and item.get('snippet')]
    )

def create_triplets(problem_statement: str, positive_snippets: List, negative_snippets: List, num_negatives_per_positive: int) -> List:
    return [{'anchor': problem_statement, 'positive': positive_doc, 'negative': random.choice(negative_snippets)}
            for positive_doc in positive_snippets
            for _ in range(min(num_negatives_per_positive, len(negative_snippets)))]

def create_triplet_dataset(dataset_path: str, snippet_folder_path: str) -> List:
    dataset = jnp.load(dataset_path, allow_pickle=True)
    instance_id_map = {item['instance_id']: item['problem_statement'] for item in dataset}
    folder_paths = [os.path.join(snippet_folder_path, f) for f in os.listdir(snippet_folder_path) if os.path.isdir(os.path.join(snippet_folder_path, f))]
    snippets = [load_json_file(os.path.join(folder_path, 'snippet.json')) for folder_path in folder_paths]
    bug_snippets, non_bug_snippets = zip(*[separate_snippets(snippet) for snippet in snippets])
    problem_statements = [instance_id_map.get(os.path.basename(folder_path)) for folder_path in folder_paths]
    triplets = [create_triplets(problem_statement, bug_snippets[i], non_bug_snippets[i], Config.NUM_NEGATIVES_PER_POSITIVE) 
                for i, problem_statement in enumerate(problem_statements)]
    return [item for sublist in triplets for item in sublist]

def load_data(dataset_path: str, snippet_folder_path: str, tokenizer):
    triplets = create_triplet_dataset(dataset_path, snippet_folder_path)
    random.shuffle(triplets)
    train_triplets, test_triplets = triplets[:int(0.8 * len(triplets))], triplets[int(0.8 * len(triplets)):]
    train_dataset = TripletDataset(train_triplets, tokenizer)
    test_dataset = TripletDataset(test_triplets, tokenizer)
    return train_dataset, test_dataset

def train(model, dataset, optimizer):
    total_loss = 0
    for batch in jax.tree_util.tree_leaves(dataset):
        anchor, positive, negative = batch
        anchor = jax.tree_util.tree_map(lambda x: x[0], anchor)
        positive = jax.tree_util.tree_map(lambda x: x[0], positive)
        negative = jax.tree_util.tree_map(lambda x: x[0], negative)
        grads = jax.grad(lambda params: model.triplet_loss(params, *model.forward(params, (anchor, positive, negative))))(model.params)
        optimizer = optimizers.adam(model.params, grads, optimizer)
        model.params = optimizer[0]
        total_loss += model.triplet_loss(model.params, *model.forward(model.params, (anchor, positive, negative)))
    return total_loss / len(dataset)

def evaluate(model, dataset):
    total_loss = 0
    for batch in jax.tree_util.tree_leaves(dataset):
        anchor, positive, negative = batch
        anchor = jax.tree_util.tree_map(lambda x: x[0], anchor)
        positive = jax.tree_util.tree_map(lambda x: x[0], positive)
        negative = jax.tree_util.tree_map(lambda x: x[0], negative)
        total_loss += model.triplet_loss(model.params, *model.forward(model.params, (anchor, positive, negative)))
    return total_loss / len(dataset)

def save_model(model, path):
    jnp.save(path, model.params)

def load_model(path):
    model = TripletModel(jax.random.PRNGKey(0))
    model.params = jnp.load(path)
    return model

def plot_history(history):
    import matplotlib.pyplot as plt
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()

def main():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    train_dataset, test_dataset = load_data(dataset_path, snippet_folder_path, tokenizer)
    model = TripletModel(jax.random.PRNGKey(0))
    optimizer = optimizers.adam(model.params, Config.LEARNING_RATE)
    model_path = 'triplet_model.npy'
    history = {'loss': [], 'val_loss': []}
    for epoch in range(Config.MAX_EPOCHS):
        loss = train(model, train_dataset, optimizer)
        val_loss = evaluate(model, test_dataset)
        history['loss'].append(loss)
        history['val_loss'].append(val_loss)
        print(f'Epoch {epoch+1}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}')
        save_model(model, model_path)
        print(f'Model saved at {model_path}')
    plot_history(history)

if __name__ == "__main__":
    main()