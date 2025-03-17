import os
import json
import random
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from transformers import T5Tokenizer

ModelConfigurations = lambda: {
    "model_name": "t5-base",
    "alpha": 16,
    "dropout_rate": 0.1,
    "decomposition_rank": 64,
    "layers_to_modify": ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
    "quantization_config": {
        "nested_quantization": False, "four_bit_dtype": "float16", "storage_dtype": "uint8",
        "quantization_strategy": "nf4", "flash_attention": False, "low_rank_peft": False,
        "eight_bit_quantization": False, "four_bit_quantization": False, "reentrant_training": False,
        "unsloth_training": False,
    },
    "use_triplet_loss": True,
    "data_source": "timdettmers/openassistant-guanaco",
    "token_flags": {"use_special_token": False, "include_special_tokens": False},
    "data_splits": ["train", "test"],
    "tokenized_file": None,
    "results_dir": "./results",
    "epochs": 3,
    "batch_sizes": {"train": 16, "eval": 64},
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "logging_dir": "./logs",
    "model_save_interval": 500,
    "max_checkpoints": 2,
    "seed": 42,
    "checkpoint_path": None,
    "negative_samples_per_batch": 5
}

TextEmbeddingModel = lambda embedding_dim, vocab_size: tf.keras.Sequential([
    layers.Embedding(vocab_size, embedding_dim),
    layers.LSTM(embedding_dim, return_sequences=True),
    lambda x: x[:, -1, :],
    layers.Dense(embedding_dim),
    layers.Dense(vocab_size)
])

calculate_triplet_loss = lambda anchor, positive, negative: tf.reduce_mean(
    tf.maximum(tf.reduce_mean(tf.square(anchor - positive), axis=-1) - 
    tf.reduce_mean(tf.square(anchor - negative), axis=-1) + 2.0, 0.0)
)

DatasetHandler = lambda data, config, tokenizer: {
    "data": data,
    "config": config,
    "tokenizer": tokenizer,
    "indices": list(range(len(data))),
    "shuffle_dataset": lambda self: random.shuffle(self["indices"]),
    "fetch_sample": lambda self, idx: (
        self["tokenizer"].encode(self["data"][self["indices"][idx]]['input'], max_length=512, padding='max_length', truncation=True),
        self["tokenizer"].encode(self["data"][self["indices"][idx]]['output'], max_length=512, padding='max_length', truncation=True),
        self["fetch_negative_samples"](idx)
    ),
    "fetch_negative_samples": lambda self, idx: [
        self["tokenizer"].encode(self["data"][random.choice([j for j in range(len(self["data"])) if j != self["indices"][idx]])]['input'],
        max_length=512, padding='max_length', truncation=True) 
        for _ in range(self["config"]["negative_samples_per_batch"])
    ],
    "create_batch": lambda self: [self["fetch_sample"](i) for i in range(len(self["data"]))]
}

DataLoaderAdapter = lambda dataset, config, tokenizer: {
    "dataset": DatasetHandler(dataset, config, tokenizer),
    "batch_size": config["batch_sizes"]['train'],
    "__len__": lambda self: len(self["dataset"]["data"]) // self["batch_size"],
    "__getitem__": lambda self, index: tuple(tf.convert_to_tensor(x) for x in zip(*self["dataset"]["create_batch"]()[index * self["batch_size"]:(index + 1) * self["batch_size"]]))
}

load_data = lambda file_path: json.load(open(file_path, 'r')) if os.path.exists(file_path) else print(f"File not found: {file_path}")

initialize_tokenizer = lambda: T5Tokenizer.from_pretrained("t5-base")

configure_optimizer = lambda model: optimizers.Adam(learning_rate=0.001)

setup_training = lambda model, optimizer: (model, optimizer, calculate_triplet_loss)

execute_training = lambda model, config, data_loader, optimizer, loss_function: (
    model.train(),
    add_early_stopping(patience=3),
    [(
        [(
            (input_ids, labels, neg_samples) := batch,
            (outputs := model(input_ids)),
            (loss := loss_function(outputs, labels, neg_samples)),
            (gradients := tf.GradientTape().gradient(loss, model.trainable_variables)),
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        ) for batch in data_loader],
        early_stopping(loss.numpy()) and print("Early stopping triggered.") and break
    ) for epoch in range(config["epochs"])]
)

assess_model = lambda model, data_loader, loss_function: (
    model.eval(),
    print(f"Mean Evaluation Loss: {sum(loss_function(model(input_ids), labels, neg_samples).numpy() for input_ids, labels, neg_samples in data_loader) / len(data_loader):.4f}")
)

store_model = lambda model, file_path: model.save_weights(file_path)

store_training_logs = lambda history, file_path: json.dump(history, open(file_path, 'w'))

add_learning_rate_scheduler = lambda optimizer, config: optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.9)

add_early_stopping = lambda patience=3: (
    lambda current_loss: (
        (current_loss < best_loss) and (best_loss := current_loss) and (counter := 0) or
        (counter := counter + 1) and (counter >= patience)
    ) if 'best_loss' in locals() else (best_loss := float('inf'), counter := 0, lambda current_loss: (
        (current_loss < best_loss) and (best_loss := current_loss) and (counter := 0) or
        (counter := counter + 1) and (counter >= patience)
    )
)

initiate_training = lambda: (
    (config := ModelConfigurations()),
    (train_data := load_data("train.json")),
    (test_data := load_data("test.json")),
    (tokenizer := initialize_tokenizer()),
    (train_loader := DataLoaderAdapter(train_data, config, tokenizer)),
    (test_loader := DataLoaderAdapter(test_data, config, tokenizer)),
    (model := TextEmbeddingModel(128, 30522)),
    (optimizer := configure_optimizer(model)),
    (model, optimizer, loss_function := setup_training(model, optimizer)),
    (scheduler := add_learning_rate_scheduler(optimizer, config)),
    execute_training(model, config, train_loader, optimizer, loss_function),
    assess_model(model, test_loader, loss_function),
    store_model(model, os.path.join(config["results_dir"], "triplet_model.h5"))
)

if __name__ == "__main__":
    initiate_training()