import os
import json
import random
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from transformers import T5Tokenizer

SETTINGS = {
    "model_name": "t5-base", "alpha": 16, "dropout_rate": 0.1, "decomposition_rank": 64,
    "layers_to_modify": ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
    "quantization_config": {"nested_quantization": False, "four_bit_dtype": "float16", "storage_dtype": "uint8",
                            "quantization_strategy": "nf4", "flash_attention": False, "low_rank_peft": False,
                            "eight_bit_quantization": False, "four_bit_quantization": False, "reentrant_training": False,
                            "unsloth_training": False},
    "use_triplet_loss": True, "data_source": "timdettmers/openassistant-guanaco",
    "token_flags": {"use_special_token": False, "include_special_tokens": False}, "data_splits": ["train", "test"],
    "tokenized_file": None, "results_dir": "./results", "epochs": 3, "batch_sizes": {"train": 16, "eval": 64},
    "warmup_steps": 500, "weight_decay": 0.01, "logging_dir": "./logs", "model_save_interval": 500,
    "max_checkpoints": 2, "seed": 42, "checkpoint_path": None, "negative_samples_per_batch": 5
}

LanguageModel = lambda vocab_size, embed_dim: tf.keras.Sequential([
    layers.Embedding(vocab_size, embed_dim),
    layers.LSTM(embed_dim, return_sequences=True),
    layers.Dense(embed_dim),
    layers.Dense(vocab_size)
])

load_data = lambda file_path: json.load(open(file_path, 'r')) if os.path.exists(file_path) else None

tokenize_data = lambda data, tokenizer, max_len=512: tf.convert_to_tensor(tokenizer.encode(data, max_length=max_len, padding='max_length', truncation=True))

TextDataset = lambda data, tokenizer, neg_samples=5: tf.data.Dataset.from_generator(
    lambda: ((tokenize_data(data[idx]['input'], tokenizer), 
             tokenize_data(data[idx]['output'], tokenizer), 
             tf.stack([tokenize_data(data[random.choice([j for j in range(len(data)) if j != idx])]['input'], tokenizer) for _ in range(neg_samples)])) 
    for idx in range(len(data))),
    output_signature=(tf.TensorSpec(shape=(None,), tf.TensorSpec(shape=(None,)), tf.TensorSpec(shape=(None,)))
)

ModelTrainer = lambda model, optimizer, loss_fn: {
    "model": model,
    "optimizer": optimizer,
    "loss_fn": loss_fn,
    "best_loss": float('inf'),
    "counter": 0,
    "train": lambda data_loader, epochs, patience=3: [
        [tf.GradientTape().__enter__().__exit__(None, None, None) for _ in range(epochs)] and None,
        [optimizer.apply_gradients(zip(tf.GradientTape().gradient(loss_fn(model(input_ids), labels, neg_samples), model.trainable_variables), model.trainable_variables)) 
         for input_ids, labels, neg_samples in data_loader] and None,
        [print("Training halted early.") if early_stopping(loss_fn(model(input_ids), labels, neg_samples).numpy(), patience) else None 
         for input_ids, labels, neg_samples in data_loader]
    ],
    "evaluate": lambda data_loader: print(f"Average Loss on Evaluation: {sum(loss_fn(model(input_ids), labels, neg_samples).numpy() for input_ids, labels, neg_samples in data_loader) / len(data_loader):.4f}"),
    "early_stopping": lambda current_loss, patience: (current_loss < best_loss and (best_loss := current_loss) and (counter := 0) or (counter := counter + 1) and counter >= patience
}

save_model = lambda model, path: model.save_weights(path)

save_history = lambda history, path: json.dump(history, open(path, 'w'))

initialize_environment = lambda: (
    SETTINGS,
    T5Tokenizer.from_pretrained("t5-base"),
    LanguageModel(30522, 128),
    optimizers.Adam(learning_rate=0.001)
)

execute_pipeline = lambda: (
    (settings, tokenizer, model, optimizer) := initialize_environment(),
    (train_data, test_data) := (load_data("train.json"), load_data("test.json")),
    (train_dataset, test_dataset) := (TextDataset(train_data, tokenizer, settings["negative_samples_per_batch"]), TextDataset(test_data, tokenizer, settings["negative_samples_per_batch"])),
    (train_loader, test_loader) := (tf.data.Dataset.from_generator(lambda: train_dataset, output_signature=(tf.TensorSpec(shape=(None,), tf.TensorSpec(shape=(None,)), tf.TensorSpec(shape=(None,))))),
                                   tf.data.Dataset.from_generator(lambda: test_dataset, output_signature=(tf.TensorSpec(shape=(None,), tf.TensorSpec(shape=(None,)), tf.TensorSpec(shape=(None,)))))),
    (trainer := ModelTrainer(model, optimizer, tf.keras.losses.TripletSemiHardLoss())),
    trainer.train(train_loader, settings["epochs"]),
    trainer.evaluate(test_loader),
    save_model(model, os.path.join(settings["results_dir"], "triplet_model.h5"))
)

add_scheduler = lambda optimizer, initial_lr, decay_steps, decay_rate: optimizers.schedules.ExponentialDecay(initial_lr, decay_steps, decay_rate)

add_checkpoint = lambda model, optimizer, checkpoint_dir, max_to_keep=2: tf.train.Checkpoint(model=model, optimizer=optimizer).save(os.path.join(checkpoint_dir, f"checkpoint_{len(os.listdir(checkpoint_dir))}.ckpt"))

if __name__ == "__main__":
    execute_pipeline()