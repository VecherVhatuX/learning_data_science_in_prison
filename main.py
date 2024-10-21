# Import necessary libraries
import logging
import sys
import traceback
from datetime import datetime
import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import BatchSamplers, SentenceTransformerTrainingArguments
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, default_data_collator, get_linear_schedule_with_warmup, AutoModelForSequenceClassification
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, PrefixTuningConfig, TaskType, LoraConfig
import torch
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from datasets import Dataset, load_dataset
from pathlib import Path
import os, json
from tqdm import tqdm
import pickle
from random import sample
from datasets import Dataset
from sentence_transformers import InputExample

def disable_ssl_warnings():
    # Disable insecure request warnings
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
    # Patch the request method to ignore SSL verification
    original_request = requests.Session.request
    def patched_request(self, *args, **kwargs):
        kwargs['verify'] = False
        return original_request(self, *args, **kwargs)
    requests.Session.request = patched_request

def setup_logging():
    # Set up basic logging configuration
    logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

def get_device_info():
    # Get the number of available GPUs
    n = torch.cuda.device_count()
    print(f"There are {n} GPUs available for torch.")
    # Print information for each GPU
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {name}")

def load_model(model_path):
    # Load the model using SentenceTransformer
    return SentenceTransformer(model_path)

def get_peft_model_instance(model):
    # Define the PEFT configuration
    peft_config = LoraConfig(
        target_modules=["dense"],
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    # Get the PEFT model
    model._modules["0"].auto_model = get_peft_model(model._modules["0"].auto_model, peft_config)
    return model

def load_pretrained_model(model_path):
    # Load the pre-trained model
    model = load_model(model_path)
    # Get the PEFT model instance
    return get_peft_model_instance(model)

def load_data(data_path):
    # Load the data from a pickle file
    with open(data_path, 'rb') as f:
        return pickle.load(f)

def prepare_dataset(data, negative_sample_size=3):
    # Initialize an empty list to store the dataset
    dataset_list = []
    # Iterate over the data
    for item in data:
        # Get the query, relevant document, and non-relevant documents
        query = item['query']
        relevant_doc = item['relevant_doc']
        non_relevant_docs = sample(item['irrelevant_docs'], min(len(item['irrelevant_docs']), negative_sample_size))
        # Create a dataset item for each non-relevant document
        for item in non_relevant_docs:
            dataset_list.append({
                "anchor": query,
                "positive": relevant_doc,
                "negative": item
            })
    return dataset_list

def create_dataset(data):
    # Prepare the dataset
    dataset_list = prepare_dataset(data)
    # Create a Dataset object from the list
    return Dataset.from_list(dataset_list)

def load_datasets():
    # Load the training dataset
    train_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="train").select(range(10000))
    # Load the evaluation dataset
    eval_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="dev").select(range(1000))
    return train_dataset, eval_dataset

def get_training_args(output_dir, train_batch_size):
    # Define the training arguments
    return SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=20,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        warmup_ratio=0.1,
        fp16=True,
        bf16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        logging_steps=100,
        run_name="nli-v2",
    )

def train_model(model, train_dataset, eval_dataset, args):
    # Define the training loss
    train_loss = losses.MultipleNegativesRankingLoss(model)
    # Create a trainer object
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss
    )
    # Train the model
    trainer.train()

def save_model(model, output_path):
    # Save the model to a pickle file
    with open(output_path + '/model.pkl', 'wb') as w:
        pickle.dump(model, w)

def main():
    # Disable SSL warnings
    disable_ssl_warnings()
    # Setup logging
    setup_logging()
    # Get device information
    get_device_info()

    # Get the data path, output path, and model path from environment variables
    data_path = os.environ.get('DATA_PATH', '/home/ma-user/data/data.pkl')
    output_path = os.environ.get('OUTPUT_PATH', '/tmp/output')
    model_path = os.environ.get('MODEL_PATH', '/home/ma-user/bert-base-uncased')

    # Set default model path if not provided
    if not model_path:
        model_path = "bert-base-uncased"

    # Define training hyperparameters
    train_batch_size = 64
    max_seq_length = 75
    num_epochs = 1

    # Load the pre-trained model
    model = load_pretrained_model(model_path)

    # Load the training and evaluation datasets
    train_dataset, eval_dataset = load_datasets()

    # Get the model name and create the output directory
    model_name = Path(model_path).stem
    output_dir = output_path + "/output/training_nli_v2_" + model_name.replace("/", "-") + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Get the training arguments
    args = get_training_args(output_dir, train_batch_size)

    # Put the model in training mode
    model.train()

    # Train the model
    train_model(model, train_dataset, eval_dataset, args)

    # Save the model
    save_model(model, output_path)

if __name__ == "__main__":
    main()