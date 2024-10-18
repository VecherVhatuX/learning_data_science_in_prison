import logging
import os
import pickle
import torch
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset, load_dataset
from random import sample
from sentence_transformers import (
    SentenceTransformer,
    losses,
    SentenceTransformerTrainingArguments,
    SentenceTransformerTrainer,
    InputExample,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.training_args import BatchSamplers
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from peft import (
    LoraConfig,
    PrefixTuningConfig,
    TaskType,
    get_peft_config,
    get_peft_model,
    PromptTuningInit,
    PeftType,
)

# Disable SSL warnings
def disable_ssl_warnings():
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
    original_request = requests.Session.request
    def patched_request(self, *args, **kwargs):
        kwargs['verify'] = False
        return original_request(self, *args, **kwargs)
    requests.Session.request = patched_request

# Set log level to INFO
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)

# Load data from environment variables
data_path = os.environ.get('DATA_PATH', '/home/ma-user/data/data.pkl')
output_path = os.environ.get('OUTPUT_PATH', '/tmp/output')
model_path = os.environ.get('MODEL_PATH', '/home/ma-user/bert-base-uncased')

def prepare_dataset(data, negative_sample_size=3):
    dataset_list = []
    for item in data:
        query = item['query']
        relevant_doc = item['relevant_doc']
        non_relevant_docs = sample(item['irrelevant_docs'], min(len(item['irrelevant_docs']), negative_sample_size))
        for item in non_relevant_docs:
            dataset_list.append({
                "anchor": query,
                "positive": relevant_doc,
                "negative": item
            })
    return dataset_list

def main():
    disable_ssl_warnings()
    
    # Get available GPUs
    n = torch.cuda.device_count()
    print(f"There are {n} GPUs available for torch.")
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {name}")

    # Load data
    with open(data_path, 'rb') as f:
        all_dataset = pickle.load(f)

    # Split data into train and eval sets
    train_data = all_dataset[0:61]
    eval_data = all_dataset[61:]

    # Prepare datasets
    train_data = prepare_dataset(train_data, negative_sample_size=10)
    eval_data = prepare_dataset(eval_data, negative_sample_size=10)

    # Convert to datasets
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    print(len(train_dataset), len(eval_dataset))

    # Load model
    model_name = Path(model_path).stem
    sentence_transformer = SentenceTransformer(model_path)

    # Get PeFT model
    peft_config = LoraConfig(
        target_modules=["dense"],
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    sentence_transformer._modules["0"].auto_model = get_peft_model(
        sentence_transformer._modules["0"].auto_model, peft_config
    )

    # Set model to train mode
    sentence_transformer.train()

    # Load training datasets
    train_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="train").select(range(10000))
    eval_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="dev").select(range(1000))

    # Define loss function
    train_loss = losses.MultipleNegativesRankingLoss(sentence_transformer)

    # Define training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=output_path + "/output/training_nli_v2_" + model_name.replace("/", "-") + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        num_train_epochs=20,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        warmup_ratio=0.1,
        fp16=True,
        bf16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=2,
        logging_steps=100,
        run_name="nli-v2",
    )

    # Create trainer and start training
    trainer = SentenceTransformerTrainer(
        model=sentence_transformer,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss
    )
    trainer.train()

if __name__ == "__main__":
    main()