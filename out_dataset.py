import json
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers

class TripletDataset:
    def __init__(self, triplets, tokenizer, max_sequence_length):
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        anchor_input_ids = self.tokenizer.encode(
            triplet['anchor'],
            max_length=self.max_sequence_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='np',
        )
        positive_input_ids = self.tokenizer.encode(
            triplet['positive'],
            max_length=self.max_sequence_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='np',
        )
        negative_input_ids = self.tokenizer.encode(
            triplet['negative'],
            max_length=self.max_sequence_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='np',
        )
        return {
            'anchor_input_ids': anchor_input_ids['input_ids'][0],
            'anchor_attention_mask': anchor_input_ids['attention_mask'][0],
            'positive_input_ids': positive_input_ids['input_ids'][0],
            'positive_attention_mask': positive_input_ids['attention_mask'][0],
            'negative_input_ids': negative_input_ids['input_ids'][0],
            'negative_attention_mask': negative_input_ids['attention_mask'][0],
        }

class TripletModel(models.Model):
    def __init__(self):
        super(TripletModel, self).__init__()
        self.bert = tf.keras.models.load_model('bert-base-uncased')
        self.fc1 = layers.Dense(128, activation='relu', input_shape=(768,))
        self.fc2 = layers.Dense(128)

    def call(self, input_ids, attention_mask):
        bert_output = self.bert([input_ids, attention_mask])
        return self.fc2(self.fc1(bert_output.pooler_output))

class TripletNetwork:
    def __init__(self, device):
        self.device = device
        self.model = TripletModel()
        self.optimizer = optimizers.Adam(learning_rate=1e-5)

    def calculate_triplet_loss(self, anchor_embeddings, positive_embeddings, negative_embeddings):
        positive_distance = tf.reduce_sum(tf.square(anchor_embeddings - positive_embeddings), axis=1)
        negative_distance = tf.reduce_sum(tf.square(anchor_embeddings - negative_embeddings), axis=1)
        return tf.reduce_mean(tf.maximum(positive_distance - negative_distance + 0.2, 0))

    def train(self, data_loader, epochs):
        for epoch in range(epochs):
            for batch in data_loader:
                with tf.GradientTape() as tape:
                    anchor_input_ids = batch['anchor_input_ids']
                    positive_input_ids = batch['positive_input_ids']
                    negative_input_ids = batch['negative_input_ids']
                    anchor_attention_mask = batch['anchor_attention_mask']
                    positive_attention_mask = batch['positive_attention_mask']
                    negative_attention_mask = batch['negative_attention_mask']
                    
                    anchor_output = self.model(anchor_input_ids, anchor_attention_mask)
                    positive_output = self.model(positive_input_ids, positive_attention_mask)
                    negative_output = self.model(negative_input_ids, negative_attention_mask)
                    
                    batch_loss = self.calculate_triplet_loss(anchor_output, positive_output, negative_output)
                gradients = tape.gradient(batch_loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                print(f'Epoch {epoch+1}, Batch Loss: {batch_loss.numpy()}')

    def evaluate(self, data_loader):
        total_correct = 0
        for batch in data_loader:
            anchor_input_ids = batch['anchor_input_ids']
            positive_input_ids = batch['positive_input_ids']
            negative_input_ids = batch['negative_input_ids']
            anchor_attention_mask = batch['anchor_attention_mask']
            positive_attention_mask = batch['positive_attention_mask']
            negative_attention_mask = batch['negative_attention_mask']
            
            anchor_output = self.model(anchor_input_ids, anchor_attention_mask)
            positive_output = self.model(positive_input_ids, positive_attention_mask)
            negative_output = self.model(negative_input_ids, negative_attention_mask)
            
            positive_similarity = tf.reduce_sum(tf.multiply(anchor_output, positive_output), axis=1)
            negative_similarity = tf.reduce_sum(tf.multiply(anchor_output, negative_output), axis=1)
            total_correct += tf.reduce_sum(tf.cast(tf.greater(positive_similarity, negative_similarity), tf.int32)).numpy()
        return total_correct / len(data_loader.dataset)

def load_data(dataset_path, snippet_folder_path):
    dataset = json.load(open(dataset_path))
    instance_id_map = {item['instance_id']: item['problem_statement'] for item in dataset}
    snippet_directories = [(os.path.join(folder, f), os.path.join(folder, f, 'snippet.json')) 
                           for f in os.listdir(snippet_folder_path) if os.path.isdir(os.path.join(snippet_folder_path, f))]
    snippets = [(folder, json.load(open(snippet_file))) for folder, snippet_file in snippet_directories]
    return instance_id_map, snippets

def create_triplets(instance_id_map, snippets):
    bug_snippets = []
    non_bug_snippets = []
    for _, snippet_file in snippets:
        snippet_data = snippet_file
        if snippet_data.get('is_bug', False):
            bug_snippets.append(snippet_data['snippet'])
        else:
            non_bug_snippets.append(snippet_data['snippet'])
    bug_snippets = [snippet for snippet in bug_snippets if snippet]
    non_bug_snippets = [snippet for snippet in non_bug_snippets if snippet]
    return [{'anchor': instance_id_map[os.path.basename(folder)], 
             'positive': positive_doc, 
             'negative': random.choice(non_bug_snippets)} 
            for folder, _ in snippets 
            for positive_doc in bug_snippets 
            for _ in range(min(1, len(non_bug_snippets)))]

def main():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    
    instance_id_map, snippets = load_data(dataset_path, snippet_folder_path)
    triplets = create_triplets(instance_id_map, snippets)
    train_triplets, test_triplets = np.array_split(np.array(triplets), 2)
    
    device = '/gpu:0' if tf.test.is_gpu_available() else '/cpu:0'
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    
    train_dataset = tf.data.Dataset.from_tensor_slices(train_triplets).map(TripletDataset(train_triplets, tokenizer, 512))
    test_dataset = tf.data.Dataset.from_tensor_slices(test_triplets).map(TripletDataset(test_triplets, tokenizer, 512))
    
    train_data_loader = tf.data.Dataset.batch(train_dataset, batch_size=32)
    test_data_loader = tf.data.Dataset.batch(test_dataset, batch_size=32)
    
    network = TripletNetwork(device)
    network.train(train_data_loader, 5)
    print(f'Test Accuracy: {network.evaluate(test_data_loader)}')

if __name__ == "__main__":
    main()