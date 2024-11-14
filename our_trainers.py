import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

def construct_triplet_dataset(samples, labels, batch_size, num_negatives):
    """
    Creates a triplet dataset from the given samples, labels, batch size and number of negatives.
    
    Args:
    samples (np.ndarray): The input samples.
    labels (np.ndarray): The labels corresponding to the samples.
    batch_size (int): The batch size.
    num_negatives (int): The number of negative samples.
    
    Returns:
    TripletDataset: The constructed triplet dataset.
    """
    return TripletDataset(samples, labels, batch_size, num_negatives)

def create_embedding_model(num_embeddings, embedding_dim):
    """
    Creates a model for computing embeddings from input IDs.
    
    Args:
    num_embeddings (int): The total number of embeddings.
    embedding_dim (int): The dimensionality of the embeddings.
    
    Returns:
    TripletModel: The created model.
    """
    return TripletModel(num_embeddings, embedding_dim)

def create_triplet_loss_function(margin=1.0):
    """
    Creates a loss function for triplet loss.
    
    Args:
    margin (float, optional): The margin for the triplet loss. Defaults to 1.0.
    
    Returns:
    TripletLoss: The created loss function.
    """
    return TripletLoss(margin)

def create_triplet_training_pipeline(model, loss_fn, epochs, lr, dataset):
    """
    Creates a training pipeline for a triplet model.
    
    Args:
    model (TripletModel): The model to train.
    loss_fn (TripletLoss): The loss function to use.
    epochs (int): The number of epochs to train for.
    lr (float): The learning rate.
    dataset (TripletDataset): The dataset to train on.
    
    Returns:
    TripletTrainer: The created training pipeline.
    """
    return TripletTrainer(model, loss_fn, epochs, lr, dataset)

def create_triplet_evaluation_pipeline(model, loss_fn, dataset):
    """
    Creates an evaluation pipeline for a triplet model.
    
    Args:
    model (TripletModel): The model to evaluate.
    loss_fn (TripletLoss): The loss function to use.
    dataset (TripletDataset): The dataset to evaluate on.
    
    Returns:
    TripletEvaluator: The created evaluation pipeline.
    """
    return TripletEvaluator(model, loss_fn, dataset)

def create_triplet_prediction_pipeline(model):
    """
    Creates a prediction pipeline for a triplet model.
    
    Args:
    model (TripletModel): The model to use for prediction.
    
    Returns:
    TripletPredictor: The created prediction pipeline.
    """
    return TripletPredictor(model)

class TripletDataset(Dataset):
    """
    A dataset for triplet learning.
    
    Args:
    samples (np.ndarray): The input samples.
    labels (np.ndarray): The labels corresponding to the samples.
    batch_size (int): The batch size.
    num_negatives (int): The number of negative samples.
    """
    def __init__(self, samples, labels, batch_size, num_negatives):
        self.samples = samples
        self.labels = labels
        self.batch_size = batch_size
        self.num_negatives = num_negatives

    def __getitem__(self, idx):
        """
        Gets the item at the given index.
        
        Args:
        idx (int): The index.
        
        Returns:
        dict: A dictionary containing the anchor, positive and negative input IDs.
        """
        anchor_idx = np.arange(idx * self.batch_size, (idx + 1) * self.batch_size)
        anchor_labels = self.labels[anchor_idx]
        positive_idx = []
        for label in anchor_labels:
            positive_idx.append(np.random.choice(np.where(self.labels == label)[0], size=1)[0])
        positive_idx = np.array(positive_idx)
        negative_idx = []
        for label in anchor_labels:
            negative_idx.append(np.random.choice(np.where(self.labels != label)[0], size=self.num_negatives, replace=False))
        negative_idx = np.array(negative_idx)
        return {
            'anchor_input_ids': torch.from_numpy(self.samples[anchor_idx]),
            'positive_input_ids': torch.from_numpy(self.samples[positive_idx]),
            'negative_input_ids': torch.from_numpy(self.samples[negative_idx]).view(self.batch_size, self.num_negatives, -1)
        }

    def __len__(self):
        """
        Gets the length of the dataset.
        
        Returns:
        int: The length of the dataset.
        """
        return len(self.samples) // self.batch_size


class TripletModel(nn.Module):
    """
    A model for computing embeddings from input IDs.
    
    Args:
    num_embeddings (int): The total number of embeddings.
    embedding_dim (int): The dimensionality of the embeddings.
    """
    def __init__(self, num_embeddings, embedding_dim):
        super(TripletModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def embed(self, input_ids):
        """
        Computes the embeddings for the given input IDs.
        
        Args:
        input_ids (torch.Tensor): The input IDs.
        
        Returns:
        torch.Tensor: The computed embeddings.
        """
        embeddings = self.embedding(input_ids)
        embeddings = self.pooling(embeddings.permute(0, 2, 1)).squeeze()
        return F.normalize(embeddings, p=2, dim=1)

    def forward(self, inputs):
        """
        The forward pass of the model.
        
        Args:
        inputs (dict): A dictionary containing the anchor, positive and negative input IDs.
        
        Returns:
        tuple: A tuple containing the anchor, positive and negative embeddings.
        """
        anchor_embeddings = self.embed(inputs['anchor_input_ids'])
        positive_embeddings = self.embed(inputs['positive_input_ids'])
        negative_embeddings = self.embed(inputs['negative_input_ids'].view(-1, inputs['negative_input_ids'].shape[2]))
        return anchor_embeddings, positive_embeddings, negative_embeddings


class TripletLoss(nn.Module):
    """
    A loss function for triplet loss.
    
    Args:
    margin (float, optional): The margin for the triplet loss. Defaults to 1.0.
    """
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        The forward pass of the loss function.
        
        Args:
        anchor (torch.Tensor): The anchor embeddings.
        positive (torch.Tensor): The positive embeddings.
        negative (torch.Tensor): The negative embeddings.
        
        Returns:
        torch.Tensor: The computed loss.
        """
        return torch.mean(torch.clamp(torch.norm(anchor - positive, p=2, dim=1) - torch.norm(anchor.unsqueeze(1) - negative, p=2, dim=2) + self.margin, min=0))


class TripletTrainer:
    """
    A training pipeline for a triplet model.
    
    Args:
    model (TripletModel): The model to train.
    loss_fn (TripletLoss): The loss function to use.
    epochs (int): The number of epochs to train for.
    lr (float): The learning rate.
    dataset (TripletDataset): The dataset to train on.
    """
    def __init__(self, model, loss_fn, epochs, lr, dataset):
        self.model = model
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.lr = lr
        self.dataset = dataset
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train_step(self, data):
        """
        A single training step.
        
        Args:
        data (dict): A dictionary containing the anchor, positive and negative input IDs.
        
        Returns:
        float: The loss for the current step.
        """
        self.optimizer.zero_grad()
        data = {key: value.to(self.device) for key, value in data.items()}
        anchor_embeddings, positive_embeddings, negative_embeddings = self.model(data)
        if len(positive_embeddings) > 0:
            loss = self.loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
            loss.backward()
            self.optimizer.step()
        return loss.item()

    def train(self):
        """
        The full training loop.
        
        Returns:
        list: A list of losses for each epoch.
        """
        return list(map(lambda epoch: self.train_epoch(epoch), range(self.epochs)))

    def train_epoch(self, epoch):
        """
        A single epoch of training.
        
        Args:
        epoch (int): The current epoch.
        """
        total_loss = 0
        for i, data in enumerate(DataLoader(self.dataset, batch_size=1, shuffle=True)):
            loss = self.train_step(data)
            total_loss += loss
        print(f'Epoch: {epoch+1}, Loss: {total_loss/(i+1):.3f}')


class TripletEvaluator:
    """
    An evaluation pipeline for a triplet model.
    
    Args:
    model (TripletModel): The model to evaluate.
    loss_fn (TripletLoss): The loss function to use.
    dataset (TripletDataset): The dataset to evaluate on.
    """
    def __init__(self, model, loss_fn, dataset):
        self.model = model
        self.loss_fn = loss_fn
        self.dataset = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def evaluate_step(self, data):
        """
        A single evaluation step.
        
        Args:
        data (dict): A dictionary containing the anchor, positive and negative input IDs.
        
        Returns:
        float: The loss for the current step.
        """
        data = {key: value.to(self.device) for key, value in data.items()}
        anchor_embeddings, positive_embeddings, negative_embeddings = self.model(data)
        if len(positive_embeddings) > 0:
            loss = self.loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
            return loss.item()
        return 0.0

    def evaluate(self):
        """
        The full evaluation loop.
        
        Returns:
        float: The average loss.
        """
        return self.evaluate_epoch()

    def evaluate_epoch(self):
        """
        A single epoch of evaluation.
        """
        total_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(DataLoader(self.dataset, batch_size=1, shuffle=True)):
                loss = self.evaluate_step(data)
                total_loss += loss
        print(f'Validation Loss: {total_loss / (i+1):.3f}')


class TripletPredictor:
    """
    A prediction pipeline for a triplet model.
    
    Args:
    model (TripletModel): The model to use for prediction.
    """
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, input_ids):
        """
        Makes a prediction for the given input IDs.
        
        Args:
        input_ids (torch.Tensor): The input IDs.
        
        Returns:
        torch.Tensor: The predicted embedding.
        """
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            return self.model({'anchor_input_ids': input_ids})[0]


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    samples = np.random.randint(0, 100, (100, 10))
    labels = np.random.randint(0, 2, (100,))
    batch_size = 32
    num_negatives = 5
    epochs = 10
    num_embeddings = 101
    embedding_dim = 10
    margin = 1.0
    lr = 1e-4

    dataset = construct_triplet_dataset(samples, labels, batch_size, num_negatives)
    model = create_embedding_model(num_embeddings, embedding_dim)
    loss_fn = create_triplet_loss_function(margin)
    trainer = create_triplet_training_pipeline(model, loss_fn, epochs, lr, dataset)
    trainer.train()

    evaluator = create_triplet_evaluation_pipeline(model, loss_fn, dataset)
    evaluator.evaluate()

    predictor = create_triplet_prediction_pipeline(model)
    input_ids = torch.tensor([1, 2, 3, 4, 5])
    output = predictor.predict(input_ids)
    print(output)


if __name__ == "__main__":
    main()