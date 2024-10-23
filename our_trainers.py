import torch
import torch.nn as nn
from transformers import Trainer
from typing import Optional, Union, Dict, Any, Callable
from trl import SFTTrainer
from transformers.trainer_utils import PredictionOutput
from torch.nn.functional import normalize

class TripletLossTrainer(Trainer):
    def __init__(self, 
                 triplet_margin: float = 1.0, 
                 triplet_loss_fn: Optional[Callable] = None,
                 layer_index=-1,
                 **kwargs):
        """
        Инициализирует TripletLossTrainer.

        Args:
            triplet_margin (float, optional): Маржа для триплетной потери. Defaults to 1.0.
            triplet_loss_fn (Callable, optional): Функция потерь. Если не указано, используется nn.TripletMarginLoss.
            **kwargs: Дополнительные аргументы для SFTTrainer.
        """
        super().__init__(**kwargs)
        self.triplet_margin = triplet_margin
        self.triplet_loss_fn = triplet_loss_fn or nn.TripletMarginLoss(margin=triplet_margin)
        self.layer_index = layer_index
        
    
    def mean_pooling(self, hidden_state, attention_mask):
        """
        Выполняет среднеарифметический пуллинг скрытых состояний.

        Args:
            model_output: Выход модели.
            attention_mask: Маска внимания.

        Returns:
            Тензор с агрегированными скрытыми состояниями.
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        sum_embeddings = torch.sum(hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Переопределение метода compute_loss для использования триплетной потери.

        Args:
            model: Модель.
            inputs: Входные данные.
            return_outputs (bool, optional): Возвращать ли выходы модели. Defaults to False.

        Returns:
            Если return_outputs=True, возвращает кортеж (loss, outputs), иначе только loss.
        """
        # Предполагаем, что входные данные содержат три набора данных: anchor, positive, negative
        # Например, inputs = {'anchor_input_ids': ..., 'anchor_attention_mask': ..., 
        #                   'positive_input_ids': ..., 'positive_attention_mask': ..., 
        #                   'negative_input_ids': ..., 'negative_attention_mask': ...}
        
        anchor_input_ids = inputs["anchor_input_ids"]
        anchor_attention_mask = inputs["anchor_attention_mask"]
        positive_input_ids = inputs["positive_input_ids"]
        positive_attention_mask = inputs["positive_attention_mask"]
        negative_input_ids = inputs["negative_input_ids"]
        negative_attention_mask = inputs["negative_attention_mask"]
        
        # Получаем скрытые представления для каждого из трёх примеров
        anchor_outputs = model(input_ids=anchor_input_ids, attention_mask=anchor_attention_mask)
        positive_outputs = model(input_ids=positive_input_ids, attention_mask=positive_attention_mask)
        negative_outputs = model(input_ids=negative_input_ids, attention_mask=negative_attention_mask)
        
                # Извлечение скрытых состояний нужного слоя
        anchor_hidden_state = anchor_outputs.hidden_states[self.layer_index]
        positive_hidden_state = positive_outputs.hidden_states[self.layer_index]
        negative_hidden_state = negative_outputs.hidden_states[self.layer_index]

        # Выполняем среднеарифметический пуллинг
        anchor_embeddings = self.mean_pooling(anchor_hidden_state, anchor_attention_mask)
        positive_embeddings = self.mean_pooling(positive_hidden_state, positive_attention_mask)
        negative_embeddings = self.mean_pooling(negative_hidden_state, negative_attention_mask)
        
        # Нормализуем векторы
        anchor_embeddings = normalize(anchor_embeddings, p=2, dim=1)
        positive_embeddings = normalize(positive_embeddings, p=2, dim=1)
        negative_embeddings = normalize(negative_embeddings, p=2, dim=1)
        
        # Вычисляем триплетную потерю
        loss = self.triplet_loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
        
        if return_outputs:
            return (loss, {
                'anchor_embeddings': anchor_embeddings, 
                'positive_embeddings': positive_embeddings, 
                'negative_embeddings': negative_embeddings
            })
        else:
            return loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Переопределяет метод prediction_step для корректной валидации.
        
        Args:
            model: Модель для предсказания.
            inputs: Входные данные.
            prediction_loss_only (bool): Возвращать ли только потери.
            ignore_keys (list, optional): Ключи, которые нужно игнорировать.
        
        Returns:
            tuple: (loss, predictions, labels)
        """
        with torch.no_grad():
            loss, predictions = self.compute_loss(model, inputs, return_outputs=True)
        # Возвращаем фиктивные метки, если они не используются
        dummy_labels = torch.zeros(inputs["anchor_input_ids"].size(0), device=inputs["anchor_input_ids"].device)
        return (loss, predictions, dummy_labels)
