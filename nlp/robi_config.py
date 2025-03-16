from dataclasses import dataclass, field
import torch
from typing import List, Tuple
import os

@dataclass
class SlidingWindowConfig:
    max_length: int = 512
    stride: int = 256

@dataclass
class RoBiConfig:
    num_classes: int = 3
    input_size: int = 1024 # must be exactly equal to RoBERTa Large's embedding dimensions!
    hidden_size: int = 512
    num_layers: int = 2
    dropout: float = 0.2
    batch_first: bool = True
    bidirectional: bool = True
    roberta_size: str = "roberta-large"

@dataclass
class TrainConfig:
    num_epochs: int = 50
    learning_rate: float = 1e-5
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    early_stopping_patience: int = 10
    gradient_clip_max_norm: float = 1.0
    train_batch_size: int = 16
    val_batch_size: int = 16
    shuffle: bool = True
    gradient_accumulation_steps: int = 8 # simulating a batch size of 16 x 8 = 128 while using a batch size of 16 for memory constraints

@dataclass
class SaveLoadConfig:
    base_path: str = "robi"
    trial_dir: str = None # "trial1", "trial2", etc.
    
    @property
    def path(self) -> str:
        if self.trial_dir:
            return os.path.join(self.base_path, self.trial_dir)
        return self.base_path

@dataclass
class PredictConfig:
    overflow_tokens: bool = True
    max_length: int = 512
    stride: int = 256
    sentiment_labels: dict = field(default_factory=lambda: {
        0: "Negative",
        1: "Neutral",
        2: "Positive"
    })

@dataclass
class BayesianOptConfig:
    n_trials: int = 50
    base_path: str = "robi"
    nu: float = 2.5 # twice differentiable 
    n_restarts_optimizer: int = 10
    random_state: int = 10
    n_random: int = 1000
    
    param_ranges = {
        'hidden_size': (384, 1024),
        'num_layers': (2, 4),
        'dropout': (0.2, 0.4),
        'learning_rate': (5e-6, 5e-4),         # slightly above and below 1e-5 as it is known to be a good learning rate for RoBERTa-BiLSTM
        'gradient_accumulation_steps': (6, 10)
    }

@dataclass
class MetricsConfig:
    decimal_places: int = 4
    table_width: int = 80
    labels: List[str] = field(default_factory=lambda: ['Negative', 'Neutral', 'Positive'])
    
@dataclass
class ConfusionMatrixConfig:
    figure_size: Tuple[int, int] = (10, 8)
    annot: bool = True
    fmt: str = 'd'
    xticklabels: List[str] = field(default_factory=lambda: ['Negative', 'Neutral', 'Positive'])
    yticklabels: List[str] = field(default_factory=lambda: ['Negative', 'Neutral', 'Positive'])
    square: bool = True
    cbar: bool = True
        
@dataclass
class ROCConfig:
    figure_size: Tuple[int, int] = (10, 8)
    line_width: float = 2.0
    colors: List[str] = field(default_factory=lambda: ['#2ecc71', '#3498db', '#e74c3c'])
    font_size: int = 12
    marker_size: int = 100
    dpi: int = 100
    labels: List[str] = field(default_factory=lambda: ['Negative', 'Neutral', 'Positive'])
    
@dataclass
class RoBiEvalConfig:
    use_sliding_window: bool = True
    trial_dir: str = "trial1"  # best trial from the optimization
    batch_size: int = 16
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes: int = 3
    labels: List[str] = field(default_factory=lambda: ["Negative", "Neutral", "Positive"])
    
@dataclass
class VaderConfig:
    model_path: str = "vader_sentiment.pkl"
    threshold_pos: float = 0.05
    threshold_neg: float = -0.05
    
@dataclass
class RoBertaBaseConfig:
    model_name: str = "roberta-base"
    model_path: str = "roberta_sentiment"
    num_labels: int = 3
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 4
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
@dataclass
class DistilBertConfig:
    model_name: str = "distilbert-base-uncased"
    model_path: str = "distilbert_sentiment"
    num_labels: int = 3
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 4
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
@dataclass
class TextBlobConfig:
    model_path: str = "textblob_model.pkl"
    threshold_pos: float = 0.1
    threshold_neg: float = -0.1