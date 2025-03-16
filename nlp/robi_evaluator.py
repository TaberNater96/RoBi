from .robi_config import (
    SaveLoadConfig,
    PredictConfig,
    MetricsConfig,
    ConfusionMatrixConfig,
    ROCConfig,
    VaderConfig,
    RoBertaBaseConfig,
    DistilBertConfig,
    TextBlobConfig
)
import numpy as np
import random
import os
from tqdm import tqdm 
import pickle
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    auc
)
from sklearn.preprocessing import label_binarize
import torch
import transformers
from torch.utils.data import (
    DataLoader,
    Dataset
)
import seaborn as sns
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)
from textblob import TextBlob
from .hybrid_sentiment_network import (
    load_RoBi, 
    SlidingWindow
)
from typing import (
    List, 
    Dict, 
    Union,
    Tuple
)

random.seed(10)
np.random.seed(10)
transformers.set_seed(10)
torch.manual_seed(10)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(10)

def calculate_metrics(
    y_true: Union[List[int], np.ndarray], 
    y_pred: Union[List[int], np.ndarray],
    config: MetricsConfig = MetricsConfig()
) -> Dict[str, Union[float, np.ndarray]]:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = {}
    
    metrics['precision'] = precision_score(y_true, y_pred, average=None)
    metrics['recall'] = recall_score(y_true, y_pred, average=None)
    metrics['f1'] = f1_score(y_true, y_pred, average=None)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    display_str = []
    
    header = f"+{'-'*(config.table_width)}+"
    display_str.append(header)
    
    metrics_header = f"|{'Metrics':^20}|"
    for label in config.labels:
        metrics_header += f"{label:^12}|"
    display_str.append(metrics_header)
    display_str.append(header)
    
    for metric in ['precision', 'recall', 'f1']:
        row = f"|{metric.title():^20}|"
        for i, score in enumerate(metrics[metric]):
            row += f"{score:^12.{config.decimal_places}f}|"
        display_str.append(row)
        
    display_str.append(header)
    accuracy_row = f"|{'Accuracy':^20}|{metrics['accuracy']:^{config.table_width-22}.{config.decimal_places}f}|"
    display_str.append(accuracy_row)
    display_str.append(header)
    
    metrics['display'] = '\n'.join(display_str)
    
    return metrics

def plot_confusion_matrix(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    config: ConfusionMatrixConfig = ConfusionMatrixConfig()
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=config.figure_size)
    sns.heatmap(
        cm, 
        annot=config.annot,
        fmt=config.fmt,
        xticklabels=config.xticklabels,
        yticklabels=config.yticklabels,
        square=config.square,
        cbar=config.cbar
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
def plot_roc_curve(
    y_true: Union[List[int], np.ndarray],
    y_pred_proba: Union[List[List[float]], np.ndarray],
    config: ROCConfig = ROCConfig()
) -> None:
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    
    y_test_bin = label_binarize(y_true, classes=list(range(len(config.labels))))
    
    plt.figure(figsize=config.figure_size, dpi=config.dpi)
    
    for i, label in enumerate(config.labels):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(
            fpr,
            tpr,
            color=config.colors[i],
            linewidth=config.line_width,
            label=f"{label} (AUC = {roc_auc:.3f})"
        )
        
    plt.plot(
        [0, 1], 
        [0, 1], 
        color='gray',
        linestyle='--',
        linewidth=config.line_width/2,
        label='Random Chance'
    )
    plt.xlabel('False Positive Rate', fontsize=config.font_size)
    plt.ylabel('True Positive Rate', fontsize=config.font_size)
    plt.title('ROC Curves by Sentiment Class', fontsize=config.font_size+2)
    plt.legend(loc='lower right', fontsize=config.font_size)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
class EvaluateRoBi:
    """
    Evaluation class for the RoBi model that properly utilizes the sliding window approach
    to analyze entire documents when making sentiment predictions. This class follows the same
    evaluation structure as other evaluation classes in the project.
    """
    def __init__(
        self,
        config: SaveLoadConfig = SaveLoadConfig(trial_dir="trial1")
    ):
        """
        Initialize the RoBi evaluator with the best trained model.
        
        Args:
            config (SaveLoadConfig): Configuration specifying where the trained model is saved.
                                    Defaults to trial1 which was the best model.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        
        # Load the trained model and tokenizer
        self.load_model()
    
    def load_model(self) -> None:
        """
        Loads the trained RoBi model and tokenizer from the specified path.
        """
        try:
            self.model, self.tokenizer = load_RoBi(config=self.config)
            self.model = self.model.to(self.device)
            self.model.eval()  # Set model to evaluation mode
            print(f"Successfully loaded RoBi model from {self.config.path}")
        except Exception as e:
            print(f"Error loading RoBi model: {str(e)}")
            raise
    
    def predict(
        self,
        texts: List[str],
        predict_config: PredictConfig = PredictConfig()
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts sentiment for a list of texts using the RoBi model with sliding window.
        This method properly maintains the relationship between chunks and their source articles,
        as well as the correct ordering of chunks within each article.
        
        Args:
            texts (List[str]): List of texts to predict sentiment for
            predict_config (PredictConfig): Configuration for prediction parameters
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing:
                - Array of predicted sentiment class indices (0=Negative, 1=Neutral, 2=Positive)
                - Array of prediction probabilities for each class
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model or tokenizer not loaded. Call load_model() first.")
        
        predictions = []
        probabilities = []
        
        # Process one document at a time
        for doc_idx, text in enumerate(tqdm(texts, desc="Generating RoBi predictions")):
            # Create the sliding window dataset for the current document
            sliding_window = SlidingWindow(
                texts=[text],
                labels=[0],  # Dummy label
                tokenizer=self.tokenizer
            )
            
            if len(sliding_window) == 0:
                print(f"Warning: Empty document after tokenization for document {doc_idx}")
                # Default prediction if document is empty
                predictions.append(1)  # Neutral
                probabilities.append([0.2, 0.6, 0.2])
                continue
            
            # Create a dataloader with batch size 1 to process one chunk at a time
            loader = torch.utils.data.DataLoader(sliding_window, batch_size=1)
            
            # Process all chunks and collect their logits
            all_logits = []
            
            with torch.no_grad():
                for batch in loader:
                    # Move tensors to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    
                    # Get RoBERTa embeddings directly
                    roberta_output = self.model.roberta(
                        input_ids=input_ids, 
                        attention_mask=attention_mask
                    )
                    sequence_output = roberta_output.last_hidden_state
                    
                    # Process through LSTM directly
                    lstm_output, _ = self.model.bilstm(sequence_output)
                    
                    # Apply attention mechanism
                    attention_weights = self.model.attention(lstm_output).squeeze(-1)
                    attention_weights = attention_weights.masked_fill(
                        attention_mask.eq(0), 
                        float('-inf')
                    )
                    attention_weights = torch.softmax(attention_weights, dim=1)
                    
                    # Apply weighted combination
                    weighted_output = torch.bmm(
                        attention_weights.unsqueeze(1),
                        lstm_output
                    ).squeeze(1)
                    
                    # Apply dropout and get logits
                    logits = self.model.fc(self.model.dropout(weighted_output))
                    all_logits.append(logits)
            
            # Combine logits from all chunks of the current document
            if all_logits:
                # Stack all logits and get the mean
                doc_logits = torch.cat(all_logits, dim=0)
                avg_logits = doc_logits.mean(dim=0)
                
                # Get probabilities with softmax
                probs = torch.softmax(avg_logits, dim=0).cpu().numpy()
                pred_class = np.argmax(probs)
                
                predictions.append(int(pred_class))
                probabilities.append(probs)
            else:
                print(f"Warning: No logits generated for document {doc_idx}")
                predictions.append(1)  # Default to neutral
                probabilities.append([0.2, 0.6, 0.2])
        
        return np.array(predictions), np.array(probabilities)
    
    def evaluate(
        self,
        texts: List[str],
        labels: List[int],
        metrics_config: MetricsConfig = MetricsConfig(),
        confusion_config: ConfusionMatrixConfig = ConfusionMatrixConfig(),
        roc_config: ROCConfig = ROCConfig(),
        predict_config: PredictConfig = PredictConfig()
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Evaluates the RoBi model on a test dataset, calculating various metrics
        and producing visualizations.
        
        Args:
            texts (List[str]): List of texts to evaluate on
            labels (List[int]): True sentiment labels for the texts
            metrics_config (MetricsConfig): Configuration for metrics calculation
            confusion_config (ConfusionMatrixConfig): Configuration for confusion matrix plot
            roc_config (ROCConfig): Configuration for ROC curve plot
            predict_config (PredictConfig): Configuration for prediction parameters
            
        Returns:
            Dict[str, Union[float, np.ndarray]]: Dictionary of evaluation metrics
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()
            
        print(f"Evaluating RoBi model on {len(texts)} documents...")
        y_pred, y_prob = self.predict(texts, predict_config)
        y_true = np.array(labels)
        
        # Calculate evaluation metrics
        metrics = calculate_metrics(y_true, y_pred, metrics_config)
        print(metrics['display'])
        
        # Plot confusion matrix
        plot_confusion_matrix(y_true, y_pred, confusion_config)
        
        # Plot ROC curve
        plot_roc_curve(y_true, y_prob, roc_config)
        
        # Add predicted vs true labels to metrics for further analysis
        metrics['predicted_labels'] = y_pred
        metrics['true_labels'] = y_true
        metrics['probabilities'] = y_prob
        
        return metrics
    
class EvaluateVader:
    def __init__(
        self,
        config: VaderConfig = VaderConfig()
    ):
        self.config = config
        self.vader = SentimentIntensityAnalyzer()
        
    def find_optimal_thresholds(
        self,
        texts: List[str],
        labels: List[int]
    ) -> Tuple[float, float]:
        scores = []
        for text in tqdm(texts, desc="Computing VADER scores"):
            scores.append(self.vader.polarity_scores(text)['compound'])
            
        scores = np.array(scores)
        labels = np.array(labels)
        
        neg_scores = scores[labels == 0]
        pos_scores = scores[labels == 2]
        
        thresh_neg = np.mean(neg_scores)
        thresh_pos = np.mean(pos_scores)
        
        return thresh_neg, thresh_pos
    
    def train(
        self,
        texts: List[str],
        labels: List[int]
    ) -> None:
        thresh_neg, thresh_pos = self.find_optimal_thresholds(texts, labels)
        
        self.config.threshold_neg = thresh_neg
        self.config.threshold_pos = thresh_pos
        
        with open(self.config.model_path, 'wb') as f:
            pickle.dump({
                'thresh_neg': thresh_neg,
                'thresh_pos': thresh_pos
            }, f)
            
    def predict(
        self,
        texts: List[str]
    ) -> Tuple[np.array, np.array]:
        predictions = []
        probabilities = []
        
        for text in tqdm(texts, desc="Generating predictions"):
            scores = self.vader.polarity_scores(text)
            compound = scores['compound']
            
            if compound <= self.config.threshold_neg:
                label = 0
            elif compound >= self.config.threshold_pos:
                label = 2
            else:
                label = 1
                
            neg_prob = scores['neg']
            neu_prob = scores['neu']
            pos_prob = scores['pos']
            probs = [neg_prob, neu_prob, pos_prob]
            
            predictions.append(label)
            probabilities.append(probs)
            
        return np.array(predictions), np.array(probabilities)
    
    def evaluate(
        self,
        texts: List[str],
        labels: List[int]
    ) -> Dict[str, Union[float, np.ndarray]]:
        try:
            with open(self.config.model_path, 'rb') as f:
                thresholds = pickle.load(f)
                self.config.threshold_neg = thresholds['thresh_neg']
                self.config.threshold_pos = thresholds['thresh_pos']
        except FileNotFoundError:
            print("No saved thresholds found. Please train the model first.")
            return None
            
        y_pred, y_prob = self.predict(texts)
        y_true = np.array(labels)
        
        metrics = calculate_metrics(y_true, y_pred)
        print(metrics['display'])
        
        plot_confusion_matrix(y_true, y_pred)
        plot_roc_curve(y_true, y_prob)
        
        return metrics
    
class RoBertaDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: RobertaTokenizer,
        max_length: int
    ):
        self.encodings = tokenizer(
            texts, 
            truncation=True, 
            padding=True, 
            max_length=max_length
        )
        self.labels = labels
        
    def __getitem__(
        self, 
        idx: int
    ) -> Dict[str, torch.Tensor]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self) -> int:
        return len(self.labels)
    
class EvaluateRoBertaBase:
    def __init__(
        self,
        config: RoBertaBaseConfig = RoBertaBaseConfig()
    ):
        self.config = config
        self.tokenizer = RobertaTokenizer.from_pretrained(config.model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(
            config.model_name, 
            num_labels=config.num_labels
        ).to(config.device)
        
    def train(
        self,
        texts: List[str],
        labels: List[int]
    ) -> None:
        train_dataset = RoBertaDataset(
            texts, 
            labels, 
            self.tokenizer, 
            self.config.max_length
        )
        
        training_args = TrainingArguments(
            output_dir=self.config.model_path,
            per_device_train_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_epochs,
            learning_rate=self.config.learning_rate,
            save_strategy="epoch",
            logging_steps=100
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset
        )
        
        trainer.train()
        self.model.save_pretrained(self.config.model_path)
        self.tokenizer.save_pretrained(self.config.model_path)
        
    def predict(
        self,
        texts: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        eval_dataset = RoBertaDataset(
            texts, 
            [0]*len(texts),
            self.tokenizer, 
            self.config.max_length
        )
        
        predictions = []
        probabilities = []
        
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(eval_dataset), self.config.batch_size), desc="Predicting"):
                batch_start = i
                batch_end = min(i + self.config.batch_size, len(eval_dataset))
                
                batch_items = [eval_dataset[j] for j in range(batch_start, batch_end)]
                
                batch = {
                    'input_ids': torch.stack([item['input_ids'] for item in batch_items]).to(self.config.device),
                    'attention_mask': torch.stack([item['attention_mask'] for item in batch_items]).to(self.config.device)
                }
                
                outputs = self.model(**batch)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(predictions), np.array(probabilities)
    
    def evaluate(
        self, 
        texts: List[str], 
        labels: List[int]
    ) -> Dict[str, Union[float, np.ndarray]]:
        if os.path.exists(self.config.model_path):
            model = RobertaForSequenceClassification.from_pretrained(
                self.config.model_path,
                num_labels=self.config.num_labels
            )
            self.model = model.to(self.config.device)
                
        y_pred, y_prob = self.predict(texts)
        y_true = np.array(labels)
        
        metrics = calculate_metrics(y_true, y_pred)
        print(metrics['display'])
        
        plot_confusion_matrix(y_true, y_pred)
        plot_roc_curve(y_true, y_prob)
        
        return metrics
    
class DistilBertDataset(Dataset):
    def __init__(
        self, 
        texts: List[str],
        labels: List[int], 
        tokenizer: DistilBertTokenizer, 
        max_length: int
    ):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
class EvaluateDistilBert:
    def __init__(
        self,
        config: DistilBertConfig = DistilBertConfig()
    ):
        self.config = config
        self.tokenizer = DistilBertTokenizer.from_pretrained(config.model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            config.model_name, 
            num_labels=config.num_labels
        ).to(config.device)
        
    def train(
        self,
        texts: List[str],
        labels: List[int]
    ) -> None:
        train_dataset = DistilBertDataset(
            texts, 
            labels, 
            self.tokenizer, 
            self.config.max_length
        )
        
        training_args = TrainingArguments(
            output_dir=self.config.model_path,
            per_device_train_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_epochs,
            learning_rate=self.config.learning_rate,
            save_strategy="epoch",
            logging_steps=100,
            weight_decay=0.01,
            warmup_steps=500
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset
        )
        
        trainer.train()
        self.model.save_pretrained(self.config.model_path)
        self.tokenizer.save_pretrained(self.config.model_path)
        
    def predict(
        self,
        texts: List[str]
    ) -> Tuple[np.array, np.array]:
        eval_dataset = DistilBertDataset(
            texts, 
            [0]*len(texts),
            self.tokenizer, 
            self.config.max_length
        )
        
        predictions = []
        probabilities = []
        
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(eval_dataset), self.config.batch_size), desc="Predicting"):
                batch_start = i
                batch_end = min(i + self.config.batch_size, len(eval_dataset))
                
                batch_items = [eval_dataset[j] for j in range(batch_start, batch_end)]
                
                batch = {
                    'input_ids': torch.stack([item['input_ids'] for item in batch_items]).to(self.config.device),
                    'attention_mask': torch.stack([item['attention_mask'] for item in batch_items]).to(self.config.device)
                }
                
                outputs = self.model(**batch)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(predictions), np.array(probabilities)
    
    def evaluate(
        self,
        texts: List[str],
        labels: List[int]
    ) -> Dict[str, Union[float, np.ndarray]]:
        if os.path.exists(self.config.model_path):
            model = DistilBertForSequenceClassification.from_pretrained(
                self.config.model_path,
                num_labels=self.config.num_labels
            )
            self.model = model.to(self.config.device)
                
        y_pred, y_prob = self.predict(texts)
        y_true = np.array(labels)
        
        metrics = calculate_metrics(y_true, y_pred)
        print(metrics['display'])
        
        plot_confusion_matrix(y_true, y_pred)
        plot_roc_curve(y_true, y_prob)
        
        return metrics
    
class EvaluateTextBlob:
    def __init__(
        self,
        config: TextBlobConfig = TextBlobConfig()
    ):
        self.config = config
        
    def find_optimal_thresholds(
        self, 
        texts: List[str], 
        labels: List[int]
    ) -> Tuple[float, float]:
        scores = []
        
        for text in tqdm(texts, desc="Computing TextBlob Scores"):
            blob = TextBlob(text)
            scores.append(blob.sentiment.polarity)
            
        scores = np.array(scores)
        labels = np.array(labels)
        
        neg_scores = scores[labels == 0]
        pos_scores = scores[labels == 2]
        
        thresh_neg = np.mean(neg_scores)
        thresh_pos = np.mean(pos_scores)
        
        return thresh_neg, thresh_pos
    
    def train(
        self,
        texts: List[str],
        labels: List[str]
    ) -> None:
        thresh_neg, thresh_pos = self.find_optimal_thresholds(texts, labels)
        
        self.config.threshold_neg = thresh_neg
        self.config.threshold_pos = thresh_pos
        
        with open(self.config.model_path, "wb") as f:
            pickle.dump({
                'thresh_neg': thresh_neg,
                "thresh_pos": thresh_pos
            }, f)
            
    def predict(
        self,
        texts: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        predictions = []
        probabilities = []
        
        for text in tqdm(texts, desc="Generating predictions"):
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity 
            subjectivity = blob.sentiment.subjectivity 
            
            if polarity <= self.config.threshold_neg:
                label = 0
                neg_prob = 0.5 + abs(polarity)/2
                remaining = 1 - neg_prob
                probs = [
                    neg_prob, 
                    remaining * 0.7,
                    remaining * 0.3
                ]
            elif polarity >= self.config.threshold_pos:
                label = 2
                pos_prob = 0.5 + polarity/2
                remaining = 1 - pos_prob
                probs = [
                    remaining * 0.3,
                    remaining * 0.7,
                    pos_prob
                ]
            else:
                label = 1
                neu_prob = 0.5 + (1 - subjectivity)/2
                remaining = 1 - neu_prob
                probs = [
                    remaining * 0.5,
                    neu_prob, 
                    remaining * 0.5 
                ]
                
            probs = np.array(probs)
            probs = probs / probs.sum()
            
            predictions.append(label)
            probabilities.append(probs)
            
        return np.array(predictions), np.array(probabilities)
    
    def evaluate(
        self,
        texts: List[str],
        labels: List[int]
    ) -> Dict[str, Union[float, np.ndarray]]:
        if os.path.exists(self.config.model_path):
            with open(self.config.model_path, 'rb') as f:
                thresholds = pickle.load(f)
                self.config.threshold_neg = thresholds['thresh_neg']
                self.config.threshold_pos = thresholds['thresh_pos']
                
        y_pred, y_prob = self.predict(texts)
        y_true = np.array(labels)
        
        metrics = calculate_metrics(y_true, y_pred)
        print(metrics['display'])
        
        plot_confusion_matrix(y_true, y_pred)
        plot_roc_curve(y_true, y_prob)
        
        return metrics