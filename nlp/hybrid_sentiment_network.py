###################################################################################################################
#                                                    RoBi                                                         #
#                                Hybrid RoBERTa-BiLSTM Sentiment Analysis Model                                   #
#                                   Using a Sliding Window Sequence Processor                                     #
#                                                                                                                 #
#                                            Author: Elijah Taber                                                 #
###################################################################################################################
"""
From Oxford Dictionary

Sentiment: a feeling or an opinion, especially one based on emotions.
"""
import torch
import transformers
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
from typing import List, Tuple, Dict, Any
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import random
import os
import json
import numpy as np
import shutil
from tqdm import tqdm
from .robi_config import (
    SlidingWindowConfig, 
    RoBiConfig, 
    TrainConfig, 
    SaveLoadConfig, 
    PredictConfig,
    BayesianOptConfig
)

# Set random seeds for reproducibility, each package must be individually addressed to lock in randomized settings under the hood
random.seed(10) # standard python
np.random.seed(10) # numpy
transformers.set_seed(10) # transformers
torch.manual_seed(10) # torch
if torch.cuda.is_available(): # GPU
    torch.cuda.manual_seed_all(10)

class SlidingWindow(Dataset):
    """
    A custom Dataset class that implements a sliding window approach for processing long texts.
    
    This class takes a list of texts and their corresponding labels, and splits long texts into overlapping chunks using the RoBERTa 
    tokenizer. This allows for processing of texts that are longer than the maximum sequence length that RoBERTa (and many other 
    encoder transformers) can handle (512 tokens). Truncation, stride, and overflowing tokens operate together a synergistic manner, 
    where all tokens are kept and continuously retokenized in a sequential stride of length 512. The return_overflowing_tokens=True 
    is what unlocks the sliding window behavior as it applies the tokenizer to the entire text, rather than the first sequence. 
    Without it, any tokens beyond the max sequence length would be dropped immediately during tokenization. The unique case here of 
    this sliding window approach, is that the chunks reatain their respective sentiment labels, allowing RoBi to process each chunk 
    independently, resulting in the BiLSTM to capture the sequential dependencies between the chunks of a single corpus.
    
    Sliding Window Flow:
    * Original Text Example: 1000 tokens, label: Positive
    * Chunk id 1: tokens 0-512, label: Positive
    * Chunk id 2: tokens 256-768, lablel: Positive
    * Chunk id 3: tokens 512-1000 (padded to 512), label: Positive

    Text:    [0.....256.....512.....768.....1000]
    Chunk 1: [0==========512]
    Chunk 2:        [256==========768]
    Chunk 3:              [512==========1000+padding]
             |--overlap--|
             
    For text that is equal to or shorter than the max sequence length, the tokenizer will return a single chunk with the text and 
    label. This allows for text of all sizes to be processed by RoBi, making sure that no information is lost during tokenization.
    For text that is shorter than the max sequence length, the tokenizer will pad the remaining tokens by filling the empty sequence
    space with padding token IDs. The tokenizer will know what is real text and what is padded text through the attention mask mechanism,
    where a 1 indicates a real token and a 0 indicates a padded token (telling RoBi to ignore all tokens with an attention mask of 0).
    This same logic applies the final sequence of the last chunk in the sliding window, where the last chunk is likely not exactly 512
    tokens long, so the tokenizer will pad the remaining tokens to maintain a consistent sequence length (as RoBi requires). 

    Attributes:
        - texts (List[str]): List of input texts.
        - labels (List[int]): List of corresponding labels for each text.
        - tokenizer (RobertaTokenizer): RoBERTa tokenizer for encoding the texts.
        - max_length (int): Maximum length of each chunk.
        - stride (int): Number of overlapping tokens between adjacent chunks.
        - chunks (List[List[int]]): List of tokenized chunks.
        - chunk_labels (List[int]): List of labels corresponding to each chunk.
    """

    def __init__(
        self, 
        texts: List[str], 
        labels: List[int], 
        tokenizer: RobertaTokenizer, 
        config: SlidingWindowConfig = SlidingWindowConfig()
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = config.max_length
        self.stride = config.stride

        self.chunks = []
        self.chunk_labels = []
        self.article_ids = [] 
        self.chunk_positions = [] 

        # Process each chunk of text and split into overlapping strides
        for article_idx, (text, label) in enumerate(zip(texts, labels)):
            # Tokenize each article with overlapping sequences of 256 tokens, text that is over RoBERTa's max sequence length will
            # consist of 256 tokens of 1 chunk and 256 tokens of the next chunk. Remaining tokens are padded to maintain same sequence length
            encodings = self.tokenizer(
                text,
                truncation=True,                # ensures each chunk is exactly 512 tokens (max sequence length of RoBERTa)
                max_length=self.max_length,
                stride=self.stride,
                return_overflowing_tokens=True, # keep creating chunks until all tokens are processed
                padding='max_length',           # pad the last chunk to max sequence length
                return_attention_mask=True,     # get mask identifying real vs padding tokens
                return_tensors=None             # return Python lists
            )

            # Extract tokenized chunks and their attention masks
            input_ids_list = encodings['input_ids']           # token IDs for each chunk
            attention_mask_list = encodings['attention_mask'] # attention masks for each chunk
            
            # Handle single-chunk case (short texts): if text fits into a single sequence, the tokenizer returns a single list instead of a nested list
            if not isinstance(input_ids_list[0], list):     # wrap in list for consistency
                input_ids_list = [input_ids_list]
                attention_mask_list = [attention_mask_list]

            # Store each chunk with its token ID and attention mask
            for chunk_idx in range(len(input_ids_list)):
                self.chunks.append({
                    'input_ids': input_ids_list[chunk_idx],
                    'attention_mask': attention_mask_list[chunk_idx]
                })
                self.chunk_labels.append(label)        # each chunk will inherit the article's sentiment
                self.article_ids.append(article_idx)   # track which article this chunk is from
                self.chunk_positions.append(chunk_idx) # track the correct order of each article's chunk (intro in front, conclusion in back)
                    
    def __len__(self) -> int:
        """Returns the total number of chunks in the dataset."""
        return len(self.chunks)

    def __getitem__(
        self, 
        idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Retrieves a specific chunk and its corresponding label. Each passing window through the transformer
        will retain is corresponding sentiment label. This method converts the chunk data into PyTorch tensors
        and returns out a reference map of tensors that is ready to be fed through RoBi.

        Args:
            - idx (int): Index of the chunk to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - 'input_ids': Tensor of token IDs for the chunk.
                - 'attention_mask': Tensor indicating which tokens should be attended to.
                - 'labels': Tensor containing the label for the chunk.
        """
        chunk = self.chunks[idx]
        label = self.chunk_labels[idx]
        
        return {
            'input_ids': torch.tensor(chunk['input_ids'], dtype=torch.long),           # token IDs (tensor shape: [512])
            'attention_mask': torch.tensor(chunk['attention_mask'], dtype=torch.long), # binary mask indicating real vs padded tokens (tensor shape: [512])
            'labels': torch.tensor(label, dtype=torch.long)                            # corresponding sentiment label (tensor shape: [1])
        }

class RoBi(nn.Module):
    """
    RoBi (RoBERTa-BiLSTM) model for sentiment analysis. A bidirectional hybrid synergy approach using sliding window tensors.

    A hybrid neural architecture that combines RoBERTa's transformer-based contextual understanding with BiLSTM's sequential 
    processing capabilities, where RoBERTa first extracts rich contextual embeddings through its self-attention mechanism 
    (capturing long-range dependencies and semantic relationships), then feeds these enhanced representations into a bidirectional 
    LSTM that processes the sequence both forwards and backwards to capture temporal dependencies and narrative flow. The model 
    incorporates a specialized attention mechanism that learns to weight the importance of different parts of the text for sentiment 
    analysis, effectively identifying sentiment-heavy phrases and their relationships, while its dual-directional nature allows it
    to understand how earlier and later parts of the text influence each other's sentiment (e.g., "I thought it would be great, but..."). 
    This architecture is particularly effective for long-form text analysis as it maintains contextual understanding across extended 
    sequences through its sliding window approach, where each window inherits both global context from RoBERTa and local sequential patterns 
    from the BiLSTM, while the attention mechanism helps focus on the most sentiment-relevant parts of each window.
    
    This can be thought of as reading a book:
    - Wrong Way: Read each chapter separately, form separate opinions, then average those opinions.
    - RoBi's Way: Read the entire book in order, form one cohesive opinion based on the whole text.

    The final classification layer integrates all the learned representations - contextual meaning, sequential patterns, and 
    attention-weighted importance - to consider both the subtle nuances of language and the broader narrative context of the text. 
    
    The sliding window and RoBi synergy process:
    1. Text Input Processing
        - Text is passed through the sliding window
        - If text ≤ 512 tokens, it creates a single chunk and pads the remaining tokens up to a sequence length of 512
        - If text > 512 tokens, it creates overlapping chunks of 512 tokens with a stride of 256 tokens
    2. For each chunk:
        - Converts tokens to IDs
        - Creates attention mask for each token (1s for real tokens, 0s for padding)
        - Tracks chunk position in the article for proper reassembly
        - Track article ID for each stride so that each chunk is assinged to the correct article along with its position and sentiment label
        - Passes through RoBi's neural pipeline for sentiment analysis:
            * RoBERTa embedding layer
            * BiLSTM for sequential understanding
            * Attention mechanism for sentiment-relevant context
            * Final classification layer for actualy sentiment prediction
    3. Final Sentiment Prediction (after training):
        - Combines logits from all chunks
        - Averages the logits to get a single sentiment prediction for the entire text
        - Returns both sentiment label and confidence scores (for each class)

    Network Architecture:
        - roberta (RobertaModel): Pre-trained RoBERTa model.
        - bilstm (nn.LSTM): Bidirectional Long-Short-Term-Memory layer.
        - attention (nn.Sequential): Attention mechanism.
        - fc (nn.Linear): Final fully connected layer for classification.
        - dropout (nn.Dropout): Dropout layer for regularization.
    """

    def __init__(
        self, 
        config: RoBiConfig = RoBiConfig()
    ):
        """
        Architecture Flow:
            Input Text → RoBERTa → BiLSTM → Attention → Dropout → Classification
                          ↓          ↓         ↓          ↓           ↓
                        Context    Sequence  Important   Prevent    Sentiment
                     Understanding  Flow       Parts   Overfitting  Prediction
                   
        Dimensional Analysis:  
        Text  → RoBERTa  →  BiLSTM  →  Attention  →  FC  →  Sentiment
        [512] →  [768]   →  [768]   →   [768]     →  [3] →    class
        
        RoBERTa Layer:   
            Input: Raw tokenized text such as: "This article was fantastic!"
            Output: Contextual embeddings
            Shape: [batch_size, sequence_length, 768]
                    
        BiLSTM Layer:
            Takes RoBERTa embeddings and processes sequence in both directions:
            Forward:  → → → →
            Backward: ← ← ← ←
            Output shape: [batch_size, seq_length, 384*2]
            
        Attention Mechanism:
            "This" "article" "was" "fantastic" "!"
            Example weights: [0.1, 0.5, 0.2, 0.8, 0.1]
                              low  med  low  high low
                              
        Final Classification:
            [0.1, 0.2, 0.7] → Positive
            (Neg, Neu, Pos)
        """
        super(RoBi, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-large')
        self.bilstm = nn.LSTM(
            input_size=config.input_size,   # same size as RoBERTa large's embedding dimensions of 1024
            hidden_size=config.hidden_size, # BiLSTM output hidden state
            num_layers=config.num_layers,   # stacked BiLSTM layers, where the second layer takes the output of the first
            batch_first=config.batch_first,
            bidirectional=config.bidirectional
        )
        self.attention = nn.Sequential(
            nn.Linear(config.hidden_size * 2, 1), # bidirectional size of 384 → 768 dimensions
            nn.Tanh()                             # squash scores to [-1, 1] range
        )
        self.fc = nn.Linear(
            config.hidden_size * 2, 
            config.num_classes                    # sentiment: [0]Negative, [1]Neutral, [2]Positive
        )
        self.dropout = nn.Dropout(config.dropout) # randomly zero out elements during training

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        article_ids: torch.Tensor = None,
        chunk_positions: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Executes the forward pass of the RoBi model through a multi-stage process. The method first uses RoBERTa's transformer 
        architecture to extract contextual embeddings. These embeddings then flow through a bidirectional LSTM that processes the 
        sequence in both directions. Then a hierarchical attention mechanism is used to first compute token-level importance within chunks, 
        which weighs the contribution of each chunk with the final sentiment, handling both local and global context in long documents.
        
        Processing Flow:
            Text Chunks → RoBERTa Embeddings → Article-Aware Processing → BiLSTM → Attention → Classification

        Args:
            - input_ids: Tokenized text sequences converted to numerical IDs that RoBERTa can process. 
                         Each sequence is padded or truncated to a fixed length of 512 tokens.
            - attention_mask: Binary indicators marking which tokens contain actual content (1) versus which 
                              are padding tokens (0). Essential for ensuring the model only processes meaningful tokens.
            - article_ids: Unique identifiers that associate each chunk with its source article, allowing chunks from the 
                           same long article to be processed together. Used to reconstruct article context across multiple chunks.
            - chunk_positions: Sequential position markers showing where each chunk belongs in its original article, 
                               allowing for correct reassembly of long texts that were split into multiple chunks.

        Returns:
            torch.Tensor: The output logits for each class.
        """
        # 1. RoBERTa Embedding: pass all chunks through RoBERTa to get contextual embeddings
        roberta_output = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask # tells RoBERTa which tokens to pay attention to
        )
        
        # Get the final hidden states from RoBERTa
        sequence_output = roberta_output.last_hidden_state # shape [batch_size, seq_length, hidden_size]
        
        # 2. Article-Aware Processing: process chunks belonging to the same article together
        if article_ids is not None:
            # Get unique article IDs in this batch
            unique_articles = torch.unique(article_ids)
            batch_outputs = []
            
            # Process each article's chunks together
            for article_id in unique_articles:
                # Create a boolean mask identifying which chunks belong to this article
                article_mask = (article_ids == article_id) # e.g., for article 1: [True, True, False, False, False]
                
                # Using the mask to select only this article's chunks from RoBERTa output
                article_chunks = sequence_output[attention_mask]
                article_attention_mask = attention_mask[article_mask]
                
                # 3. Chunk Ordering: get positions of each chunk within this article
                chunk_pos = chunk_positions[article_mask]
                sorted_indicies = torch.argsort(chunk_pos) # e.g., if chunks were [2, 1, 3], sorted_indices would be [1, 0, 2]
                
                # Reorder chunks and attention mask to match original article flow
                article_chunks = article_chunks[sorted_indicies]
                article_attention_mask = article_attention_mask[sorted_indicies]
                
                # 4. BiLSTM Processing: process all chunks of this article through BiLSTM together, where BiLSTM sees chunks 
                # in the correct order, maintaining article flow through long-short-term memory
                lstm_output, _ = self.bilstm(article_chunks)
                
                # 5. Attention Mechanism: compute attention scores for each token, applying attention across the entire article's content
                attention_weights = self.attention(lstm_output).squeeze(-1)
                
                # Mask out padding tokens by setting their attention weights to negative infinity, ensuring they'll have zero weight after softmax
                attention_weights = attention_weights.masked_fill(
                    article_attention_mask.eq(0),
                    float("-inf")
                )
                
                # Convert attention weights to probabilities using softmax
                attention_weights = torch.softmax(attention_weights, dim=1) # each token now has a weight between 0 and 1, sum of weights = 1
                
                # 6. Weighted Combination: combine token representations based on their attention weights
                article_output = torch.bmm(
                    attention_weights.unsqueeze(1), # add dimension for matrix multiplication
                    lstm_output
                ).squeeze(1)                        # remove extra dimension
                
                # Average the outputs if there were multiple chunks, this gives one representation for the entire article
                article_output = article_output.mean(0, keepdim=True)
                batch_outputs.append(article_output)
                
            # 7. Combine All Article Outputs: stack all article outputs back together into a batch
            output = torch.cat(batch_outputs, dim=0)
            
        else:
            # If no article IDs provided, process each chunk independently
            lstm_output, _ = self.bilstm(sequence_output)
            attention_weights = self.attention(lstm_output).squeeze(-1)
            attention_weights = attention_weights.masked_fill(
                attention_mask.eq(0),
                float('-inf')
            )
            attention_weights = torch.softmax(attention_weights, dim=1)
            output = torch.bmm(
                attention_weights.unsqueeze(1),
                lstm_output
            ).squeeze(1)
            
        # 8. Final Classification: apply dropout to prevent overfitting
        return self.fc(self.dropout(output)) # pass through final linear layer to get sentiment scores

def train_RoBi(
    model: nn.Module, 
    train_dataloader: DataLoader, 
    val_dataloader: DataLoader, 
    config: TrainConfig = TrainConfig(),
    save_path: str = None
) -> nn.Module:
    """
    Trains RoBi using GPU acceleration and memory management techniques.
    
    Implements a training pipeline for RoBi incorporating gradient accumulation, learning rate scheduling, and early stopping. 
    The training process utilizes a one-cycle learning rate policy which initially increases the learning rate to help escape 
    local minima, then gradually decreases it for fine-tuned convergence. This method uses gradient accumulation to simulate 
    larger batch sizes while maintaining memory efficiency, and implements early stopping to prevent overfitting by monitoring
    validation loss trends.

    Training Pipeline:
        1. Forward Pass → Loss Computation → Backward Pass (repeated for gradient_accumulation_steps)
        2. Gradient Clipping → Optimizer Step → LR Scheduling
        3. Validation → Early Stopping Check
        
    Learning Rate Optimization:
    
           Learning Rate                   Validation Loss   
           
        ↑      Peak LR                ↑                     
        │      ╭───╮                  │╲                     
        │     /     ╲                 │ ╲                    
        │    /       ╲                │  ╲                   
        │   /         ╲               │   |   ← Peak LR               
        │  /           ╲              │    ╲__                 
        │ /             ╲             │       ╲__  ╭──╮ 
        │/               ╲            │          ╰─╯← best_model.pth                  
        └─────────────────┘           └──────────────────     
       Start            End           Start           End     
      Warm-up        Cool-down        Loss            Loss          
      
    Start:   lr/10
    Peak:    lr 
    End:     lr/1000

    Args:
        model (nn.Module): The RoBi model to train.
        train_dataloader (DataLoader): DataLoader for the training data.
        val_dataloader (DataLoader): DataLoader for the validation data.
        config (TrainConfig): Configuration for training parameters.
        save_path (str, optional): Path to save the best model to. Defaults to None.

    Returns:
        Tuple[nn.Module, float]: The trained RoBi model and its best validation loss.
    """
    model.to(config.device)
    
    # Initialize cross entropy loss - this measures how far off the predictions are from true labels 
    criterion = nn.CrossEntropyLoss()
    
    # Adam optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), 
                            lr=config.learning_rate) # this is the maximum that the learning rate will ramp up to before decaying
    
    # Calculate total training steps for learning rate scheduler, accounts for gradient accumulation in step count
    total_steps = len(train_dataloader) // config.gradient_accumulation_steps * config.num_epochs
    
    # This scheduler implements the 1cycle policy from the paper "Super-Convergence"
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        steps_per_epoch=len(train_dataloader) // config.gradient_accumulation_steps,
        epochs=config.num_epochs
    )

    # Initialize tracking variables for early stopping
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0 # if val_loss does not decrease for 5 epochs, stop training
    
    for epoch in range(config.num_epochs):
        model.train()          # set the model in training mode (enables dropout, batch norm updates)
        total_loss = 0         # tracks cumulative loss for this epoch
        optimizer.zero_grad()  # reset gradients at start of epoch
        
        # Inner loop - processes each batch in the epoch with a progress bar
        for batch_idx, batch in tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch + 1}/{config.num_epochs}"):
            
            # Move batch data to same device as model (GPU/CPU)
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)
            
            # Forward pass - calls model.forward() internally
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)                 # loss between model predictions and true labels
            loss = loss / config.gradient_accumulation_steps  # scaling ensures consistent gradients regardless of accumulation
            
            # Backward pass - computes gradients of loss with respect to model parameters
            loss.backward()
            
            # Accumulate loss for logging (unscale for true loss value)
            total_loss += loss.item() * config.gradient_accumulation_steps
            
            # Gradient clipping: scales down gradient values to prevent the "exploding gradient problem", which RNNs are susceptible to
            # Very important in sentiment models due to the wide range of emotions humans write their text in, causing the model to overcompensate
            # Step optimizer and scheduler every accumulation_steps batches
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    config.gradient_clip_max_norm
                )
                
                # Update weights and learning rate
                optimizer.step()       # apply accumulated gradients
                scheduler.step()       # adjust learning rate
                optimizer.zero_grad()  # clear gradients for next accumulation
                
            # Free up memory
            del outputs
            torch.cuda.empty_cache()

        # Compute the average loss for the epoch
        avg_train_loss = total_loss / len(train_dataloader)
        
        # Validation phase
        model.eval()  # set the model in evaluation mode
        val_loss = 0
        
        # Disable gradient calculation during evaluation
        with torch.no_grad():            
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(config.device)
                attention_mask = batch['attention_mask'].to(config.device)
                labels = batch['labels'].to(config.device)

                # Forward pass only (no backward pass needed)
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Clear memory
                del outputs
                torch.cuda.empty_cache()

        # Compute the average validation loss for the epoch
        avg_val_loss = val_loss / len(val_dataloader)
        
        print(f"Epoch {epoch + 1}/{config.num_epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}")

        # Early stopping check - is this the best model so far?
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss                  # update
            best_model_state = model.state_dict().copy()  # save weights
            epochs_without_improvement = 0                # reset
            
            # Save best model if a path is provided
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(best_model_state, save_path)
        else:
            epochs_without_improvement += 1

        # Early stopping trigger check
        if epochs_without_improvement >= config.early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Load the best model state
    if best_model_state:
        model.load_state_dict(best_model_state)
        
    return model, best_val_loss

def save_RoBi(
    model: nn.Module, 
    tokenizer: RobertaTokenizer, 
    config: SaveLoadConfig = SaveLoadConfig(), 
    best_model_path: str = 'best_model.pth'
):
    """
    Saves the RoBi model and its tokenizer to a directory. The model will be saved as a 
    pytorch file, which uses python's pickle utility for serialization. This function
    saves the model state from best_model_path to the specified directory along with
    the tokenizer and configuration.

    Args:
        model (nn.Module): The RoBi model to save.
        tokenizer (RobertaTokenizer): The tokenizer used with the model.
        path (str): The directory path to save the model and tokenizer to.
    """
    os.makedirs(config.path, exist_ok=True)
    
    # Check if the best model checkpoint exists
    if os.path.exists(best_model_path):
        # If we're given a pre-saved model checkpoint, copy it
        model_state = torch.load(best_model_path)
        torch.save(model_state, os.path.join(config.path, 'robi_model.pt'))
    else:
        # If no checkpoint exists, save the current model state
        torch.save(model.state_dict(), os.path.join(config.path, 'robi_model.pt'))
    
    tokenizer.save_pretrained(config.path)
    
    if config.trial_dir:
        config_dict = {
            'base_path': config.base_path,
            'trial_dir': config.trial_dir
        }
        with open(os.path.join(config.path, 'save_config.json'), 'w') as f:
            json.dump(config_dict, f)

def load_RoBi(
    config: SaveLoadConfig = SaveLoadConfig(),
    robi_config: RoBiConfig = RoBiConfig()
) -> Tuple[nn.Module, RobertaTokenizer]:
    """
    Loads the saved RoBi model from the RoBi directory and loads in the RoBERTa tokenizer
    that was used to train the model.

    Args:
        path (str): The directory path to load the model and tokenizer from.
        num_classes (int, optional): Number of classes for the model. Defaults to 5.

    Returns:
        Tuple[nn.Module, RobertaTokenizer]: The loaded RoBi model and its tokenizer.
    """
    model = RoBi(config=robi_config)
    
    if not os.path.exists(config.path):
        raise FileNotFoundError(f"Model directory not found: {config.path}")
        
    model_path = os.path.join(config.path, 'robi_model.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.load_state_dict(torch.load(model_path))
    
    tokenizer = RobertaTokenizer.from_pretrained(config.path)
    
    return model, tokenizer

def predict_sentiment(
    model: nn.Module, 
    tokenizer: RobertaTokenizer, 
    text: str, 
    device: torch.device,
    config: PredictConfig = PredictConfig()
) -> str:
    """
    Predicts sentiment by processing text through RoBi's neural pipeline. Handles both single-chunk and multi-chunk 
    texts through a sliding window approach. RoBERTa will output logits that represent confidence scores for each class,
    which for this model is 3 (Negative, Neutral, Positive). The higher the logit value, the higher the model places 
    confidence in that class. The highest value is considered the actual sentiment. Technically, this function can handle
    a list of texts to predict multiple sentiment labels itteratively, but this specific use case is designed for a single corpus. 

    Output Processing:
        Logits → Softmax → Max Index → Label
        
    Example:
        logits:           [1.2, 4.5, 0.8] 
        indices:            0    1    2  
        class:             Neg  Neu  Pos
        softmax          [0.15, 0.75, 0.10]
        
        → max(logits) = 4.5 → Neutral (final classification) 
        
        Confidence in classification: 75%
    
    Args:
        model (nn.Module): The trained RoBi model.
        tokenizer (RobertaTokenizer): The tokenizer used with the model.
        text (str): Unseen input text to analyze.
        device (torch.device): The device to run the model on.
        max_length (int, optional): Maximum length of each chunk. Defaults to 512.
        stride (int, optional): Number of overlapping tokens between chunks. Defaults to 256.

    Returns:
        int: The predicted sentiment class: 0 (Negative), 1 (Neutral), or 2 (Positive).
    """
    model = model.to(device)
    
    # Set the model to evaluation mode (disables dropout and freezes batch norm statistics)
    model.eval()
    
    # The sliding window requires a label for each chunk as was designed during training, a new label will be assigned for final prediction
    sliding_window_dataset = SlidingWindow(
        texts=[text],
        labels=[1], # dummy label
        tokenizer=tokenizer,
        config=SlidingWindowConfig()
    )
    
    if len(sliding_window_dataset) == 0:
        raise ValueError("Empty or invalid text provided")
    
    article_ids = torch.zeros(len(sliding_window_dataset))  # all chunks belong to same article (0)
    chunk_positions = torch.tensor(sliding_window_dataset.chunk_positions) # track the order of each chunk
    
    all_logits = []
    
    # Disable gradient calculation for inference ,this reduces memory usage and speeds up computation
    with torch.no_grad():
        for i in range(len(sliding_window_dataset)):
            chunk = sliding_window_dataset[i]
            input_ids = chunk['input_ids'].unsqueeze(0).to(device)
            attention_mask = chunk['attention_mask'].unsqueeze(0).to(device)
            
            curr_article_ids = article_ids[i].unsqueeze(0).to(device)
            curr_chunk_positions = chunk_positions[i].unsqueeze(0).to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                article_ids=curr_article_ids,
                chunk_positions=curr_chunk_positions
            )
            all_logits.append(outputs)
    
    combined_logits = torch.cat(all_logits, dim=0)
    final_logits = torch.mean(combined_logits, dim=0)
    probabilities = torch.softmax(final_logits, dim=0)
    predicted_class = torch.argmax(probabilities).item()
    
    results = {
        'sentiment': config.sentiment_labels[predicted_class], # map the predicted class index to the sentiment label
        'probabilities': {
            'negative': probabilities[0].item(),
            'neutral': probabilities[1].item(),
            'positive': probabilities[2].item()
        }
    }
    
    return results

class BayesianTuner:
    def __init__(
        self, 
        config: BayesianOptConfig = BayesianOptConfig()
    ):
        """
        A Bayesian Optimization tuner for hyperparameter optimization of RoBi.
        
        This class implements Bayesian Optimization using Gaussian Process Regression (GPR) to search through the 
        hyperparameter space of RoBi. The tuner uses a Matérn kernel for the GPR, which is particularly well-suited 
        for hyperparameter optimization as it makes fewer smoothness assumptions by maintaining uncertainty in unexplored 
        regions compared to the RBF kernel. Gaussian Process Regression in Bayesian optimization assumes that similar 
        hyperparameter values will likely give similar performance results (a smoothness assumption), which helps it make 
        educated guesses about untested hyperparameter combinations based on the results it has already seen. The twice-
        differentiable property (nu=2.5) is used because it provides just the right amount of smoothness - not too rigid 
        to miss important variations in performance, but smooth enough to make reasonable predictions about how well nearby 
        hyperparameter values will perform. 
        
        The optimization process consists of two phases:
        1. Initial exploration: Random sampling of hyperparameters to build initial understanding
        2. Guided optimization: Using GPR to predict promising hyperparameter combinations

        Attributes:
            config (BayesianOptConfig): Configuration object containing optimization parameters
            trials_results (list): Storage for results of each optimization trial
            gpr (GaussianProcessRegressor): Gaussian Process model for optimization
            X (list): Storage for tried hyperparameter configurations
            y (list): Storage for corresponding performance metrics
        """
        self.config = config
        self.trials_results = []                              # will contain metrics, hyperparameters, and trial tracking
        self.gpr = GaussianProcessRegressor(
            kernel=Matern(nu=config.nu),                      # differentiate twice over to generate relatively smooth functions
            n_restarts_optimizer=config.n_restarts_optimizer, # multiple restarts help avoid local optima
            random_state=config.random_state
        )
        self.first_trial = True
        self.X = []  # stores hyperparameter configurations, X will store the actual hyperparameter values tried
        self.y = []  # stores corresponding validation losses, y will store the validation loss for each configuration
        
        # Create base directory
        os.makedirs(self.config.base_path, exist_ok=True)
        
    def _sample_parameters(self) -> Dict[str, Any]:
        """
        Samples the next set of hyperparameters to evaluate using Gaussian Process optimization.
        Uses a default baseline configuration for the first trial, then uses Bayesian optimization
        to explore the hyperparameter space intelligently using probabilty based on previous results.
        
        Processing Flow:
            First Trial → Use Baseline Configuration (from RoBiConfig/TrainConfig)
                ↓
            Subsequent Trials → GPR Optimization
                1. Normalize Previous Results
                2. Fit GPR Model
                3. Generate Candidate Points
                4. Calculate Acquisition Function (UCB)
                5. Select Most Promising Point
        
        The method uses an Upper Confidence Bound (UCB) acquisition function:
            UCB = μ(x) + κσ(x)
            where:
                μ(x) is the predicted mean performance
                σ(x) is the predicted uncertainty
                κ is the exploration-exploitation trade-off parameter
        
        Returns:
            Dict[str, Any]: A dictionary containing the sampled hyperparameters where:
                - 'hidden_size': Size of LSTM hidden layers (int)
                - 'num_layers': Number of stacked LSTM layers (int)
                - 'dropout': Dropout probability (float)
                - 'learning_rate': Learning rate for optimization (float)
                - 'gradient_accumulation_steps': Number of steps to accumulate gradients (int)
        """
        # For first trial, use default configurations as an informed starting point rather than random guessing
        if self.first_trial:
            self.first_trial = False
            robi_defaults = RoBiConfig()
            train_defaults = TrainConfig()
            
            return {
                'hidden_size': robi_defaults.hidden_size,
                'num_layers': robi_defaults.num_layers,
                'dropout': robi_defaults.dropout,
                'learning_rate': train_defaults.learning_rate,
                'gradient_accumulation_steps': train_defaults.gradient_accumulation_steps
            }
            
        # Normalize parameters to [0,1] range for stable GP fitting
        X_norm = self._normalize_params(self.X)
        y_norm = (self.y - np.mean(self.y)) / np.std(self.y) if len(self.y) > 1 else self.y
        
        self.gpr.fit(X_norm, y_norm)
        
        # Generate random candidate points for the acquisition function
        n_random = self.config.n_random
        random_points = []
        for param_name, (low, high) in self.config.param_ranges.items():
            # Generate candidates respecting parameter types
            if param_name == 'num_layers' or param_name == 'gradient_accumulation_steps':
                random_points.append(np.random.randint(low, high + 1, n_random))
            else:
                random_points.append(np.random.uniform(low, high, n_random))
        random_points = np.array(random_points).T
        
        # Normalize candidate points for GP prediction
        random_points_norm = self._normalize_params(random_points)
        
        # Predict mean and std by getting GP predictions for candidates
        mu, std = self.gpr.predict(random_points_norm, return_std=True)
        
        # Calculate Upper Confidence Bound acquisition function
        kappa = 2.0  # exploration-exploitation trade-off
        acq = mu + kappa * std
        
        # Select the point with the highest acquisition value
        best_idx = np.argmax(acq)
        best_point = random_points[best_idx]
        
        # Convert selected point back to parameter dictionary
        params = {}
        for i, (param_name, _) in enumerate(self.config.param_ranges.items()):
            if param_name == 'num_layers' or param_name == 'gradient_accumulation_steps':
                params[param_name] = int(best_point[i])
            else:
                params[param_name] = float(best_point[i])
                
        return params
    
    def _normalize_params(
        self, 
        X
    ) -> np.ndarray:
        """
        Normalizes hyperparameter values to the range [0,1] for effective Gaussian Process Regression (GPR).
        Normalization is crucial for GPR as it makes sure all hyperparameters contribute equally to the
        optimization process, which prevents parameters with larger scales from dominating the search space. Each
        parameter is normalized independently using its defined range with the formula (x - min) / (max - min).

        Args:
            X: Input hyperparameter values in one of these forms:
                - Empty array: Returns empty numpy array
                - 1D array: Single set of hyperparameters, reshaped to 2D
                - 2D array: Multiple sets of hyperparameters [n_samples, n_params]
        
        Returns:
            np.ndarray: Normalized hyperparameters where each parameter is scaled to [0,1]:
                - Empty array if input is empty
                - 2D array of shape [n_samples, n_params] otherwise
        """
        if len(X) == 0:
            return np.array([])
            
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        X_norm = np.zeros_like(X, dtype=np.float32)
        
        for i, (param_name, (low, high)) in enumerate(self.config.param_ranges.items()):
            X_norm[:, i] = (X[:, i] - low) / (high - low)
            
        return X_norm
    
    def _create_trial_configs(
        self, 
        params: Dict[str, Any], trial_num: int
    ) -> Tuple[RoBiConfig, TrainConfig, SaveLoadConfig]:
        """
        Creates configuration objects for each trial by mapping Bayesian-optimized hyperparameters to their 
        respective configuration classes. This method serves as a bridge between the tuner's parameter space
        and RoBi's configuration system, ensuring that optimized parameters are properly integrated into
        the model's architecture and training process. Each trial will be contained in its own subdirectory so 
        model parameters are isolated, the the final results will be compared between all 5 trials and their 
        respective subdirectories. Note that non-optimized parameters use defaults from respective config classes.
        
        Processing Flow:
            Parameter Dictionary → Configuration Objects → Trial Directory Setup
                ↓                           ↓                      ↓ 
            Optimized Values → RoBiConfig/TrainConfig → SaveLoadConfig
        
        Configuration Mapping:
            RoBiConfig:
                params['hidden_size']  → robi_config.hidden_size
                params['num_layers']   → robi_config.num_layers
                params['dropout']      → robi_config.dropout
                
            TrainConfig:
                params['learning_rate'] → train_config.learning_rate
                params['gradient_accumulation_steps'] → train_config.gradient_accumulation_steps
                
            SaveLoadConfig:
                Creates trial-specific directory for model artifacts
        
        Args:
            - params: Dictionary of optimized hyperparameters containing:
                - hidden_size: Size of LSTM hidden layers (int)
                - num_layers: Number of stacked LSTM layers (int)
                - dropout: Dropout probability (float)
                - learning_rate: Learning rate for optimization (float)
                - gradient_accumulation_steps: Steps for gradient accumulation (int)
            - trial_num: Current trial number for directory organization
                
        Returns:
            Tuple containing:
                - RoBiConfig: Model architecture configuration
                - TrainConfig: Training process configuration
                - SaveLoadConfig: Model saving/loading configuration
        """
        robi_config = RoBiConfig(
            hidden_size=int(params['hidden_size']),
            num_layers=int(params['num_layers']),
            dropout=float(params['dropout'])
        )
        
        train_config = TrainConfig(
            learning_rate=float(params['learning_rate']),
            gradient_accumulation_steps=int(params['gradient_accumulation_steps'])
        )
        
        save_config = SaveLoadConfig(
            base_path=self.config.base_path,
            trial_dir=f"trial{trial_num}"
        )
        
        return robi_config, train_config, save_config
    
    def _save_trial_metadata(
        self, 
        trial_num: int, 
        params: Dict[str, Any], 
        val_loss: float
    ) -> None:
        """
        Saves trial metadata to a JSON file in the trial directory.
        
        Args:
            trial_num: The trial number
            params: The hyperparameters used for this trial
            val_loss: The best validation loss achieved in this trial
        """
        trial_dir = os.path.join(self.config.base_path, f"trial{trial_num}")
        os.makedirs(trial_dir, exist_ok=True)
        
        metadata = {
            "trial_number": trial_num,
            "hyperparameters": params,
            "best_val_loss": val_loss
        }
        
        with open(os.path.join(trial_dir, 'trial_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
        
    def optimize(
        self, 
        train_dataloader: DataLoader, 
        val_dataloader: DataLoader, 
        tokenizer: RobertaTokenizer
    ) -> Tuple[RoBi, Dict[str, Any]]:
        """
        Executes the Bayesian optimization process to find optimal hyperparameters for RoBi. This method implements
        Bayesian optimization by treating hyperparameter tuning as a sequential decision problem, where Bayes' Theorem is 
        used to update beliefs about the performance of different hyperparameter configurations based on observed results.
        The optimization process iteratively updates a Gaussian Process model with observed validation performance, using 
        an Upper Confidence Bound acquisition function to balance exploration and exploitation when selecting new 
        hyperparameters to evaluate.
        
        Bayesian Optimization Theory:
            Using Bayes' Theorem: P(A|B) = P(B|A)P(A)/P(B), where:
                P(A) = Prior belief about hyperparameter performance
                P(B|A) = Likelihood of observing validation loss given hyperparameters
                P(A|B) = Posterior belief about hyperparameter performance
                P(B) = Normalization constant
        
        Optimization Pipeline:
            Trial 1 (Baseline):
                → Use proven RoBiConfig/TrainConfig values
                → Establish performance benchmark
                → Initialize Gaussian Process with Matérn kernel (ν=2.5)
                
            Trials 2-10 (Optimization):
                → Sample promising hyperparameters using normalized GP predictions
                → Train RoBi with new configuration (monitoring validation loss)
                → Save best model state and clear GPU memory 
                → Update GP with performance results
                → Select next configuration via UCB acquisition
                
            Final Cleanup:
                → Identify best performing trial
                → Remove all other suboptimal trial directories
                → Save comprehensive result summary
        
        Args:
            - train_dataloader: DataLoader containing training data batches using SlidingWindow
            - val_dataloader: DataLoader containing validation data batches using SlidingWindow
            - tokenizer: RoBERTa tokenizer for text processing
                
        Returns:
            Tuple containing:
                - RoBi: Best performing model across all trials
                - Dict[str, Any]: Corresponding optimal hyperparameters
        """
        best_val_loss = float('inf')
        best_trial_num = None
        best_params = None
        
        for trial in range(1, self.config.n_trials + 1):
            print(f"\nStarting Trial {trial}/{self.config.n_trials}")
            
            # Sample next hyperparameter set based on Gaussian Process predictions
            params = self._sample_parameters()
            print("Trial hyperparameters:", json.dumps(params, indent=2))
            
            # Convert raw parameters into structured configs for model architecture and training
            robi_config, train_config, save_config = self._create_trial_configs(params, trial)
            
            # Create trial directory
            trial_dir = os.path.join(self.config.base_path, f"trial{trial}")
            os.makedirs(trial_dir, exist_ok=True)
            
            # Define model checkpoint path
            model_checkpoint_path = os.path.join(trial_dir, 'best_model.pth')
            
            # Initialize fresh model for this trial to avoid any state leakage
            model = RoBi(config=robi_config)
            model = model.to(train_config.device)
            
            # Execute this trial's training loop with current hyperparameter configuration
            trained_model, trial_val_loss = train_RoBi(
                model,
                train_dataloader,
                val_dataloader,
                train_config,
                save_path=model_checkpoint_path
            )
            
            # Save tokenizer and model configuration for future loading
            save_RoBi(
                trained_model,
                tokenizer,
                save_config,
                best_model_path=model_checkpoint_path
            )
            
            # Save trial metadata
            self._save_trial_metadata(trial, params, trial_val_loss)
            
            # Update Gaussian Process with new observation for next sampling
            self.X.append([params[name] for name in self.config.param_ranges.keys()])
            self.y.append(trial_val_loss)
            
            # Track best performing configuration seen so far
            if trial_val_loss < best_val_loss:
                best_val_loss = trial_val_loss
                best_trial_num = trial
                best_params = params
            
            # Explicitly free GPU cache to prevent CUDA OOM between trials
            del model
            del trained_model
            torch.cuda.empty_cache()
            
        # After all trials, load the best model
        print(f"\nBest trial: {best_trial_num} with validation loss: {best_val_loss:.4f}")
        print("Best hyperparameters:", json.dumps(best_params, indent=2))
        
        # Clean up suboptimal trial artifacts to save storage, removing the need to manually delete other trials
        for trial in range(1, self.config.n_trials + 1):
            if trial != best_trial_num:
                trial_dir = os.path.join(self.config.base_path, f"trial{trial}")
                shutil.rmtree(trial_dir)
        
        # Load and reconstruct best performing configuration
        best_config = SaveLoadConfig(
            base_path=self.config.base_path,
            trial_dir=f"trial{best_trial_num}"
        )
        best_model, _ = load_RoBi(config=best_config)
        
        return best_model, best_params