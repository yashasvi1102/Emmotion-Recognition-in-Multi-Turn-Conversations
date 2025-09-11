import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class BERTEmotionClassifier(nn.Module):
    """BERT-based emotion classifier for single utterances"""
    
    def __init__(self, config):
        super(BERTEmotionClassifier, self).__init__()
        self.config = config
        
        # Load pre-trained BERT
        self.bert_config = AutoConfig.from_pretrained(config.MODEL_NAME)
        self.bert = AutoModel.from_pretrained(config.MODEL_NAME)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Classification head
        self.classifier = nn.Linear(self.bert_config.hidden_size, config.NUM_CLASSES)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of classification layer"""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        Args:
            input_ids: Tensor of token ids
            attention_mask: Tensor of attention masks
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Get logits
        logits = self.classifier(pooled_output)
        
        return logits

class BERTWithAttention(nn.Module):
    """BERT with additional attention mechanism for emotion classification"""
    
    def __init__(self, config):
        super(BERTWithAttention, self).__init__()
        self.config = config
        
        # Load pre-trained BERT
        self.bert_config = AutoConfig.from_pretrained(config.MODEL_NAME)
        self.bert = AutoModel.from_pretrained(config.MODEL_NAME)
        
        # Self-attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=self.bert_config.hidden_size,
            num_heads=8,
            dropout=0.1
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.bert_config.hidden_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Classification head
        self.classifier = nn.Linear(self.bert_config.hidden_size, config.NUM_CLASSES)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of classification layer"""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass with attention
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get sequence output
        sequence_output = outputs.last_hidden_state
        
        # Apply self-attention
        # Transpose for attention (seq_len, batch_size, hidden_size)
        sequence_output = sequence_output.transpose(0, 1)
        attn_output, _ = self.attention(
            sequence_output, sequence_output, sequence_output,
            key_padding_mask=~attention_mask.bool()
        )
        
        # Transpose back and apply residual connection
        attn_output = attn_output.transpose(0, 1)
        sequence_output = self.layer_norm(sequence_output.transpose(0, 1) + attn_output)
        
        # Pool by taking mean of non-padded tokens
        mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
        sum_embeddings = torch.sum(sequence_output * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Get logits
        logits = self.classifier(pooled_output)
        
        return logits

# Model initialization helper
def create_bert_model(config, use_attention=False):
    """Create BERT model based on configuration"""
    if use_attention:
        model = BERTWithAttention(config)
    else:
        model = BERTEmotionClassifier(config)
    
    model.to(config.DEVICE)
    return model