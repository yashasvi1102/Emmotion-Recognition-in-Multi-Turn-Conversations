import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
import numpy as np

class DialogueRNN(nn.Module):
    """
    DialogueRNN model for emotion recognition in conversations
    Based on: https://arxiv.org/abs/1811.00405
    """
    
    def __init__(self, config, feature_extractor='bert'):
        super(DialogueRNN, self).__init__()
        self.config = config
        
        # Feature extractor (BERT)
        if feature_extractor == 'bert':
            self.bert_config = AutoConfig.from_pretrained(config.MODEL_NAME)
            self.bert = AutoModel.from_pretrained(config.MODEL_NAME)
            self.feature_dim = self.bert_config.hidden_size
        
        # Dimensions
        self.hidden_dim = 128
        self.speaker_dim = 100
        self.emotion_dim = 100
        
        # Global GRU for context
        self.global_gru = nn.GRU(
            self.feature_dim,
            self.hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        
        # Speaker GRU
        self.speaker_gru = nn.GRU(
            self.feature_dim + 2 * self.hidden_dim,
            self.speaker_dim,
            batch_first=True
        )
        
        # Emotion GRU
        self.emotion_gru = nn.GRU(
            self.speaker_dim,
            self.emotion_dim,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = SimpleAttention(2 * self.hidden_dim)
        
        # Output layers
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(
            self.emotion_dim + self.speaker_dim,
            config.NUM_CLASSES
        )
        
        # Speaker embeddings (assuming max 10 speakers)
        self.speaker_embeddings = nn.Embedding(10, 50)
    
    def extract_features(self, input_ids, attention_mask):
        """Extract features using BERT"""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        # Use CLS token
        return outputs.pooler_output
    
    def forward(self, dialogue_data):
        """
        Forward pass for dialogue
        Args:
            dialogue_data: Dict containing:
                - utterances: List of (input_ids, attention_mask) tuples
                - speakers: List of speaker IDs
        """
        utterances = dialogue_data['utterances']
        speakers = dialogue_data['speakers']
        
        # Extract features for all utterances
        utterance_features = []
        for input_ids, attention_mask in utterances:
            features = self.extract_features(input_ids, attention_mask)
            utterance_features.append(features)
        
        # Stack features
        utterance_features = torch.stack(utterance_features, dim=1)  # (batch, seq_len, feature_dim)
        
        # Global context
        global_output, _ = self.global_gru(utterance_features)
        
        # Process each utterance with speaker and emotion modeling
        outputs = []
        speaker_states = {}  # Track speaker states
        
        for t in range(len(utterances)):
            # Get current utterance and speaker
            current_features = utterance_features[:, t, :]
            speaker_id = speakers[t]
            
            # Get global context up to current utterance
            if t > 0:
                context = global_output[:, :t, :]
                # Apply attention to get weighted context
                context_vector = self.attention(
                    global_output[:, t, :].unsqueeze(1),
                    context
                )
            else:
                context_vector = torch.zeros(
                    utterance_features.size(0),
                    2 * self.hidden_dim
                ).to(utterance_features.device)
            
            # Combine features with context
            combined = torch.cat([current_features, context_vector], dim=-1)
            
            # Update speaker state
            if speaker_id not in speaker_states:
                speaker_states[speaker_id] = None
            
            speaker_input = combined.unsqueeze(1)
            speaker_output, speaker_states[speaker_id] = self.speaker_gru(
                speaker_input,
                speaker_states[speaker_id]
            )
            speaker_output = speaker_output.squeeze(1)
            
            # Emotion modeling
            emotion_output, _ = self.emotion_gru(speaker_output.unsqueeze(1))
            emotion_output = emotion_output.squeeze(1)
            
            # Combine speaker and emotion representations
            final_repr = torch.cat([speaker_output, emotion_output], dim=-1)
            final_repr = self.dropout(final_repr)
            
            # Classification
            logits = self.classifier(final_repr)
            outputs.append(logits)
        
        return torch.stack(outputs, dim=1)  # (batch, seq_len, num_classes)

class SimpleAttention(nn.Module):
    """Simple attention mechanism"""
    
    def __init__(self, hidden_dim):
        super(SimpleAttention, self).__init__()
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, query, keys):
        """
        Args:
            query: (batch, 1, hidden_dim)
            keys: (batch, seq_len, hidden_dim)
        """
        # Project query and keys
        query_proj = self.query_projection(query)
        keys_proj = self.projection(keys)
        
        # Compute attention scores
        scores = torch.bmm(query_proj, keys_proj.transpose(1, 2))
        scores = scores / np.sqrt(query.size(-1))
        
        # Apply softmax
        weights = F.softmax(scores, dim=-1)
        
        # Weighted sum
        context = torch.bmm(weights, keys).squeeze(1)
        
        return context

class BERTDialogueRNN(nn.Module):
    """
    Simplified DialogueRNN using BERT features
    Processes conversations sequentially considering context
    """

    def __init__(self, config):
        super(BERTDialogueRNN, self).__init__()
        self.config = config

        # BERT feature extractor
        self.bert_config = AutoConfig.from_pretrained(config.MODEL_NAME)
        self.bert = AutoModel.from_pretrained(config.MODEL_NAME)

        # Context GRU
        self.context_gru = nn.GRU(
            self.bert_config.hidden_size,
            256,
            batch_first=True,
            bidirectional=True
        )

        # Classification head
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(512, config.NUM_CLASSES)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        Args:
            input_ids: (batch, seq_len, max_length)
            attention_mask: (batch, seq_len, max_length)
        """
        batch_size, seq_len, max_length = input_ids.shape

        # Extract features for each utterance in the dialogue
        utterance_features = []
        for i in range(seq_len):
            outputs = self.bert(
                input_ids[:, i, :],           # (batch, max_length)
                attention_mask[:, i, :],      # (batch, max_length)
                return_dict=True
            )
            utterance_features.append(outputs.pooler_output)  # (batch, hidden_size)
        # Stack features: (batch, seq_len, hidden_size)
        utterance_features = torch.stack(utterance_features, dim=1)

        # Pass through GRU
        gru_output, _ = self.context_gru(utterance_features)  # (batch, seq_len, 2*hidden_size)

        # Apply dropout
        gru_output = self.dropout(gru_output)

        # Classify each utterance
        logits = self.classifier(gru_output)  # (batch, seq_len, num_classes)

        return logits