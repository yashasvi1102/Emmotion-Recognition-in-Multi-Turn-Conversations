import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import numpy as np
from typing import List, Dict, Tuple

class MELDDataset(Dataset):
    """Dataset class for MELD emotion recognition"""
    
    def __init__(self, csv_path: str, tokenizer, config, use_context: bool = False):
        """
        Args:
            csv_path: Path to CSV file
            tokenizer: BERT tokenizer
            config: Configuration object
            use_context: Whether to include conversation context
        """
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.config = config
        self.use_context = use_context
        
        # Create emotion to index mapping
        self.emotion2idx = {emotion: idx for idx, emotion in enumerate(config.EMOTIONS)}
        
        # Group by dialogue for context
        if self.use_context:
            self.dialogue_groups = self._group_by_dialogue()
    
    def _group_by_dialogue(self) -> Dict[int, List[int]]:
        """Group utterances by dialogue ID"""
        dialogue_groups = {}
        for idx, row in self.data.iterrows():
            dialogue_id = row['Dialogue_ID']
            if dialogue_id not in dialogue_groups:
                dialogue_groups[dialogue_id] = []
            dialogue_groups[dialogue_id].append(idx)
        return dialogue_groups
    
    def _get_context_utterances(self, idx: int) -> List[str]:
        """Get previous utterances in the same dialogue"""
        current_row = self.data.iloc[idx]
        dialogue_id = current_row['Dialogue_ID']
        utterance_id = current_row['Utterance_ID']
        
        context_utterances = []
        dialogue_indices = self.dialogue_groups[dialogue_id]
        
        # Find current position in dialogue
        current_pos = dialogue_indices.index(idx)
        
        # Get previous utterances (up to context window)
        for i in range(max(0, current_pos - self.config.CONTEXT_WINDOW), current_pos):
            context_idx = dialogue_indices[i]
            context_row = self.data.iloc[context_idx]
            speaker = context_row['Speaker']
            utterance = context_row['Utterance']
            context_utterances.append(f"{speaker}: {utterance}")
        
        return context_utterances
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        utterance = row['Utterance']
        emotion = row['Emotion']
        speaker = row['Speaker']
        
        # Prepare text input
        if self.use_context:
            context = self._get_context_utterances(idx)
            if context:
                context_text = " [SEP] ".join(context)
                text = f"{context_text} [SEP] {speaker}: {utterance}"
            else:
                text = f"{speaker}: {utterance}"
        else:
            text = utterance
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.config.MAX_LENGTH,
            return_tensors='pt'
        )
        
        # Get emotion label
        label = self.emotion2idx.get(emotion.lower(), 0)  # Default to neutral if not found
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long),
            'dialogue_id': row['Dialogue_ID'],
            'utterance_id': row['Utterance_ID']
        }

def create_data_loaders(config, use_context=False):
    """Create train, validation, and test data loaders"""
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Create datasets
    train_dataset = MELDDataset(config.TRAIN_PATH, tokenizer, config, use_context)
    val_dataset = MELDDataset(config.DEV_PATH, tokenizer, config, use_context)
    test_dataset = MELDDataset(config.TEST_PATH, tokenizer, config, use_context)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader, test_loader, tokenizer

# Additional preprocessing utilities
def preprocess_meld_data(csv_path: str) -> pd.DataFrame:
    """Preprocess MELD dataset if needed"""
    df = pd.read_csv(csv_path)
    
    # Clean text
    df['Utterance'] = df['Utterance'].str.strip()
    
    # Ensure emotion labels are lowercase
    df['Emotion'] = df['Emotion'].str.lower()
    
    # Remove any rows with missing values
    df = df.dropna(subset=['Utterance', 'Emotion'])
    
    return df

def get_class_weights(train_path: str, config) -> torch.Tensor:
    """Calculate class weights for handling imbalanced dataset"""
    df = pd.read_csv(train_path)
    emotion_counts = df['Emotion'].str.lower().value_counts()
    
    weights = []
    total_samples = len(df)
    
    for emotion in config.EMOTIONS:
        count = emotion_counts.get(emotion, 1)  # Avoid division by zero
        weight = total_samples / (len(config.EMOTIONS) * count)
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)