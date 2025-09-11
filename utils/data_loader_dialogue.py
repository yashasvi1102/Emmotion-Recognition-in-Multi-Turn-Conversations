import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
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

class MELDDialogueDataset(Dataset):
    """Dataset for batching dialogues for DialogueRNN"""

    def __init__(self, csv_path, tokenizer, config):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.config = config
        self.emotion2idx = {emotion: idx for idx, emotion in enumerate(config.EMOTIONS)}
        # Group utterances by dialogue
        self.dialogues = self.data.groupby('Dialogue_ID')
        self.dialogue_ids = list(self.dialogues.groups.keys())

    def __len__(self):
        return len(self.dialogue_ids)

    def __getitem__(self, idx):
        dialogue_id = self.dialogue_ids[idx]
        utterances = self.dialogues.get_group(dialogue_id)
        texts = [f"{row['Speaker']}: {row['Utterance']}" for _, row in utterances.iterrows()]
        labels = [self.emotion2idx.get(row['Emotion'].lower(), 0) for _, row in utterances.iterrows()]

        # Tokenize all utterances in the dialogue
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=self.config.MAX_LENGTH,
            return_tensors='pt'
        )
        return {
            'input_ids': encodings['input_ids'],         # [seq_len, max_length]
            'attention_mask': encodings['attention_mask'],
            'labels': torch.tensor(labels, dtype=torch.long),
            'seq_len': len(texts)
        }

def dialogue_collate_fn(batch):
    # Pad sequences to the same seq_len in the batch
    max_seq_len = max(item['seq_len'] for item in batch)
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    for item in batch:
        pad_len = max_seq_len - item['seq_len']
        batch_input_ids.append(
            torch.cat([item['input_ids'], torch.zeros((pad_len, item['input_ids'].shape[1]), dtype=torch.long)], dim=0)
        )
        batch_attention_mask.append(
            torch.cat([item['attention_mask'], torch.zeros((pad_len, item['attention_mask'].shape[1]), dtype=torch.long)], dim=0)
        )
        batch_labels.append(
            torch.cat([item['labels'], torch.zeros(pad_len, dtype=torch.long)], dim=0)
        )
    return {
        'input_ids': torch.stack(batch_input_ids),         # [batch_size, seq_len, max_length]
        'attention_mask': torch.stack(batch_attention_mask),
        'label': torch.stack(batch_labels)                 # [batch_size, seq_len]
    }

# Usage in your loader creation:
def create_data_loaders(config, use_context=False):
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    train_dataset = MELDDialogueDataset(config.TRAIN_PATH, tokenizer, config)
    val_dataset = MELDDialogueDataset(config.DEV_PATH, tokenizer, config)
    test_dataset = MELDDialogueDataset(config.TEST_PATH, tokenizer, config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=dialogue_collate_fn,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=dialogue_collate_fn,
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=dialogue_collate_fn,
        num_workers=4
    )
    return train_loader, val_loader, test_loader, tokenizer