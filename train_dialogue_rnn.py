import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import os
import random
from datetime import datetime

from config import Config
from utils.data_loader_dialogue import create_data_loaders
from utils.data_loader import get_class_weights
from model.dialogue_rnn import BERTDialogueRNN
from utils.metrics import calculate_metrics, plot_confusion_matrix

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_epoch(model, train_loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc="Training")

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)         # [batch, seq_len, max_length]
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)                # [batch, seq_len]

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)         # [batch, seq_len, num_classes]

        # Flatten logits and labels for loss
        logits = logits.view(-1, logits.size(-1))         # [(batch * seq_len), num_classes]
        labels = labels.view(-1)                          # [(batch * seq_len)]

        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        total_loss += loss.item()
        avg_loss = total_loss / (progress_bar.n + 1)
        accuracy = correct / total
        progress_bar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'acc': f'{accuracy:.4f}'
        })

    return total_loss / len(train_loader), correct / total

def evaluate(model, val_loader, criterion, device, return_predictions=False):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)

            loss = criterion(logits, labels)
            total_loss += loss.item()

            _, predicted = torch.max(logits, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = calculate_metrics(all_labels, all_predictions)
    avg_loss = total_loss / len(val_loader)

    if return_predictions:
        return avg_loss, metrics, all_predictions, all_labels

    return avg_loss, metrics

def train_model(model_type='dialogue_rnn', use_context=False):
    config = Config()
    set_seed(config.SEED)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'{config.LOGS_DIR}/{model_type}_{timestamp}')

    print("Loading data...")
    train_loader, val_loader, test_loader, tokenizer = create_data_loaders(config, use_context=use_context)

    class_weights = get_class_weights(config.TRAIN_PATH, config)
    class_weights = class_weights.to(config.DEVICE)

    print(f"Initializing {model_type} model...")
    if model_type == 'dialogue_rnn':
        model = BERTDialogueRNN(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * config.NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.WARMUP_STEPS,
        num_training_steps=total_steps
    )

    best_val_f1 = 0
    patience = 3
    patience_counter = 0

    print("Starting training...")
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, config.DEVICE
        )
        val_loss, val_metrics = evaluate(model, val_loader, criterion, config.DEVICE)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
        writer.add_scalar('F1/val', val_metrics['f1_weighted'], epoch)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
              f"Val F1: {val_metrics['f1_weighted']:.4f}")

        if val_metrics['f1_weighted'] > best_val_f1:
            best_val_f1 = val_metrics['f1_weighted']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'config': config
            }, f"{config.OUTPUT_DIR}/{model_type}_best_model.pth")
            print(f"Saved best model with F1: {best_val_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    print("\nEvaluating on test set...")
    checkpoint = torch.load(f"{config.OUTPUT_DIR}/{model_type}_best_model.pth", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_metrics, predictions, labels = evaluate(
        model, test_loader, criterion, config.DEVICE, return_predictions=True
    )

    print(f"\nTest Results:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"F1 (weighted): {test_metrics['f1_weighted']:.4f}")
    print(f"F1 (macro): {test_metrics['f1_macro']:.4f}")
    print("\nClassification Report:")
    print(test_metrics['classification_report'])

    plot_confusion_matrix(
        labels, predictions, config.EMOTIONS,
        save_path=f"{config.OUTPUT_DIR}/{model_type}_confusion_matrix.png"
    )

    writer.close()
    return model, test_metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train emotion recognition model")
    parser.add_argument('--model', type=str, default='dialogue_rnn',
                        choices=['dialogue_rnn'],
                        help='Model type to train')
    parser.add_argument('--use_context', action='store_true',
                        help='Use conversation context')
    args = parser.parse_args()
    model, metrics = train_model(args.model, args.use_context)