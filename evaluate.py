import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

from config import Config
from utils.data_loader import create_data_loaders as create_data_loaders_bert
from utils.data_loader_dialogue import create_data_loaders as create_data_loaders_dialogue
from model.bert_baseline import create_bert_model
from model.dialogue_rnn import BERTDialogueRNN
from utils.metrics import calculate_metrics, plot_confusion_matrix
from utils.visualization import (
    plot_emotion_trajectory, 
    analyze_errors,
    create_performance_report
)

def evaluate_model(model_path: str, model_type: str = 'bert_baseline'):
    """Evaluate a trained model"""
    config = Config()
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=config.DEVICE, weights_only=False)

    # Initialize model
    if model_type == 'bert_baseline':
        model = create_bert_model(config, use_attention=False)
        _, _, test_loader, _ = create_data_loaders_bert(config)
    elif model_type == 'bert_attention':
        model = create_bert_model(config, use_attention=True)
        _, _, test_loader, _ = create_data_loaders_bert(config)
    elif model_type == 'dialogue_rnn':
        model = BERTDialogueRNN(config)
        _, _, test_loader, _ = create_data_loaders_dialogue(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_predictions = []
    all_labels = []
    all_dialogue_ids = []
    all_utterance_ids = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            labels = batch['label']

            logits = model(input_ids, attention_mask)
            # Flatten if output is 3D
            if logits.dim() == 3:
                logits = logits.view(-1, logits.size(-1))
                if labels.dim() == 2:
                    labels = labels.view(-1)

            _, predicted = torch.max(logits, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            # For dialogue_rnn, you may not have dialogue_id/utterance_id in batch
            if 'dialogue_id' in batch:
                all_dialogue_ids.extend(batch['dialogue_id'].numpy())
            else:
                all_dialogue_ids.extend([None]*len(predicted))
            if 'utterance_id' in batch:
                all_utterance_ids.extend(batch['utterance_id'].numpy())
            else:
                all_utterance_ids.extend([None]*len(predicted))

    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_predictions)

    # Create results dataframe
    results_df = pd.DataFrame({
        'dialogue_id': all_dialogue_ids,
        'utterance_id': all_utterance_ids,
        'true_label': all_labels,
        'predicted_label': all_predictions,
        'true_emotion': [config.EMOTIONS[l] for l in all_labels],
        'predicted_emotion': [config.EMOTIONS[p] for p in all_predictions]
    })

    # Save results
    results_df.to_csv(f"{config.OUTPUT_DIR}/{model_type}_predictions.csv", index=False)

    print("\nEvaluation Metrics:")
    print(metrics)
    print("\nSample Predictions:")
    print(results_df.head())

    return metrics, results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate emotion recognition model")
    parser.add_argument('--model', type=str, default='bert_baseline',
                        choices=['bert_baseline', 'bert_attention', 'dialogue_rnn'],
                        help='Model type to evaluate')
    args = parser.parse_args()
    model_path = f"{Config().OUTPUT_DIR}/{args.model}_best_model.pth"
    metrics, results_df = evaluate_model(model_path, args.model)