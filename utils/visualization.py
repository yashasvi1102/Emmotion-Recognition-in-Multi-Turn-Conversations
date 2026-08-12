import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import pandas as pd
import seaborn as sns

def plot_emotion_trajectory(dialogue_emotions: List[str], 
                          dialogue_predictions: List[str],
                          dialogue_id: int,
                          save_path=None):
    """Plot emotion trajectory for a single dialogue"""
    emotions = ['neutral', 'joy', 'surprise', 'anger', 'sadness', 'disgust', 'fear']
    emotion_to_idx = {e: i for i, e in enumerate(emotions)}
    
    # Convert emotions to indices
    true_indices = [emotion_to_idx[e] for e in dialogue_emotions]
    pred_indices = [emotion_to_idx[e] for e in dialogue_predictions]
    
    utterance_indices = list(range(len(dialogue_emotions)))
    
    plt.figure(figsize=(12, 6))
    plt.plot(utterance_indices, true_indices, 'b-o', label='True Emotions', markersize=8)
    plt.plot(utterance_indices, pred_indices, 'r--s', label='Predicted Emotions', markersize=6)
    
    plt.yticks(range(len(emotions)), emotions)
    plt.xlabel('Utterance Index')
    plt.ylabel('Emotion')
    plt.title(f'Emotion Trajectory for Dialogue {dialogue_id}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_training_curves(log_file: str, save_path=None):
    """Plot training and validation curves from log file"""
    # Example: Load metrics from a CSV log file with columns:
    # epoch,train_loss,val_loss,train_acc,val_acc,val_f1
    df = pd.read_csv(log_file)
    train_losses = df['train_loss'].tolist()
    val_losses = df['val_loss'].tolist()
    train_accs = df['train_acc'].tolist()
    val_accs = df['val_acc'].tolist()
    val_f1s = df['val_f1'].tolist()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy/F1 curves
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Val Accuracy')
    ax2.plot(val_f1s, label='Val F1')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('Training and Validation Metrics')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def analyze_errors(predictions, labels, utterances, emotions):
    """Analyze prediction errors"""
    errors = []
    
    for i, (pred, true) in enumerate(zip(predictions, labels)):
        if pred != true:
            errors.append({
                'utterance': utterances[i],
                'true_emotion': emotions[true],
                'predicted_emotion': emotions[pred],
                'index': i
            })
    
    # Create error dataframe
    error_df = pd.DataFrame(errors)
    
    # Error analysis by emotion
    error_matrix = pd.crosstab(
        error_df['true_emotion'],
        error_df['predicted_emotion'],
        margins=True
    )
    
    return error_df, error_matrix

def plot_speaker_emotion_patterns(data_path: str, save_path=None):
    """Analyze emotion patterns per speaker"""
    df = pd.read_csv(data_path)
    
    # Get emotion distribution per speaker
    speaker_emotions = pd.crosstab(df['Speaker'], df['Emotion'])
    
    # Normalize to percentages
    speaker_emotions_pct = speaker_emotions.div(
        speaker_emotions.sum(axis=1), axis=0
    ) * 100
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(speaker_emotions_pct, annot=True, fmt='.1f', cmap='YlOrRd')
    plt.title('Emotion Distribution by Speaker (%)')
    plt.xlabel('Emotion')
    plt.ylabel('Speaker')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def create_performance_report(test_metrics: Dict, model_name: str, save_path: str):
    """Create a comprehensive performance report"""
    report = f"""
    Emotion Recognition Performance Report
    =====================================
    Model: {model_name}
    
    Overall Metrics:
    ---------------
    Accuracy: {test_metrics['accuracy']:.4f}
    F1 Score (Weighted): {test_metrics['f1_weighted']:.4f}
    F1 Score (Macro): {test_metrics['f1_macro']:.4f}
    
    Per-Class Performance:
    ---------------------
    {test_metrics['classification_report']}
    
    Key Observations:
    ----------------
    - Best performing emotion: [Analyze from results]
    - Worst performing emotion: [Analyze from results]
    - Common confusion pairs: [Analyze from confusion matrix]
    """
    
    with open(save_path, 'w') as f:
        f.write(report)
    
    return report