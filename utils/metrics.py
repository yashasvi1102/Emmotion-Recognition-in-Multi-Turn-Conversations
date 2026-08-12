import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def calculate_metrics(y_true, y_pred):
    """Calculate various metrics for emotion classification"""
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    # Weighted and macro averages
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    # Classification report
    report = classification_report(y_true, y_pred, digits=4)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'classification_report': report
    }
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_emotion_distribution(csv_path, save_path=None):
    """Plot emotion distribution in dataset"""
    df = pd.read_csv(csv_path)
    emotion_counts = df['Emotion'].value_counts()
    
    plt.figure(figsize=(10, 6))
    emotion_counts.plot(kind='bar')
    plt.title('Emotion Distribution in Dataset')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
