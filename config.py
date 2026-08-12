import torch

class Config:
    # Data paths
    TRAIN_PATH = 'data/train_sent_emo.csv'
    DEV_PATH = 'data/dev_sent_emo.csv'
    TEST_PATH = 'data/test_sent_emo.csv'
    
    # Model parameters
    MODEL_NAME = 'bert-base-uncased'
    MAX_LENGTH =64
    BATCH_SIZE = 2
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 5
    WARMUP_STEPS = 500
    
    # Training parameters
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42
    
    # Emotion labels (MELD dataset)
    EMOTIONS = ['neutral', 'joy', 'surprise', 'anger', 'sadness', 'disgust', 'fear']
    NUM_CLASSES = len(EMOTIONS)
    
    # Context parameters
    CONTEXT_WINDOW = 3  # Number of previous utterances to consider
    
    # Output paths
    OUTPUT_DIR = 'outputs/'
    MODEL_SAVE_PATH = 'outputs/best_model.pth'
    LOGS_DIR = 'logs/'