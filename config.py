"""
Configuration settings for Fake News Detection System
"""

# Model Configuration
MODEL_CONFIG = {
    'vectorizer_max_features': 5000,
    'ngram_range': (1, 2),
    'min_df': 5,
    'max_df': 0.8,
    'test_size': 0.2,
    'random_state': 42,
}

# Classifier Configuration
CLASSIFIER_CONFIG = {
    'logistic_regression': {
        'max_iter': 200,
        'random_state': 42,
        'solver': 'lbfgs',
    },
    'random_forest': {
        'n_estimators': 100,
        'random_state': 42,
        'n_jobs': -1,
        'max_depth': 15,
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'random_state': 42,
        'learning_rate': 0.1,
        'max_depth': 5,
    },
    'svm': {
        'random_state': 42,
        'max_iter': 2000,
        'C': 1.0,
    }
}

# Text Preprocessing Config
PREPROCESSING_CONFIG = {
    'remove_urls': True,
    'remove_emails': True,
    'remove_special_chars': True,
    'lowercase': True,
    'tokenize': True,
    'remove_stopwords': True,
    'stemming': True,
}

# API Configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True,
    'cors_enabled': True,
}

# Model Paths
MODEL_PATHS = {
    'model_file': 'fake_news_detector_model.pkl',
    'dataset_file': 'fake_news_dataset.csv',
    'sample_dataset': 'sample_fake_news.csv',
}

# Prediction Thresholds
PREDICTION_THRESHOLDS = {
    'high_confidence': 0.80,
    'medium_confidence': 0.50,
    'low_confidence': 0.0,
}

# Performance Metrics to Track
METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'roc_auc',
]

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'fake_news_detection.log',
}

# Data Configuration
DATA_CONFIG = {
    'sample_size': 2000,
    'train_split': 0.8,
    'validation_split': 0.1,
    'test_split': 0.1,
}

# Feature Engineering
FEATURE_CONFIG = {
    'use_tfidf': True,
    'use_word_embeddings': False,
    'embedding_dim': 100,
    'max_sequence_length': 500,
}