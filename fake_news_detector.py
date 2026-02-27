import numpy as np
import pandas as pd
import pickle
import re
import string 
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class TextPreprocessor:
    """Handles text cleaning and preprocessing"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text: str) -> str:
        """
        clean and normalize text
        - Convert to lowercase
        - Remove special characters
        - Remove URLs
        - Remove extra whitespace
        """
        if not isinstance(text, str):
            return ""
        
        # convert to lowercase
        text = text.lower()

        # remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text
    
    def tokenize_and_stem(self, text: str) -> str:
        """Tokenize and apply stemming"""
        tokens = word_tokenize(text)
        stemmed = [self.stemmer.stem(token) for token in tokens
                   if token not in self.stop_words]
        return ' '.join(stemmed)
    
    def preprocess(self, text: str) -> str:
        """Complete preprocessing pipeline"""
        text = self.clean_text(text)
        text = self.tokenize_and_stem(text)
        return text
    
class FakeNewsDetector:
    """Main detector class for using multiple classifiers"""
    def __init__(self, vectorizer_max_features=5000, ngram_range=(1, 2)):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(
            max_features=vectorizer_max_features,
            ngram_range=ngram_range,
            min_df=5,
            max_df=0.8
        )

        # Initialize multiple classifiers for ensemble
        self.classifiers = {
            'logistic_regression': LogisticRegression(max_iter=200, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'svm': LinearSVC(random_state=42, max_iter=2000)
        }

        self.models_trained = False
        self.performance_metrics = {}

    def preprocess_data(self, texts: List[str]) -> List[str]:
        """Preprocess list of texts"""
        return [self.preprocessor.preprocess(text) for text in texts]
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray):
        """
        Train all classifiers and evaluate
        
        Args:
            X_train: Training feature vectors
            y_train: Training labels
            X_test: Test feature vectors
            y_test: Test labels
        """
        print("🚀 Training Fake News Detection Models...\n")

        for name, classifier in self.classifiers.items():
            print(f"Training {name.replace('_', ' ').title()}...")

            # train
            classifier.fit(X_train, y_train)

            # pridict
            y_pred = classifier.predict(X_test)

            # evaluate
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            self.performance_metrics[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'y_pred': y_pred,
                'y_test': y_test
            }

            print(f" Accuracy: {accuracy:.4f}")
            print(f" Precision: {precision:.4f}")
            print(f" Recall: {recall:.4f}")
            print(f" F1 Score: {f1:.4f}\n")

        self.models_trained = True

    def predict(self, texts: List[str]) -> Dict:
        """
        Predict if news is fake or real using ensemble voting
        
        Args:
            texts: List of news headlines/articles
        
        Returns:
            Dictionary with predictions and confidence scores
        """
        if not self.models_trained:
            raise ValueError("Models not trained yet. call train() first.")
        processed_texts = self.preprocess_data(texts)
        X = self.vectorizer.transform(processed_texts)

        predictions = {}

        for idx, text in enumerate(texts):
            votes = []
            probs = []

            for name, classifier in self.classifiers.items():
                prediction = classifier.predict(X[idx:idx+1])[0]
                votes.append(prediction)

                # Get confidence if available
                if hasattr(classifier, 'predict_proba'):
                    prob = classifier.predict_proba(X[idx:idx+1])[0]
                    probs.append(prob[1])
            
            # Ensemble voting
            final_prediction = max(set(votes), key=votes.count)
            confidence = np.mean(probs) if probs else 0.5
            
            predictions[text] = {
                'prediction': 'FAKE' if final_prediction == 1 else 'REAL',
                'confidence': float(confidence),
                'votes': votes
            }
        
        return predictions
    
    def print_performance_report(self):
        """Print detailed performance report"""
        print("\n" + "="*70)
        print("MODEL PERFORMANCE REPORT")
        print("="*70 + "\n")
        
        for model_name, metrics in self.performance_metrics.items():
            print(f"\n{model_name.replace('_', ' ').upper()}")
            print("-" * 70)
            print(f"Accuracy:  {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1 Score:  {metrics['f1_score']:.4f}")
            print(f"\nConfusion Matrix:\n{confusion_matrix(metrics['y_test'], metrics['y_pred'])}")
    
    def save_model(self, filepath: str):
        """Save trained model and vectorizer"""
        model_data = {
            'classifiers': self.classifiers,
            'vectorizer': self.vectorizer,
            'preprocessor': self.preprocessor
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model and vectorizer"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.classifiers = model_data['classifiers']
        self.vectorizer = model_data['vectorizer']
        self.preprocessor = model_data['preprocessor']
        self.models_trained = True
        print(f"✅ Model loaded from {filepath}")


def prepare_dataset(filepath: str) -> Tuple[List[str], np.ndarray]:
    """
    Prepare dataset for training
    Expected CSV format: 'title', 'text', 'label' (0=Real, 1=Fake)
    """
    df = pd.read_csv(filepath)
    
    # Handle missing values
    df = df.dropna(subset=['title', 'label'])
    
    # Combine title and text
    texts = df['title'].astype(str) + " " + df['text'].astype(str)
    labels = df['label'].values
    
    return texts.tolist(), labels


def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print("FAKE NEWS DETECTION SYSTEM")
    print("="*70 + "\n")
    
    # Load dataset
    print("📂 Loading dataset...")
    try:
        texts, labels = prepare_dataset('fake_news_dataset.csv')
        print(f"✅ Loaded {len(texts)} news articles")
        print(f"   Real news: {sum(labels == 0)}")
        print(f"   Fake news: {sum(labels == 1)}\n")
    except FileNotFoundError:
        print("❌ Dataset file not found!")
        print("Please ensure 'fake_news_dataset.csv' exists in the current directory")
        return
    
    # Initialize detector
    detector = FakeNewsDetector()
    
    # Preprocess data
    print("🔄 Preprocessing text data...")
    processed_texts = detector.preprocess_data(texts)
    print("✅ Preprocessing complete\n")
    
    # Vectorize
    print("📊 Vectorizing text (TF-IDF)...")
    X = detector.vectorizer.fit_transform(processed_texts)
    print(f"✅ Created feature matrix: {X.shape}\n")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Train models
    detector.train(X_train, y_train, X_test, y_test)
    
    # Print report
    detector.print_performance_report()
    
    # Save model
    detector.save_model('fake_news_detector_model.pkl')
    
    # Test predictions
    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS")
    print("="*70 + "\n")
    
    sample_texts = [
        "Trump wins 2024 election with historic landslide victory",
        "Scientists discover new miraculous cure that doctors hate",
        "New climate study shows rising global temperatures",
    ]
    
    predictions = detector.predict(sample_texts)
    
    for text, pred in predictions.items():
        print(f"Text: {text[:60]}...")
        print(f"Prediction: {pred['prediction']}")
        print(f"Confidence: {pred['confidence']:.2%}\n")


if __name__ == "__main__":
    main()
