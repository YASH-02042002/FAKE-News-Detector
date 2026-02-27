import pytest
import numpy as np
import pandas as pd
from fake_news_detector import TextPreprocessor, FakeNewsDetector


class TestTextPreprocessor:
    """Test text preprocessing functionality"""
    
    @pytest.fixture
    def preprocessor(self):
        return TextPreprocessor()
    
    def test_clean_text_lowercase(self, preprocessor):
        """Test lowercase conversion"""
        text = "HELLO WORLD"
        result = preprocessor.clean_text(text)
        assert result == "hello world"
    
    def test_clean_text_remove_urls(self, preprocessor):
        """Test URL removal"""
        text = "Check this https://example.com for more"
        result = preprocessor.clean_text(text)
        assert "https" not in result
        assert "example.com" not in result
    
    def test_clean_text_remove_special_chars(self, preprocessor):
        """Test special character removal"""
        text = "Hello! @World #123"
        result = preprocessor.clean_text(text)
        assert "!" not in result
        assert "@" not in result
        assert "#" not in result
    
    def test_clean_text_whitespace(self, preprocessor):
        """Test extra whitespace removal"""
        text = "Hello    World    Test"
        result = preprocessor.clean_text(text)
        assert "    " not in result
    
    def test_preprocess_pipeline(self, preprocessor):
        """Test complete preprocessing"""
        text = "COVID-19 VACCINE APPROVED BY FDA!!!1"
        result = preprocessor.preprocess(text)
        assert isinstance(result, str)
        assert len(result) > 0
        assert result == result.lower()
    
    def test_empty_text(self, preprocessor):
        """Test handling of empty text"""
        result = preprocessor.preprocess("")
        assert result == ""
    
    def test_non_string_input(self, preprocessor):
        """Test handling of non-string input"""
        result = preprocessor.clean_text(123)
        assert result == ""


class TestFakeNewsDetector:
    """Test fake news detection functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data"""
        real_texts = [
            "Scientists discover new species in Amazon",
            "FDA approves new vaccine after trials",
            "Climate summit reaches historic agreement",
            "Companies invest in renewable energy"
        ]
        fake_texts = [
            "Miracle cure discovered - doctors hate this",
            "Secret government conspiracy revealed",
            "Celebrity shocking confession exposed",
            "This trick will shock you immediately"
        ]
        
        texts = real_texts + fake_texts
        labels = [0] * len(real_texts) + [1] * len(fake_texts)
        
        return texts, np.array(labels)
    
    @pytest.fixture
    def detector(self):
        return FakeNewsDetector()
    
    def test_initialization(self, detector):
        """Test detector initialization"""
        assert detector.preprocessor is not None
        assert detector.vectorizer is not None
        assert len(detector.classifiers) == 4
        assert not detector.models_trained
    
    def test_preprocess_data(self, detector, sample_data):
        """Test data preprocessing"""
        texts, _ = sample_data
        processed = detector.preprocess_data(texts)
        
        assert len(processed) == len(texts)
        assert all(isinstance(p, str) for p in processed)
        assert all(len(p) > 0 for p in processed)
    
    def test_vectorization(self, detector, sample_data):
        """Test TF-IDF vectorization"""
        texts, _ = sample_data
        processed = detector.preprocess_data(texts)
        X = detector.vectorizer.fit_transform(processed)
        
        assert X.shape[0] == len(texts)
        assert X.shape[1] > 0
        assert X.shape[1] <= 5000  # max_features
    
    def test_training(self, detector, sample_data):
        """Test model training"""
        texts, labels = sample_data
        processed = detector.preprocess_data(texts)
        X = detector.vectorizer.fit_transform(processed)
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = labels[:split_idx]
        y_test = labels[split_idx:]
        
        # Train
        detector.train(X_train, y_train, X_test, y_test)
        
        assert detector.models_trained
        assert len(detector.performance_metrics) == 4
    
    def test_prediction_before_training(self, detector):
        """Test that prediction fails before training"""
        with pytest.raises(ValueError):
            detector.predict(["test text"])
    
    def test_prediction_after_training(self, detector, sample_data):
        """Test prediction after training"""
        texts, labels = sample_data
        processed = detector.preprocess_data(texts)
        X = detector.vectorizer.fit_transform(processed)
        
        # Train
        split_idx = int(len(X) * 0.8)
        detector.train(
            X[:split_idx], labels[:split_idx],
            X[split_idx:], labels[split_idx:]
        )
        
        # Predict
        test_texts = ["New vaccine approved", "Miracle cure revealed"]
        results = detector.predict(test_texts)
        
        assert len(results) == len(test_texts)
        for text, pred in results.items():
            assert 'prediction' in pred
            assert 'confidence' in pred
            assert pred['prediction'] in ['REAL', 'FAKE']
            assert 0 <= pred['confidence'] <= 1
    
    def test_model_save_load(self, detector, sample_data, tmp_path):
        """Test model saving and loading"""
        # Train
        texts, labels = sample_data
        processed = detector.preprocess_data(texts)
        X = detector.vectorizer.fit_transform(processed)
        split_idx = int(len(X) * 0.8)
        detector.train(
            X[:split_idx], labels[:split_idx],
            X[split_idx:], labels[split_idx:]
        )
        
        # Save
        model_path = tmp_path / "test_model.pkl"
        detector.save_model(str(model_path))
        assert model_path.exists()
        
        # Load
        new_detector = FakeNewsDetector()
        new_detector.load_model(str(model_path))
        
        assert new_detector.models_trained


class TestPerformanceMetrics:
    """Test performance metrics"""
    
    @pytest.fixture
    def detector_with_results(self, sample_data):
        """Create detector with trained models"""
        detector = FakeNewsDetector()
        texts, labels = sample_data
        processed = detector.preprocess_data(texts)
        X = detector.vectorizer.fit_transform(processed)
        
        split_idx = int(len(X) * 0.8)
        detector.train(
            X[:split_idx], labels[:split_idx],
            X[split_idx:], labels[split_idx:]
        )
        return detector
    
    @pytest.fixture
    def sample_data(self):
        real_texts = [
            "Scientists discover new species",
            "FDA approves vaccine",
        ]
        fake_texts = [
            "Miracle cure doctors hate",
            "Secret conspiracy revealed",
        ]
        texts = real_texts + fake_texts
        labels = [0] * len(real_texts) + [1] * len(fake_texts)
        return texts, np.array(labels)
    
    def test_metrics_generated(self, detector_with_results):
        """Test that performance metrics are generated"""
        assert len(detector_with_results.performance_metrics) > 0
        
        for model_name, metrics in detector_with_results.performance_metrics.items():
            assert 'accuracy' in metrics
            assert 'precision' in metrics
            assert 'recall' in metrics
            assert 'f1_score' in metrics
    
    def test_metrics_valid_ranges(self, detector_with_results):
        """Test that metrics are in valid ranges"""
        for model_name, metrics in detector_with_results.performance_metrics.items():
            assert 0 <= metrics['accuracy'] <= 1
            assert 0 <= metrics['precision'] <= 1
            assert 0 <= metrics['recall'] <= 1
            assert 0 <= metrics['f1_score'] <= 1


def test_integration():
    """Integration test - full pipeline"""
    # Create detector
    detector = FakeNewsDetector()
    
    # Create sample data
    texts = [
        "New breakthrough in cancer research",
        "Miracle supplement cures all diseases",
        "Climate scientists warn of rising temperatures",
        "Secret government cover-up exposed",
    ]
    labels = np.array([0, 1, 0, 1])
    
    # Preprocess
    processed = detector.preprocess_data(texts)
    X = detector.vectorizer.fit_transform(processed)
    
    # Train
    split_idx = 3
    detector.train(X[:split_idx], labels[:split_idx], X[split_idx:], labels[split_idx:])
    
    # Predict
    test_texts = ["FDA approves new vaccine", "Doctors hate this one trick"]
    results = detector.predict(test_texts)
    
    assert len(results) == len(test_texts)
    assert all('confidence' in pred for pred in results.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])