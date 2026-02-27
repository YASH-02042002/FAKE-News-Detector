from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
from fake_news_detector import FakeNewsDetector, TextPreprocessor
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Global detector instance
detector = None
MODEL_PATH = 'fake_news_detector_model.pkl'

def load_model():
    """Load the trained model"""
    global detector
    if detector is None:
        if os.path.exists(MODEL_PATH):
            detector = FakeNewsDetector()
            detector.load_model(MODEL_PATH)
        else:
            return False
    return True

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': detector is not None
    }), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict if news is fake or real
    Expected JSON:
    {
        "text": ["news headline 1", "news headline 2"],
        "threshold": 0.5 # optional confidence threshold
    }
    """
    try:
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({
                'error': 'Missing "texts" field in request'
            }), 400

        texts = data.get('texts')
        threshold = data.get('threshold', 0.5)

        # Validate input
        if not isinstance(texts, list):
            texts = [texts]

        if not texts or any(not isinstance(t, str) for t in texts):
            return jsonify({
                'error': 'All texts must be non-empty strings'
            }), 400
        
        # Make predictions
        predictions = detector.predict(texts)

        # format response
        results = []
        for text, pred in predictions.items():
            confidence = pred['confidence']
            prediction = pred['prediction']

            # Apply threshold
            if confidence < threshold:
                reliability = 'UNCERTAIN'
            else:
                reliability = 'HIGH CONFIDENCE' if confidence > 0.8 else 'MEDIUM CONFIDENCE'

            results.append({
                'text': text,
                'prediction': prediction,
                'confidence': round(confidence, 4),
                'reliability': reliability,
                'is_fake': prediction == 'FAKE'
            })

        return jsonify({
            'status': 'success',
            'total_articles': len(results),
            'results': results
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500
    
@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """
    Predict multiple news items with detailed analysis
    
    Expected JSON:
    {
        "articles": [
            {"title": "...", "content": "..."},
            ...
        ]
    }
    """
    try:
        data = request.get_json()

        if not data or 'articles' not in data:
            return jsonify({'error': 'Missing "articles" field'}), 400
        
        articles = data['articles']

        # Combine title & contant
        texts = []
        for article in articles:
            title = article.get('title', '')
            content = article.get('content', '')
            combined = f"{title} {content}".strip()
            if combined:
                texts.append(combined)

        if not texts:
            return jsonify({'error': 'No valid articles provided'}), 400
        
        # Make predictions
        predictions = detector.predict(texts)

        # Format response
        results = []
        for article, pred in zip(articles, predictions.values()):
            results.append({
                'article': article,
                'prediction': pred['prediction'],
                'confidence': round(pred['confidence'], 4),
                'is_fake': pred['prediction'] == 'FAKE'
            })

        return jsonify({
            'status': 'success',
            'total_articles': len(results),
            'fake_count': sum(1 for r in results if r['is_fake']),
            'results': results
        }), 200
    
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500
    
@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Detailed analysis of a news article
    Returns prediction, confidence, and key statistics
    """
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'Missing text field'}), 400
        
        # Preprocess
        preprocessor = TextPreprocessor()
        processed = preprocessor.preprocess(text)
        
        # Get prediction
        predictions = detector.predict([text])
        pred = list(predictions.values())[0]
        
        # Analyze text
        word_count = len(text.split())
        processed_word_count = len(processed.split())
        unique_words = len(set(processed.split()))

        analysis = {
            'original_text': text[:500],  # First 500 chars
            'prediction': pred['prediction'],
            'confidence': round(pred['confidence'], 4),
            'statistics': {
                'word_count': word_count,
                'unique_words': unique_words,
                'processed_words': processed_word_count,
                'text_length': len(text)
            },
            'recommendation': (
                '⚠️  Likely Fake - Verify with trusted sources'
                if pred['prediction'] == 'FAKE'
                else '✅ Likely Real - But always verify important claims'
            )
        }

        return jsonify({
            'status': 'success',
            'analysis': analysis
        }), 200
    
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500
    
@app.route('/api/info', methods=['GET'])
def info():
    """Get information about the model"""
    return jsonify({
        'model_name': 'Fake News Detection System',
        'version': '1.0',
        'description': 'ML-based system for detecting fake news using multiple classifiers',
        'endpoints': {
            '/api/health': 'Check API health',
            '/api/predict': 'Single/multiple prediction',
            '/api/batch-predict': 'Batch prediction with article structure',
            '/api/analyze': 'Detailed analysis of text'
        },
        'models_used': [
            'Logistic Regression',
            'Random Forest',
            'Gradient Boosting',
            'Support Vector Machine'
        ]
    }), 200

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        'message': 'Fake News Detection API',
        'version': '1.0',
        'docs': 'Visit /api/info for documentation'
    }), 200

if __name__ == '__main__':
    print("Loading Model...")
    if load_model():
        print("Model loaded successfully")
        app.run(debug=True, port=5000)
    else:
        print("Failed to load model. please train the model first.")