"""
Pytest Configuration and Fixtures for Entertainment Investment Intelligence Platform
==================================================================================

Shared fixtures, configuration, and test utilities for the entertainment
investment intelligence platform test suite.

Author: Emilio Cardenas
License: MIT
"""

import pytest
import os
import sys
import tempfile
import sqlite3
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Generator

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure pytest
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture(scope="session")
def test_config():
    """Test configuration fixture."""
    return {
        'testing': True,
        'debug': False,
        'database_url': 'sqlite:///:memory:',
        'api_timeout': 30,
        'max_concurrent_requests': 10,
        'cache_timeout': 300,
        'log_level': 'WARNING'
    }


@pytest.fixture(scope="session")
def app():
    """Flask application fixture for testing."""
    try:
        from main import app as flask_app
        
        # Configure app for testing
        flask_app.config.update({
            'TESTING': True,
            'DEBUG': False,
            'SECRET_KEY': 'test-secret-key',
            'WTF_CSRF_ENABLED': False,
        })
        
        return flask_app
    except ImportError:
        # Return mock if app not available
        mock_app = Mock()
        mock_app.config = {}
        mock_app.test_client = Mock(return_value=Mock())
        return mock_app


@pytest.fixture
def client(app):
    """Flask test client fixture."""
    if hasattr(app, 'test_client'):
        return app.test_client()
    else:
        return Mock()


@pytest.fixture(scope="session")
def test_database():
    """Test database fixture."""
    # Create temporary database file
    db_fd, db_path = tempfile.mkstemp(suffix='.db')
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create test schema
        cursor.executescript('''
            CREATE TABLE movies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                genre TEXT NOT NULL,
                budget INTEGER NOT NULL,
                predicted_gross INTEGER,
                actual_gross INTEGER,
                director TEXT,
                studio TEXT,
                release_date TEXT,
                rating TEXT,
                runtime INTEGER,
                franchise BOOLEAN DEFAULT 0,
                sequel BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                movie_id INTEGER,
                predicted_gross INTEGER NOT NULL,
                confidence_score REAL,
                confidence_interval_lower INTEGER,
                confidence_interval_upper INTEGER,
                model_version TEXT,
                prediction_method TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (movie_id) REFERENCES movies (id)
            );
            
            CREATE TABLE portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                movie_id INTEGER,
                investment_amount INTEGER NOT NULL,
                investment_type TEXT,
                investment_date TEXT,
                expected_roi REAL,
                actual_roi REAL,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (movie_id) REFERENCES movies (id)
            );
            
            CREATE TABLE market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                genre TEXT,
                market_condition TEXT,
                box_office_total INTEGER,
                average_ticket_price REAL,
                theater_count INTEGER,
                seasonal_factor REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        ''')
        
        conn.commit()
        conn.close()
        
        yield db_path
        
    finally:
        os.close(db_fd)
        os.unlink(db_path)


@pytest.fixture
def sample_movie_data():
    """Sample movie data for testing."""
    return {
        'title': 'Test Blockbuster',
        'genre': 'Action',
        'budget': 150000000,
        'director': 'Christopher Nolan',
        'cast': ['Leonardo DiCaprio', 'Tom Hardy', 'Marion Cotillard'],
        'studio': 'Warner Bros',
        'release_date': '2025-07-15',
        'rating': 'PG-13',
        'runtime': 148,
        'franchise': True,
        'sequel': False,
        'description': 'A mind-bending action thriller',
        'genre_tags': ['action', 'thriller', 'sci-fi']
    }


@pytest.fixture
def sample_prediction_response():
    """Sample prediction response for testing."""
    return {
        'movie_title': 'Test Blockbuster',
        'predicted_gross': 375000000,
        'confidence_interval': {
            'lower': 325000000,
            'upper': 425000000,
            'confidence_level': 0.95
        },
        'roi_analysis': {
            'predicted_roi': 1.50,
            'investment_grade': 'A',
            'payback_period_months': 8.5,
            'risk_adjusted_return': 1.25,
            'profit_margin': 0.60
        },
        'risk_assessment': {
            'overall_risk_score': 4.2,
            'risk_category': 'Moderate Risk',
            'risk_factors': {
                'budget_risk': 6.0,
                'genre_risk': 4.0,
                'timing_risk': 3.0,
                'competition_risk': 5.0,
                'execution_risk': 4.5
            }
        },
        'market_analysis': {
            'competition_level': 'Medium',
            'genre_performance': 'Strong',
            'seasonal_factor': 'Favorable',
            'market_saturation': 'Low',
            'audience_demand': 'High'
        },
        'recommendations': [
            'Proceed with investment - strong ROI potential',
            'Consider premium marketing budget allocation',
            'Monitor competitor release schedules',
            'Leverage franchise elements in marketing'
        ],
        'model_metadata': {
            'model_version': '2.1.0',
            'ensemble_models': ['xgboost', 'lightgbm', 'neural_network'],
            'feature_importance': {
                'budget': 0.25,
                'genre': 0.20,
                'director': 0.18,
                'cast': 0.15,
                'franchise': 0.12,
                'timing': 0.10
            }
        },
        'timestamp': datetime.now().isoformat(),
        'processing_time_ms': 234
    }


@pytest.fixture
def sample_portfolio_data():
    """Sample portfolio data for testing."""
    return {
        'portfolio_summary': {
            'total_investments': 500000000,
            'current_value': 1235000000,
            'total_roi': 2.47,
            'active_projects': 23,
            'completed_projects': 67,
            'upcoming_releases': 8
        },
        'performance_metrics': {
            'sharpe_ratio': 2.84,
            'sortino_ratio': 3.21,
            'max_drawdown': -0.156,
            'win_rate': 0.734,
            'average_roi': 1.89,
            'volatility': 0.28
        },
        'genre_breakdown': {
            'action': {'weight': 0.32, 'performance': 0.287, 'count': 21},
            'comedy': {'weight': 0.24, 'performance': 0.198, 'count': 18},
            'drama': {'weight': 0.18, 'performance': 0.234, 'count': 15},
            'sci_fi': {'weight': 0.15, 'performance': 0.345, 'count': 12},
            'horror': {'weight': 0.11, 'performance': 0.167, 'count': 9}
        },
        'budget_distribution': {
            'micro_budget': {'weight': 0.15, 'range': '1M-10M', 'count': 12},
            'low_budget': {'weight': 0.28, 'range': '10M-50M', 'count': 25},
            'medium_budget': {'weight': 0.34, 'range': '50M-150M', 'count': 28},
            'high_budget': {'weight': 0.18, 'range': '150M-300M', 'count': 15},
            'blockbuster': {'weight': 0.05, 'range': '300M+', 'count': 5}
        },
        'recent_performance': [
            {
                'project': 'AI Revolution',
                'budget': 75000000,
                'gross': 189200000,
                'roi': 1.523,
                'status': 'completed'
            },
            {
                'project': 'Space Odyssey',
                'budget': 120000000,
                'gross': 345000000,
                'roi': 1.875,
                'status': 'completed'
            },
            {
                'project': 'Comedy Gold',
                'budget': 25000000,
                'gross': 82300000,
                'roi': 2.292,
                'status': 'completed'
            }
        ],
        'timestamp': datetime.now().isoformat()
    }


@pytest.fixture
def mock_ml_models():
    """Mock ML models for testing."""
    class MockMLModel:
        def __init__(self, name):
            self.name = name
            self.is_trained = True
            
        def predict(self, features):
            # Simple mock prediction based on budget
            budget = features.get('budget', 100000000)
            multiplier = {
                'xgboost': 2.1,
                'lightgbm': 2.0,
                'random_forest': 2.2,
                'neural_network': 1.9
            }.get(self.name, 2.0)
            
            return budget * multiplier
        
        def predict_proba(self, features):
            prediction = self.predict(features)
            confidence = 0.85 + (hash(str(features)) % 100) / 1000
            return {'prediction': prediction, 'confidence': confidence}
    
    return {
        'xgboost': MockMLModel('xgboost'),
        'lightgbm': MockMLModel('lightgbm'),
        'random_forest': MockMLModel('random_forest'),
        'neural_network': MockMLModel('neural_network')
    }


@pytest.fixture
def mock_external_apis():
    """Mock external API responses for testing."""
    def mock_box_office_api(movie_title):
        return {
            'title': movie_title,
            'total_gross': 187500000,
            'domestic_gross': 125000000,
            'international_gross': 62500000,
            'opening_weekend': 45000000,
            'theater_count': 3500,
            'weeks_in_theaters': 12,
            'last_updated': datetime.now().isoformat()
        }
    
    def mock_social_sentiment_api(movie_title):
        return {
            'overall_sentiment': 0.67,
            'platforms': {
                'twitter': {'mentions': 15420, 'sentiment': 0.72, 'engagement': 0.045},
                'instagram': {'mentions': 8934, 'sentiment': 0.68, 'engagement': 0.067},
                'facebook': {'mentions': 12567, 'sentiment': 0.63, 'engagement': 0.034},
                'youtube': {'mentions': 3421, 'sentiment': 0.71, 'engagement': 0.089},
                'tiktok': {'mentions': 24789, 'sentiment': 0.69, 'engagement': 0.156}
            },
            'trending_hashtags': ['#TestBlockbuster', '#MustWatch', '#BlockbusterMovie'],
            'last_updated': datetime.now().isoformat()
        }
    
    def mock_news_sentiment_api(movie_title):
        return {
            'articles_count': 156,
            'overall_sentiment': 0.67,
            'sentiment_breakdown': {
                'positive': 0.68,
                'neutral': 0.22,
                'negative': 0.10
            },
            'key_topics': [
                'box office performance',
                'critical reception',
                'audience reaction',
                'visual effects',
                'cast performance'
            ],
            'credibility_score': 0.78,
            'source_breakdown': {
                'entertainment_news': 45,
                'mainstream_media': 67,
                'trade_publications': 28,
                'blog_reviews': 16
            },
            'last_updated': datetime.now().isoformat()
        }
    
    return {
        'box_office': mock_box_office_api,
        'social_sentiment': mock_social_sentiment_api,
        'news_sentiment': mock_news_sentiment_api
    }


@pytest.fixture
def performance_test_data():
    """Generate performance test datasets."""
    def generate_test_dataset(size: int) -> pd.DataFrame:
        np.random.seed(42)  # For reproducible results
        
        data = {
            'movie_id': range(size),
            'title': [f'Movie {i}' for i in range(size)],
            'genre': np.random.choice(['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi'], size),
            'budget': np.random.uniform(5000000, 300000000, size),
            'director_score': np.random.uniform(3.0, 10.0, size),
            'cast_popularity': np.random.uniform(4.0, 9.0, size),
            'franchise': np.random.binomial(1, 0.2, size),
            'sequel': np.random.binomial(1, 0.15, size),
            'release_month': np.random.randint(1, 13, size),
            'studio_score': np.random.uniform(5.0, 9.0, size),
            'competition_factor': np.random.uniform(0.5, 1.5, size)
        }
        
        return pd.DataFrame(data)
    
    return generate_test_dataset


@pytest.fixture
def test_data_manager():
    """Test data manager with mocked database operations."""
    class TestDataManager:
        def __init__(self):
            self._movies = []
            self._predictions = []
            self._portfolio = []
        
        def add_movie(self, movie_data):
            movie_id = len(self._movies) + 1
            movie = {**movie_data, 'id': movie_id}
            self._movies.append(movie)
            return movie_id
        
        def get_movie(self, movie_id):
            for movie in self._movies:
                if movie['id'] == movie_id:
                    return movie
            return None
        
        def add_prediction(self, movie_id, prediction_data):
            prediction_id = len(self._predictions) + 1
            prediction = {
                **prediction_data,
                'id': prediction_id,
                'movie_id': movie_id,
                'created_at': datetime.now().isoformat()
            }
            self._predictions.append(prediction)
            return prediction_id
        
        def get_predictions(self, movie_id=None):
            if movie_id:
                return [p for p in self._predictions if p['movie_id'] == movie_id]
            return self._predictions
        
        def get_portfolio_summary(self):
            return {
                'total_movies': len(self._movies),
                'total_predictions': len(self._predictions),
                'avg_budget': sum(m.get('budget', 0) for m in self._movies) / len(self._movies) if self._movies else 0
            }
    
    return TestDataManager()


@pytest.fixture
def benchmark_timer():
    """Benchmark timer for performance tests."""
    class BenchmarkTimer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.perf_counter()
        
        def stop(self):
            self.end_time = time.perf_counter()
            return self.elapsed_time
        
        @property
        def elapsed_time(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return BenchmarkTimer()


@pytest.fixture
def mock_cache():
    """Mock cache for testing caching functionality."""
    class MockCache:
        def __init__(self):
            self._cache = {}
            self._ttl = {}
        
        def get(self, key):
            if key in self._cache:
                # Check TTL
                if key in self._ttl and time.time() > self._ttl[key]:
                    self.delete(key)
                    return None
                return self._cache[key]
            return None
        
        def set(self, key, value, ttl=None):
            self._cache[key] = value
            if ttl:
                self._ttl[key] = time.time() + ttl
        
        def delete(self, key):
            self._cache.pop(key, None)
            self._ttl.pop(key, None)
        
        def clear(self):
            self._cache.clear()
            self._ttl.clear()
        
        def size(self):
            return len(self._cache)
    
    return MockCache()


# Pytest markers for test categorization
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests as API tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# Test utilities
class TestUtils:
    """Utility functions for tests."""
    
    @staticmethod
    def assert_valid_prediction_response(response_data):
        """Assert that a prediction response has valid structure."""
        required_fields = [
            'movie_title', 'predicted_gross', 'confidence_interval',
            'roi_analysis', 'risk_assessment', 'timestamp'
        ]
        
        for field in required_fields:
            assert field in response_data, f"Missing required field: {field}"
        
        # Validate specific field types and ranges
        assert isinstance(response_data['predicted_gross'], (int, float))
        assert response_data['predicted_gross'] > 0
        
        ci = response_data['confidence_interval']
        assert 'lower' in ci and 'upper' in ci
        assert ci['lower'] < response_data['predicted_gross'] < ci['upper']
        
        roi = response_data['roi_analysis']
        assert 'predicted_roi' in roi
        assert isinstance(roi['predicted_roi'], (int, float))
        
        risk = response_data['risk_assessment']
        assert 'overall_risk_score' in risk
        assert isinstance(risk['overall_risk_score'], (int, float))
    
    @staticmethod
    def generate_random_movie_data():
        """Generate random movie data for testing."""
        import random
        
        genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi', 'Romance', 'Thriller']
        directors = ['Christopher Nolan', 'Quentin Tarantino', 'Steven Spielberg', 'Martin Scorsese']
        studios = ['Warner Bros', 'Disney', 'Universal', 'Sony', 'Paramount']
        ratings = ['G', 'PG', 'PG-13', 'R']
        
        return {
            'title': f'Random Movie {random.randint(1, 10000)}',
            'genre': random.choice(genres),
            'budget': random.randint(10000000, 300000000),
            'director': random.choice(directors),
            'studio': random.choice(studios),
            'rating': random.choice(ratings),
            'runtime': random.randint(90, 180),
            'franchise': random.choice([True, False]),
            'sequel': random.choice([True, False]),
            'release_date': f"2025-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
        }
    
    @staticmethod
    def validate_api_response_time(response_time, max_time=1.0):
        """Validate API response time."""
        assert response_time < max_time, f"Response time {response_time:.3f}s exceeds maximum {max_time}s"
        assert response_time >= 0, "Response time cannot be negative"
    
    @staticmethod
    def validate_json_structure(data, expected_keys):
        """Validate JSON response structure."""
        for key in expected_keys:
            assert key in data, f"Missing expected key: {key}"


# Make TestUtils available to all tests
@pytest.fixture
def test_utils():
    """Test utilities fixture."""
    return TestUtils()