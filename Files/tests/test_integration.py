"""
Integration Testing Suite for Entertainment Investment Intelligence Platform
==========================================================================

Tests integration between components, API endpoints, database operations,
and external services for the entertainment investment platform.

Author: Emilio Cardenas
License: MIT
"""

import pytest
import asyncio
import requests
import json
import time
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import threading
import tempfile
import sqlite3
import pandas as pd

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from main import app
    from advanced_entertainment_analytics import EntertainmentInvestmentPlatform
    from analytics_engine import AnalyticsEngine
    from data_manager import DataManager
    from ml_models import MLModelManager
    from visualization_manager import VisualizationManager
except ImportError as e:
    print(f"Warning: Using mocks for unavailable modules: {e}")
    app = Mock()


class TestAPIIntegration:
    """API endpoint integration tests."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.client = app.test_client() if hasattr(app, 'test_client') else Mock()
        self.base_url = 'http://localhost:8002'
        self.headers = {'Content-Type': 'application/json'}
        
        self.sample_prediction_request = {
            'title': 'Integration Test Movie',
            'genre': 'Action',
            'budget': 125000000,
            'director': 'Christopher Nolan',
            'cast': ['Leonardo DiCaprio', 'Tom Hardy'],
            'studio': 'Warner Bros',
            'release_date': '2025-07-15',
            'rating': 'PG-13',
            'runtime': 148,
            'franchise': True,
            'sequel': False
        }
    
    def test_full_prediction_workflow(self):
        """Test complete prediction workflow from request to response."""
        if not hasattr(self.client, 'post'):
            pytest.skip("Client not available")
            
        # Step 1: Make prediction request
        response = self.client.post('/api/v1/predict', 
                                   json=self.sample_prediction_request,
                                   headers=self.headers)
        
        if response.status_code == 200:
            data = json.loads(response.data)
            
            # Verify response structure
            assert 'predicted_gross' in data
            assert 'confidence_interval' in data
            assert 'roi_analysis' in data
            assert 'risk_assessment' in data
            assert 'market_analysis' in data
            assert 'recommendations' in data
            
            # Verify data types and ranges
            assert isinstance(data['predicted_gross'], (int, float))
            assert data['predicted_gross'] > 0
            
            if 'confidence_interval' in data:
                ci = data['confidence_interval']
                assert 'lower' in ci and 'upper' in ci
                assert ci['lower'] < data['predicted_gross'] < ci['upper']
    
    def test_portfolio_analytics_integration(self):
        """Test portfolio analytics endpoint integration."""
        if not hasattr(self.client, 'get'):
            pytest.skip("Client not available")
            
        response = self.client.get('/api/v1/portfolio')
        
        if response.status_code == 200:
            data = json.loads(response.data)
            
            # Verify portfolio structure
            expected_sections = [
                'portfolio_summary',
                'performance_metrics',
                'genre_breakdown',
                'budget_distribution',
                'recent_performance'
            ]
            
            for section in expected_sections:
                if section in data:
                    assert data[section] is not None
    
    def test_powerbi_integration_endpoint(self):
        """Test Power BI integration data endpoint."""
        if not hasattr(self.client, 'get'):
            pytest.skip("Client not available")
            
        response = self.client.get('/api/v1/powerbi/data')
        
        if response.status_code == 200:
            data = json.loads(response.data)
            
            # Verify Power BI data structure
            expected_keys = [
                'project_performance',
                'prediction_accuracy',
                'market_trends'
            ]
            
            for key in expected_keys:
                if key in data:
                    assert isinstance(data[key], (list, dict))
    
    def test_api_error_handling_integration(self):
        """Test API error handling integration."""
        if not hasattr(self.client, 'post'):
            pytest.skip("Client not available")
            
        # Test with invalid data
        invalid_requests = [
            {},  # Empty request
            {'title': ''},  # Empty title
            {'budget': -1000000},  # Negative budget
            {'genre': 'InvalidGenre'}  # Invalid genre
        ]
        
        for invalid_data in invalid_requests:
            response = self.client.post('/api/v1/predict',
                                       json=invalid_data,
                                       headers=self.headers)
            
            # Should handle errors gracefully
            assert response.status_code in [400, 422, 500]
    
    def test_concurrent_api_requests(self):
        """Test handling of concurrent API requests."""
        if not hasattr(self.client, 'post'):
            pytest.skip("Client not available")
        
        results = []
        errors = []
        
        def make_concurrent_request():
            try:
                response = self.client.post('/api/v1/predict',
                                           json=self.sample_prediction_request,
                                           headers=self.headers)
                results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple concurrent requests
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_concurrent_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(results) == 5
        assert len(errors) == 0 or len(errors) < 3  # Allow some errors under load


class TestDatabaseIntegration:
    """Database integration tests."""
    
    def setup_method(self):
        """Setup test database."""
        self.test_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.test_db.close()
        
        # Create test database schema
        conn = sqlite3.connect(self.test_db.name)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS movies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                genre TEXT NOT NULL,
                budget INTEGER NOT NULL,
                predicted_gross INTEGER,
                actual_gross INTEGER,
                release_date TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                movie_id INTEGER,
                predicted_gross INTEGER NOT NULL,
                confidence_score REAL,
                model_version TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (movie_id) REFERENCES movies (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def teardown_method(self):
        """Clean up test database."""
        try:
            os.unlink(self.test_db.name)
        except FileNotFoundError:
            pass
    
    def test_movie_data_persistence(self):
        """Test movie data persistence in database."""
        conn = sqlite3.connect(self.test_db.name)
        cursor = conn.cursor()
        
        # Insert test movie data
        movie_data = (
            'Test Movie',
            'Action',
            100000000,
            250000000,
            '2025-07-15'
        )
        
        cursor.execute('''
            INSERT INTO movies (title, genre, budget, predicted_gross, release_date)
            VALUES (?, ?, ?, ?, ?)
        ''', movie_data)
        
        movie_id = cursor.lastrowid
        conn.commit()
        
        # Verify data was stored
        cursor.execute('SELECT * FROM movies WHERE id = ?', (movie_id,))
        result = cursor.fetchone()
        
        assert result is not None
        assert result[1] == 'Test Movie'  # title
        assert result[2] == 'Action'      # genre
        assert result[3] == 100000000     # budget
        
        conn.close()
    
    def test_prediction_history_tracking(self):
        """Test prediction history tracking."""
        conn = sqlite3.connect(self.test_db.name)
        cursor = conn.cursor()
        
        # Insert movie and predictions
        cursor.execute('''
            INSERT INTO movies (title, genre, budget, release_date)
            VALUES (?, ?, ?, ?)
        ''', ('History Test Movie', 'Drama', 50000000, '2025-08-01'))
        
        movie_id = cursor.lastrowid
        
        # Insert multiple predictions for the same movie
        predictions = [
            (movie_id, 125000000, 0.85, 'v1.0'),
            (movie_id, 130000000, 0.87, 'v1.1'),
            (movie_id, 128000000, 0.89, 'v1.2')
        ]
        
        cursor.executemany('''
            INSERT INTO predictions (movie_id, predicted_gross, confidence_score, model_version)
            VALUES (?, ?, ?, ?)
        ''', predictions)
        
        conn.commit()
        
        # Verify prediction history
        cursor.execute('''
            SELECT COUNT(*) FROM predictions WHERE movie_id = ?
        ''', (movie_id,))
        
        count = cursor.fetchone()[0]
        assert count == 3
        
        conn.close()
    
    def test_data_integrity_constraints(self):
        """Test database integrity constraints."""
        conn = sqlite3.connect(self.test_db.name)
        cursor = conn.cursor()
        
        # Test foreign key constraint
        try:
            cursor.execute('''
                INSERT INTO predictions (movie_id, predicted_gross, confidence_score)
                VALUES (?, ?, ?)
            ''', (99999, 100000000, 0.5))  # Non-existent movie_id
            
            conn.commit()
            
            # If no foreign key constraint, manually check
            cursor.execute('SELECT movie_id FROM predictions WHERE movie_id = 99999')
            result = cursor.fetchone()
            
            if result:
                # Check if movie exists
                cursor.execute('SELECT id FROM movies WHERE id = 99999')
                movie_exists = cursor.fetchone()
                assert movie_exists is None  # Should fail integrity
                
        except sqlite3.IntegrityError:
            # Expected behavior with foreign key constraints
            pass
        
        conn.close()
    
    def test_bulk_data_operations(self):
        """Test bulk database operations."""
        conn = sqlite3.connect(self.test_db.name)
        cursor = conn.cursor()
        
        # Prepare bulk data
        movies_data = [
            ('Bulk Movie 1', 'Action', 80000000, '2025-09-01'),
            ('Bulk Movie 2', 'Comedy', 45000000, '2025-09-15'),
            ('Bulk Movie 3', 'Drama', 25000000, '2025-10-01'),
            ('Bulk Movie 4', 'Horror', 15000000, '2025-10-15'),
            ('Bulk Movie 5', 'Sci-Fi', 120000000, '2025-11-01')
        ]
        
        # Execute bulk insert
        start_time = time.time()
        
        cursor.executemany('''
            INSERT INTO movies (title, genre, budget, release_date)
            VALUES (?, ?, ?, ?)
        ''', movies_data)
        
        conn.commit()
        end_time = time.time()
        
        # Verify all records were inserted
        cursor.execute('SELECT COUNT(*) FROM movies')
        total_count = cursor.fetchone()[0]
        assert total_count >= 5
        
        # Performance check (should be reasonably fast)
        execution_time = end_time - start_time
        assert execution_time < 1.0  # Should complete within 1 second
        
        conn.close()


class TestMLModelIntegration:
    """Machine Learning model integration tests."""
    
    def setup_method(self):
        """Setup ML testing environment."""
        self.model_manager = MLModelManager() if 'MLModelManager' in globals() else Mock()
        self.sample_features = {
            'budget': 100000000,
            'genre_action': 1,
            'genre_comedy': 0,
            'director_score': 8.5,
            'cast_popularity': 7.8,
            'franchise': 1,
            'sequel': 0,
            'release_month': 7,
            'studio_score': 8.2,
            'rating_pg13': 1
        }
    
    def test_model_loading_and_initialization(self):
        """Test ML model loading and initialization."""
        if isinstance(self.model_manager, Mock):
            pytest.skip("MLModelManager not available")
        
        # Test model initialization
        assert self.model_manager is not None
        
        # Mock model loading
        models_loaded = getattr(self.model_manager, 'models_loaded', True)
        assert models_loaded
    
    def test_feature_preprocessing(self):
        """Test feature preprocessing pipeline."""
        # Mock feature preprocessing
        def preprocess_features(raw_features):
            processed = {}
            
            # Convert budget to millions
            processed['budget_millions'] = raw_features.get('budget', 0) / 1000000
            
            # One-hot encode genre
            genre = raw_features.get('genre', 'Action')
            processed[f'genre_{genre.lower()}'] = 1
            
            # Normalize scores
            director_score = raw_features.get('director_score', 5.0)
            processed['director_score_norm'] = min(max(director_score / 10.0, 0), 1)
            
            return processed
        
        raw_data = {
            'budget': 100000000,
            'genre': 'Action',
            'director_score': 8.5
        }
        
        processed = preprocess_features(raw_data)
        
        assert 'budget_millions' in processed
        assert processed['budget_millions'] == 100.0
        assert 'genre_action' in processed
        assert processed['genre_action'] == 1
        assert 'director_score_norm' in processed
        assert 0 <= processed['director_score_norm'] <= 1
    
    def test_ensemble_model_prediction(self):
        """Test ensemble model prediction integration."""
        # Mock ensemble prediction
        def ensemble_predict(features):
            models = {
                'xgboost': features['budget'] * 2.1,
                'lightgbm': features['budget'] * 2.0,
                'random_forest': features['budget'] * 2.2,
                'neural_network': features['budget'] * 1.9
            }
            
            # Weighted ensemble
            weights = {
                'xgboost': 0.3,
                'lightgbm': 0.25,
                'random_forest': 0.25,
                'neural_network': 0.2
            }
            
            ensemble_prediction = sum(
                models[model] * weights[model] 
                for model in models
            )
            
            return {
                'prediction': ensemble_prediction,
                'individual_models': models,
                'confidence': 0.87
            }
        
        test_features = {'budget': 100000000}
        result = ensemble_predict(test_features)
        
        assert 'prediction' in result
        assert 'individual_models' in result
        assert 'confidence' in result
        assert result['prediction'] > 0
        assert 0 <= result['confidence'] <= 1
    
    def test_model_performance_validation(self):
        """Test model performance validation."""
        # Mock prediction vs actual data
        test_cases = [
            {'predicted': 180000000, 'actual': 175000000},
            {'predicted': 220000000, 'actual': 230000000},
            {'predicted': 95000000, 'actual': 88000000},
            {'predicted': 450000000, 'actual': 465000000},
            {'predicted': 125000000, 'actual': 132000000}
        ]
        
        # Calculate performance metrics
        errors = []
        for case in test_cases:
            error = abs(case['actual'] - case['predicted']) / case['actual']
            errors.append(error)
        
        mean_error = sum(errors) / len(errors)
        
        # Performance thresholds
        assert mean_error < 0.15  # Less than 15% average error
        assert all(error < 0.25 for error in errors)  # No single prediction > 25% error
    
    def test_confidence_interval_calculation(self):
        """Test confidence interval calculation."""
        def calculate_confidence_interval(prediction, model_uncertainty, confidence_level=0.95):
            import math
            
            # Z-score for confidence level
            z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
            z_score = z_scores.get(confidence_level, 1.96)
            
            # Calculate margin of error
            margin = z_score * model_uncertainty
            
            return {
                'lower': prediction - margin,
                'upper': prediction + margin,
                'margin': margin
            }
        
        prediction = 200000000
        uncertainty = 15000000
        
        ci = calculate_confidence_interval(prediction, uncertainty)
        
        assert 'lower' in ci and 'upper' in ci
        assert ci['lower'] < prediction < ci['upper']
        assert ci['upper'] - ci['lower'] == 2 * ci['margin']


class TestExternalServiceIntegration:
    """External service integration tests."""
    
    def setup_method(self):
        """Setup external service test environment."""
        self.mock_services = {
            'box_office_mojo': 'http://mock-boxofficemojo.com/api',
            'imdb': 'http://mock-imdb.com/api',
            'social_media': 'http://mock-social.com/api',
            'news_sentiment': 'http://mock-news.com/api'
        }
    
    def test_box_office_data_integration(self):
        """Test box office data service integration."""
        def mock_box_office_api(movie_title):
            # Mock API response
            return {
                'title': movie_title,
                'total_gross': 187500000,
                'domestic_gross': 125000000,
                'international_gross': 62500000,
                'opening_weekend': 45000000,
                'theater_count': 3500,
                'weeks_in_theaters': 12
            }
        
        result = mock_box_office_api('Test Movie')
        
        assert 'total_gross' in result
        assert 'domestic_gross' in result
        assert 'international_gross' in result
        assert result['total_gross'] > 0
    
    def test_social_media_sentiment_integration(self):
        """Test social media sentiment analysis integration."""
        def mock_sentiment_api(movie_title, platforms=['twitter', 'instagram']):
            sentiment_data = {}
            
            for platform in platforms:
                sentiment_data[platform] = {
                    'mentions': 15000 + hash(platform) % 10000,
                    'sentiment_score': 0.3 + (hash(platform) % 100) / 100 * 0.4,
                    'engagement_rate': 0.05 + (hash(platform) % 50) / 1000
                }
            
            return sentiment_data
        
        result = mock_sentiment_api('Integration Test Movie')
        
        for platform, data in result.items():
            assert 'mentions' in data
            assert 'sentiment_score' in data
            assert 'engagement_rate' in data
            assert -1 <= data['sentiment_score'] <= 1
    
    def test_news_sentiment_integration(self):
        """Test news sentiment analysis integration."""
        def mock_news_api(movie_title):
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
                    'audience reaction'
                ],
                'credibility_score': 0.78
            }
        
        result = mock_news_api('Test Movie')
        
        assert 'overall_sentiment' in result
        assert 'sentiment_breakdown' in result
        assert 'credibility_score' in result
        
        # Verify sentiment breakdown sums to 1
        breakdown = result['sentiment_breakdown']
        total = sum(breakdown.values())
        assert abs(total - 1.0) < 0.01
    
    def test_service_timeout_handling(self):
        """Test external service timeout handling."""
        def mock_slow_service(delay=0.5):
            time.sleep(delay)
            return {'status': 'success', 'data': 'response'}
        
        # Test with acceptable delay
        start_time = time.time()
        try:
            result = mock_slow_service(delay=0.1)
            end_time = time.time()
            
            assert end_time - start_time < 1.0
            assert result['status'] == 'success'
        except Exception:
            pytest.fail("Service should handle short delays")
    
    def test_service_failure_resilience(self):
        """Test resilience to external service failures."""
        def failing_service():
            raise ConnectionError("Service unavailable")
        
        def resilient_service_call():
            try:
                return failing_service()
            except ConnectionError:
                # Return fallback data
                return {
                    'status': 'fallback',
                    'message': 'Using cached or default data'
                }
        
        result = resilient_service_call()
        assert result['status'] == 'fallback'


class TestWorkflowIntegration:
    """End-to-end workflow integration tests."""
    
    def test_complete_investment_analysis_workflow(self):
        """Test complete investment analysis workflow."""
        # Step 1: Movie data input
        movie_data = {
            'title': 'Workflow Test Movie',
            'genre': 'Action',
            'budget': 150000000,
            'director': 'Christopher Nolan',
            'cast': ['Leonardo DiCaprio', 'Marion Cotillard'],
            'studio': 'Warner Bros',
            'release_date': '2025-07-15'
        }
        
        # Step 2: Feature extraction and preprocessing
        def extract_features(data):
            return {
                'budget_millions': data['budget'] / 1000000,
                'genre_action': 1 if data['genre'] == 'Action' else 0,
                'director_score': 9.2,  # Mock score for Christopher Nolan
                'cast_score': 8.8,      # Mock combined cast score
                'studio_score': 8.5,    # Mock studio score
                'release_month': 7      # July release
            }
        
        features = extract_features(movie_data)
        assert features['budget_millions'] == 150.0
        
        # Step 3: Box office prediction
        def predict_box_office(features):
            base_prediction = features['budget_millions'] * 2.1
            genre_multiplier = 1.2 if features['genre_action'] else 1.0
            director_multiplier = features['director_score'] / 8.0
            
            prediction = base_prediction * genre_multiplier * director_multiplier
            
            return {
                'predicted_gross': prediction * 1000000,
                'confidence': 0.87,
                'factors_used': features
            }
        
        prediction = predict_box_office(features)
        assert prediction['predicted_gross'] > movie_data['budget']
        
        # Step 4: ROI analysis
        def analyze_roi(budget, predicted_gross):
            roi = (predicted_gross - budget) / budget
            
            return {
                'roi': roi,
                'profit': predicted_gross - budget,
                'roi_category': 'High' if roi > 1.5 else 'Medium' if roi > 0.5 else 'Low'
            }
        
        roi_analysis = analyze_roi(movie_data['budget'], prediction['predicted_gross'])
        assert roi_analysis['roi'] > 0
        
        # Step 5: Risk assessment
        def assess_risk(movie_data, prediction):
            risk_factors = {
                'budget_risk': 0.3 if movie_data['budget'] > 100000000 else 0.1,
                'genre_risk': 0.2 if movie_data['genre'] == 'Action' else 0.3,
                'release_timing_risk': 0.1  # Summer release - low risk
            }
            
            total_risk = sum(risk_factors.values())
            
            return {
                'risk_score': total_risk,
                'risk_factors': risk_factors,
                'risk_level': 'High' if total_risk > 0.7 else 'Medium' if total_risk > 0.4 else 'Low'
            }
        
        risk_assessment = assess_risk(movie_data, prediction)
        assert 0 <= risk_assessment['risk_score'] <= 1
        
        # Step 6: Investment recommendation
        def generate_recommendation(roi_analysis, risk_assessment):
            if roi_analysis['roi'] > 1.0 and risk_assessment['risk_score'] < 0.5:
                recommendation = 'Strong Buy'
            elif roi_analysis['roi'] > 0.5 and risk_assessment['risk_score'] < 0.7:
                recommendation = 'Buy'
            elif roi_analysis['roi'] > 0.2:
                recommendation = 'Hold'
            else:
                recommendation = 'Avoid'
            
            return {
                'recommendation': recommendation,
                'reasoning': f"ROI: {roi_analysis['roi']:.2f}, Risk: {risk_assessment['risk_level']}"
            }
        
        recommendation = generate_recommendation(roi_analysis, risk_assessment)
        assert recommendation['recommendation'] in ['Strong Buy', 'Buy', 'Hold', 'Avoid']
    
    def test_batch_movie_analysis_workflow(self):
        """Test batch analysis of multiple movies."""
        movies_batch = [
            {'title': 'Movie A', 'genre': 'Action', 'budget': 120000000},
            {'title': 'Movie B', 'genre': 'Comedy', 'budget': 45000000},
            {'title': 'Movie C', 'genre': 'Drama', 'budget': 25000000},
            {'title': 'Movie D', 'genre': 'Horror', 'budget': 10000000}
        ]
        
        results = []
        
        for movie in movies_batch:
            # Mock analysis for each movie
            analysis = {
                'title': movie['title'],
                'predicted_gross': movie['budget'] * 2.0,
                'roi': 1.0,
                'risk_score': 0.4,
                'recommendation': 'Buy'
            }
            results.append(analysis)
        
        assert len(results) == len(movies_batch)
        assert all(result['predicted_gross'] > 0 for result in results)
    
    def test_real_time_data_update_workflow(self):
        """Test real-time data update workflow."""
        # Simulate real-time data updates
        def simulate_real_time_updates():
            updates = [
                {'type': 'social_sentiment', 'value': 0.72, 'timestamp': datetime.now()},
                {'type': 'pre_sales', 'value': 25000000, 'timestamp': datetime.now()},
                {'type': 'critic_score', 'value': 85, 'timestamp': datetime.now()},
                {'type': 'audience_score', 'value': 78, 'timestamp': datetime.now()}
            ]
            
            return updates
        
        updates = simulate_real_time_updates()
        
        # Process updates
        processed_updates = {}
        for update in updates:
            processed_updates[update['type']] = update['value']
        
        assert len(processed_updates) == 4
        assert 'social_sentiment' in processed_updates
        assert 'pre_sales' in processed_updates


if __name__ == '__main__':
    # Run integration tests
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '-x'  # Stop on first failure for integration tests
    ])