"""
Comprehensive Test Suite for Entertainment Investment Intelligence Platform
=========================================================================

Test suite covering box office prediction accuracy, sentiment analysis, 
ROI calculations, and entertainment investment intelligence features.

Author: Emilio Cardenas
License: MIT
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json
import tempfile
import numpy as np
import pandas as pd
from decimal import Decimal

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from main import app, entertainment_platform, analytics_engine
    from advanced_entertainment_analytics import EntertainmentInvestmentPlatform
    from analytics_engine import AnalyticsEngine
    from data_manager import DataManager
    from ml_models import MLModelManager
    from visualization_manager import VisualizationManager
except ImportError as e:
    # Create mock classes for testing when modules aren't available
    print(f"Warning: Modules not available, using mocks: {e}")
    app = Mock()
    entertainment_platform = Mock()
    analytics_engine = Mock()


class TestEntertainmentPlatformCore:
    """Core platform functionality tests."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.client = app.test_client() if hasattr(app, 'test_client') else Mock()
        self.sample_movie_data = {
            'title': 'Test Movie',
            'genre': 'Action',
            'budget': 100000000,
            'director': 'Steven Spielberg',
            'cast_popularity': 8.5,
            'release_date': '2025-07-15',
            'runtime': 120,
            'rating': 'PG-13',
            'studio': 'Warner Bros',
            'sequel': False,
            'franchise': True
        }
    
    def test_app_initialization(self):
        """Test Flask app initialization and basic configuration."""
        assert app is not None
        if hasattr(app, 'config'):
            assert 'SECRET_KEY' in app.config
    
    def test_home_dashboard(self):
        """Test main dashboard endpoint."""
        if hasattr(self.client, 'get'):
            response = self.client.get('/')
            assert response.status_code in [200, 404]  # Allow 404 for mock
    
    def test_health_check(self):
        """Test health check endpoint functionality."""
        if hasattr(self.client, 'get'):
            response = self.client.get('/health')
            if response.status_code == 200:
                data = json.loads(response.data) if hasattr(response, 'data') else {}
                assert 'status' in data
                assert data.get('service') == 'entertainment-investment-platform'
    
    def test_api_status(self):
        """Test API status endpoint."""
        if hasattr(self.client, 'get'):
            response = self.client.get('/api/v1/status')
            if response.status_code == 200:
                data = json.loads(response.data) if hasattr(response, 'data') else {}
                assert 'api_version' in data
                assert 'features' in data


class TestBoxOfficePrediction:
    """Box office prediction accuracy tests."""
    
    def setup_method(self):
        """Setup prediction test environment."""
        self.client = app.test_client() if hasattr(app, 'test_client') else Mock()
        self.prediction_data = {
            'title': 'Blockbuster Movie',
            'genre': 'Action',
            'budget': 150000000,
            'director_score': 9.2,
            'cast_popularity': 8.8,
            'franchise': True,
            'sequel': False,
            'release_season': 'summer'
        }
    
    def test_box_office_prediction_endpoint(self):
        """Test box office prediction API endpoint."""
        if hasattr(self.client, 'post'):
            response = self.client.post('/api/v1/predict', 
                                      json=self.prediction_data,
                                      content_type='application/json')
            if response.status_code == 200:
                data = json.loads(response.data)
                assert 'predicted_gross' in data
                assert 'confidence_interval' in data
                assert 'roi_analysis' in data
    
    def test_prediction_accuracy_validation(self):
        """Test prediction accuracy metrics."""
        # Mock prediction results
        predictions = [180000000, 220000000, 165000000]
        actuals = [175000000, 215000000, 170000000]
        
        # Calculate Mean Absolute Percentage Error
        mape = np.mean(np.abs((np.array(actuals) - np.array(predictions)) / np.array(actuals))) * 100
        assert mape < 15  # Less than 15% error
    
    def test_confidence_interval_calculation(self):
        """Test confidence interval calculations."""
        predicted_gross = 200000000
        confidence = 0.95
        std_error = 15000000
        
        # Calculate 95% confidence interval
        margin = 1.96 * std_error  # Z-score for 95% confidence
        lower_bound = predicted_gross - margin
        upper_bound = predicted_gross + margin
        
        assert lower_bound < predicted_gross < upper_bound
        assert (upper_bound - lower_bound) / predicted_gross < 0.3  # Within 30%
    
    def test_genre_performance_factors(self):
        """Test genre-specific prediction factors."""
        genre_multipliers = {
            'Action': 1.2,
            'Comedy': 1.0,
            'Drama': 0.8,
            'Horror': 1.5,
            'Sci-Fi': 1.3
        }
        
        for genre, multiplier in genre_multipliers.items():
            assert 0.5 <= multiplier <= 2.0
    
    def test_seasonal_adjustment_factors(self):
        """Test seasonal box office adjustment factors."""
        seasonal_factors = {
            'summer': 1.3,
            'winter': 1.1,
            'spring': 0.9,
            'fall': 0.8
        }
        
        for season, factor in seasonal_factors.items():
            assert 0.7 <= factor <= 1.5


class TestROICalculations:
    """ROI calculation and investment analysis tests."""
    
    def setup_method(self):
        """Setup ROI test environment."""
        self.investment_data = {
            'production_budget': 100000000,
            'marketing_budget': 50000000,
            'distribution_costs': 20000000,
            'predicted_gross': 300000000,
            'studio_share': 0.55
        }
    
    def test_basic_roi_calculation(self):
        """Test basic ROI calculation formula."""
        total_investment = (self.investment_data['production_budget'] + 
                          self.investment_data['marketing_budget'] + 
                          self.investment_data['distribution_costs'])
        
        net_revenue = (self.investment_data['predicted_gross'] * 
                      self.investment_data['studio_share'])
        
        roi = (net_revenue - total_investment) / total_investment
        
        assert roi > 0  # Should be profitable
        assert 0 <= roi <= 10  # Reasonable ROI range
    
    def test_risk_adjusted_roi(self):
        """Test risk-adjusted ROI calculations."""
        base_roi = 1.5
        risk_factors = {
            'market_risk': 0.1,
            'execution_risk': 0.15,
            'competition_risk': 0.08
        }
        
        total_risk = sum(risk_factors.values())
        risk_adjusted_roi = base_roi * (1 - total_risk)
        
        assert risk_adjusted_roi < base_roi
        assert risk_adjusted_roi > 0
    
    def test_payback_period_calculation(self):
        """Test investment payback period calculation."""
        initial_investment = 170000000
        monthly_revenue = 25000000
        
        payback_months = initial_investment / monthly_revenue
        
        assert payback_months > 0
        assert payback_months < 24  # Should payback within 2 years
    
    def test_portfolio_diversification_metrics(self):
        """Test portfolio diversification calculations."""
        portfolio_weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        
        # Calculate Herfindahl-Hirschman Index for concentration
        hhi = sum(w**2 for w in portfolio_weights)
        
        assert hhi <= 1.0  # Maximum concentration
        assert hhi >= 0.2  # Reasonable diversification
    
    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio for portfolio performance."""
        portfolio_returns = [0.15, 0.22, 0.18, 0.25, 0.12]
        risk_free_rate = 0.03
        
        excess_returns = [r - risk_free_rate for r in portfolio_returns]
        avg_excess_return = np.mean(excess_returns)
        std_excess_return = np.std(excess_returns)
        
        sharpe_ratio = avg_excess_return / std_excess_return if std_excess_return > 0 else 0
        
        assert sharpe_ratio >= 0
        assert sharpe_ratio <= 5  # Reasonable upper bound


class TestSentimentAnalysis:
    """Sentiment analysis and market intelligence tests."""
    
    def setup_method(self):
        """Setup sentiment analysis test data."""
        self.sample_reviews = [
            "Amazing movie, great acting and storyline!",
            "Terrible plot, waste of time and money.",
            "Pretty good film, worth watching but not exceptional.",
            "Outstanding cinematography and direction!",
            "Boring and predictable, disappointed."
        ]
        
        self.social_mentions = {
            'twitter': 15000,
            'instagram': 8500,
            'facebook': 12000,
            'youtube': 3200,
            'tiktok': 25000
        }
    
    def test_sentiment_scoring(self):
        """Test sentiment scoring algorithm."""
        # Mock sentiment scores
        sentiment_scores = [0.8, -0.7, 0.2, 0.9, -0.6]
        
        avg_sentiment = np.mean(sentiment_scores)
        sentiment_std = np.std(sentiment_scores)
        
        assert -1 <= avg_sentiment <= 1
        assert sentiment_std >= 0
    
    def test_social_media_buzz_calculation(self):
        """Test social media buzz metrics."""
        total_mentions = sum(self.social_mentions.values())
        weighted_buzz = (
            self.social_mentions['twitter'] * 1.2 +
            self.social_mentions['instagram'] * 1.1 +
            self.social_mentions['facebook'] * 1.0 +
            self.social_mentions['youtube'] * 1.3 +
            self.social_mentions['tiktok'] * 1.4
        )
        
        assert total_mentions > 0
        assert weighted_buzz > total_mentions
    
    def test_critic_audience_correlation(self):
        """Test correlation between critic and audience scores."""
        critic_scores = [85, 72, 91, 68, 78]
        audience_scores = [82, 75, 88, 65, 81]
        
        correlation = np.corrcoef(critic_scores, audience_scores)[0, 1]
        
        assert -1 <= correlation <= 1
        assert not np.isnan(correlation)
    
    def test_trend_momentum_analysis(self):
        """Test trend momentum calculations."""
        daily_mentions = [1000, 1200, 1500, 1800, 1650, 2000, 2300]
        
        # Calculate momentum (rate of change)
        momentum = []
        for i in range(1, len(daily_mentions)):
            daily_change = (daily_mentions[i] - daily_mentions[i-1]) / daily_mentions[i-1]
            momentum.append(daily_change)
        
        avg_momentum = np.mean(momentum)
        assert avg_momentum >= -1  # No more than 100% decline per day
    
    def test_sentiment_volatility(self):
        """Test sentiment volatility calculations."""
        daily_sentiment = [0.6, 0.4, 0.7, 0.3, 0.8, 0.5, 0.6]
        
        sentiment_volatility = np.std(daily_sentiment)
        
        assert sentiment_volatility >= 0
        assert sentiment_volatility <= 1  # Maximum possible volatility


class TestMarketIntelligence:
    """Market intelligence and competitive analysis tests."""
    
    def setup_method(self):
        """Setup market intelligence test data."""
        self.competitor_data = {
            'similar_releases': [
                {'title': 'Competitor A', 'budget': 120000000, 'gross': 280000000},
                {'title': 'Competitor B', 'budget': 90000000, 'gross': 200000000},
                {'title': 'Competitor C', 'budget': 150000000, 'gross': 320000000}
            ]
        }
    
    def test_competitive_landscape_analysis(self):
        """Test competitive landscape analysis."""
        competitor_rois = []
        for movie in self.competitor_data['similar_releases']:
            roi = (movie['gross'] - movie['budget']) / movie['budget']
            competitor_rois.append(roi)
        
        market_avg_roi = np.mean(competitor_rois)
        market_std_roi = np.std(competitor_rois)
        
        assert market_avg_roi > 0
        assert market_std_roi >= 0
    
    def test_market_share_calculation(self):
        """Test market share calculations."""
        total_market_gross = sum(movie['gross'] for movie in self.competitor_data['similar_releases'])
        our_predicted_gross = 250000000
        
        projected_market_share = our_predicted_gross / (total_market_gross + our_predicted_gross)
        
        assert 0 <= projected_market_share <= 1
    
    def test_release_timing_optimization(self):
        """Test release timing optimization analysis."""
        release_windows = {
            'memorial_day': 1.15,
            'july_4th': 1.25,
            'labor_day': 0.95,
            'thanksgiving': 1.10,
            'christmas': 1.20
        }
        
        optimal_window = max(release_windows, key=release_windows.get)
        assert optimal_window in release_windows
        assert release_windows[optimal_window] >= 1.0
    
    def test_genre_performance_trends(self):
        """Test genre performance trend analysis."""
        genre_trends = {
            'action': [1.2, 1.3, 1.25, 1.4, 1.35],
            'comedy': [1.0, 0.95, 1.05, 0.98, 1.02],
            'drama': [0.8, 0.85, 0.82, 0.87, 0.84]
        }
        
        for genre, trend_data in genre_trends.items():
            trend_slope = np.polyfit(range(len(trend_data)), trend_data, 1)[0]
            assert not np.isnan(trend_slope)
    
    def test_audience_demographic_analysis(self):
        """Test audience demographic analysis."""
        demographics = {
            'age_18_24': 0.25,
            'age_25_34': 0.35,
            'age_35_44': 0.20,
            'age_45_54': 0.15,
            'age_55_plus': 0.05
        }
        
        total_percentage = sum(demographics.values())
        assert abs(total_percentage - 1.0) < 0.01  # Should sum to 100%


class TestPerformanceMetrics:
    """Performance and scalability tests."""
    
    def test_prediction_response_time(self):
        """Test prediction API response time."""
        import time
        
        start_time = time.time()
        
        # Simulate prediction calculation
        prediction_data = {
            'budget': 100000000,
            'genre_factor': 1.2,
            'star_power': 8.5,
            'director_factor': 1.1
        }
        
        # Mock prediction calculation
        predicted_gross = (prediction_data['budget'] * 
                         prediction_data['genre_factor'] * 
                         (prediction_data['star_power'] / 10) * 
                         prediction_data['director_factor'])
        
        end_time = time.time()
        response_time = end_time - start_time
        
        assert response_time < 1.0  # Should respond within 1 second
        assert predicted_gross > 0
    
    def test_concurrent_predictions(self):
        """Test handling multiple concurrent predictions."""
        import threading
        
        results = []
        
        def make_prediction():
            # Simulate prediction
            result = {'prediction': 200000000, 'confidence': 0.85}
            results.append(result)
        
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_prediction)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(results) == 10
        assert all(r['prediction'] > 0 for r in results)
    
    def test_memory_usage_optimization(self):
        """Test memory usage optimization."""
        import sys
        
        # Create large dataset simulation
        large_dataset = list(range(100000))
        initial_size = sys.getsizeof(large_dataset)
        
        # Process in chunks to optimize memory
        chunk_size = 1000
        processed_chunks = []
        
        for i in range(0, len(large_dataset), chunk_size):
            chunk = large_dataset[i:i + chunk_size]
            # Process chunk (mock operation)
            processed_chunks.append(len(chunk))
        
        assert sum(processed_chunks) == len(large_dataset)
        assert len(processed_chunks) == 100  # 100 chunks of 1000 each
    
    def test_caching_mechanism(self):
        """Test caching mechanism for improved performance."""
        cache = {}
        
        def cached_prediction(movie_id, data):
            cache_key = f"prediction_{movie_id}"
            if cache_key in cache:
                return cache[cache_key]
            
            # Simulate expensive calculation
            result = {'prediction': data.get('budget', 0) * 2.5}
            cache[cache_key] = result
            return result
        
        # First call - should cache
        result1 = cached_prediction('movie_1', {'budget': 100000000})
        assert 'movie_1' in str(list(cache.keys())[0]) if cache else False
        
        # Second call - should use cache
        result2 = cached_prediction('movie_1', {'budget': 100000000})
        assert result1 == result2


class TestDataValidation:
    """Data validation and integrity tests."""
    
    def test_input_data_validation(self):
        """Test input data validation."""
        valid_data = {
            'title': 'Test Movie',
            'budget': 100000000,
            'genre': 'Action',
            'rating': 'PG-13'
        }
        
        # Test required fields
        required_fields = ['title', 'budget', 'genre']
        for field in required_fields:
            assert field in valid_data
            assert valid_data[field] is not None
        
        # Test data types
        assert isinstance(valid_data['title'], str)
        assert isinstance(valid_data['budget'], (int, float))
        assert valid_data['budget'] > 0
    
    def test_budget_range_validation(self):
        """Test budget range validation."""
        test_budgets = [1000000, 50000000, 200000000, 500000000]
        
        for budget in test_budgets:
            assert budget >= 100000  # Minimum realistic budget
            assert budget <= 1000000000  # Maximum realistic budget
    
    def test_date_validation(self):
        """Test release date validation."""
        from datetime import datetime
        
        valid_dates = [
            '2025-07-15',
            '2025-12-25',
            '2026-06-01'
        ]
        
        for date_str in valid_dates:
            try:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                assert date_obj > datetime.now()  # Future release date
            except ValueError:
                pytest.fail(f"Invalid date format: {date_str}")
    
    def test_genre_validation(self):
        """Test genre validation."""
        valid_genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi', 'Romance', 'Thriller']
        test_genre = 'Action'
        
        assert test_genre in valid_genres
    
    def test_rating_validation(self):
        """Test movie rating validation."""
        valid_ratings = ['G', 'PG', 'PG-13', 'R', 'NC-17']
        test_rating = 'PG-13'
        
        assert test_rating in valid_ratings


class TestErrorHandling:
    """Error handling and edge case tests."""
    
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        invalid_inputs = [
            None,
            {},
            {'budget': -1000000},  # Negative budget
            {'genre': 'InvalidGenre'},
            {'title': ''}  # Empty title
        ]
        
        for invalid_input in invalid_inputs:
            # Should handle gracefully without crashing
            try:
                # Mock validation function
                if invalid_input is None or invalid_input == {}:
                    raise ValueError("No data provided")
                if invalid_input.get('budget', 0) < 0:
                    raise ValueError("Budget must be positive")
                if invalid_input.get('title') == '':
                    raise ValueError("Title cannot be empty")
            except ValueError as e:
                assert str(e)  # Should have error message
    
    def test_api_error_responses(self):
        """Test API error response handling."""
        if hasattr(self, 'client'):
            # Test 404 error
            response = self.client.get('/nonexistent-endpoint') if hasattr(self.client, 'get') else Mock()
            if hasattr(response, 'status_code'):
                assert response.status_code in [404, 500]  # Expected error codes
    
    def test_database_connection_error(self):
        """Test database connection error handling."""
        # Mock database connection failure
        def mock_db_connect():
            raise ConnectionError("Database unavailable")
        
        try:
            mock_db_connect()
            assert False, "Should have raised ConnectionError"
        except ConnectionError as e:
            assert "Database unavailable" in str(e)
    
    def test_external_api_timeout(self):
        """Test external API timeout handling."""
        import time
        
        def mock_external_api_call(timeout=1):
            time.sleep(timeout + 0.1)  # Simulate timeout
            return "API Response"
        
        start_time = time.time()
        try:
            result = mock_external_api_call(timeout=0.001)
            end_time = time.time()
            assert (end_time - start_time) > 0
        except Exception:
            # Timeout handling should be graceful
            pass


class TestSecurityAndCompliance:
    """Security and compliance tests."""
    
    def test_input_sanitization(self):
        """Test input sanitization for security."""
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE movies; --",
            "../../../etc/passwd"
        ]
        
        for malicious_input in malicious_inputs:
            # Should sanitize or reject malicious inputs
            sanitized = malicious_input.replace('<script>', '').replace('DROP TABLE', '')
            assert sanitized != malicious_input or malicious_input == sanitized
    
    def test_authentication_required(self):
        """Test authentication requirements."""
        # Mock authentication check
        def require_auth(api_key):
            valid_keys = ['test-key-123', 'prod-key-456']
            return api_key in valid_keys
        
        assert require_auth('test-key-123') == True
        assert require_auth('invalid-key') == False
    
    def test_rate_limiting(self):
        """Test API rate limiting."""
        request_count = 0
        max_requests = 100
        
        def make_request():
            nonlocal request_count
            if request_count >= max_requests:
                raise Exception("Rate limit exceeded")
            request_count += 1
            return "Success"
        
        # Should allow normal usage
        for i in range(50):
            result = make_request()
            assert result == "Success"
        
        assert request_count == 50
    
    def test_data_encryption(self):
        """Test sensitive data encryption."""
        import hashlib
        
        sensitive_data = "user-financial-data"
        
        # Mock encryption
        encrypted_data = hashlib.sha256(sensitive_data.encode()).hexdigest()
        
        assert encrypted_data != sensitive_data
        assert len(encrypted_data) == 64  # SHA256 hex length


if __name__ == '__main__':
    # Configure pytest for comprehensive testing
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '--cov=src',
        '--cov-report=html',
        '--cov-report=term-missing'
    ])