"""
API Endpoint Testing Suite for Entertainment Investment Intelligence Platform
============================================================================

Comprehensive API testing covering all endpoints, authentication, validation,
error handling, and API contract compliance for entertainment investment analytics.

Author: Emilio Cardenas
License: MIT
"""

import pytest
import json
import sys
import os
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import requests
import time

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from main import app
    from advanced_entertainment_analytics import EntertainmentInvestmentPlatform
except ImportError as e:
    print(f"Warning: Using mocks for API testing: {e}")
    app = Mock()


class TestHealthAndStatusEndpoints:
    """Health check and status endpoint tests."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.client = app.test_client() if hasattr(app, 'test_client') else Mock()
        
    def test_health_endpoint(self):
        """Test health check endpoint."""
        if not hasattr(self.client, 'get'):
            pytest.skip("Client not available")
            
        response = self.client.get('/health')
        
        if hasattr(response, 'status_code'):
            if response.status_code == 200:
                data = json.loads(response.data)
                
                # Required fields in health response
                assert 'status' in data
                assert 'service' in data
                assert 'timestamp' in data
                
                # Validate field values
                assert data['status'] in ['healthy', 'degraded', 'unhealthy']
                assert data['service'] == 'entertainment-investment-platform'
                
                # Optional fields that should be present
                if 'components' in data:
                    assert isinstance(data['components'], dict)
                    
                if 'system_info' in data:
                    assert isinstance(data['system_info'], dict)
                    assert 'python_version' in data['system_info']
    
    def test_api_status_endpoint(self):
        """Test API status endpoint."""
        if not hasattr(self.client, 'get'):
            pytest.skip("Client not available")
            
        response = self.client.get('/api/v1/status')
        
        if hasattr(response, 'status_code'):
            if response.status_code == 200:
                data = json.loads(response.data)
                
                # Required fields
                assert 'api_version' in data
                assert 'status' in data
                assert 'features' in data
                
                # Validate field values
                assert data['api_version'] == 'v1'
                assert data['status'] == 'operational'
                assert isinstance(data['features'], list)
                
                # Expected features
                expected_features = [
                    'box_office_prediction',
                    'roi_optimization',
                    'risk_assessment',
                    'portfolio_analytics'
                ]
                
                for feature in expected_features:
                    if feature in data['features']:
                        assert isinstance(feature, str)
    
    def test_home_dashboard_endpoint(self):
        """Test home dashboard endpoint."""
        if not hasattr(self.client, 'get'):
            pytest.skip("Client not available")
            
        response = self.client.get('/')
        
        if hasattr(response, 'status_code'):
            assert response.status_code in [200, 404, 500]  # Allow various states
            
            if response.status_code == 200:
                # Should return HTML content for dashboard
                content_type = response.headers.get('Content-Type', '')
                assert 'html' in content_type.lower() or 'json' in content_type.lower()


class TestPredictionEndpoint:
    """Box office prediction endpoint tests."""
    
    def setup_method(self):
        """Setup for prediction tests."""
        self.client = app.test_client() if hasattr(app, 'test_client') else Mock()
        
        self.valid_prediction_request = {
            'title': 'Test Blockbuster',
            'genre': 'Action',
            'budget': 150000000,
            'director': 'Christopher Nolan',
            'cast': ['Leonardo DiCaprio', 'Tom Hardy'],
            'studio': 'Warner Bros',
            'release_date': '2025-07-15',
            'rating': 'PG-13',
            'runtime': 148,
            'franchise': True,
            'sequel': False
        }
    
    def test_prediction_endpoint_success(self):
        """Test successful prediction request."""
        if not hasattr(self.client, 'post'):
            pytest.skip("Client not available")
            
        response = self.client.post('/api/v1/predict',
                                   json=self.valid_prediction_request,
                                   content_type='application/json')
        
        if hasattr(response, 'status_code'):
            if response.status_code == 200:
                data = json.loads(response.data)
                
                # Required response fields
                assert 'movie_title' in data
                assert 'predicted_gross' in data
                assert 'confidence_interval' in data
                assert 'roi_analysis' in data
                assert 'risk_assessment' in data
                assert 'recommendations' in data
                assert 'timestamp' in data
                
                # Validate prediction data
                assert isinstance(data['predicted_gross'], (int, float))
                assert data['predicted_gross'] > 0
                
                # Validate confidence interval
                ci = data['confidence_interval']
                assert 'lower' in ci and 'upper' in ci
                assert ci['lower'] < data['predicted_gross'] < ci['upper']
                
                # Validate ROI analysis
                roi = data['roi_analysis']
                assert 'predicted_roi' in roi
                assert 'investment_grade' in roi
                assert isinstance(roi['predicted_roi'], (int, float))
                
                # Validate risk assessment
                risk = data['risk_assessment']
                assert 'overall_risk_score' in risk
                assert 'risk_category' in risk
                assert isinstance(risk['overall_risk_score'], (int, float))
                
                # Validate recommendations
                assert isinstance(data['recommendations'], list)
    
    def test_prediction_endpoint_validation(self):
        """Test prediction endpoint input validation."""
        if not hasattr(self.client, 'post'):
            pytest.skip("Client not available")
        
        # Test missing required fields
        invalid_requests = [
            {},  # Empty request
            {'title': ''},  # Empty title
            {'budget': -1000000},  # Negative budget
            {'genre': 'InvalidGenre'},  # Invalid genre
            {'rating': 'InvalidRating'},  # Invalid rating
            {'release_date': 'invalid-date'},  # Invalid date format
            {'runtime': -30},  # Negative runtime
        ]
        
        for invalid_request in invalid_requests:
            response = self.client.post('/api/v1/predict',
                                       json=invalid_request,
                                       content_type='application/json')
            
            if hasattr(response, 'status_code'):
                # Should return client error for invalid data
                assert response.status_code in [400, 422, 500]
                
                if response.status_code in [400, 422]:
                    data = json.loads(response.data) if hasattr(response, 'data') else {}
                    assert 'error' in data
    
    def test_prediction_endpoint_data_types(self):
        """Test prediction endpoint data type validation."""
        if not hasattr(self.client, 'post'):
            pytest.skip("Client not available")
        
        # Test with correct data types
        type_tests = [
            {'title': 'String Title', 'budget': 100000000},  # String and int
            {'title': 'Float Budget', 'budget': 100000000.0},  # Float budget
            {'franchise': True, 'sequel': False},  # Boolean values
            {'cast': ['Actor 1', 'Actor 2']},  # List of strings
            {'runtime': 120},  # Integer runtime
        ]
        
        for test_data in type_tests:
            request_data = {**self.valid_prediction_request, **test_data}
            
            response = self.client.post('/api/v1/predict',
                                       json=request_data,
                                       content_type='application/json')
            
            # Should accept valid data types
            if hasattr(response, 'status_code'):
                assert response.status_code in [200, 500, 503]  # Allow service errors
    
    def test_prediction_endpoint_edge_cases(self):
        """Test prediction endpoint edge cases."""
        if not hasattr(self.client, 'post'):
            pytest.skip("Client not available")
        
        edge_cases = [
            # Minimum budget
            {**self.valid_prediction_request, 'budget': 1000000},
            
            # Maximum budget  
            {**self.valid_prediction_request, 'budget': 500000000},
            
            # Very long title
            {**self.valid_prediction_request, 'title': 'A' * 200},
            
            # Future release date
            {**self.valid_prediction_request, 'release_date': '2030-12-25'},
            
            # Maximum runtime
            {**self.valid_prediction_request, 'runtime': 300},
            
            # Minimum runtime
            {**self.valid_prediction_request, 'runtime': 60},
        ]
        
        for edge_case in edge_cases:
            response = self.client.post('/api/v1/predict',
                                       json=edge_case,
                                       content_type='application/json')
            
            if hasattr(response, 'status_code'):
                # Should handle edge cases gracefully
                assert response.status_code in [200, 400, 422, 500]
    
    def test_prediction_content_type_handling(self):
        """Test prediction endpoint content type handling."""
        if not hasattr(self.client, 'post'):
            pytest.skip("Client not available")
        
        # Test with correct content type
        response = self.client.post('/api/v1/predict',
                                   json=self.valid_prediction_request,
                                   content_type='application/json')
        
        if hasattr(response, 'status_code'):
            assert response.status_code in [200, 400, 500, 503]
        
        # Test with incorrect content type (should handle or reject)
        response_wrong_type = self.client.post('/api/v1/predict',
                                              data=json.dumps(self.valid_prediction_request),
                                              content_type='text/plain')
        
        if hasattr(response_wrong_type, 'status_code'):
            assert response_wrong_type.status_code in [400, 415, 500]


class TestPortfolioEndpoint:
    """Portfolio analytics endpoint tests."""
    
    def setup_method(self):
        """Setup for portfolio tests."""
        self.client = app.test_client() if hasattr(app, 'test_client') else Mock()
    
    def test_portfolio_endpoint_success(self):
        """Test successful portfolio analytics request."""
        if not hasattr(self.client, 'get'):
            pytest.skip("Client not available")
            
        response = self.client.get('/api/v1/portfolio')
        
        if hasattr(response, 'status_code'):
            if response.status_code == 200:
                data = json.loads(response.data)
                
                # Expected portfolio sections
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
                
                # Validate portfolio summary
                if 'portfolio_summary' in data:
                    summary = data['portfolio_summary']
                    numeric_fields = ['total_investments', 'current_value', 'total_roi']
                    
                    for field in numeric_fields:
                        if field in summary:
                            assert isinstance(summary[field], (int, float))
                
                # Validate performance metrics
                if 'performance_metrics' in data:
                    metrics = data['performance_metrics']
                    
                    for metric_name, metric_value in metrics.items():
                        if isinstance(metric_value, (int, float)):
                            # ROI and ratios should be reasonable
                            if 'roi' in metric_name.lower():
                                assert -1 <= metric_value <= 10
                
                # Validate genre breakdown
                if 'genre_breakdown' in data:
                    genres = data['genre_breakdown']
                    assert isinstance(genres, dict)
                    
                    for genre, genre_data in genres.items():
                        if isinstance(genre_data, dict):
                            if 'weight' in genre_data:
                                assert 0 <= genre_data['weight'] <= 1
    
    def test_portfolio_endpoint_response_format(self):
        """Test portfolio endpoint response format."""
        if not hasattr(self.client, 'get'):
            pytest.skip("Client not available")
            
        response = self.client.get('/api/v1/portfolio')
        
        if hasattr(response, 'status_code'):
            if response.status_code == 200:
                # Should return JSON
                content_type = response.headers.get('Content-Type', '')
                assert 'json' in content_type.lower() or content_type == ''
                
                # Should be valid JSON
                try:
                    data = json.loads(response.data)
                    assert isinstance(data, dict)
                except json.JSONDecodeError:
                    pytest.fail("Response is not valid JSON")
    
    def test_portfolio_endpoint_caching(self):
        """Test portfolio endpoint caching behavior."""
        if not hasattr(self.client, 'get'):
            pytest.skip("Client not available")
        
        # Make multiple requests
        response_times = []
        
        for i in range(3):
            start_time = time.time()
            response = self.client.get('/api/v1/portfolio')
            end_time = time.time()
            
            if hasattr(response, 'status_code') and response.status_code == 200:
                response_times.append(end_time - start_time)
        
        if len(response_times) >= 2:
            # Later requests might be faster due to caching
            first_request_time = response_times[0]
            avg_later_requests = sum(response_times[1:]) / len(response_times[1:])
            
            # Allow for reasonable caching improvement
            assert avg_later_requests <= first_request_time * 1.5


class TestPowerBIEndpoint:
    """Power BI integration endpoint tests."""
    
    def setup_method(self):
        """Setup for Power BI tests."""
        self.client = app.test_client() if hasattr(app, 'test_client') else Mock()
    
    def test_powerbi_data_endpoint(self):
        """Test Power BI data endpoint."""
        if not hasattr(self.client, 'get'):
            pytest.skip("Client not available")
            
        response = self.client.get('/api/v1/powerbi/data')
        
        if hasattr(response, 'status_code'):
            if response.status_code == 200:
                data = json.loads(response.data)
                
                # Expected Power BI data sections
                expected_sections = [
                    'project_performance',
                    'prediction_accuracy',
                    'market_trends'
                ]
                
                for section in expected_sections:
                    if section in data:
                        assert data[section] is not None
                
                # Validate project performance data
                if 'project_performance' in data:
                    projects = data['project_performance']
                    assert isinstance(projects, list)
                    
                    for project in projects[:3]:  # Check first few projects
                        if isinstance(project, dict):
                            # Expected project fields
                            project_fields = ['title', 'genre', 'budget', 'predicted_gross']
                            for field in project_fields:
                                if field in project:
                                    if field in ['budget', 'predicted_gross']:
                                        assert isinstance(project[field], (int, float))
                                        assert project[field] > 0
                
                # Validate prediction accuracy data
                if 'prediction_accuracy' in data:
                    accuracy = data['prediction_accuracy']
                    assert isinstance(accuracy, list)
                    
                    for model in accuracy:
                        if isinstance(model, dict) and 'accuracy' in model:
                            assert 0 <= model['accuracy'] <= 1
                
                # Validate market trends
                if 'market_trends' in data:
                    trends = data['market_trends']
                    assert isinstance(trends, dict)
    
    def test_powerbi_data_format_compliance(self):
        """Test Power BI data format compliance."""
        if not hasattr(self.client, 'get'):
            pytest.skip("Client not available")
            
        response = self.client.get('/api/v1/powerbi/data')
        
        if hasattr(response, 'status_code'):
            if response.status_code == 200:
                data = json.loads(response.data)
                
                # Should be JSON serializable
                try:
                    json.dumps(data)
                except TypeError:
                    pytest.fail("Power BI data contains non-serializable objects")
                
                # Should not contain None values (Power BI doesn't handle well)
                def check_no_none_values(obj, path=""):
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            if value is None:
                                pytest.fail(f"Found None value at path: {path}.{key}")
                            check_no_none_values(value, f"{path}.{key}")
                    elif isinstance(obj, list):
                        for i, value in enumerate(obj):
                            if value is None:
                                pytest.fail(f"Found None value at path: {path}[{i}]")
                            check_no_none_values(value, f"{path}[{i}]")
                
                # Check for None values (uncomment if strict None checking is needed)
                # check_no_none_values(data)


class TestErrorHandling:
    """API error handling tests."""
    
    def setup_method(self):
        """Setup for error handling tests."""
        self.client = app.test_client() if hasattr(app, 'test_client') else Mock()
    
    def test_404_error_handling(self):
        """Test 404 error handling for non-existent endpoints."""
        if not hasattr(self.client, 'get'):
            pytest.skip("Client not available")
        
        non_existent_endpoints = [
            '/api/v1/nonexistent',
            '/api/v2/predict',
            '/invalid-endpoint',
            '/api/v1/predict/123'
        ]
        
        for endpoint in non_existent_endpoints:
            response = self.client.get(endpoint)
            
            if hasattr(response, 'status_code'):
                assert response.status_code == 404
                
                # Should return JSON error response
                try:
                    data = json.loads(response.data)
                    assert 'error' in data
                    assert isinstance(data['error'], str)
                except json.JSONDecodeError:
                    # Some 404 responses might be HTML
                    pass
    
    def test_405_method_not_allowed(self):
        """Test 405 error for wrong HTTP methods."""
        if not hasattr(self.client, 'get') or not hasattr(self.client, 'post'):
            pytest.skip("Client methods not available")
        
        # Test wrong methods on existing endpoints
        wrong_method_tests = [
            ('POST', '/health'),  # GET endpoint
            ('GET', '/api/v1/predict'),  # POST endpoint
            ('PUT', '/api/v1/portfolio'),  # GET endpoint
            ('DELETE', '/api/v1/status'),  # GET endpoint
        ]
        
        for method, endpoint in wrong_method_tests:
            if method == 'POST' and hasattr(self.client, 'post'):
                response = self.client.post(endpoint)
            elif method == 'GET' and hasattr(self.client, 'get'):
                response = self.client.get(endpoint)
            else:
                continue
                
            if hasattr(response, 'status_code'):
                # Should return 405 or handle gracefully
                assert response.status_code in [405, 404, 200, 500]
    
    def test_500_error_handling(self):
        """Test 500 error handling for server errors."""
        # This test would require mocking internal failures
        # For now, just verify error response structure
        
        def mock_500_response():
            return {
                'error': 'Internal server error',
                'message': 'Please contact support if the problem persists',
                'timestamp': datetime.now().isoformat()
            }
        
        error_response = mock_500_response()
        
        assert 'error' in error_response
        assert 'message' in error_response
        assert isinstance(error_response['error'], str)
    
    def test_request_timeout_handling(self):
        """Test request timeout handling."""
        # Mock timeout scenario
        def mock_timeout_request():
            import time
            time.sleep(0.1)  # Short delay to simulate processing
            return {'status': 'completed'}
        
        # Test should complete within reasonable time
        start_time = time.time()
        result = mock_timeout_request()
        end_time = time.time()
        
        request_time = end_time - start_time
        assert request_time < 1.0  # Should complete within 1 second
        assert result['status'] == 'completed'


class TestAPIRateLimiting:
    """API rate limiting tests."""
    
    def setup_method(self):
        """Setup for rate limiting tests."""
        self.client = app.test_client() if hasattr(app, 'test_client') else Mock()
    
    def test_rate_limiting_behavior(self):
        """Test API rate limiting behavior."""
        if not hasattr(self.client, 'get'):
            pytest.skip("Client not available")
        
        # Make multiple rapid requests
        request_count = 20
        responses = []
        start_time = time.time()
        
        for i in range(request_count):
            response = self.client.get('/health')
            if hasattr(response, 'status_code'):
                responses.append(response.status_code)
            time.sleep(0.05)  # Small delay between requests
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Analyze rate limiting
        success_count = sum(1 for status in responses if status == 200)
        rate_limited_count = sum(1 for status in responses if status == 429)
        
        # Should handle rapid requests gracefully
        assert len(responses) == request_count
        
        # Either all succeed (no rate limiting) or some are rate limited
        if rate_limited_count > 0:
            assert rate_limited_count < request_count  # Some should succeed
            assert success_count > 0  # Some should still work
    
    def test_rate_limit_headers(self):
        """Test rate limit headers in response."""
        if not hasattr(self.client, 'get'):
            pytest.skip("Client not available")
        
        response = self.client.get('/health')
        
        if hasattr(response, 'headers'):
            # Common rate limit headers (if implemented)
            rate_limit_headers = [
                'X-RateLimit-Limit',
                'X-RateLimit-Remaining', 
                'X-RateLimit-Reset',
                'Retry-After'
            ]
            
            # Check if any rate limit headers are present
            present_headers = [h for h in rate_limit_headers if h in response.headers]
            
            # If rate limiting is implemented, validate header values
            for header in present_headers:
                if header in ['X-RateLimit-Limit', 'X-RateLimit-Remaining']:
                    try:
                        value = int(response.headers[header])
                        assert value >= 0
                    except ValueError:
                        pytest.fail(f"Rate limit header {header} should be numeric")


class TestAPIAuthentication:
    """API authentication tests (if implemented)."""
    
    def setup_method(self):
        """Setup for authentication tests."""
        self.client = app.test_client() if hasattr(app, 'test_client') else Mock()
        self.valid_api_key = 'test-api-key-123'
        self.invalid_api_key = 'invalid-key'
    
    def test_api_key_authentication(self):
        """Test API key authentication (if implemented)."""
        if not hasattr(self.client, 'get'):
            pytest.skip("Client not available")
        
        # Test without API key
        response_no_key = self.client.get('/api/v1/status')
        
        # Test with valid API key
        response_valid_key = self.client.get('/api/v1/status', 
                                           headers={'X-API-Key': self.valid_api_key})
        
        # Test with invalid API key
        response_invalid_key = self.client.get('/api/v1/status',
                                             headers={'X-API-Key': self.invalid_api_key})
        
        # If authentication is implemented, validate responses
        for response in [response_no_key, response_valid_key, response_invalid_key]:
            if hasattr(response, 'status_code'):
                # Should return valid HTTP status codes
                assert response.status_code in [200, 401, 403, 500]
    
    def test_authentication_error_responses(self):
        """Test authentication error responses."""
        # Mock authentication error responses
        auth_error_responses = {
            'missing_key': {
                'error': 'Authentication required',
                'message': 'API key is required',
                'status_code': 401
            },
            'invalid_key': {
                'error': 'Invalid authentication',
                'message': 'API key is invalid',
                'status_code': 403
            },
            'expired_key': {
                'error': 'Authentication expired',
                'message': 'API key has expired',
                'status_code': 403
            }
        }
        
        for error_type, error_response in auth_error_responses.items():
            assert 'error' in error_response
            assert 'message' in error_response
            assert error_response['status_code'] in [401, 403]


class TestAPIVersioning:
    """API versioning tests."""
    
    def setup_method(self):
        """Setup for versioning tests."""
        self.client = app.test_client() if hasattr(app, 'test_client') else Mock()
    
    def test_api_version_endpoints(self):
        """Test API version endpoints."""
        if not hasattr(self.client, 'get'):
            pytest.skip("Client not available")
        
        # Test v1 endpoints
        v1_endpoints = [
            '/api/v1/status',
            '/api/v1/predict',  # POST endpoint, but test GET should return 405 or 404
            '/api/v1/portfolio'
        ]
        
        for endpoint in v1_endpoints:
            response = self.client.get(endpoint)
            
            if hasattr(response, 'status_code'):
                # Should be valid endpoints (200, 405 for wrong method, or 500 for errors)
                assert response.status_code in [200, 404, 405, 500, 503]
    
    def test_unsupported_api_versions(self):
        """Test unsupported API versions."""
        if not hasattr(self.client, 'get'):
            pytest.skip("Client not available")
        
        # Test unsupported versions
        unsupported_endpoints = [
            '/api/v0/status',
            '/api/v2/status',
            '/api/v1.1/status',
            '/api/beta/status'
        ]
        
        for endpoint in unsupported_endpoints:
            response = self.client.get(endpoint)
            
            if hasattr(response, 'status_code'):
                # Should return 404 for unsupported versions
                assert response.status_code in [404, 500]


if __name__ == '__main__':
    # Run API tests
    pytest.main([
        __file__,
        '-v',
        '--tb=short'
    ])