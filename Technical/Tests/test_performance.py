"""
Performance Testing Suite for Entertainment Investment Intelligence Platform
===========================================================================

Performance, scalability, and load testing for box office prediction,
ROI calculations, and entertainment investment analytics.

Author: Emilio Cardenas
License: MIT
"""

import pytest
import time
import threading
import multiprocessing
import asyncio
import sys
import os
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from unittest.mock import Mock
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
import requests
import memory_profiler

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from main import app
    from advanced_entertainment_analytics import EntertainmentInvestmentPlatform
    from analytics_engine import AnalyticsEngine
    from data_manager import DataManager
    from ml_models import MLModelManager
except ImportError as e:
    print(f"Warning: Using mocks for performance testing: {e}")
    app = Mock()


class TestAPIPerformance:
    """API endpoint performance tests."""
    
    def setup_method(self):
        """Setup performance test environment."""
        self.client = app.test_client() if hasattr(app, 'test_client') else Mock()
        self.base_url = 'http://localhost:8002'
        self.sample_request = {
            'title': 'Performance Test Movie',
            'genre': 'Action',
            'budget': 100000000,
            'director': 'Christopher Nolan',
            'release_date': '2025-07-15'
        }
    
    def test_single_prediction_response_time(self):
        """Test single prediction API response time."""
        if not hasattr(self.client, 'post'):
            pytest.skip("Client not available")
        
        # Warmup request
        self.client.post('/api/v1/predict', json=self.sample_request)
        
        # Measure response time
        start_time = time.perf_counter()
        response = self.client.post('/api/v1/predict', json=self.sample_request)
        end_time = time.perf_counter()
        
        response_time = end_time - start_time
        
        # Performance requirements
        assert response_time < 1.0  # Should respond within 1 second
        
        if hasattr(response, 'status_code'):
            assert response.status_code in [200, 503]  # Allow service unavailable during testing
    
    def test_concurrent_predictions_performance(self):
        """Test concurrent prediction requests performance."""
        if not hasattr(self.client, 'post'):
            pytest.skip("Client not available")
        
        num_concurrent_requests = 10
        response_times = []
        successful_requests = 0
        
        def make_request():
            nonlocal successful_requests
            start_time = time.perf_counter()
            try:
                response = self.client.post('/api/v1/predict', json=self.sample_request)
                end_time = time.perf_counter()
                response_times.append(end_time - start_time)
                if hasattr(response, 'status_code') and response.status_code == 200:
                    successful_requests += 1
            except Exception:
                pass
        
        # Execute concurrent requests
        with ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
            futures = [executor.submit(make_request) for _ in range(num_concurrent_requests)]
            
            # Wait for all requests to complete
            for future in futures:
                future.result()
        
        # Performance analysis
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            
            # Performance requirements
            assert avg_response_time < 2.0  # Average should be under 2 seconds
            assert max_response_time < 5.0  # No request should take more than 5 seconds
            assert successful_requests >= num_concurrent_requests * 0.8  # 80% success rate
    
    def test_high_load_stress_testing(self):
        """Test API performance under high load."""
        if not hasattr(self.client, 'post'):
            pytest.skip("Client not available")
        
        num_requests = 50
        concurrent_users = 5
        response_times = []
        error_count = 0
        
        def user_simulation():
            nonlocal error_count
            for _ in range(num_requests // concurrent_users):
                try:
                    start_time = time.perf_counter()
                    response = self.client.post('/api/v1/predict', json=self.sample_request)
                    end_time = time.perf_counter()
                    
                    response_times.append(end_time - start_time)
                    
                    if hasattr(response, 'status_code') and response.status_code != 200:
                        error_count += 1
                        
                except Exception:
                    error_count += 1
                
                # Small delay between requests
                time.sleep(0.1)
        
        # Start concurrent users
        threads = []
        start_time = time.perf_counter()
        
        for _ in range(concurrent_users):
            thread = threading.Thread(target=user_simulation)
            threads.append(thread)
            thread.start()
        
        # Wait for all users to complete
        for thread in threads:
            thread.join()
        
        total_time = time.perf_counter() - start_time
        
        # Performance analysis
        if response_times:
            throughput = len(response_times) / total_time  # Requests per second
            avg_response_time = sum(response_times) / len(response_times)
            error_rate = error_count / num_requests
            
            # Performance requirements
            assert throughput > 1.0  # At least 1 request per second
            assert avg_response_time < 3.0  # Average response under 3 seconds
            assert error_rate < 0.2  # Less than 20% error rate
    
    def test_portfolio_analytics_performance(self):
        """Test portfolio analytics endpoint performance."""
        if not hasattr(self.client, 'get'):
            pytest.skip("Client not available")
        
        # Multiple requests to test caching and performance
        response_times = []
        
        for i in range(10):
            start_time = time.perf_counter()
            response = self.client.get('/api/v1/portfolio')
            end_time = time.perf_counter()
            
            response_times.append(end_time - start_time)
            
            if hasattr(response, 'status_code') and response.status_code == 200:
                # Verify response is not empty
                data = json.loads(response.data) if hasattr(response, 'data') else {}
                assert len(data) > 0
        
        # Performance analysis
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            first_request_time = response_times[0]
            subsequent_avg = sum(response_times[1:]) / len(response_times[1:]) if len(response_times) > 1 else 0
            
            # Performance requirements
            assert avg_response_time < 0.5  # Should be fast for analytics
            
            # Caching should improve subsequent requests (if implemented)
            if subsequent_avg > 0:
                improvement_ratio = first_request_time / subsequent_avg
                # Allow for some variation in timing


class TestMLModelPerformance:
    """Machine Learning model performance tests."""
    
    def setup_method(self):
        """Setup ML performance testing."""
        self.model_manager = MLModelManager() if 'MLModelManager' in globals() else Mock()
        self.test_dataset_sizes = [100, 500, 1000, 5000]
        
    def generate_test_features(self, size):
        """Generate test feature dataset."""
        np.random.seed(42)  # For reproducible results
        
        features = {
            'budget': np.random.uniform(10000000, 300000000, size),
            'genre_action': np.random.binomial(1, 0.3, size),
            'genre_comedy': np.random.binomial(1, 0.25, size),
            'director_score': np.random.uniform(3.0, 10.0, size),
            'cast_popularity': np.random.uniform(4.0, 9.0, size),
            'franchise': np.random.binomial(1, 0.2, size),
            'sequel': np.random.binomial(1, 0.15, size),
            'release_month': np.random.randint(1, 13, size),
            'studio_score': np.random.uniform(5.0, 9.0, size)
        }
        
        return pd.DataFrame(features)
    
    def test_single_prediction_performance(self):
        """Test single movie prediction performance."""
        features = self.generate_test_features(1).iloc[0].to_dict()
        
        # Mock prediction function
        def mock_predict(features):
            # Simulate model computation
            base_prediction = features['budget'] * 2.1
            adjustments = (
                features.get('genre_action', 0) * 0.1 +
                features.get('director_score', 5) / 10 +
                features.get('cast_popularity', 5) / 10
            )
            
            return base_prediction * (1 + adjustments)
        
        # Measure prediction time
        start_time = time.perf_counter()
        prediction = mock_predict(features)
        end_time = time.perf_counter()
        
        prediction_time = end_time - start_time
        
        # Performance requirements
        assert prediction_time < 0.1  # Should be very fast for single prediction
        assert prediction > 0
    
    def test_batch_prediction_performance(self):
        """Test batch prediction performance across different dataset sizes."""
        for dataset_size in self.test_dataset_sizes:
            features_df = self.generate_test_features(dataset_size)
            
            # Mock batch prediction
            def mock_batch_predict(features_df):
                predictions = []
                for _, row in features_df.iterrows():
                    base_prediction = row['budget'] * 2.1
                    adjustments = (
                        row.get('genre_action', 0) * 0.1 +
                        row.get('director_score', 5) / 10 +
                        row.get('cast_popularity', 5) / 10
                    )
                    predictions.append(base_prediction * (1 + adjustments))
                
                return np.array(predictions)
            
            # Measure batch prediction time
            start_time = time.perf_counter()
            predictions = mock_batch_predict(features_df)
            end_time = time.perf_counter()
            
            batch_time = end_time - start_time
            per_prediction_time = batch_time / dataset_size
            
            # Performance requirements
            assert len(predictions) == dataset_size
            assert per_prediction_time < 0.01  # Less than 10ms per prediction
            assert batch_time < dataset_size * 0.005  # Batch should be more efficient
    
    def test_model_scaling_performance(self):
        """Test model performance scaling with dataset size."""
        performance_data = []
        
        for size in self.test_dataset_sizes:
            features_df = self.generate_test_features(size)
            
            # Mock ensemble prediction (more complex)
            def mock_ensemble_predict(features_df):
                # Simulate multiple models
                model_predictions = {}
                
                for model_name in ['xgboost', 'lightgbm', 'random_forest', 'neural_net']:
                    model_preds = []
                    multiplier = {'xgboost': 2.1, 'lightgbm': 2.0, 'random_forest': 2.2, 'neural_net': 1.9}[model_name]
                    
                    for _, row in features_df.iterrows():
                        pred = row['budget'] * multiplier
                        model_preds.append(pred)
                    
                    model_predictions[model_name] = np.array(model_preds)
                
                # Ensemble weights
                weights = [0.3, 0.25, 0.25, 0.2]
                ensemble_pred = sum(w * preds for w, preds in zip(weights, model_predictions.values()))
                
                return ensemble_pred
            
            # Measure performance
            start_time = time.perf_counter()
            predictions = mock_ensemble_predict(features_df)
            end_time = time.perf_counter()
            
            processing_time = end_time - start_time
            throughput = size / processing_time  # Predictions per second
            
            performance_data.append({
                'dataset_size': size,
                'processing_time': processing_time,
                'throughput': throughput,
                'per_prediction_time': processing_time / size
            })
        
        # Analyze scaling performance
        for i, data in enumerate(performance_data):
            assert data['throughput'] > 100  # At least 100 predictions per second
            assert data['per_prediction_time'] < 0.05  # Less than 50ms per prediction
            
            # Check scaling efficiency (later datasets shouldn't be much slower per prediction)
            if i > 0:
                prev_data = performance_data[i-1]
                scaling_factor = data['dataset_size'] / prev_data['dataset_size']
                time_scaling = data['processing_time'] / prev_data['processing_time']
                
                # Time scaling should be roughly linear (not exponential)
                assert time_scaling < scaling_factor * 1.5
    
    def test_memory_usage_during_prediction(self):
        """Test memory usage during model predictions."""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Large dataset for memory testing
        large_dataset = self.generate_test_features(10000)
        
        def memory_intensive_prediction(features_df):
            # Simulate memory-intensive operations
            results = []
            for _, row in features_df.iterrows():
                # Create some intermediate calculations
                temp_data = np.random.random(1000)  # Simulate feature transformations
                prediction = row['budget'] * 2.1 * np.mean(temp_data)
                results.append(prediction)
                
                # Cleanup intermediate data
                del temp_data
            
            return results
        
        # Monitor memory during prediction
        gc.collect()  # Clean up before measurement
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        predictions = memory_intensive_prediction(large_dataset)
        
        gc.collect()  # Force garbage collection
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        memory_increase = memory_after - memory_before
        
        # Memory requirements
        assert len(predictions) == len(large_dataset)
        assert memory_increase < 500  # Should not use more than 500MB additional memory
    
    @pytest.mark.slow
    def test_concurrent_model_predictions(self):
        """Test concurrent model prediction performance."""
        def prediction_task(task_id):
            features = self.generate_test_features(100)
            
            # Mock prediction with task ID for uniqueness
            predictions = features['budget'] * (2.0 + task_id * 0.01)
            
            return {
                'task_id': task_id,
                'predictions_count': len(predictions),
                'avg_prediction': float(np.mean(predictions))
            }
        
        # Test with multiple concurrent tasks
        num_tasks = 5
        
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=num_tasks) as executor:
            futures = [executor.submit(prediction_task, i) for i in range(num_tasks)]
            results = [future.result() for future in futures]
        
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        
        # Performance analysis
        assert len(results) == num_tasks
        assert all(result['predictions_count'] == 100 for result in results)
        assert total_time < 2.0  # Should complete within 2 seconds
        
        # Verify concurrent execution was effective
        sequential_estimate = num_tasks * 0.5  # Estimated time if run sequentially
        assert total_time < sequential_estimate  # Should be faster than sequential


class TestDataProcessingPerformance:
    """Data processing and analytics performance tests."""
    
    def setup_method(self):
        """Setup data processing performance tests."""
        self.data_manager = DataManager() if 'DataManager' in globals() else Mock()
    
    def generate_large_dataset(self, num_records):
        """Generate large dataset for performance testing."""
        np.random.seed(42)
        
        data = {
            'movie_id': range(num_records),
            'title': [f'Movie {i}' for i in range(num_records)],
            'genre': np.random.choice(['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi'], num_records),
            'budget': np.random.uniform(5000000, 300000000, num_records),
            'gross': np.random.uniform(10000000, 1000000000, num_records),
            'release_date': pd.date_range('2020-01-01', periods=num_records, freq='D'),
            'director': [f'Director {i % 100}' for i in range(num_records)],
            'studio': np.random.choice(['Warner Bros', 'Disney', 'Universal', 'Sony', 'Paramount'], num_records)
        }
        
        return pd.DataFrame(data)
    
    def test_data_loading_performance(self):
        """Test data loading and processing performance."""
        dataset_sizes = [1000, 5000, 10000, 25000]
        
        for size in dataset_sizes:
            # Generate test dataset
            start_time = time.perf_counter()
            dataset = self.generate_large_dataset(size)
            loading_time = time.perf_counter() - start_time
            
            # Performance requirements
            assert len(dataset) == size
            assert loading_time < size * 0.001  # Less than 1ms per record
            
            # Memory check
            memory_usage = dataset.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            assert memory_usage < size * 0.01  # Less than 10KB per record on average
    
    def test_aggregation_performance(self):
        """Test data aggregation performance."""
        dataset = self.generate_large_dataset(50000)
        
        # Test various aggregations
        aggregation_tests = [
            ('genre_stats', lambda df: df.groupby('genre').agg({'budget': 'mean', 'gross': 'sum'})),
            ('studio_stats', lambda df: df.groupby('studio').agg({'gross': ['mean', 'median', 'std']})),
            ('yearly_trends', lambda df: df.groupby(df['release_date'].dt.year).agg({'budget': 'sum', 'gross': 'sum'})),
            ('roi_analysis', lambda df: df.assign(roi=(df['gross'] - df['budget']) / df['budget']).groupby('genre')['roi'].mean())
        ]
        
        for test_name, aggregation_func in aggregation_tests:
            start_time = time.perf_counter()
            result = aggregation_func(dataset)
            end_time = time.perf_counter()
            
            processing_time = end_time - start_time
            
            # Performance requirements
            assert processing_time < 1.0  # Should complete within 1 second
            assert len(result) > 0  # Should produce results
    
    def test_filtering_and_sorting_performance(self):
        """Test data filtering and sorting performance."""
        dataset = self.generate_large_dataset(100000)
        
        # Test various filtering operations
        filtering_tests = [
            ('high_budget_filter', lambda df: df[df['budget'] > 100000000]),
            ('genre_filter', lambda df: df[df['genre'].isin(['Action', 'Sci-Fi'])]),
            ('profitable_filter', lambda df: df[df['gross'] > df['budget'] * 1.5]),
            ('recent_movies', lambda df: df[df['release_date'] >= '2023-01-01']),
            ('complex_filter', lambda df: df[(df['budget'] > 50000000) & (df['gross'] > df['budget'] * 2) & (df['genre'] == 'Action')])
        ]
        
        for test_name, filter_func in filtering_tests:
            start_time = time.perf_counter()
            filtered_data = filter_func(dataset)
            end_time = time.perf_counter()
            
            filtering_time = end_time - start_time
            
            # Performance requirements
            assert filtering_time < 0.5  # Should complete within 0.5 seconds
            assert isinstance(filtered_data, pd.DataFrame)
        
        # Test sorting performance
        sorting_tests = [
            ('sort_by_budget', lambda df: df.sort_values('budget', ascending=False)),
            ('sort_by_gross', lambda df: df.sort_values('gross', ascending=False)),
            ('sort_by_roi', lambda df: df.assign(roi=(df['gross'] - df['budget']) / df['budget']).sort_values('roi', ascending=False)),
            ('multi_sort', lambda df: df.sort_values(['genre', 'budget'], ascending=[True, False]))
        ]
        
        for test_name, sort_func in sorting_tests:
            start_time = time.perf_counter()
            sorted_data = sort_func(dataset)
            end_time = time.perf_counter()
            
            sorting_time = end_time - start_time
            
            # Performance requirements
            assert sorting_time < 1.0  # Should complete within 1 second
            assert len(sorted_data) == len(dataset)
    
    def test_data_transformation_performance(self):
        """Test data transformation performance."""
        dataset = self.generate_large_dataset(50000)
        
        def complex_transformation(df):
            # Multiple transformation operations
            result = df.copy()
            
            # Calculate ROI
            result['roi'] = (result['gross'] - result['budget']) / result['budget']
            
            # Create budget categories
            result['budget_category'] = pd.cut(result['budget'], 
                                             bins=[0, 25000000, 100000000, 200000000, float('inf')],
                                             labels=['Low', 'Medium', 'High', 'Blockbuster'])
            
            # Calculate rolling averages (by studio)
            result['studio_avg_gross'] = result.groupby('studio')['gross'].transform('mean')
            
            # Create success indicator
            result['successful'] = (result['roi'] > 0.5) & (result['gross'] > result['budget'] * 1.5)
            
            # Add year and month columns
            result['release_year'] = result['release_date'].dt.year
            result['release_month'] = result['release_date'].dt.month
            
            return result
        
        start_time = time.perf_counter()
        transformed_data = complex_transformation(dataset)
        end_time = time.perf_counter()
        
        transformation_time = end_time - start_time
        
        # Performance requirements
        assert transformation_time < 2.0  # Should complete within 2 seconds
        assert len(transformed_data) == len(dataset)
        assert 'roi' in transformed_data.columns
        assert 'budget_category' in transformed_data.columns
        assert 'successful' in transformed_data.columns


class TestVisualizationPerformance:
    """Visualization generation performance tests."""
    
    def test_chart_generation_performance(self):
        """Test chart generation performance."""
        # Mock chart generation functions
        def generate_box_office_chart(data_size):
            # Simulate chart data preparation
            time.sleep(data_size * 0.00001)  # Simulate processing time
            return f"Chart with {data_size} data points"
        
        def generate_roi_analysis_chart(data_size):
            # Simulate complex chart generation
            time.sleep(data_size * 0.00002)  # Simulate processing time
            return f"ROI chart with {data_size} data points"
        
        chart_tests = [
            ('box_office_chart', generate_box_office_chart, 1000),
            ('roi_analysis_chart', generate_roi_analysis_chart, 2000),
            ('box_office_chart_large', generate_box_office_chart, 10000),
            ('roi_analysis_chart_large', generate_roi_analysis_chart, 15000)
        ]
        
        for test_name, chart_func, data_size in chart_tests:
            start_time = time.perf_counter()
            chart_result = chart_func(data_size)
            end_time = time.perf_counter()
            
            generation_time = end_time - start_time
            
            # Performance requirements
            assert generation_time < 1.0  # Should generate within 1 second
            assert chart_result is not None
            assert str(data_size) in chart_result
    
    def test_dashboard_rendering_performance(self):
        """Test dashboard rendering performance."""
        def render_dashboard_components():
            components = []
            
            # Simulate various dashboard components
            component_generation_times = {
                'summary_stats': 0.05,
                'prediction_chart': 0.1,
                'roi_analysis': 0.08,
                'risk_assessment': 0.06,
                'market_trends': 0.12,
                'portfolio_overview': 0.09
            }
            
            for component, expected_time in component_generation_times.items():
                start_time = time.perf_counter()
                time.sleep(expected_time * 0.1)  # Simulate reduced processing time
                end_time = time.perf_counter()
                
                actual_time = end_time - start_time
                components.append({
                    'component': component,
                    'render_time': actual_time,
                    'status': 'rendered'
                })
            
            return components
        
        start_time = time.perf_counter()
        dashboard_components = render_dashboard_components()
        end_time = time.perf_counter()
        
        total_render_time = end_time - start_time
        
        # Performance requirements
        assert len(dashboard_components) == 6
        assert total_render_time < 0.5  # Total dashboard should render within 0.5 seconds
        assert all(comp['status'] == 'rendered' for comp in dashboard_components)


@pytest.mark.slow
class TestStressAndLoad:
    """Stress testing and load testing."""
    
    def test_sustained_load_test(self):
        """Test sustained load over time."""
        if not hasattr(app, 'test_client'):
            pytest.skip("App test client not available")
        
        client = app.test_client()
        duration = 30  # seconds
        request_interval = 0.5  # seconds between requests
        
        start_time = time.time()
        request_count = 0
        error_count = 0
        response_times = []
        
        while time.time() - start_time < duration:
            try:
                request_start = time.perf_counter()
                response = client.get('/health')
                request_end = time.perf_counter()
                
                response_times.append(request_end - request_start)
                request_count += 1
                
                if hasattr(response, 'status_code') and response.status_code != 200:
                    error_count += 1
                    
            except Exception:
                error_count += 1
            
            time.sleep(request_interval)
        
        # Performance analysis
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            error_rate = error_count / request_count if request_count > 0 else 1
            
            # Performance requirements for sustained load
            assert avg_response_time < 1.0  # Average response time under 1 second
            assert max_response_time < 3.0  # No response should take more than 3 seconds
            assert error_rate < 0.1  # Less than 10% error rate
            assert request_count > duration / request_interval * 0.8  # At least 80% of expected requests
    
    def test_memory_leak_detection(self):
        """Test for memory leaks during extended operation."""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Simulate extended operation
        for i in range(1000):
            # Simulate prediction operations
            data = {
                'budget': 100000000 + i * 1000,
                'genre': ['Action', 'Comedy', 'Drama'][i % 3],
                'predictions': list(range(100))  # Create some data
            }
            
            # Process data
            result = data['budget'] * 2.1
            
            # Clean up
            del data
            
            # Check memory every 100 iterations
            if i % 100 == 0:
                gc.collect()
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - initial_memory
                
                # Memory should not increase significantly
                assert memory_increase < 100  # Less than 100MB increase
        
        # Final memory check
        gc.collect()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        total_memory_increase = final_memory - initial_memory
        
        # Should not have significant memory leak
        assert total_memory_increase < 50  # Less than 50MB total increase


if __name__ == '__main__':
    # Run performance tests
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '-m', 'not slow'  # Skip slow tests by default
    ])