"""
Test Data Management Suite for Entertainment Investment Intelligence Platform
============================================================================

Test data generation, validation, and management utilities for comprehensive
testing of the entertainment investment intelligence platform.

Author: Emilio Cardenas
License: MIT
"""

import pytest
import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import random
import string
from unittest.mock import Mock

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from data_manager import DataManager
    from ml_models import MLModelManager
except ImportError as e:
    print(f"Warning: Using mocks for data testing: {e}")
    DataManager = Mock
    MLModelManager = Mock


class TestDataGeneration:
    """Test data generation and validation."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.data_generator = EntertainmentTestDataGenerator()
    
    def test_movie_data_generation(self):
        """Test movie data generation with various parameters."""
        # Test single movie generation
        movie = self.data_generator.generate_movie_data()
        
        # Validate required fields
        required_fields = ['title', 'genre', 'budget', 'director', 'studio', 'release_date']
        for field in required_fields:
            assert field in movie
            assert movie[field] is not None
        
        # Validate data types and ranges
        assert isinstance(movie['title'], str)
        assert len(movie['title']) > 0
        assert isinstance(movie['budget'], (int, float))
        assert movie['budget'] > 0
        assert movie['genre'] in self.data_generator.GENRES
        assert movie['rating'] in self.data_generator.RATINGS
    
    def test_batch_movie_data_generation(self):
        """Test batch movie data generation."""
        batch_size = 100
        movies = self.data_generator.generate_movie_batch(batch_size)
        
        assert len(movies) == batch_size
        assert all(isinstance(movie, dict) for movie in movies)
        
        # Check for unique titles
        titles = [movie['title'] for movie in movies]
        assert len(set(titles)) == batch_size  # All titles should be unique
        
        # Validate budget distribution
        budgets = [movie['budget'] for movie in movies]
        assert min(budgets) >= self.data_generator.MIN_BUDGET
        assert max(budgets) <= self.data_generator.MAX_BUDGET
    
    def test_realistic_movie_data_distribution(self):
        """Test realistic distribution of movie data."""
        movies = self.data_generator.generate_movie_batch(1000)
        df = pd.DataFrame(movies)
        
        # Test genre distribution (should be somewhat balanced)
        genre_counts = df['genre'].value_counts()
        for genre in self.data_generator.GENRES:
            assert genre_counts[genre] > 0  # All genres represented
            assert genre_counts[genre] < len(movies) * 0.8  # No single genre dominates
        
        # Test budget distribution
        budget_stats = df['budget'].describe()
        assert budget_stats['std'] > 0  # Should have variation
        assert budget_stats['min'] >= self.data_generator.MIN_BUDGET
        assert budget_stats['max'] <= self.data_generator.MAX_BUDGET
        
        # Test franchise/sequel distribution
        franchise_rate = df['franchise'].mean()
        sequel_rate = df['sequel'].mean()
        assert 0.1 <= franchise_rate <= 0.4  # Reasonable franchise rate
        assert 0.05 <= sequel_rate <= 0.3   # Reasonable sequel rate
    
    def test_temporal_data_generation(self):
        """Test temporal movie data generation."""
        # Generate movies with different time periods
        current_movies = self.data_generator.generate_movie_batch(50, 
                                                                 date_range=('2024-01-01', '2024-12-31'))
        future_movies = self.data_generator.generate_movie_batch(50,
                                                                date_range=('2025-01-01', '2025-12-31'))
        
        # Validate date ranges
        for movie in current_movies:
            release_date = datetime.strptime(movie['release_date'], '%Y-%m-%d')
            assert release_date.year == 2024
        
        for movie in future_movies:
            release_date = datetime.strptime(movie['release_date'], '%Y-%m-%d')
            assert release_date.year == 2025
    
    def test_prediction_data_generation(self):
        """Test prediction data generation."""
        movie = self.data_generator.generate_movie_data()
        prediction = self.data_generator.generate_prediction_data(movie)
        
        # Validate prediction structure
        assert 'predicted_gross' in prediction
        assert 'confidence_score' in prediction
        assert 'confidence_interval' in prediction
        assert 'roi_analysis' in prediction
        assert 'risk_assessment' in prediction
        
        # Validate prediction values
        assert prediction['predicted_gross'] > 0
        assert 0 <= prediction['confidence_score'] <= 1
        
        ci = prediction['confidence_interval']
        assert ci['lower'] < prediction['predicted_gross'] < ci['upper']
        
        # ROI should be reasonable for the budget
        roi = prediction['roi_analysis']['predicted_roi']
        assert -1 <= roi <= 10  # Reasonable ROI range
    
    def test_portfolio_data_generation(self):
        """Test investment portfolio data generation."""
        portfolio = self.data_generator.generate_portfolio_data(50)
        
        # Validate portfolio structure
        assert 'movies' in portfolio
        assert 'summary' in portfolio
        assert 'performance_metrics' in portfolio
        
        # Validate movies in portfolio
        assert len(portfolio['movies']) == 50
        for movie in portfolio['movies']:
            assert 'investment_amount' in movie
            assert 'expected_roi' in movie
            assert movie['investment_amount'] > 0
        
        # Validate summary metrics
        summary = portfolio['summary']
        assert 'total_investment' in summary
        assert 'expected_returns' in summary
        assert summary['total_investment'] > 0
    
    def test_market_data_generation(self):
        """Test market and trend data generation."""
        market_data = self.data_generator.generate_market_data(365)  # One year of data
        
        assert len(market_data) == 365
        
        for day_data in market_data:
            assert 'date' in day_data
            assert 'box_office_total' in day_data
            assert 'theater_count' in day_data
            assert 'average_ticket_price' in day_data
            
            # Validate ranges
            assert day_data['box_office_total'] >= 0
            assert day_data['theater_count'] > 0
            assert day_data['average_ticket_price'] > 0
    
    def test_data_quality_validation(self):
        """Test data quality validation functions."""
        # Generate test data
        movies = self.data_generator.generate_movie_batch(100)
        
        # Test validation functions
        validator = DataQualityValidator()
        
        for movie in movies[:10]:  # Test first 10 movies
            validation_result = validator.validate_movie_data(movie)
            
            assert validation_result['is_valid'] == True
            assert len(validation_result['errors']) == 0
            assert validation_result['data_quality_score'] > 0.8
    
    def test_data_corruption_detection(self):
        """Test detection of corrupted or invalid data."""
        validator = DataQualityValidator()
        
        # Test with corrupted data
        corrupted_movies = [
            {'title': '', 'genre': 'Action', 'budget': 100000000},  # Empty title
            {'title': 'Test', 'genre': 'InvalidGenre', 'budget': 100000000},  # Invalid genre
            {'title': 'Test', 'genre': 'Action', 'budget': -1000000},  # Negative budget
            {'title': 'Test', 'genre': 'Action', 'budget': '100M'},  # Wrong type
            {'title': 'Test', 'genre': 'Action'},  # Missing budget
        ]
        
        for corrupted_movie in corrupted_movies:
            validation_result = validator.validate_movie_data(corrupted_movie)
            
            assert validation_result['is_valid'] == False
            assert len(validation_result['errors']) > 0
            assert validation_result['data_quality_score'] < 0.8


class EntertainmentTestDataGenerator:
    """Test data generator for entertainment investment platform."""
    
    GENRES = ['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi', 'Romance', 'Thriller', 'Animation', 'Documentary']
    RATINGS = ['G', 'PG', 'PG-13', 'R', 'NC-17']
    
    DIRECTORS = [
        'Christopher Nolan', 'Quentin Tarantino', 'Steven Spielberg', 'Martin Scorsese',
        'Denis Villeneuve', 'Jordan Peele', 'Greta Gerwig', 'Chloe Zhao',
        'Ryan Coogler', 'Rian Johnson', 'Taika Waititi', 'Patty Jenkins'
    ]
    
    STUDIOS = [
        'Warner Bros', 'Disney', 'Universal', 'Sony Pictures', 'Paramount',
        'Fox', 'Lionsgate', 'A24', 'Netflix', 'Amazon Studios'
    ]
    
    ACTORS = [
        'Leonardo DiCaprio', 'Scarlett Johansson', 'Robert Downey Jr.', 'Emma Stone',
        'Ryan Gosling', 'Margot Robbie', 'Tom Hanks', 'Meryl Streep',
        'Denzel Washington', 'Jennifer Lawrence', 'Brad Pitt', 'Charlize Theron',
        'Will Smith', 'Sandra Bullock', 'Matt Damon', 'Amy Adams'
    ]
    
    MIN_BUDGET = 1000000      # $1M
    MAX_BUDGET = 400000000    # $400M
    MIN_RUNTIME = 80          # 80 minutes
    MAX_RUNTIME = 200         # 200 minutes
    
    def __init__(self, seed=None):
        """Initialize with optional random seed."""
        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_movie_data(self, **kwargs) -> Dict[str, Any]:
        """Generate single movie data."""
        # Basic movie information
        movie = {
            'title': self._generate_movie_title(),
            'genre': random.choice(self.GENRES),
            'budget': self._generate_budget(),
            'director': random.choice(self.DIRECTORS),
            'studio': random.choice(self.STUDIOS),
            'rating': random.choice(self.RATINGS),
            'runtime': random.randint(self.MIN_RUNTIME, self.MAX_RUNTIME),
            'release_date': self._generate_release_date(**kwargs),
            'franchise': random.random() < 0.25,  # 25% franchise
            'sequel': random.random() < 0.15,     # 15% sequel
        }
        
        # Generate cast
        cast_size = random.randint(3, 6)
        movie['cast'] = random.sample(self.ACTORS, cast_size)
        
        # Additional metadata
        movie['description'] = self._generate_description(movie)
        movie['genre_tags'] = self._generate_genre_tags(movie['genre'])
        
        # Apply any overrides
        movie.update(kwargs)
        
        return movie
    
    def generate_movie_batch(self, count: int, **kwargs) -> List[Dict[str, Any]]:
        """Generate batch of movie data."""
        return [self.generate_movie_data(**kwargs) for _ in range(count)]
    
    def generate_prediction_data(self, movie: Dict[str, Any]) -> Dict[str, Any]:
        """Generate prediction data for a movie."""
        budget = movie['budget']
        genre = movie['genre']
        
        # Base prediction factors
        genre_multipliers = {
            'Action': 2.2, 'Sci-Fi': 2.1, 'Adventure': 2.0,
            'Comedy': 1.8, 'Drama': 1.5, 'Horror': 2.5,
            'Romance': 1.4, 'Thriller': 1.7, 'Animation': 2.3,
            'Documentary': 0.8
        }
        
        base_multiplier = genre_multipliers.get(genre, 1.8)
        
        # Add randomness and other factors
        director_factor = 1.0 + (random.random() * 0.4 - 0.2)  # ±20%
        franchise_factor = 1.2 if movie.get('franchise') else 1.0
        sequel_factor = 1.1 if movie.get('sequel') else 1.0
        
        predicted_gross = budget * base_multiplier * director_factor * franchise_factor * sequel_factor
        
        # Add some randomness
        predicted_gross *= (0.8 + random.random() * 0.4)  # ±20% variation
        
        # Calculate confidence and intervals
        confidence = 0.7 + random.random() * 0.25  # 70-95% confidence
        margin_of_error = predicted_gross * (0.15 + random.random() * 0.10)  # 15-25% margin
        
        # ROI analysis
        roi = (predicted_gross - budget) / budget
        
        return {
            'predicted_gross': int(predicted_gross),
            'confidence_score': round(confidence, 3),
            'confidence_interval': {
                'lower': int(predicted_gross - margin_of_error),
                'upper': int(predicted_gross + margin_of_error),
                'confidence_level': 0.95
            },
            'roi_analysis': {
                'predicted_roi': round(roi, 3),
                'investment_grade': self._get_investment_grade(roi),
                'payback_period_months': max(6, int(24 / (roi + 1))),
                'profit_margin': round(roi / (1 + roi), 3)
            },
            'risk_assessment': {
                'overall_risk_score': round(random.uniform(2.0, 8.0), 1),
                'risk_category': random.choice(['Low Risk', 'Moderate Risk', 'High Risk']),
                'risk_factors': {
                    'budget_risk': round(random.uniform(1.0, 9.0), 1),
                    'genre_risk': round(random.uniform(1.0, 9.0), 1),
                    'timing_risk': round(random.uniform(1.0, 9.0), 1),
                    'competition_risk': round(random.uniform(1.0, 9.0), 1)
                }
            },
            'model_version': f"v{random.randint(1,3)}.{random.randint(0,9)}.{random.randint(0,9)}",
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_portfolio_data(self, movie_count: int) -> Dict[str, Any]:
        """Generate portfolio data with multiple movies."""
        movies = []
        total_investment = 0
        total_expected_returns = 0
        
        for _ in range(movie_count):
            movie = self.generate_movie_data()
            prediction = self.generate_prediction_data(movie)
            
            investment_amount = movie['budget']  # Assume full budget as investment
            expected_return = prediction['predicted_gross']
            
            portfolio_movie = {
                **movie,
                'investment_amount': investment_amount,
                'expected_return': expected_return,
                'expected_roi': prediction['roi_analysis']['predicted_roi'],
                'risk_score': prediction['risk_assessment']['overall_risk_score'],
                'investment_status': random.choice(['active', 'completed', 'planned'])
            }
            
            movies.append(portfolio_movie)
            total_investment += investment_amount
            total_expected_returns += expected_return
        
        portfolio_roi = (total_expected_returns - total_investment) / total_investment
        
        return {
            'movies': movies,
            'summary': {
                'total_investment': total_investment,
                'expected_returns': total_expected_returns,
                'portfolio_roi': round(portfolio_roi, 3),
                'movie_count': movie_count,
                'avg_investment': total_investment // movie_count,
                'genre_distribution': self._calculate_genre_distribution(movies)
            },
            'performance_metrics': {
                'sharpe_ratio': round(random.uniform(1.5, 3.5), 2),
                'sortino_ratio': round(random.uniform(2.0, 4.0), 2),
                'max_drawdown': round(random.uniform(-0.3, -0.1), 3),
                'win_rate': round(random.uniform(0.6, 0.85), 3),
                'volatility': round(random.uniform(0.15, 0.35), 3)
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_market_data(self, days: int) -> List[Dict[str, Any]]:
        """Generate market trend data for specified number of days."""
        market_data = []
        base_date = datetime.now() - timedelta(days=days)
        
        # Base values
        base_box_office = 50000000  # $50M daily base
        base_theater_count = 40000
        base_ticket_price = 12.50
        
        for i in range(days):
            current_date = base_date + timedelta(days=i)
            
            # Add seasonal variations
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * i / 365)  # Yearly cycle
            weekend_factor = 1.5 if current_date.weekday() >= 5 else 1.0  # Weekend boost
            
            # Add random variation
            daily_variation = 0.8 + random.random() * 0.4
            
            total_factor = seasonal_factor * weekend_factor * daily_variation
            
            day_data = {
                'date': current_date.strftime('%Y-%m-%d'),
                'box_office_total': int(base_box_office * total_factor),
                'theater_count': int(base_theater_count * (0.95 + random.random() * 0.1)),
                'average_ticket_price': round(base_ticket_price * (0.9 + random.random() * 0.2), 2),
                'seasonal_factor': round(seasonal_factor, 3),
                'day_of_week': current_date.strftime('%A'),
                'is_weekend': current_date.weekday() >= 5
            }
            
            market_data.append(day_data)
        
        return market_data
    
    def _generate_movie_title(self) -> str:
        """Generate realistic movie title."""
        title_templates = [
            "{adjective} {noun}",
            "The {adjective} {noun}",
            "{noun} {verb_suffix}",
            "{adjective} {noun}: {subtitle}",
            "The {noun} of {noun}",
            "{character_name}",
            "{location} {noun}"
        ]
        
        adjectives = ['Dark', 'Lost', 'Hidden', 'Final', 'Secret', 'Eternal', 'Golden', 'Silent', 'Deadly', 'Wild']
        nouns = ['Kingdom', 'Warrior', 'Legend', 'Quest', 'Journey', 'Mystery', 'Empire', 'Revolution', 'Adventure', 'Prophecy']
        verb_suffixes = ['Returns', 'Rises', 'Awakens', 'Begins', 'Strikes', 'Reborn', 'Unleashed', 'Rising', 'Forever']
        subtitles = ['The Beginning', 'Revenge', 'Redemption', 'The Final Chapter', 'Origins', 'Revolution']
        character_names = ['Phoenix', 'Blade Runner', 'Wonder', 'Storm', 'Infinity', 'Genesis', 'Eclipse', 'Vortex']
        locations = ['Pacific', 'Atlantic', 'Arctic', 'Desert', 'Mountain', 'Cosmic', 'Underground', 'Skyline']
        
        template = random.choice(title_templates)
        
        title = template.format(
            adjective=random.choice(adjectives),
            noun=random.choice(nouns),
            verb_suffix=random.choice(verb_suffixes),
            subtitle=random.choice(subtitles),
            character_name=random.choice(character_names),
            location=random.choice(locations)
        )
        
        return title
    
    def _generate_budget(self) -> int:
        """Generate realistic movie budget."""
        # Log-normal distribution for realistic budget spread
        log_mean = np.log(75000000)  # $75M median
        log_std = 0.8
        
        budget = int(np.random.lognormal(log_mean, log_std))
        
        # Clamp to reasonable range
        return max(self.MIN_BUDGET, min(self.MAX_BUDGET, budget))
    
    def _generate_release_date(self, date_range=None, **kwargs) -> str:
        """Generate release date."""
        if date_range:
            start_date, end_date = date_range
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            
            random_days = random.randint(0, (end - start).days)
            release_date = start + timedelta(days=random_days)
        else:
            # Default to future dates
            base_date = datetime.now()
            future_days = random.randint(30, 730)  # 1 month to 2 years
            release_date = base_date + timedelta(days=future_days)
        
        return release_date.strftime('%Y-%m-%d')
    
    def _generate_description(self, movie: Dict[str, Any]) -> str:
        """Generate movie description."""
        templates = [
            "A thrilling {genre} about {plot_element}.",
            "An epic {genre} featuring {plot_element}.",
            "{director} directs this {genre} about {plot_element}.",
            "A {rating}-rated {genre} that explores {plot_element}."
        ]
        
        plot_elements = [
            "a hero's journey", "forbidden love", "family secrets", "time travel",
            "alien invasion", "corporate conspiracy", "supernatural forces",
            "redemption story", "coming of age", "survival against odds"
        ]
        
        template = random.choice(templates)
        return template.format(
            genre=movie['genre'].lower(),
            director=movie['director'],
            rating=movie['rating'],
            plot_element=random.choice(plot_elements)
        )
    
    def _generate_genre_tags(self, primary_genre: str) -> List[str]:
        """Generate genre tags based on primary genre."""
        genre_tag_map = {
            'Action': ['action', 'adventure', 'thriller'],
            'Comedy': ['comedy', 'humor', 'funny'],
            'Drama': ['drama', 'emotional', 'character-driven'],
            'Horror': ['horror', 'scary', 'supernatural'],
            'Sci-Fi': ['sci-fi', 'futuristic', 'technology'],
            'Romance': ['romance', 'love', 'relationship'],
            'Thriller': ['thriller', 'suspense', 'mystery'],
            'Animation': ['animation', 'family', 'animated'],
            'Documentary': ['documentary', 'real-life', 'educational']
        }
        
        base_tags = genre_tag_map.get(primary_genre, [primary_genre.lower()])
        additional_tags = ['blockbuster', 'award-worthy', 'must-see', 'critically-acclaimed']
        
        # Select 2-4 tags
        selected_tags = base_tags + random.sample(additional_tags, random.randint(0, 2))
        return selected_tags[:4]  # Max 4 tags
    
    def _get_investment_grade(self, roi: float) -> str:
        """Get investment grade based on ROI."""
        if roi >= 2.0:
            return 'A+ (Exceptional)'
        elif roi >= 1.5:
            return 'A (Excellent)'
        elif roi >= 1.0:
            return 'B (Good)'
        elif roi >= 0.5:
            return 'C (Fair)'
        elif roi >= 0.0:
            return 'D (Poor)'
        else:
            return 'F (Loss)'
    
    def _calculate_genre_distribution(self, movies: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate genre distribution in portfolio."""
        genre_counts = {}
        total_movies = len(movies)
        
        for movie in movies:
            genre = movie['genre']
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        return {genre: count / total_movies for genre, count in genre_counts.items()}


class DataQualityValidator:
    """Validate data quality for entertainment investment platform."""
    
    def validate_movie_data(self, movie: Dict[str, Any]) -> Dict[str, Any]:
        """Validate movie data quality."""
        errors = []
        warnings = []
        quality_score = 1.0
        
        # Required fields check
        required_fields = ['title', 'genre', 'budget', 'director', 'studio', 'release_date']
        for field in required_fields:
            if field not in movie or movie[field] is None:
                errors.append(f"Missing required field: {field}")
                quality_score -= 0.2
        
        # Data type validation
        if 'title' in movie:
            if not isinstance(movie['title'], str) or len(movie['title'].strip()) == 0:
                errors.append("Title must be a non-empty string")
                quality_score -= 0.1
        
        if 'budget' in movie:
            if not isinstance(movie['budget'], (int, float)) or movie['budget'] <= 0:
                errors.append("Budget must be a positive number")
                quality_score -= 0.15
        
        if 'genre' in movie:
            valid_genres = EntertainmentTestDataGenerator.GENRES
            if movie['genre'] not in valid_genres:
                errors.append(f"Invalid genre: {movie['genre']}. Valid genres: {valid_genres}")
                quality_score -= 0.1
        
        if 'rating' in movie:
            valid_ratings = EntertainmentTestDataGenerator.RATINGS
            if movie['rating'] not in valid_ratings:
                warnings.append(f"Unusual rating: {movie['rating']}")
                quality_score -= 0.05
        
        # Business logic validation
        if 'budget' in movie and movie['budget'] > 500000000:
            warnings.append("Unusually high budget (>$500M)")
            quality_score -= 0.05
        
        if 'runtime' in movie:
            if not isinstance(movie['runtime'], int) or movie['runtime'] < 60 or movie['runtime'] > 300:
                warnings.append("Unusual runtime (should be 60-300 minutes)")
                quality_score -= 0.05
        
        # Date validation
        if 'release_date' in movie:
            try:
                release_date = datetime.strptime(movie['release_date'], '%Y-%m-%d')
                if release_date < datetime(1900, 1, 1):
                    warnings.append("Very old release date")
                    quality_score -= 0.02
            except ValueError:
                errors.append("Invalid date format for release_date (should be YYYY-MM-DD)")
                quality_score -= 0.1
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'data_quality_score': max(0.0, quality_score),
            'validation_timestamp': datetime.now().isoformat()
        }
    
    def validate_prediction_data(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Validate prediction data quality."""
        errors = []
        warnings = []
        quality_score = 1.0
        
        # Required fields
        required_fields = ['predicted_gross', 'confidence_score', 'roi_analysis']
        for field in required_fields:
            if field not in prediction:
                errors.append(f"Missing required field: {field}")
                quality_score -= 0.2
        
        # Value range validation
        if 'predicted_gross' in prediction:
            if prediction['predicted_gross'] <= 0:
                errors.append("Predicted gross must be positive")
                quality_score -= 0.2
        
        if 'confidence_score' in prediction:
            score = prediction['confidence_score']
            if not (0 <= score <= 1):
                errors.append("Confidence score must be between 0 and 1")
                quality_score -= 0.15
            elif score < 0.5:
                warnings.append("Low confidence score")
                quality_score -= 0.05
        
        # ROI validation
        if 'roi_analysis' in prediction:
            roi_data = prediction['roi_analysis']
            if 'predicted_roi' in roi_data:
                roi = roi_data['predicted_roi']
                if roi < -1:
                    warnings.append("Extremely negative ROI")
                    quality_score -= 0.1
                elif roi > 10:
                    warnings.append("Extremely high ROI (>1000%)")
                    quality_score -= 0.05
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'data_quality_score': max(0.0, quality_score),
            'validation_timestamp': datetime.now().isoformat()
        }


# Test fixtures for common test data scenarios
@pytest.fixture
def sample_movies_dataset():
    """Generate sample movies dataset for testing."""
    generator = EntertainmentTestDataGenerator(seed=42)
    return generator.generate_movie_batch(100)


@pytest.fixture
def sample_predictions_dataset():
    """Generate sample predictions dataset for testing."""
    generator = EntertainmentTestDataGenerator(seed=42)
    movies = generator.generate_movie_batch(50)
    predictions = []
    
    for movie in movies:
        prediction = generator.generate_prediction_data(movie)
        prediction['movie'] = movie
        predictions.append(prediction)
    
    return predictions


@pytest.fixture
def sample_portfolio():
    """Generate sample portfolio for testing."""
    generator = EntertainmentTestDataGenerator(seed=42)
    return generator.generate_portfolio_data(25)


@pytest.fixture
def data_validator():
    """Data quality validator instance."""
    return DataQualityValidator()


if __name__ == '__main__':
    # Run data management tests
    pytest.main([
        __file__,
        '-v',
        '--tb=short'
    ])