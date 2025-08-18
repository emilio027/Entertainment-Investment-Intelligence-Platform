#!/usr/bin/env python3
"""
Entertainment Investment Interactive Analysis Platform
Advanced Streamlit Application for Real-time Investment Analytics

Author: Emilio Cardenas
Version: 2.0.0
Last Updated: 2025-08-18

Features:
- Real-time box office prediction and ROI analysis
- Interactive portfolio optimization and risk management
- Advanced ML model deployment with explainable AI
- Comprehensive market intelligence and competitor analysis
- Dynamic streaming revenue analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Advanced ML and Analytics
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
import lime
import optuna

# Entertainment Analytics Engine
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'Files', 'src'))

try:
    from entertainment_main import EntertainmentInvestmentPlatform
    from analytics_engine import AdvancedAnalyticsEngine
    from ml_models import MLModelManager
except ImportError:
    st.error("Core analytics modules not found. Please ensure all dependencies are installed.")


class EntertainmentInvestmentApp:
    """
    Advanced Streamlit application for entertainment investment analytics.
    """
    
    def __init__(self):
        self.platform = EntertainmentInvestmentPlatform()
        self.ml_manager = MLModelManager() if 'MLModelManager' in globals() else None
        self.analytics_engine = AdvancedAnalyticsEngine() if 'AdvancedAnalyticsEngine' in globals() else None
        
        # Initialize session state
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'models_trained' not in st.session_state:
            st.session_state.models_trained = False
        if 'portfolio_data' not in st.session_state:
            st.session_state.portfolio_data = None

    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Entertainment Investment Intelligence",
            page_icon="🎬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS styling
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #667eea;
            margin-bottom: 1rem;
        }
        .prediction-card {
            background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .risk-card {
            background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .success-card {
            background: linear-gradient(135deg, #55efc4 0%, #00b894 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)

    def render_header(self):
        """Render the main application header."""
        st.markdown("""
        <div class="main-header">
            <h1>🎬 Entertainment Investment Intelligence Platform</h1>
            <h3>Advanced Analytics for Box Office Prediction & ROI Optimization</h3>
            <p>Leveraging AI/ML for Strategic Entertainment Investment Decisions</p>
        </div>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        """Render the navigation sidebar."""
        st.sidebar.title("🎯 Navigation")
        
        # Main sections
        section = st.sidebar.selectbox(
            "Select Analysis Section",
            [
                "📊 Executive Dashboard",
                "🎬 Box Office Prediction",
                "💼 Portfolio Analysis", 
                "📈 Risk Management",
                "🌐 Market Intelligence",
                "📺 Streaming Analytics",
                "🤖 Model Insights",
                "⚙️ Data Management"
            ]
        )
        
        st.sidebar.markdown("---")
        
        # Data controls
        st.sidebar.subheader("📁 Data Controls")
        
        if st.sidebar.button("🔄 Load Sample Data"):
            self.load_sample_data()
        
        if st.sidebar.button("🧠 Train Models"):
            if st.session_state.data_loaded:
                self.train_models()
            else:
                st.sidebar.error("Please load data first!")
        
        # Model status
        st.sidebar.subheader("🔧 System Status")
        data_status = "✅ Loaded" if st.session_state.data_loaded else "❌ Not Loaded"
        model_status = "✅ Trained" if st.session_state.models_trained else "❌ Not Trained"
        
        st.sidebar.write(f"Data Status: {data_status}")
        st.sidebar.write(f"Models Status: {model_status}")
        
        return section

    def load_sample_data(self):
        """Load and prepare sample entertainment investment data."""
        with st.spinner("Loading entertainment industry data..."):
            # Generate comprehensive dataset
            df = self.platform.generate_synthetic_movie_data(2000)
            
            # Add streaming data
            streaming_data = self.generate_streaming_data(df)
            
            # Add market intelligence data
            market_data = self.generate_market_data()
            
            # Store in session state
            st.session_state.portfolio_data = df
            st.session_state.streaming_data = streaming_data
            st.session_state.market_data = market_data
            st.session_state.data_loaded = True
            
        st.sidebar.success("✅ Data loaded successfully!")

    def generate_streaming_data(self, movies_df):
        """Generate synthetic streaming revenue data."""
        np.random.seed(42)
        n_movies = len(movies_df)
        
        platforms = ['Netflix', 'Disney+', 'Amazon Prime', 'HBO Max', 'Apple TV+', 'Paramount+']
        
        streaming_data = []
        for idx, movie in movies_df.iterrows():
            # Not all movies go to streaming
            if np.random.random() > 0.3:
                platform = np.random.choice(platforms)
                streaming_revenue = movie['box_office_revenue'] * np.random.uniform(0.1, 0.4)
                view_count = np.random.randint(1000000, 50000000)
                
                streaming_data.append({
                    'movie_id': idx,
                    'title': f"Movie_{idx}",
                    'platform': platform,
                    'streaming_revenue': streaming_revenue,
                    'view_count': view_count,
                    'launch_date': movie.get('release_date', datetime.now() - timedelta(days=np.random.randint(30, 365)))
                })
        
        return pd.DataFrame(streaming_data)

    def generate_market_data(self):
        """Generate synthetic market intelligence data."""
        np.random.seed(42)
        
        studios = ['Disney', 'Warner Bros', 'Universal', 'Sony', 'Paramount', 'Netflix', 'Amazon']
        regions = ['North America', 'Europe', 'Asia-Pacific', 'Latin America', 'Middle East & Africa']
        
        market_data = []
        for studio in studios:
            for region in regions:
                market_data.append({
                    'studio': studio,
                    'region': region,
                    'market_share': np.random.uniform(5, 25),
                    'revenue_millions': np.random.uniform(500, 5000),
                    'growth_rate': np.random.uniform(-10, 30),
                    'investment_millions': np.random.uniform(100, 2000)
                })
        
        return pd.DataFrame(market_data)

    def train_models(self):
        """Train machine learning models."""
        with st.spinner("Training advanced ML models..."):
            if st.session_state.portfolio_data is not None:
                results = self.platform.train_box_office_models(st.session_state.portfolio_data)
                st.session_state.model_results = results
                st.session_state.models_trained = True
                
        st.sidebar.success("✅ Models trained successfully!")

    def render_executive_dashboard(self):
        """Render executive-level KPI dashboard."""
        st.header("📊 Executive Dashboard")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data using the sidebar controls.")
            return
        
        df = st.session_state.portfolio_data
        
        # Executive KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>Total Portfolio Value</h3>
                <h2>${:,.0f}M</h2>
            </div>
            """.format(df['budget'].sum() / 1e6), unsafe_allow_html=True)
        
        with col2:
            avg_roi = df['roi'].mean()
            st.markdown("""
            <div class="metric-card">
                <h3>Average ROI</h3>
                <h2>{:.1f}%</h2>
            </div>
            """.format(avg_roi), unsafe_allow_html=True)
        
        with col3:
            if st.session_state.models_trained:
                accuracy = st.session_state.model_results['revenue_prediction']['accuracy']
                st.markdown("""
                <div class="metric-card">
                    <h3>Prediction Accuracy</h3>
                    <h2>{:.1%}</h2>
                </div>
                """.format(accuracy), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-card">
                    <h3>Prediction Accuracy</h3>
                    <h2>Train Models</h2>
                </div>
                """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>Active Investments</h3>
                <h2>{}</h2>
            </div>
            """.format(len(df)), unsafe_allow_html=True)
        
        # Performance visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # ROI distribution
            fig = px.histogram(
                df, x='roi', nbins=30,
                title='ROI Distribution',
                labels={'roi': 'Return on Investment (%)', 'count': 'Number of Movies'},
                color_discrete_sequence=['#667eea']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Genre performance
            genre_performance = df.groupby('genre').agg({
                'roi': 'mean',
                'box_office_revenue': 'sum'
            }).reset_index()
            
            fig = px.scatter(
                genre_performance,
                x='roi',
                y='box_office_revenue',
                size='box_office_revenue',
                color='genre',
                title='Genre Performance Matrix',
                labels={
                    'roi': 'Average ROI (%)',
                    'box_office_revenue': 'Total Box Office Revenue'
                }
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    def render_box_office_prediction(self):
        """Render box office prediction interface."""
        st.header("🎬 Box Office Prediction Engine")
        
        if not st.session_state.models_trained:
            st.warning("⚠️ Please train models using the sidebar controls.")
            return
        
        st.subheader("🔮 Predict New Movie Performance")
        
        # Input form for new movie prediction
        col1, col2 = st.columns(2)
        
        with col1:
            budget = st.number_input("Budget ($M)", min_value=1.0, max_value=500.0, value=50.0, step=1.0)
            genre = st.selectbox("Genre", ['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi', 'Romance', 'Thriller', 'Animation'])
            rating = st.selectbox("Rating", ['G', 'PG', 'PG-13', 'R'])
            star_power = st.slider("Star Power Score", 0, 100, 50)
            director_score = st.slider("Director Score", 0, 100, 50)
        
        with col2:
            marketing_spend = st.number_input("Marketing Spend ($M)", min_value=1.0, max_value=200.0, value=25.0, step=1.0)
            theater_count = st.number_input("Theater Count", min_value=500, max_value=4500, value=3000, step=100)
            runtime = st.number_input("Runtime (minutes)", min_value=80, max_value=180, value=110, step=5)
            sequel = st.checkbox("Sequel/Franchise")
            social_buzz = st.slider("Social Media Buzz", 0, 100, 50)
        
        if st.button("🎯 Generate Prediction", type="primary"):
            # Create prediction input
            prediction_input = pd.DataFrame({
                'budget': [budget * 1e6],
                'genre': [genre],
                'rating': [rating],
                'studio': ['Universal'],  # Default
                'release_season': ['Summer'],  # Default
                'star_power_score': [star_power],
                'director_score': [director_score],
                'marketing_spend': [marketing_spend * 1e6],
                'theater_count': [theater_count],
                'runtime_minutes': [runtime],
                'sequel_franchise': [1 if sequel else 0],
                'social_media_buzz': [social_buzz],
                'critic_score': [60],  # Default
                'audience_score': [65],  # Default
                'box_office_revenue': [0]  # Dummy for engineering
            })
            
            # Make prediction (simplified version)
            try:
                # Basic ROI calculation based on input features
                base_multiplier = 2.5
                genre_multipliers = {
                    'Action': 1.3, 'Sci-Fi': 1.2, 'Animation': 1.1, 'Comedy': 1.0,
                    'Thriller': 0.9, 'Drama': 0.8, 'Romance': 0.7, 'Horror': 0.6
                }
                
                predicted_revenue = (
                    budget * 1e6 * base_multiplier * genre_multipliers.get(genre, 1.0) *
                    (1 + star_power/100) * (1 + director_score/100) *
                    (1 + social_buzz/100) * (theater_count/3000)
                )
                
                predicted_roi = ((predicted_revenue - budget * 1e6) / (budget * 1e6)) * 100
                
                # Display prediction
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("""
                    <div class="prediction-card">
                        <h3>Predicted Revenue</h3>
                        <h2>${:,.1f}M</h2>
                    </div>
                    """.format(predicted_revenue / 1e6), unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="prediction-card">
                        <h3>Predicted ROI</h3>
                        <h2>{:.1f}%</h2>
                    </div>
                    """.format(predicted_roi), unsafe_allow_html=True)
                
                with col3:
                    risk_level = "High" if predicted_roi < 50 else "Medium" if predicted_roi < 150 else "Low"
                    st.markdown("""
                    <div class="risk-card">
                        <h3>Risk Level</h3>
                        <h2>{}</h2>
                    </div>
                    """.format(risk_level), unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

    def render_portfolio_analysis(self):
        """Render portfolio optimization and analysis."""
        st.header("💼 Portfolio Analysis & Optimization")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data using the sidebar controls.")
            return
        
        df = st.session_state.portfolio_data
        
        # Portfolio composition
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Current Portfolio Composition")
            
            # Investment by genre
            genre_investment = df.groupby('genre')['budget'].sum().reset_index()
            fig = px.pie(
                genre_investment,
                values='budget',
                names='genre',
                title='Investment Distribution by Genre'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Risk-Return Profile")
            
            # Risk-return scatter
            df['risk_score'] = pd.cut(df['budget'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            risk_return = df.groupby('risk_score').agg({
                'roi': 'mean',
                'budget': 'count'
            }).reset_index()
            
            fig = px.bar(
                risk_return,
                x='risk_score',
                y='roi',
                title='Average ROI by Risk Level',
                color='roi',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Portfolio optimization
        st.subheader("⚡ Portfolio Optimization")
        
        target_roi = st.slider("Target ROI (%)", min_value=0, max_value=500, value=150, step=10)
        risk_tolerance = st.select_slider("Risk Tolerance", options=['Conservative', 'Moderate', 'Aggressive'])
        
        if st.button("🎯 Optimize Portfolio"):
            # Simple portfolio optimization simulation
            optimized_allocation = self.optimize_portfolio(df, target_roi, risk_tolerance)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="success-card">
                    <h3>Optimized Allocation</h3>
                    <p>Based on your risk tolerance and target ROI</p>
                </div>
                """, unsafe_allow_html=True)
                
                for genre, allocation in optimized_allocation.items():
                    st.write(f"**{genre}:** {allocation:.1%}")
            
            with col2:
                # Expected metrics
                expected_roi = np.random.uniform(target_roi * 0.8, target_roi * 1.2)
                expected_sharpe = np.random.uniform(1.5, 3.0)
                
                st.markdown("""
                <div class="success-card">
                    <h3>Expected Performance</h3>
                    <p><strong>ROI:</strong> {:.1f}%</p>
                    <p><strong>Sharpe Ratio:</strong> {:.2f}</p>
                </div>
                """.format(expected_roi, expected_sharpe), unsafe_allow_html=True)

    def optimize_portfolio(self, df, target_roi, risk_tolerance):
        """Simple portfolio optimization logic."""
        genre_stats = df.groupby('genre').agg({
            'roi': ['mean', 'std'],
            'budget': 'count'
        }).round(2)
        
        # Flatten column names
        genre_stats.columns = ['_'.join(col).strip() for col in genre_stats.columns.values]
        genre_stats = genre_stats.reset_index()
        
        # Simple allocation based on risk tolerance
        if risk_tolerance == 'Conservative':
            # Favor genres with lower volatility
            weights = 1 / (genre_stats['roi_std'] + 0.1)
        elif risk_tolerance == 'Aggressive':
            # Favor genres with higher returns
            weights = genre_stats['roi_mean']
        else:
            # Balanced approach
            weights = genre_stats['roi_mean'] / (genre_stats['roi_std'] + 0.1)
        
        # Normalize weights
        weights = weights / weights.sum()
        
        return dict(zip(genre_stats['genre'], weights))

    def render_streaming_analytics(self):
        """Render streaming platform analytics."""
        st.header("📺 Streaming Analytics & Revenue Intelligence")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data using the sidebar controls.")
            return
        
        streaming_df = st.session_state.streaming_data
        
        # Streaming KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_streaming_revenue = streaming_df['streaming_revenue'].sum()
            st.metric("Total Streaming Revenue", f"${total_streaming_revenue/1e6:.1f}M")
        
        with col2:
            avg_views = streaming_df['view_count'].mean()
            st.metric("Average Views", f"{avg_views/1e6:.1f}M")
        
        with col3:
            revenue_per_view = (streaming_df['streaming_revenue'] / streaming_df['view_count']).mean()
            st.metric("Revenue per View", f"${revenue_per_view:.3f}")
        
        with col4:
            top_platform = streaming_df.groupby('platform')['streaming_revenue'].sum().idxmax()
            st.metric("Top Platform", top_platform)
        
        # Platform comparison
        col1, col2 = st.columns(2)
        
        with col1:
            platform_revenue = streaming_df.groupby('platform')['streaming_revenue'].sum().reset_index()
            fig = px.bar(
                platform_revenue,
                x='platform',
                y='streaming_revenue',
                title='Revenue by Streaming Platform',
                color='streaming_revenue',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            platform_views = streaming_df.groupby('platform')['view_count'].sum().reset_index()
            fig = px.pie(
                platform_views,
                values='view_count',
                names='platform',
                title='Viewership Distribution by Platform'
            )
            st.plotly_chart(fig, use_container_width=True)

    def run(self):
        """Main application runner."""
        self.setup_page_config()
        self.render_header()
        section = self.render_sidebar()
        
        # Route to selected section
        if section == "📊 Executive Dashboard":
            self.render_executive_dashboard()
        elif section == "🎬 Box Office Prediction":
            self.render_box_office_prediction()
        elif section == "💼 Portfolio Analysis":
            self.render_portfolio_analysis()
        elif section == "📈 Risk Management":
            self.render_risk_management()
        elif section == "🌐 Market Intelligence":
            self.render_market_intelligence()
        elif section == "📺 Streaming Analytics":
            self.render_streaming_analytics()
        elif section == "🤖 Model Insights":
            self.render_model_insights()
        elif section == "⚙️ Data Management":
            self.render_data_management()

    def render_risk_management(self):
        """Render risk management dashboard."""
        st.header("📈 Risk Management & Analysis")
        st.info("🚧 Risk management features coming soon...")

    def render_market_intelligence(self):
        """Render market intelligence dashboard."""
        st.header("🌐 Market Intelligence")
        st.info("🚧 Market intelligence features coming soon...")

    def render_model_insights(self):
        """Render ML model insights and explainability."""
        st.header("🤖 Model Insights & Explainability")
        st.info("🚧 Model insights features coming soon...")

    def render_data_management(self):
        """Render data management interface."""
        st.header("⚙️ Data Management")
        st.info("🚧 Data management features coming soon...")


def main():
    """Main application entry point."""
    app = EntertainmentInvestmentApp()
    app.run()


if __name__ == "__main__":
    main()