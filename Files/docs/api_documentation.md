# Entertainment Investment Intelligence Platform
## API Documentation

### Version 2.0.0 Enterprise
### Author: API Documentation Team  
### Date: August 2025

---

## Overview

The Entertainment Investment Intelligence Platform provides comprehensive APIs for box office prediction, content analytics, and entertainment investment optimization. All APIs support high-volume requests with industry-leading prediction accuracy.

**Base URL**: `https://api.entertainment.enterprise.com/v2`
**Authentication**: Bearer Token (OAuth 2.0)
**Rate Limiting**: 5,000 requests/minute per API key

## Authentication

```bash
Authorization: Bearer {access_token}
```

### Get Access Token

**Endpoint**: `POST /auth/token`

**Request**:
```json
{
  "grant_type": "client_credentials",
  "client_id": "your_client_id", 
  "client_secret": "your_client_secret",
  "scope": "entertainment:read entertainment:write analytics:full"
}
```

## Core Entertainment APIs

### 1. Box Office Prediction

#### Predict Box Office Performance

**Endpoint**: `POST /box-office/predict`

**Request Body**:
```json
{
  "movie_data": {
    "title": "AI Revolution",
    "genre": ["Sci-Fi", "Action"],
    "budget": 75000000,
    "rating": "PG-13",
    "runtime": 142,
    "release_date": "2025-07-04",
    "theater_count": 3500,
    "cast": [
      {"name": "Actor Name", "star_power": 85},
      {"name": "Actress Name", "star_power": 92}
    ],
    "director": {
      "name": "Director Name",
      "track_record": 88,
      "previous_films": ["Film A", "Film B"]
    },
    "marketing_spend": 45000000,
    "social_buzz": {
      "twitter_mentions": 125000,
      "youtube_trailer_views": 15000000,
      "instagram_engagement": 89000
    }
  }
}
```

**Response**:
```json
{
  "prediction_id": "PRED-2025-08-001",
  "movie_title": "AI Revolution",
  "predictions": {
    "opening_weekend": {
      "domestic": 52400000,
      "confidence_interval": [45200000, 59600000],
      "confidence": 0.89
    },
    "total_domestic": {
      "amount": 187500000,
      "confidence_interval": [165000000, 210000000],
      "confidence": 0.85
    },
    "international": {
      "amount": 312800000,
      "confidence_interval": [280000000, 345000000],
      "confidence": 0.82
    },
    "total_worldwide": {
      "amount": 500300000,
      "confidence_interval": [445000000, 555000000],
      "confidence": 0.84
    }
  },
  "roi_analysis": {
    "total_investment": 120000000,
    "projected_profit": 380300000,
    "roi_percentage": 3.17,
    "break_even_point": 240000000,
    "risk_assessment": "MODERATE"
  },
  "success_factors": [
    {
      "factor": "Star Power",
      "impact": 0.23,
      "description": "Strong cast with proven box office appeal"
    },
    {
      "factor": "Genre Appeal", 
      "impact": 0.19,
      "description": "Sci-Fi action appeals to core demographic"
    },
    {
      "factor": "Release Date",
      "impact": 0.16,
      "description": "July 4th weekend maximizes audience"
    }
  ],
  "risk_factors": [
    {
      "factor": "Competition",
      "impact": -0.08,
      "description": "Two major releases same weekend"
    }
  ],
  "model_confidence": 0.91,
  "prediction_date": "2025-08-18T15:30:45Z"
}
```

### 2. Content Analytics

#### Analyze Content Performance

**Endpoint**: `POST /content/analyze`

**Request Body**:
```json
{
  "content_id": "CONTENT-2025-001",
  "content_type": "FEATURE_FILM",
  "analysis_scope": ["script", "cast", "visual", "market"],
  "content_data": {
    "script_text": "Full script content...",
    "cast_list": ["Actor A", "Actor B"],
    "director": "Director Name",
    "genre": ["Drama", "Thriller"],
    "target_audience": "R-rated adult drama",
    "production_budget": 25000000
  }
}
```

**Response**:
```json
{
  "analysis_id": "ANA-2025-08-001",
  "content_quality_score": 87.3,
  "script_analysis": {
    "theme_strength": 0.89,
    "character_development": 0.84,
    "dialogue_quality": 0.91,
    "narrative_structure": 0.86,
    "genre_authenticity": 0.88,
    "emotional_impact": 0.92
  },
  "cast_analysis": {
    "overall_star_power": 78.5,
    "chemistry_potential": 0.83,
    "audience_appeal": 0.87,
    "award_potential": 0.74
  },
  "market_positioning": {
    "target_demographic": "Adults 25-54",
    "competition_level": "MODERATE",
    "release_window_optimization": "Fall awards season",
    "international_appeal": 0.72
  },
  "financial_projections": {
    "estimated_budget_range": [20000000, 30000000],
    "box_office_potential": 145000000,
    "profitability_score": 0.85,
    "investment_grade": "B+"
  }
}
```

### 3. Investment Portfolio Management

#### Optimize Investment Portfolio

**Endpoint**: `POST /portfolio/optimize`

**Request Body**:
```json
{
  "portfolio_id": "PORT-ENT-001",
  "optimization_objective": "MAXIMIZE_ROI",
  "constraints": {
    "max_single_investment": 0.15,
    "min_diversification": 0.75,
    "risk_tolerance": "MODERATE",
    "time_horizon_months": 36
  },
  "available_projects": [
    {
      "project_id": "PROJ-001",
      "type": "FEATURE_FILM",
      "budget": 50000000,
      "expected_roi": 2.4,
      "risk_score": 0.3,
      "genre": "Action"
    }
  ]
}
```

**Response**:
```json
{
  "optimization_id": "OPT-2025-08-001",
  "recommended_allocation": [
    {
      "project_id": "PROJ-001",
      "allocation_percentage": 0.12,
      "investment_amount": 6000000,
      "expected_return": 14400000,
      "risk_contribution": 0.08
    }
  ],
  "portfolio_metrics": {
    "expected_annual_return": 0.247,
    "portfolio_volatility": 0.156,
    "sharpe_ratio": 1.58,
    "max_drawdown": 0.089,
    "diversification_score": 0.82
  }
}
```

## Data Models

### Movie Data Schema

```json
{
  "type": "object",
  "properties": {
    "title": {"type": "string"},
    "genre": {"type": "array", "items": {"type": "string"}},
    "budget": {"type": "number", "minimum": 100000},
    "rating": {"type": "string", "enum": ["G", "PG", "PG-13", "R", "NC-17"]},
    "runtime": {"type": "number", "minimum": 60, "maximum": 300},
    "release_date": {"type": "string", "format": "date"},
    "theater_count": {"type": "number", "minimum": 1}
  },
  "required": ["title", "genre", "budget", "rating", "release_date"]
}
```

## Error Handling

### HTTP Status Codes

| Code | Description | Usage |
|------|-------------|-------|
| 200 | OK | Successful request |
| 201 | Created | Analysis/prediction created |
| 400 | Bad Request | Invalid parameters |
| 401 | Unauthorized | Authentication required |
| 429 | Too Many Requests | Rate limit exceeded |

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_MOVIE_DATA",
    "message": "Budget must be positive number",
    "details": {
      "field": "budget",
      "requirement": "minimum 100000"
    }
  },
  "request_id": "REQ-2025-08-001",
  "timestamp": "2025-08-18T15:30:45Z"
}
```

## Rate Limiting

- **Default Rate Limit**: 5,000 requests/minute
- **Burst Limit**: 500 requests/10 seconds
- **Headers**: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`

This API documentation provides comprehensive guidance for integrating with the Entertainment Investment Intelligence Platform's prediction and analytics capabilities.