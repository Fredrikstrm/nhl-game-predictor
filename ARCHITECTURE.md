# System Architecture

## Overview

The NHL Game Predictor follows the Feature, Training, Inference (FTI) pipeline pattern for scalable machine learning.

## Data Flow

```
NHL Stats API
    ↓
Feature Pipeline (Daily)
    ↓
Hopsworks Feature Store
    ↓
Training Pipeline (Periodic)
    ↓
Hopsworks Model Registry
    ↓
Inference Pipeline (Daily)
    ↓
Predictions (CSV/API)
    ↓
Dashboard UI
```

## Components

### 1. Feature Pipeline (`src/feature_pipeline/`)

**Purpose**: Extract, transform, and load (ETL) NHL game data into features

**Process**:
1. Fetches historical game data from NHL Stats API (last 90 days)
2. Computes rolling features for each team:
   - Win/loss streaks
   - Recent performance metrics
   - Goal differentials
   - Special teams efficiency
   - Home/away records
   - Head-to-head statistics
3. Stores features in Hopsworks Feature Store

**Key Files**:
- `main.py`: Pipeline orchestration
- Uses `utils/nhl_api_client.py` for API calls
- Uses `utils/feature_engineering.py` for feature computation
- Uses `utils/hopsworks_client.py` for feature store operations

**Execution**: Run daily (can be scheduled as serverless job)

### 2. Training Pipeline (`src/training_pipeline/`)

**Purpose**: Train ML models using historical features

**Process**:
1. Retrieves features from Hopsworks Feature Store
2. Prepares training/validation/test splits
3. Trains gradient-boosted tree model (XGBoost or LightGBM)
4. Evaluates model performance
5. Registers model in Hopsworks Model Registry

**Key Files**:
- `main.py`: Training orchestration
- Uses `utils/hopsworks_client.py` for feature retrieval and model registration

**Execution**: Run periodically (weekly) or when sufficient new data is available

### 3. Inference Pipeline (`src/inference_pipeline/`)

**Purpose**: Generate predictions for upcoming games

**Process**:
1. Loads latest trained model from Model Registry
2. Fetches upcoming games from NHL API (next 1-3 days)
3. Computes features for upcoming games using historical data
4. Generates win probability predictions
5. Saves predictions to CSV file

**Key Files**:
- `main.py`: Inference orchestration
- Uses all utility modules

**Execution**: Run daily to get fresh predictions

### 4. Dashboard (`src/dashboard/`)

**Purpose**: Display predictions in a web interface

**Process**:
1. Flask web server serves HTML dashboard
2. Reads predictions from CSV file
3. Displays upcoming games with win probabilities
4. Auto-refreshes every 5 minutes

**Key Files**:
- `app.py`: Flask application
- `templates/index.html`: Dashboard UI

**Execution**: Run continuously (can be deployed as web service)

## Utility Modules (`src/utils/`)

### `nhl_api_client.py`
- Client for NHL Stats API
- Methods for fetching schedules, game details, team stats

### `feature_engineering.py`
- Feature computation logic
- Rolling window calculations
- Team performance metrics

### `hopsworks_client.py`
- Feature Store operations (create, insert, retrieve)
- Model Registry operations (register, retrieve, download)

### `test_connection.py`
- Utility script to test API and Hopsworks connections

## Configuration

### `config.yaml`
- NHL API settings
- Feature engineering parameters
- Model hyperparameters
- Feature Store and Model Registry names
- Dashboard settings

### `.env` (from `env.example`)
- Hopsworks credentials
- API keys
- Environment-specific settings

## Feature Engineering

### Team Features (per team)
- `win_streak`: Current win streak
- `loss_streak`: Current loss streak
- `recent_win_pct`: Win percentage in last N games
- `recent_goal_differential`: Goal difference in last N games
- `avg_goals_for`: Average goals scored
- `avg_goals_against`: Average goals allowed
- `power_play_pct`: Power play success rate
- `penalty_kill_pct`: Penalty kill success rate
- `home_win_pct`: Home win percentage
- `away_win_pct`: Away win percentage

### Game Features
- `h2h_team1_wins`: Head-to-head wins for team 1
- `h2h_team2_wins`: Head-to-head wins for team 2
- `h2h_team1_win_pct`: Head-to-head win percentage

### Feature Naming Convention
- Home team features: `home_*`
- Away team features: `away_*`
- Head-to-head features: `h2h_*`

## Model

### Type
- Gradient Boosted Trees (XGBoost or LightGBM)

### Target
- Binary classification: Home team wins (1) or loses (0)

### Output
- Win probability for home team
- Win probability for away team (1 - home_probability)

### Evaluation Metrics
- Accuracy
- Log Loss
- ROC AUC

## Scalability Considerations

1. **Feature Pipeline**: Can be run as serverless job (AWS Lambda, Google Cloud Functions)
2. **Training Pipeline**: Can be scheduled on compute instances
3. **Inference Pipeline**: Can be containerized and run on-demand
4. **Dashboard**: Can be deployed on cloud platforms (Heroku, AWS, GCP)
5. **Feature Store**: Hopsworks provides scalable feature storage and retrieval
6. **Model Registry**: Centralized model versioning and management

## Future Enhancements

- Real-time feature updates
- Model retraining automation
- A/B testing framework
- Additional features (player injuries, rest days, etc.)
- Ensemble models
- Confidence intervals for predictions
- Historical prediction accuracy tracking

