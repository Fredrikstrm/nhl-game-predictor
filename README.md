# NHL Game Predictor

**Course**: Scalable Machine Learning (ID2223)  
**Author**: Fredrik Ström (frest@kth.se)

A scalable ML system that predicts NHL game outcomes using the Feature, Training, Inference (FTI) pipeline pattern.

## Project Overview

### 1. Dynamic Data Source
Uses the **NHL Stats API** (`https://api-web.nhle.com`) which provides:
- Daily updated game schedules and results
- Real-time team statistics
- Historical game data
- No static datasets - all data is fetched dynamically

### 2. Prediction Problem
Predicts **NHL game outcomes** (home team win probability) for upcoming games **1-3 days ahead**, using:
- Team performance metrics (win streaks, goal differentials)
- Head-to-head historical records
- Home/away performance statistics

### 3. User Interface
**Web Dashboard** displaying:
- Upcoming NHL games with predicted win probabilities
- Visual probability indicators for each team
- Predicted winners
- Available at: [GitHub Pages](https://fredrikstrm.github.io/nhl-game-predictor/) or locally at `http://localhost:5001`

### 4. Technologies
- **Python 3.11** - Core language
- **Hopsworks** - Feature Store & Model Registry
- **XGBoost** - Gradient boosted tree model
- **Flask** - Web dashboard framework
- **NHL Stats API** - Dynamic data source
- **GitHub Actions** - Automated inference and deployment

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   - Copy `.env.example` to `.env`
   - Add your Hopsworks API key and project name

3. **Run pipelines:**
   ```bash
   # 1. Fetch data and compute features
   python src/feature_pipeline/main.py
   
   # 2. Train model
   python src/training_pipeline/main.py
   
   # 3. Generate predictions
   python src/inference_pipeline/main.py
   
   # 4. Start dashboard (optional)
   python src/dashboard/app.py
   ```

## Project Structure

```
src/
├── feature_pipeline/    # Fetches NHL data, computes features
├── training_pipeline/   # Trains XGBoost model
├── inference_pipeline/  # Generates predictions
├── dashboard/          # Web UI for predictions
└── utils/              # Shared utilities (API client, feature engineering)
```

## Features

The model uses **21 working features** including:
- Team win/loss streaks
- Recent goal differentials
- Head-to-head records
- Home/away performance

Note: Power play and penalty kill stats default to 0.0 (API limitation).

## Configuration

Edit `config.yaml` to adjust:
- Prediction horizon (default: 3 days)
- Training data window (default: 365 days)
- Model hyperparameters
