# NHL Game Outcome and Win Probability Predictor

A scalable machine learning system that predicts NHL game outcomes (1-3 days ahead) using dynamic data from the NHL Stats API and follows the Feature, Training, Inference (FTI) pipeline pattern.

## Project Structure

```
project/
├── src/
│   ├── feature_pipeline/      # Feature engineering and data ingestion
│   ├── training_pipeline/      # Model training and registration
│   ├── inference_pipeline/     # Prediction generation
│   ├── dashboard/              # Web dashboard UI
│   ├── utils/                  # Shared utilities
│   └── config/                 # Configuration files
├── data/                       # Local data storage (optional)
├── requirements.txt            # Python dependencies
├── config.yaml                 # Configuration settings
├── .env.example               # Environment variables template
└── README.md                  # This file
```

## Components

### 1. Feature Pipeline
- Fetches daily NHL data from the NHL Stats API
- Computes rolling features:
  - Recent team performance (win/loss streaks)
  - Goal differential
  - Special teams efficiency (power play, penalty kill)
  - Goalie form and save percentage
  - Head-to-head records
- Stores features in Hopsworks Feature Store

### 2. Training Pipeline
- Trains gradient-boosted tree models (XGBoost/LightGBM)
- Uses historical game data and features from feature store
- Evaluates model performance
- Registers trained models in Hopsworks Model Registry

### 3. Inference Pipeline
- Generates predictions for upcoming games (1-3 days ahead)
- Uses registered model from Model Registry
- Outputs win probabilities for each team

### 4. Dashboard
- Simple web UI displaying:
  - List of upcoming games
  - Predicted win probabilities for each team
  - Model confidence metrics

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your Hopsworks API key and project settings
```

3. Configure settings in `config.yaml`

## Usage

### Run Feature Pipeline
```bash
python src/feature_pipeline/main.py
```

### Run Training Pipeline
```bash
python src/training_pipeline/main.py
```

### Run Inference Pipeline
```bash
python src/inference_pipeline/main.py
```

### Start Dashboard
```bash
python src/dashboard/app.py
```

## Data Source

- **NHL Stats API**: https://statsapi.web.nhl.com/api/v1/
- Provides game schedules, results, player statistics, team metrics

## Technologies

- Python 3.8+
- Hopsworks (Feature Store & Model Registry)
- XGBoost/LightGBM (ML models)
- Flask/FastAPI (Dashboard)
- Pandas, NumPy (Data processing)

