# Setup Guide

## Prerequisites

- Python 3.8 or higher
- Hopsworks account and API key
- Internet connection for NHL API access

## Installation Steps

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy the example environment file and fill in your credentials:

```bash
cp env.example .env
```

Edit `.env` and add:
- Your Hopsworks API key
- Your Hopsworks project name

### 4. Configure Settings

Edit `config.yaml` to adjust:
- Feature engineering parameters (rolling window size, etc.)
- Model hyperparameters
- Feature group and model names
- Dashboard settings

## Running the Pipelines

### Feature Pipeline

Fetches NHL data and computes features:

```bash
python src/feature_pipeline/main.py
```

This should be run daily (can be scheduled as a serverless job).

### Training Pipeline

Trains the model using features from the feature store:

```bash
python src/training_pipeline/main.py
```

Run this periodically (e.g., weekly) or when you have enough new data.

### Inference Pipeline

Generates predictions for upcoming games:

```bash
python src/inference_pipeline/main.py
```

Run this daily to get fresh predictions.

### Dashboard

Start the web dashboard:

```bash
python src/dashboard/app.py
```

Then open your browser to `http://localhost:5000`

## Pipeline Execution Order

1. **Feature Pipeline** → Fetches data and stores features
2. **Training Pipeline** → Trains model using stored features
3. **Inference Pipeline** → Generates predictions using trained model
4. **Dashboard** → Displays predictions (can run continuously)

## Notes

- The feature pipeline needs historical data to compute rolling features. The first run may need to fetch more historical data.
- Make sure you have sufficient data in the feature store before running the training pipeline.
- The inference pipeline requires a trained model in the Model Registry.

## Troubleshooting

### Hopsworks Connection Issues
- Verify your API key is correct
- Check that your project name matches exactly
- Ensure you have the necessary permissions

### NHL API Issues
- The NHL API is public and free, but be respectful with request rates
- If you encounter rate limiting, add delays between requests

### Missing Features
- Some features may not be available for all games (e.g., special teams stats)
- The code handles missing data with default values

