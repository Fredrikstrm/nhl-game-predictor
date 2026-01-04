"""
Training Pipeline
Trains ML model using features from Feature Store and registers it in Model Registry
"""
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import xgboost as xgb

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.hopsworks_client import HopsworksClient
from config.config_loader import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare features and target for training"""
    # Exclude non-feature columns
    exclude_cols = ['game_id', 'game_date', 'home_team_id', 'away_team_id', 'home_team_won']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].fillna(0)
    y = df['home_team_won']
    
    # Ensure target is binary integer (0 or 1) for XGBoost
    y = y.astype(int)
    
    # #region agent log
    import json
    try:
        with open('/Users/fredrikstrom/Documents/KTH_Dokument/Scalable ML/project/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H","location":"training_pipeline/main.py:prepare_features","message":"Function exit","data":{"y_dtype":str(y.dtype),"y_min":int(y.min()) if len(y) > 0 else None,"y_max":int(y.max()) if len(y) > 0 else None,"y_unique":list(y.unique()) if len(y) > 0 else None},"timestamp":int(__import__('time').time()*1000)})+'\n')
    except: pass
    # #endregion
    
    return X, y, feature_cols


def train_model(X_train, y_train, X_val, y_val, config: dict):
    """Train XGBoost model"""
    hyperparams = config['training']['hyperparameters']
    
    # #region agent log
    import json
    try:
        with open('/Users/fredrikstrom/Documents/KTH_Dokument/Scalable ML/project/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"I","location":"training_pipeline/main.py:train_model","message":"Function entry","data":{"y_train_dtype":str(y_train.dtype),"y_train_min":int(y_train.min()) if len(y_train) > 0 else None,"y_train_max":int(y_train.max()) if len(y_train) > 0 else None,"y_train_unique":list(y_train.unique()) if len(y_train) > 0 else None},"timestamp":int(__import__('time').time()*1000)})+'\n')
    except: pass
    # #endregion
    
    # Ensure target values are in [0, 1] range for binary classification
    y_train_binary = y_train.astype(int)
    y_val_binary = y_val.astype(int)
    
    # #region agent log
    try:
        with open('/Users/fredrikstrom/Documents/KTH_Dokument/Scalable ML/project/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"I","location":"training_pipeline/main.py:train_model","message":"After binary conversion","data":{"y_train_binary_dtype":str(y_train_binary.dtype),"y_train_binary_min":int(y_train_binary.min()) if len(y_train_binary) > 0 else None,"y_train_binary_max":int(y_train_binary.max()) if len(y_train_binary) > 0 else None,"y_train_binary_unique":list(y_train_binary.unique()) if len(y_train_binary) > 0 else None,"y_train_mean":float(y_train_binary.mean()) if len(y_train_binary) > 0 else None},"timestamp":int(__import__('time').time()*1000)})+'\n')
    except: pass
    # #endregion
    
    # Verify target is binary
    if not set(y_train_binary.unique()).issubset({0, 1}):
        raise ValueError(f"Target must be binary (0/1), but found values: {y_train_binary.unique()}")
    
    # Calculate base_score as the mean of positive class (must be in [0,1])
    base_score = float(y_train_binary.mean())
    # Ensure base_score is in valid range [0,1]
    base_score = max(0.0, min(1.0, base_score))
    if base_score == 0.0:
        base_score = 0.01  # Avoid exactly 0
    elif base_score == 1.0:
        base_score = 0.99  # Avoid exactly 1
    
    # #region agent log
    try:
        with open('/Users/fredrikstrom/Documents/KTH_Dokument/Scalable ML/project/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"I","location":"training_pipeline/main.py:train_model","message":"Before model creation","data":{"base_score":base_score},"timestamp":int(__import__('time').time()*1000)})+'\n')
    except: pass
    # #endregion
    
    # Remove base_score from hyperparams if present to avoid conflicts
    hyperparams_clean = {k: v for k, v in hyperparams.items() if k != 'base_score'}
    
    model = xgb.XGBClassifier(
        **hyperparams_clean,
        random_state=config['training']['random_state'],
        eval_metric='logloss',
        objective='binary:logistic',  # Explicitly set binary classification objective
        base_score=base_score  # Explicitly set base_score to avoid auto-calculation issues
    )
    model.fit(
        X_train, y_train_binary,
        eval_set=[(X_val, y_val_binary)],
        verbose=False
    )
    
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    # Ensure target is binary integer (0 or 1) for evaluation
    y_test_binary = y_test.astype(int)
    
    # Log test set class distribution before evaluation
    y_test_counts = y_test_binary.value_counts().to_dict()
    y_test_pct = (y_test_binary.value_counts(normalize=True) * 100).to_dict()
    logger.info(f"Test set class distribution before evaluation: {y_test_counts} ({y_test_pct.get(0, 0):.1f}% class 0, {y_test_pct.get(1, 0):.1f}% class 1)")
    
    # Check if test set has only one class
    if len(y_test_counts) == 1:
        logger.warning(f"WARNING: Test set contains only one class ({list(y_test_counts.keys())[0]}). Metrics may be unreliable.")
    
    # #region agent log
    import json
    try:
        with open('/Users/fredrikstrom/Documents/KTH_Dokument/Scalable ML/project/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"J","location":"training_pipeline/main.py:evaluate_model","message":"Function entry","data":{"y_test_dtype":str(y_test.dtype),"y_test_binary_dtype":str(y_test_binary.dtype),"y_test_binary_min":int(y_test_binary.min()) if len(y_test_binary) > 0 else None,"y_test_binary_max":int(y_test_binary.max()) if len(y_test_binary) > 0 else None,"y_test_binary_unique":list(y_test_binary.unique()) if len(y_test_binary) > 0 else None,"y_test_binary_counts":dict(y_test_binary.value_counts().to_dict()) if len(y_test_binary) > 0 else None},"timestamp":int(__import__('time').time()*1000)})+'\n')
    except: pass
    # #endregion
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test_binary, y_pred)
    # Explicitly provide labels to log_loss to handle edge cases where test set might have only one class
    logloss = log_loss(y_test_binary, y_pred_proba, labels=[0, 1])
    
    # Calculate ROC AUC, but handle case where it's undefined (only one class in test set)
    try:
        roc_auc = roc_auc_score(y_test_binary, y_pred_proba)
        # Check if ROC AUC is NaN (happens when test set has only one class)
        if pd.isna(roc_auc) or np.isnan(roc_auc):
            roc_auc = None
            logger.warning("ROC AUC is undefined (test set has only one class), excluding from metrics")
        else:
            roc_auc = float(roc_auc)
    except ValueError as e:
        # ROC AUC can't be computed (only one class)
        roc_auc = None
        logger.warning(f"ROC AUC cannot be computed: {e}, excluding from metrics")
    
    # Build metrics dict, excluding NaN/None values
    metrics = {
        'accuracy': float(accuracy),
        'log_loss': float(logloss)
    }
    # Only include ROC AUC if it's a valid number
    if roc_auc is not None:
        metrics['roc_auc'] = roc_auc
    
    # #region agent log
    try:
        with open('/Users/fredrikstrom/Documents/KTH_Dokument/Scalable ML/project/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"J","location":"training_pipeline/main.py:evaluate_model","message":"Function exit","data":{"metrics":metrics,"roc_auc_value":roc_auc},"timestamp":int(__import__('time').time()*1000)})+'\n')
    except: pass
    # #endregion
    
    logger.info(f"Model Metrics: {metrics}")
    return metrics


def main():
    """Main training pipeline execution"""
    logger.info("Starting Training Pipeline")
    
    # Load configuration
    config = load_config()
    
    # Initialize Hopsworks client
    hopsworks_client = HopsworksClient()
    
    # Get feature group
    fg_config = config['feature_store']
    feature_group = hopsworks_client.get_or_create_feature_group(
        name=fg_config['feature_group_name'],
        version=fg_config['feature_group_version']
    )
    
    # Retrieve features from feature store
    logger.info("Retrieving features from Feature Store")
    features_df = hopsworks_client.get_features(feature_group)
    logger.info(f"Retrieved {len(features_df)} rows from Feature Store")
    
    if len(features_df) == 0:
        logger.error("No features found in Feature Store. Run feature pipeline first.")
        return
    
    # Prepare features
    X, y, feature_cols = prepare_features(features_df)
    logger.info(f"Prepared {len(feature_cols)} features for {len(X)} samples")
    
    # Log full dataset class distribution
    y_counts = y.value_counts().to_dict()
    y_pct = (y.value_counts(normalize=True) * 100).to_dict()
    logger.info(f"Full dataset class distribution: {y_counts} ({y_pct[0]:.1f}% class 0, {y_pct[1]:.1f}% class 1)")
    
    # Check if stratification is possible (need at least 2 samples per class)
    min_class_count = min(y_counts.values())
    if min_class_count < 2:
        logger.warning(f"Stratification may fail: minimum class count is {min_class_count} (need at least 2 per class)")
    
    # Split data
    train_config = config['training']
    try:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(train_config['test_size'] + train_config['validation_size']),
            random_state=train_config['random_state'],
            stratify=y
        )
        logger.info("First split (train vs temp) completed with stratification")
    except ValueError as e:
        logger.warning(f"Stratification failed in first split: {e}. Retrying without stratification.")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(train_config['test_size'] + train_config['validation_size']),
            random_state=train_config['random_state'],
            stratify=None
        )
    
    # Log train and temp distributions
    y_train_counts = y_train.value_counts().to_dict()
    y_train_pct = (y_train.value_counts(normalize=True) * 100).to_dict()
    logger.info(f"Train set class distribution: {y_train_counts} ({y_train_pct.get(0, 0):.1f}% class 0, {y_train_pct.get(1, 0):.1f}% class 1)")
    
    y_temp_counts = y_temp.value_counts().to_dict()
    y_temp_pct = (y_temp.value_counts(normalize=True) * 100).to_dict()
    logger.info(f"Temp set (val+test) class distribution: {y_temp_counts} ({y_temp_pct.get(0, 0):.1f}% class 0, {y_temp_pct.get(1, 0):.1f}% class 1)")
    
    val_size = train_config['validation_size'] / (train_config['test_size'] + train_config['validation_size'])
    try:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=train_config['random_state'],
            stratify=y_temp
        )
        logger.info("Second split (val vs test) completed with stratification")
    except ValueError as e:
        logger.warning(f"Stratification failed in second split: {e}. Retrying without stratification.")
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=train_config['random_state'],
            stratify=None
        )
    
    # Log validation and test distributions
    y_val_counts = y_val.value_counts().to_dict()
    y_val_pct = (y_val.value_counts(normalize=True) * 100).to_dict()
    logger.info(f"Val set class distribution: {y_val_counts} ({y_val_pct.get(0, 0):.1f}% class 0, {y_val_pct.get(1, 0):.1f}% class 1)")
    
    y_test_counts = y_test.value_counts().to_dict()
    y_test_pct = (y_test.value_counts(normalize=True) * 100).to_dict()
    logger.info(f"Test set class distribution: {y_test_counts} ({y_test_pct.get(0, 0):.1f}% class 0, {y_test_pct.get(1, 0):.1f}% class 1)")
    
    logger.info(f"Train set: {len(X_train)}, Val set: {len(X_val)}, Test set: {len(X_test)}")
    
    # Train model
    logger.info("Training model...")
    model = train_model(X_train, y_train, X_val, y_val, config)
    
    # Evaluate model
    logger.info("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    # Register model
    mr_config = config['model_registry']
    logger.info("Registering model in Model Registry...")
    hopsworks_client.register_model(
        model=model,
        model_name=mr_config['model_name'],
        description=mr_config.get('model_description', ''),
        metrics=metrics
    )
    
    logger.info("Training Pipeline completed successfully")


if __name__ == "__main__":
    main()

