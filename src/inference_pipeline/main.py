"""
Inference Pipeline
Generates predictions for upcoming games using trained model
"""
import sys
import logging
from pathlib import Path
import pandas as pd
import joblib
from datetime import datetime, timedelta, timezone

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.nhl_api_client import NHLAPIClient
from utils.feature_engineering import FeatureEngineer
from utils.hopsworks_client import HopsworksClient
from config.config_loader import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_historical_data_for_features(api_client: NHLAPIClient, days_back: int = 90) -> pd.DataFrame:
    """
    Fetch historical games needed for feature computation.
    
    Why we need this: To compute rolling features (win streaks, recent performance, 
    goal differentials, head-to-head records) for upcoming games, we need historical 
    game data from the past 90 days.
    
    Note: This is NOT for training - it's for computing features for each upcoming game.
    For example, to know "Team A's win streak" or "Team A's recent goal differential",
    we need their recent game history. The 90 days provides enough data to compute
    meaningful rolling statistics (e.g., last 10 games performance).
    
    The training data (640 games) was used to train the model. This historical data
    (also ~640 games over 90 days) is used to compute features for NEW games that
    haven't been played yet. This is the correct approach.
    """
    from datetime import timedelta
    import pandas as pd
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)
    
    logger.info(f"Fetching historical games from {start_date} to {end_date} for feature computation")
    
    # Use get_daily_scores instead of get_schedule to get actual scores (same as feature pipeline)
    games = []
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        try:
            daily_games = api_client.get_daily_scores(date=date_str)
            games.extend(daily_games)
        except Exception as e:
            logger.debug(f"Error fetching scores for {date_str}: {e}")
        current_date += timedelta(days=1)
    
    games_data = []
    for game in games:
        # Check if game is final - API structure may vary
        game_status = game.get("gameState") or game.get("status", {}).get("detailedState") or game.get("gameStatus")
        is_final = game_status in ["OFF", "FINAL", "Final", "OFFICIAL"]
        
        if not is_final:
            continue
        
        # Extract team information - handle different response structures
        home_team = game.get('homeTeam') or game.get('teams', {}).get('home', {}).get('team', {})
        away_team = game.get('awayTeam') or game.get('teams', {}).get('away', {}).get('team', {})
        
        # Get scores - handle different response structures
        home_score = (game.get('homeTeamScore') or 
                     game.get('homeScore') or 
                     game.get('teams', {}).get('home', {}).get('score') or
                     game.get('homeTeam', {}).get('score') or
                     game.get('home', {}).get('score'))
        away_score = (game.get('awayTeamScore') or 
                     game.get('awayScore') or 
                     game.get('teams', {}).get('away', {}).get('score') or
                     game.get('awayTeam', {}).get('score') or
                     game.get('away', {}).get('score'))
        
        # Get team IDs
        home_team_id = home_team.get('id') if isinstance(home_team, dict) else None
        away_team_id = away_team.get('id') if isinstance(away_team, dict) else None
        
        # Get game ID and date
        game_id = game.get('id') or game.get('gamePk') or game.get('gameId')
        game_date_str = game.get('startTimeUTC') or game.get('gameDate') or game.get('date')
        
        if game_id and home_team_id and away_team_id:
            # Convert scores to integers
            home_score_int = int(home_score) if home_score is not None and home_score != '' else 0
            away_score_int = int(away_score) if away_score is not None and away_score != '' else 0
            
            games_data.append({
                'game_id': game_id,
                'game_date': game_date_str,
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'home_team_score': home_score_int,
                'away_team_score': away_score_int,
                'home_team_won': 1 if home_score_int > away_score_int else 0
            })
    
    df = pd.DataFrame(games_data)
    logger.info(f"Fetched {len(df)} historical games for feature computation")
    return df


def _transform_games_for_team(team_games: pd.DataFrame, team_id: int) -> pd.DataFrame:
    """
    Transform games DataFrame to team perspective (add won, goals_for, goals_against, is_home columns)
    Same function as in feature_pipeline/main.py
    """
    if len(team_games) == 0:
        return pd.DataFrame()
    
    # Create a copy to avoid modifying original
    team_df = team_games.copy()
    
    # Determine if team was home or away, and compute team-specific columns
    team_df['is_home'] = (team_df['home_team_id'] == team_id).astype(int)
    team_df['is_away'] = (team_df['away_team_id'] == team_id).astype(int)
    
    # Compute goals_for and goals_against from team's perspective
    team_df['goals_for'] = (
        team_df['home_team_score'] * team_df['is_home'] + 
        team_df['away_team_score'] * team_df['is_away']
    )
    team_df['goals_against'] = (
        team_df['away_team_score'] * team_df['is_home'] + 
        team_df['home_team_score'] * team_df['is_away']
    )
    
    # Compute won column from team's perspective
    # Team wins if: (team is home and home_team_won=1) OR (team is away and home_team_won=0)
    team_df['won'] = (
        (team_df['is_home'] == 1) & (team_df['home_team_won'] == 1)
    ) | (
        (team_df['is_away'] == 1) & (team_df['home_team_won'] == 0)
    )
    team_df['won'] = team_df['won'].astype(int)
    
    return team_df


def build_team_name_mapping(games: list, historical_games: pd.DataFrame = None) -> dict:
    """
    Build a mapping from team_id to team_name from game data.
    Tries multiple possible field names for team names.
    """
    team_mapping = {}
    
    # Extract from upcoming games
    for game in games:
        home_team = game.get('homeTeam') or game.get('teams', {}).get('home', {}).get('team', {})
        away_team = game.get('awayTeam') or game.get('teams', {}).get('away', {}).get('team', {})
        
        if isinstance(home_team, dict):
            team_id = home_team.get('id')
            # Try multiple possible name fields
            team_name = (home_team.get('name') or 
                        home_team.get('teamName') or 
                        home_team.get('name', {}).get('default') or
                        home_team.get('abbrev') or
                        '')
            if team_id and team_name:
                team_mapping[team_id] = team_name
        
        if isinstance(away_team, dict):
            team_id = away_team.get('id')
            team_name = (away_team.get('name') or 
                        away_team.get('teamName') or 
                        away_team.get('name', {}).get('default') or
                        away_team.get('abbrev') or
                        '')
            if team_id and team_name:
                team_mapping[team_id] = team_name
    
    # Also try to extract from historical games if available
    # Historical games might have team names in a different format
    # For now, we'll rely on upcoming games for the mapping
    
    return team_mapping


def compute_features_for_upcoming_game(
    game: dict,
    historical_games: pd.DataFrame,
    feature_engineer: FeatureEngineer,
    team_name_mapping: dict = None
) -> pd.DataFrame:
    """Compute features for a single upcoming game"""
    # Extract team information - handle different response structures
    home_team = game.get('homeTeam') or game.get('teams', {}).get('home', {}).get('team', {})
    away_team = game.get('awayTeam') or game.get('teams', {}).get('away', {}).get('team', {})
    
    game_id = game.get('id') or game.get('gamePk') or game.get('gameId')
    home_team_id = home_team.get('id') if isinstance(home_team, dict) else None
    away_team_id = away_team.get('id') if isinstance(away_team, dict) else None
    game_date_str = game.get('startTimeUTC') or game.get('gameDate') or game.get('date')
    
    if not game_id or not home_team_id or not away_team_id:
        logger.warning(f"Missing required game data: game_id={game_id}, home_team_id={home_team_id}, away_team_id={away_team_id}")
        return pd.DataFrame()
    
    game_date = pd.to_datetime(game_date_str).date()
    
    # Get historical games before this game date
    historical = historical_games[
        pd.to_datetime(historical_games['game_date']).dt.date < game_date
    ]
    
    # Compute home team features
    home_team_games_raw = historical[
        (historical['home_team_id'] == home_team_id) |
        (historical['away_team_id'] == home_team_id)
    ]
    # Transform to team perspective
    home_team_games = _transform_games_for_team(home_team_games_raw, home_team_id)
    home_features = feature_engineer.compute_team_features(home_team_games, home_team_id)
    
    # Compute away team features
    away_team_games_raw = historical[
        (historical['home_team_id'] == away_team_id) |
        (historical['away_team_id'] == away_team_id)
    ]
    # Transform to team perspective
    away_team_games = _transform_games_for_team(away_team_games_raw, away_team_id)
    away_features = feature_engineer.compute_team_features(away_team_games, away_team_id)
    
    # Compute head-to-head features
    h2h_features = feature_engineer.compute_head_to_head_features(
        historical, home_team_id, away_team_id
    )
    
    # Get team names - try multiple sources
    home_team_name = ''
    away_team_name = ''
    
    if isinstance(home_team, dict):
        home_team_name = (home_team.get('name') or 
                         home_team.get('teamName') or 
                         (home_team.get('name', {}).get('default') if isinstance(home_team.get('name'), dict) else None) or
                         home_team.get('abbrev') or
                         '')
    
    if isinstance(away_team, dict):
        away_team_name = (away_team.get('name') or 
                         away_team.get('teamName') or 
                         (away_team.get('name', {}).get('default') if isinstance(away_team.get('name'), dict) else None) or
                         away_team.get('abbrev') or
                         '')
    
    # Fallback to team_name_mapping if names are still empty
    if not home_team_name and team_name_mapping and home_team_id in team_name_mapping:
        home_team_name = team_name_mapping[home_team_id]
    if not away_team_name and team_name_mapping and away_team_id in team_name_mapping:
        away_team_name = team_name_mapping[away_team_id]
    
    # Log feature values for debugging (first game only)
    if not hasattr(compute_features_for_upcoming_game, '_logged_first'):
        logger.info(f"Sample features for game {game_id}: home_win_streak={home_features.get('win_streak', 0)}, away_win_streak={away_features.get('win_streak', 0)}")
        logger.info(f"Home team games: {len(home_team_games)}, Away team games: {len(away_team_games)}")
        compute_features_for_upcoming_game._logged_first = True
    
    # Combine features
    feature_row = {
        'game_id': game_id,
        'game_date': game_date_str,
        'home_team_id': home_team_id,
        'away_team_id': away_team_id,
        'home_team_name': home_team_name,
        'away_team_name': away_team_name,
        # Home team features
        **{f'home_{k}': v for k, v in home_features.items()},
        # Away team features
        **{f'away_{k}': v for k, v in away_features.items()},
        # Head-to-head features
        **h2h_features
    }
    
    return pd.DataFrame([feature_row])


def generate_predictions(model, features_df: pd.DataFrame, training_feature_cols: list = None) -> pd.DataFrame:
    """Generate predictions using the trained model"""
    # Exclude non-feature columns (must match training pipeline)
    exclude_cols = ['game_id', 'game_date', 'home_team_id', 'away_team_id', 
                    'home_team_name', 'away_team_name', 'home_team_won']
    
    # Get feature columns - use training feature order if provided
    available_feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    
    if training_feature_cols:
        # Ensure we use the same columns in the same order as training
        feature_cols = [col for col in training_feature_cols if col in available_feature_cols]
        missing_cols = [col for col in training_feature_cols if col not in available_feature_cols]
        extra_cols = [col for col in available_feature_cols if col not in training_feature_cols]
        
        if missing_cols:
            logger.warning(f"Missing feature columns from training: {missing_cols}")
        if extra_cols:
            logger.warning(f"Extra feature columns not in training: {extra_cols}")
        
        # Reorder to match training
        X = features_df[feature_cols].fillna(0)
    else:
        # Fallback: use available columns
        feature_cols = available_feature_cols
        X = features_df[feature_cols].fillna(0)
        logger.warning("No training feature columns provided - using inference columns (may cause mismatch)")
    
    # Log feature statistics for debugging
    logger.info(f"Using {len(feature_cols)} features for prediction")
    logger.debug(f"Feature columns: {feature_cols[:10]}...")  # First 10
    
    # Check if all features are the same (would explain identical predictions)
    feature_variance = X.var()
    zero_variance_features = feature_variance[feature_variance == 0].index.tolist()
    if zero_variance_features:
        logger.warning(f"Features with zero variance (all same values): {len(zero_variance_features)} features")
        logger.warning(f"Zero variance features: {zero_variance_features}")
        # Log the actual values for these features
        for feat in zero_variance_features[:5]:  # First 5
            logger.warning(f"  {feat}: all values = {X[feat].iloc[0]}")
    
    # Get predictions
    probabilities = model.predict_proba(X)[:, 1]  # Probability of home team winning
    
    predictions_df = features_df[['game_id', 'game_date', 'home_team_name', 'away_team_name']].copy()
    predictions_df['home_win_probability'] = probabilities
    predictions_df['away_win_probability'] = 1 - probabilities
    predictions_df['predicted_winner'] = predictions_df.apply(
        lambda row: row['home_team_name'] if row['home_win_probability'] > 0.5 else row['away_team_name'],
        axis=1
    )
    
    return predictions_df


def main():
    """Main inference pipeline execution"""
    logger.info("Starting Inference Pipeline")
    
    # Load configuration
    config = load_config()
    
    # Initialize clients
    api_client = NHLAPIClient(
        base_url=config['nhl_api']['base_url'],
        timeout=config['nhl_api']['timeout']
    )
    
    hopsworks_client = HopsworksClient()
    feature_engineer = FeatureEngineer(
        rolling_window=config['features']['rolling_window']
    )
    
    # Get trained model from Model Registry
    mr_config = config['model_registry']
    model_name = mr_config['model_name']
    model_version = mr_config.get('model_version', None)  # None means latest
    
    if model_version is not None:
        logger.info(f"Loading model: {model_name} version {model_version}")
        model_registry_entry = hopsworks_client.get_model(model_name, version=model_version)
    else:
        logger.info(f"Loading latest version of model: {model_name}")
        model_registry_entry = hopsworks_client.get_latest_model(model_name)
    
    # Download and load model
    model_path = hopsworks_client.download_model(model_registry_entry)
    model = joblib.load(model_path)
    logger.info("Model loaded successfully")
    
    # Fetch upcoming games
    days_ahead = config['features']['prediction_horizon_days']
    logger.info(f"Fetching upcoming games for next {days_ahead} days")
    all_upcoming_games = api_client.get_upcoming_games(days_ahead=days_ahead)
    
    # Filter to only include scheduled games (not final, not cancelled) within date range
    # Use UTC for consistent date comparison (GitHub Actions runs in UTC)
    now_utc = datetime.now(timezone.utc)
    today_utc = now_utc.date()
    end_date = today_utc + timedelta(days=days_ahead)
    
    logger.info(f"Filtering games from {today_utc} to {end_date} (UTC)")
    
    upcoming_games = []
    for game in all_upcoming_games:
        # Check game status - only include scheduled/upcoming games
        game_status = game.get("gameState") or game.get("status", {}).get("detailedState") or game.get("gameStatus")
        # Exclude games that are already finished (FINAL/OFFICIAL)
        is_final = game_status in ["OFF", "FINAL", "Final", "OFFICIAL"]
        is_cancelled = game_status in ["CANCELLED", "POSTPONED", "DELAYED"]
        
        # Exclude games that are already final or cancelled
        if is_final or is_cancelled:
            logger.debug(f"Skipping game {game.get('gamePk', 'unknown')}: status={game_status}")
            continue
        
        # Check game date is in the future and within range
        game_date_str = game.get('startTimeUTC') or game.get('gameDate') or game.get('date')
        if game_date_str:
            try:
                # Parse game date (assume UTC if timezone not specified)
                game_datetime = pd.to_datetime(game_date_str, utc=True)
                game_date = game_datetime.date()
                
                # Exclude games from past dates (before today)
                if game_date < today_utc:
                    logger.debug(f"Skipping past game {game.get('gamePk', 'unknown')}: date={game_date} < today={today_utc}")
                    continue
                
                # Include if within date range (today to end_date)
                if game_date <= end_date:
                    upcoming_games.append(game)
                else:
                    logger.debug(f"Skipping future game {game.get('gamePk', 'unknown')}: date={game_date} > end_date={end_date}")
            except Exception as e:
                logger.debug(f"Could not parse game date {game_date_str}: {e}")
                continue
        else:
            logger.debug(f"Skipping game {game.get('gamePk', 'unknown')}: no date found")
    
    logger.info(f"Found {len(upcoming_games)} upcoming games (filtered from {len(all_upcoming_games)} total)")
    
    if len(upcoming_games) == 0:
        logger.warning("No upcoming games found in the specified date range")
        return
    
    # Fetch historical data for feature computation (use inference_historical_days from config)
    inference_days = config['features'].get('inference_historical_days', 90)
    logger.info(f"Fetching {inference_days} days of historical data for feature computation")
    historical_games = fetch_historical_data_for_features(api_client, days_back=inference_days)
    logger.info(f"Fetched {len(historical_games)} historical games for feature computation")
    
    # Build team name mapping from upcoming games
    team_name_mapping = build_team_name_mapping(upcoming_games, historical_games)
    logger.info(f"Built team name mapping with {len(team_name_mapping)} teams")
    if len(team_name_mapping) > 0:
        logger.debug(f"Sample team mappings: {dict(list(team_name_mapping.items())[:5])}")
    
    # Compute features for each upcoming game
    all_features = []
    for game in upcoming_games:
        features = compute_features_for_upcoming_game(
            game, historical_games, feature_engineer, team_name_mapping
        )
        if not features.empty:
            all_features.append(features)
    
    if len(all_features) == 0:
        logger.error("No features computed for any games. Check feature computation logic.")
        return
    
    features_df = pd.concat(all_features, ignore_index=True)
    
    # Remove duplicates based on game_id (same game might appear multiple times from API)
    initial_count = len(features_df)
    features_df = features_df.drop_duplicates(subset=['game_id'], keep='first')
    if len(features_df) < initial_count:
        logger.info(f"Removed {initial_count - len(features_df)} duplicate games (kept {len(features_df)} unique games)")
    
    logger.info(f"Computed features for {len(features_df)} unique games")
    
    # Log sample features to debug identical probabilities
    if len(features_df) > 0:
        feature_cols = [col for col in features_df.columns if col not in 
                       ['game_id', 'game_date', 'home_team_id', 'away_team_id', 
                        'home_team_name', 'away_team_name']]
        logger.info(f"Feature columns: {len(feature_cols)}")
        # Check if features are all the same
        sample_features = features_df[feature_cols].iloc[0]
        logger.info(f"Sample features (first game): {dict(sample_features.head(10))}")
        # Check feature variance
        for col in feature_cols[:5]:  # Check first 5 features
            unique_vals = features_df[col].nunique()
            logger.debug(f"Feature {col}: {unique_vals} unique values, mean={features_df[col].mean():.4f}")
    
    # Get feature column names from model if available (XGBoost stores them)
    training_feature_cols = None
    try:
        if hasattr(model, 'feature_names_in_'):
            training_feature_cols = list(model.feature_names_in_)
            logger.info(f"Retrieved {len(training_feature_cols)} feature names from model")
        elif hasattr(model, 'get_booster'):
            # XGBoost sklearn wrapper
            booster = model.get_booster()
            if hasattr(booster, 'feature_names'):
                training_feature_cols = booster.feature_names
                logger.info(f"Retrieved {len(training_feature_cols)} feature names from XGBoost booster")
    except Exception as e:
        logger.warning(f"Could not retrieve feature names from model: {e}")
    
    # Generate predictions
    logger.info("Generating predictions...")
    predictions_df = generate_predictions(model, features_df, training_feature_cols)
    
    # Log prediction statistics
    if len(predictions_df) > 0:
        logger.info(f"Prediction statistics:")
        logger.info(f"  Home win prob - min: {predictions_df['home_win_probability'].min():.4f}, "
                   f"max: {predictions_df['home_win_probability'].max():.4f}, "
                   f"mean: {predictions_df['home_win_probability'].mean():.4f}")
        logger.info(f"  Unique probabilities: {predictions_df['home_win_probability'].nunique()}")
    
    # Remove duplicates from predictions (in case they still exist)
    initial_pred_count = len(predictions_df)
    predictions_df = predictions_df.drop_duplicates(subset=['game_id'], keep='first')
    if len(predictions_df) < initial_pred_count:
        logger.info(f"Removed {initial_pred_count - len(predictions_df)} duplicate predictions")
    
    # Save predictions (can be stored in feature store or database for dashboard)
    output_path = Path(__file__).parent.parent.parent / "data" / "predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(predictions_df)} unique predictions to {output_path}")
    
    # Print predictions
    print("\n" + "="*80)
    print("UPCOMING GAME PREDICTIONS")
    print("="*80)
    for _, pred in predictions_df.iterrows():
        print(f"\n{pred['away_team_name']} @ {pred['home_team_name']}")
        print(f"  Date: {pred['game_date']}")
        print(f"  Home Win Probability: {pred['home_win_probability']:.2%}")
        print(f"  Away Win Probability: {pred['away_win_probability']:.2%}")
        print(f"  Predicted Winner: {pred['predicted_winner']}")
    
    logger.info("Inference Pipeline completed successfully")


if __name__ == "__main__":
    main()

