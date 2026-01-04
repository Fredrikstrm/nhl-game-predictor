"""
Feature Pipeline
Fetches NHL data from API and computes features, then stores in Hopsworks Feature Store
"""
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

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


def fetch_historical_games(api_client: NHLAPIClient, days_back: int = 90) -> pd.DataFrame:
    """Fetch historical game data with scores"""
    logger.info(f"Fetching historical games from last {days_back} days")
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)
    
    logger.info(f"Date range: {start_date} to {end_date}")
    
    # Use get_daily_scores instead of get_schedule to get actual scores
    games = []
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        try:
            daily_games = api_client.get_daily_scores(date=date_str)
            games.extend(daily_games)
            logger.debug(f"Fetched {len(daily_games)} games for {date_str}")
        except Exception as e:
            logger.warning(f"Error fetching scores for {date_str}: {e}")
        current_date += timedelta(days=1)
    
    logger.info(f"API returned {len(games)} raw game objects")
    
    # Convert to DataFrame
    games_data = []
    skipped_not_final = 0
    skipped_missing_data = 0
    skipped_out_of_range = 0
    seen_game_ids = set()
    
    for game in games:
        game_status = game.get("gameState")
        is_final = game_status in ["OFF", "FINAL", "Final", "OFFICIAL"]
        
        if not is_final:
            skipped_not_final += 1
            continue
        
        # Extract team information (API structure: homeTeam/awayTeam objects)
        home_team = game.get('homeTeam', {})
        away_team = game.get('awayTeam', {})
        
        # Get scores
        home_score = home_team.get('score')
        away_score = away_team.get('score')
        
        # Get team IDs
        home_team_id = home_team.get('id')
        away_team_id = away_team.get('id')
        
        # Get game ID and date
        game_id = game.get('id')
        game_date_str = game.get('startTimeUTC') or game.get('gameDate')
        
        # Parse and validate game date
        try:
            if game_date_str:
                # Parse the date
                if isinstance(game_date_str, str):
                    if 'T' in game_date_str:
                        game_date = pd.to_datetime(game_date_str).date()
                    else:
                        game_date = pd.to_datetime(game_date_str).date()
                else:
                    game_date = pd.to_datetime(game_date_str).date()
                
                if game_date < start_date or game_date > end_date:
                    skipped_out_of_range += 1
                    continue
            else:
                skipped_missing_data += 1
                continue
        except Exception as e:
            logger.debug(f"Could not parse game date {game_date_str}: {e}")
            skipped_missing_data += 1
            continue
        
        # Check for duplicates
        if game_id in seen_game_ids:
            continue
        seen_game_ids.add(game_id)
        
        if game_id and home_team_id and away_team_id:
            home_score_int = int(home_score) if home_score is not None and home_score != '' else 0
            away_score_int = int(away_score) if away_score is not None and away_score != '' else 0
            
            # Determine winner
            if home_score_int > away_score_int:
                home_team_won = 1
            elif away_score_int > home_score_int:
                home_team_won = 0
            else:
                # Tie game - log it and mark as 0
                logger.debug(f"Tie game detected: game_id={game_id}, home_score={home_score_int}, away_score={away_score_int}")
                home_team_won = 0  # Could also be handled differently
            
            games_data.append({
                'game_id': game_id,
                'game_date': game_date_str,
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'home_team_score': home_score_int,
                'away_team_score': away_score_int,
                'home_team_won': home_team_won
            })
        else:
            skipped_missing_data += 1
    
    logger.info(f"Processed games: {len(games_data)} valid, {skipped_not_final} not final, {skipped_missing_data} missing data, {skipped_out_of_range} out of date range")
    
    df = pd.DataFrame(games_data)
    if len(df) > 0:
        logger.info(f"Date range in final data: {pd.to_datetime(df['game_date']).min().date()} to {pd.to_datetime(df['game_date']).max().date()}")
        logger.info(f"Total unique games: {len(df)}")
        
        # Log score and outcome statistics
        logger.info(f"Score statistics - Home: min={df['home_team_score'].min()}, max={df['home_team_score'].max()}, mean={df['home_team_score'].mean():.2f}")
        logger.info(f"Score statistics - Away: min={df['away_team_score'].min()}, max={df['away_team_score'].max()}, mean={df['away_team_score'].mean():.2f}")
        
        # Log outcome distribution
        outcome_counts = df['home_team_won'].value_counts().to_dict()
        outcome_pct = (df['home_team_won'].value_counts(normalize=True) * 100).to_dict()
        logger.info(f"Outcome distribution: {outcome_counts} ({outcome_pct.get(0, 0):.1f}% away wins, {outcome_pct.get(1, 0):.1f}% home wins)")
        
        # Check for potential issues
        if df['home_team_score'].sum() == 0 and df['away_team_score'].sum() == 0:
            logger.warning("WARNING: All scores are 0! The API might not be returning scores from the schedule endpoint.")
        if len(outcome_counts) == 1:
            logger.warning(f"WARNING: Only one outcome class found ({list(outcome_counts.keys())[0]}). This will cause training issues.")
    
    return df


def _transform_games_for_team(team_games: pd.DataFrame, team_id: int) -> pd.DataFrame:
    """
    Transform games DataFrame to team perspective (add won, goals_for, goals_against, is_home columns)
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
    team_df['won'] = (
        (team_df['is_home'] == 1) & (team_df['home_team_won'] == 1)
    ) | (
        (team_df['is_away'] == 1) & (team_df['home_team_won'] == 0)
    )
    team_df['won'] = team_df['won'].astype(int)
    
    return team_df


def compute_game_features(
    games_df: pd.DataFrame,
    feature_engineer: FeatureEngineer,
    api_client: NHLAPIClient
) -> pd.DataFrame:
    """Compute features for each game"""
    logger.info("Computing features for games")
    
    features_list = []
    
    for _, game in games_df.iterrows():
        game_id = game['game_id']
        home_team_id = game['home_team_id']
        away_team_id = game['away_team_id']
        game_date = pd.to_datetime(game['game_date']).date()
        
        # Get historical games before this game date
        historical_games = games_df[
            pd.to_datetime(games_df['game_date']).dt.date < game_date
        ]
        
        # Compute home team features
        home_team_games_raw = historical_games[
            (historical_games['home_team_id'] == home_team_id) |
            (historical_games['away_team_id'] == home_team_id)
        ]
        # Transform to team perspective
        home_team_games = _transform_games_for_team(home_team_games_raw, home_team_id)
        home_features = feature_engineer.compute_team_features(home_team_games, home_team_id)
        
        # Compute away team features
        away_team_games_raw = historical_games[
            (historical_games['home_team_id'] == away_team_id) |
            (historical_games['away_team_id'] == away_team_id)
        ]
        # Transform to team perspective
        away_team_games = _transform_games_for_team(away_team_games_raw, away_team_id)
        away_features = feature_engineer.compute_team_features(away_team_games, away_team_id)
        
        # Compute head-to-head features
        h2h_features = feature_engineer.compute_head_to_head_features(
            historical_games, home_team_id, away_team_id
        )
        
        # Combine all features
        feature_row = {
            'game_id': game_id,
            'game_date': game['game_date'],
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'home_team_won': game['home_team_won'],
            **{f'home_{k}': v for k, v in home_features.items()},
            **{f'away_{k}': v for k, v in away_features.items()},
            **h2h_features
        }
        
        features_list.append(feature_row)
    
    result_df = pd.DataFrame(features_list)
    return result_df


def main():
    """Main feature pipeline execution"""
    logger.info("Starting Feature Pipeline")
    
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
    
    # Fetch historical games
    training_days = config['features'].get('training_data_days', 365)
    logger.info(f"Fetching {training_days} days of historical data for training")
    games_df = fetch_historical_games(api_client, days_back=training_days)
    logger.info(f"Fetched {len(games_df)} historical games")
    
    if len(games_df) == 0:
        logger.warning("No games found. Exiting.")
        return
    
    # Compute features
    features_df = compute_game_features(games_df, feature_engineer, api_client)
    logger.info(f"Computed features for {len(features_df)} games")
    
    # Get or create feature group
    fg_config = config['feature_store']
    feature_group = hopsworks_client.get_or_create_feature_group(
        name=fg_config['feature_group_name'],
        version=fg_config['feature_group_version'],
        description=fg_config.get('feature_group_description', ''),
        primary_key=['game_id']
    )
    
    # Insert features into feature store
    hopsworks_client.insert_features(feature_group, features_df)
    
    logger.info("Feature Pipeline completed successfully")


if __name__ == "__main__":
    main()

