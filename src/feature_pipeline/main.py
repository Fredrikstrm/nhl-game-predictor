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
    # The schedule endpoint may not include scores for completed games
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
    seen_game_ids = set()  # Track duplicates
    
    for game in games:
        # Check if game is final - API structure may vary
        game_status = game.get("gameState") or game.get("status", {}).get("detailedState") or game.get("gameStatus")
        is_final = game_status in ["OFF", "FINAL", "Final", "OFFICIAL"]
        
        if not is_final:
            skipped_not_final += 1
            continue
        
        # Extract team information - handle different response structures
        home_team = game.get('homeTeam') or game.get('teams', {}).get('home', {}).get('team', {})
        away_team = game.get('awayTeam') or game.get('teams', {}).get('away', {}).get('team', {})
        
        # Get scores - handle different response structures
        # Try multiple possible score fields
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
        
        # Log first few games to debug score extraction
        if len(games_data) < 3:
            logger.debug(f"Game {game.get('id', 'unknown')} score extraction - home_score: {home_score}, away_score: {away_score}")
            logger.debug(f"Game keys: {list(game.keys())}")
            if 'homeTeam' in game:
                logger.debug(f"homeTeam keys: {list(game['homeTeam'].keys()) if isinstance(game['homeTeam'], dict) else 'not a dict'}")
            if 'awayTeam' in game:
                logger.debug(f"awayTeam keys: {list(game['awayTeam'].keys()) if isinstance(game['awayTeam'], dict) else 'not a dict'}")
        
        # Get team IDs
        home_team_id = home_team.get('id') if isinstance(home_team, dict) else None
        away_team_id = away_team.get('id') if isinstance(away_team, dict) else None
        
        # Get game ID and date
        game_id = game.get('id') or game.get('gamePk') or game.get('gameId')
        game_date_str = game.get('startTimeUTC') or game.get('gameDate') or game.get('date')
        
        # Parse and validate game date
        try:
            if game_date_str:
                # Parse the date (handle different formats)
                if isinstance(game_date_str, str):
                    # Try ISO format first (2024-01-15T19:00:00Z)
                    if 'T' in game_date_str:
                        game_date = pd.to_datetime(game_date_str).date()
                    else:
                        game_date = pd.to_datetime(game_date_str).date()
                else:
                    game_date = pd.to_datetime(game_date_str).date()
                
                # Filter by date range
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
            # Convert scores to integers, defaulting to 0 if None
            home_score_int = int(home_score) if home_score is not None and home_score != '' else 0
            away_score_int = int(away_score) if away_score is not None and away_score != '' else 0
            
            # Determine winner (handle ties as 0 for now, but log them)
            if home_score_int > away_score_int:
                home_team_won = 1
            elif away_score_int > home_score_int:
                home_team_won = 0
            else:
                # Tie game - log it and mark as 0 (away team didn't win, but neither did home)
                logger.debug(f"Tie game detected: game_id={game_id}, home_score={home_score_int}, away_score={away_score_int}")
                home_team_won = 0  # Could also be handled differently
            
            games_data.append({
                'game_id': game_id,
                'game_date': game_date_str,  # Keep original string format
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
    # Team wins if: (team is home and home_team_won=1) OR (team is away and home_team_won=0)
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
    
    # #region agent log
    import json
    try:
        with open('/Users/fredrikstrom/Documents/KTH_Dokument/Scalable ML/project/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"feature_pipeline/main.py:compute_game_features","message":"Function entry","data":{"games_df_len":len(games_df),"columns":list(games_df.columns)},"timestamp":int(__import__('time').time()*1000)})+'\n')
    except: pass
    # #endregion
    
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
            # Home team features (prefix with home_)
            **{f'home_{k}': v for k, v in home_features.items()},
            # Away team features (prefix with away_)
            **{f'away_{k}': v for k, v in away_features.items()},
            # Head-to-head features
            **h2h_features
        }
        
        features_list.append(feature_row)
    
    result_df = pd.DataFrame(features_list)
    
    # #region agent log
    import json
    try:
        with open('/Users/fredrikstrom/Documents/KTH_Dokument/Scalable ML/project/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"feature_pipeline/main.py:compute_game_features","message":"Function exit","data":{"result_df_is_none":result_df is None,"result_df_len":len(result_df) if result_df is not None else 0,"result_df_type":str(type(result_df)),"result_df_columns":list(result_df.columns) if result_df is not None else None},"timestamp":int(__import__('time').time()*1000)})+'\n')
    except: pass
    # #endregion
    
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
    # Get training data window from config (default to 365 days for full season)
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
    
    # #region agent log
    import json
    try:
        with open('/Users/fredrikstrom/Documents/KTH_Dokument/Scalable ML/project/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"feature_pipeline/main.py:main","message":"Before get_or_create_feature_group","data":{"features_df_is_none":features_df is None,"features_df_len":len(features_df) if features_df is not None else 0},"timestamp":int(__import__('time').time()*1000)})+'\n')
    except: pass
    # #endregion
    
    # Get or create feature group
    fg_config = config['feature_store']
    feature_group = hopsworks_client.get_or_create_feature_group(
        name=fg_config['feature_group_name'],
        version=fg_config['feature_group_version'],
        description=fg_config.get('feature_group_description', ''),
        primary_key=['game_id']
    )
    
    # #region agent log
    try:
        with open('/Users/fredrikstrom/Documents/KTH_Dokument/Scalable ML/project/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"feature_pipeline/main.py:main","message":"Before insert_features","data":{"feature_group_is_none":feature_group is None,"feature_group_type":str(type(feature_group)) if feature_group is not None else "None","features_df_is_none":features_df is None},"timestamp":int(__import__('time').time()*1000)})+'\n')
    except: pass
    # #endregion
    
    # Insert features into feature store
    hopsworks_client.insert_features(feature_group, features_df)
    
    logger.info("Feature Pipeline completed successfully")


if __name__ == "__main__":
    main()

