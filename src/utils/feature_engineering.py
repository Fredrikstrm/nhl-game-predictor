"""
Feature Engineering Utilities
Computes rolling features for NHL games
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Engineers features for NHL game prediction"""
    
    def __init__(self, rolling_window: int = 10):
        self.rolling_window = rolling_window
    
    def compute_team_features(self, team_games: pd.DataFrame, team_id: int) -> Dict:
        """
        Compute rolling features for a team based on recent games
        Args:
            team_games: DataFrame with team's historical games
            team_id: Team identifier
        Returns:
            Dictionary of computed features
        """
        if len(team_games) == 0:
            return self._get_default_features()
        
        # Sort by date
        team_games = team_games.sort_values('game_date')
        
        # Get recent games (last N games)
        recent_games = team_games.tail(self.rolling_window)
        
        features = {}
        
        # Win/Loss streaks
        features['win_streak'] = self._compute_win_streak(team_games)
        features['loss_streak'] = self._compute_loss_streak(team_games)
        
        # Recent performance
        features['recent_wins'] = recent_games['won'].sum() if 'won' in recent_games.columns else 0
        features['recent_win_pct'] = features['recent_wins'] / len(recent_games) if len(recent_games) > 0 else 0
        
        # Goal differential
        if 'goals_for' in recent_games.columns and 'goals_against' in recent_games.columns:
            features['recent_goal_differential'] = (
                recent_games['goals_for'].sum() - recent_games['goals_against'].sum()
            )
            features['avg_goals_for'] = recent_games['goals_for'].mean()
            features['avg_goals_against'] = recent_games['goals_against'].mean()
        else:
            features['recent_goal_differential'] = 0
            features['avg_goals_for'] = 0
            features['avg_goals_against'] = 0
        
        # Special teams (if available)
        # NOTE: These features default to 0.0 because the NHL API endpoints we use
        # (get_daily_scores) don't provide power play/penalty kill statistics.
        # To improve model performance, these would need to be fetched from different
        # API endpoints or computed from boxscore data if available.
        features['power_play_pct'] = self._compute_power_play_pct(recent_games)
        features['penalty_kill_pct'] = self._compute_penalty_kill_pct(recent_games)
        
        # Home/Away performance
        if 'is_home' in recent_games.columns:
            home_games = recent_games[recent_games['is_home'] == 1]
            away_games = recent_games[recent_games['is_home'] == 0]
            
            features['home_win_pct'] = home_games['won'].mean() if len(home_games) > 0 else 0
            features['away_win_pct'] = away_games['won'].mean() if len(away_games) > 0 else 0
        else:
            features['home_win_pct'] = 0
            features['away_win_pct'] = 0
        
        return features
    
    def compute_goalie_features(self, goalie_stats: pd.DataFrame) -> Dict:
        """
        Compute goalie performance features
        
        NOTE: Currently returns default values because the NHL API endpoints we use
        (get_daily_scores) don't provide goalie statistics. To improve model performance,
        goalie stats would need to be fetched from boxscore endpoints or team stats endpoints.
        """
        if len(goalie_stats) == 0:
            return {
                'goalie_save_pct': 0.9,  # Default NHL average
                'goalie_goals_against_avg': 2.5,  # Default NHL average
                'recent_saves': 0
            }
        
        recent_stats = goalie_stats.tail(self.rolling_window)
        
        return {
            'goalie_save_pct': recent_stats['save_pct'].mean() if 'save_pct' in recent_stats.columns else 0.9,
            'goalie_goals_against_avg': recent_stats['goals_against_avg'].mean() if 'goals_against_avg' in recent_stats.columns else 2.5,
            'recent_saves': recent_stats['saves'].sum() if 'saves' in recent_stats.columns else 0
        }
    
    def compute_head_to_head_features(
        self, 
        historical_games: pd.DataFrame, 
        team1_id: int, 
        team2_id: int
    ) -> Dict:
        """Compute head-to-head features between two teams"""
        if len(historical_games) == 0:
            return {
                'h2h_team1_wins': 0,
                'h2h_team2_wins': 0,
                'h2h_team1_win_pct': 0.5
            }
        
        # Filter games between these two teams
        h2h_games = historical_games[
            ((historical_games['home_team_id'] == team1_id) & (historical_games['away_team_id'] == team2_id)) |
            ((historical_games['home_team_id'] == team2_id) & (historical_games['away_team_id'] == team1_id))
        ]
        
        if len(h2h_games) == 0:
            return {
                'h2h_team1_wins': 0,
                'h2h_team2_wins': 0,
                'h2h_team1_win_pct': 0.5
            }
        
        # Count wins for each team
        # team1 wins when: (team1 is home and home won) OR (team1 is away and away won)
        team1_wins = len(h2h_games[
            ((h2h_games['home_team_id'] == team1_id) & (h2h_games['home_team_won'] == 1)) |
            ((h2h_games['away_team_id'] == team1_id) & (h2h_games['home_team_won'] == 0))
        ])
        
        return {
            'h2h_team1_wins': team1_wins,
            'h2h_team2_wins': len(h2h_games) - team1_wins,
            'h2h_team1_win_pct': team1_wins / len(h2h_games) if len(h2h_games) > 0 else 0.5
        }
    
    def _compute_win_streak(self, games: pd.DataFrame) -> int:
        """Compute current win streak"""
        if len(games) == 0 or 'won' not in games.columns:
            return 0
        
        streak = 0
        for won in reversed(games['won'].values):
            if won == 1:
                streak += 1
            else:
                break
        return streak
    
    def _compute_loss_streak(self, games: pd.DataFrame) -> int:
        """Compute current loss streak"""
        if len(games) == 0 or 'won' not in games.columns:
            return 0
        
        streak = 0
        for won in reversed(games['won'].values):
            if won == 0:
                streak += 1
            else:
                break
        return streak
    
    def _compute_power_play_pct(self, games: pd.DataFrame) -> float:
        """Compute power play percentage"""
        if 'power_play_goals' not in games.columns or 'power_play_opportunities' not in games.columns:
            return 0.0
        
        total_goals = games['power_play_goals'].sum()
        total_opportunities = games['power_play_opportunities'].sum()
        
        return total_goals / total_opportunities if total_opportunities > 0 else 0.0
    
    def _compute_penalty_kill_pct(self, games: pd.DataFrame) -> float:
        """Compute penalty kill percentage"""
        if 'penalty_kill_goals_against' not in games.columns or 'penalty_kill_opportunities' not in games.columns:
            return 0.0
        
        goals_against = games['penalty_kill_goals_against'].sum()
        opportunities = games['penalty_kill_opportunities'].sum()
        
        return 1 - (goals_against / opportunities) if opportunities > 0 else 0.0
    
    def _get_default_features(self) -> Dict:
        """Return default feature values when no data is available"""
        return {
            'win_streak': 0,
            'loss_streak': 0,
            'recent_wins': 0,
            'recent_win_pct': 0.5,
            'recent_goal_differential': 0,
            'avg_goals_for': 2.5,
            'avg_goals_against': 2.5,
            'power_play_pct': 0.2,
            'penalty_kill_pct': 0.8,
            'home_win_pct': 0.5,
            'away_win_pct': 0.5
        }

