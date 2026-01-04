"""
NHL API Client
Fetches data from the NHL Web API (api-web.nhle.com)
Based on official NHL API documentation
"""
import requests
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta, timezone
import logging

logger = logging.getLogger(__name__)


class NHLAPIClient:
    """Client for interacting with the NHL Web API"""
    
    def __init__(self, base_url: str = "https://api-web.nhle.com", timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.max_redirects = 5
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None, follow_redirects: bool = False) -> Dict:
        """Make a request to the NHL API with retry logic"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            if follow_redirects:
                response = self.session.get(url, params=params, timeout=self.timeout, allow_redirects=True)
            else:
                response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            raise
    
    def get_schedule(self, start_date: str = None, end_date: str = None) -> List[Dict]:
        """
        Get game schedule for a date range or current schedule
        Args:
            start_date: Start date in YYYY-MM-DD format (optional)
            end_date: End date in YYYY-MM-DD format (optional)
        Returns:
            List of game dictionaries
        """
        games = []
        
        if start_date and end_date:
            # Get schedule for date range
            current_date = datetime.strptime(start_date, "%Y-%m-%d").date()
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
            
            while current_date <= end_date_obj:
                date_str = current_date.strftime("%Y-%m-%d")
                endpoint = f"v1/schedule/{date_str}"
                data = self._make_request(endpoint)
                
                if "gameWeek" in data and len(data["gameWeek"]) > 0:
                    for day in data["gameWeek"]:
                        if "games" in day:
                            games.extend(day["games"])
                
                current_date += timedelta(days=1)
        else:
            # Get current schedule
            endpoint = "v1/schedule/now"
            data = self._make_request(endpoint, follow_redirects=True)
            
            if "gameWeek" in data:
                for day in data["gameWeek"]:
                    if "games" in day:
                        games.extend(day["games"])
        
        return games
    
    def get_upcoming_games(self, days_ahead: int = 3) -> List[Dict]:
        """
        Get upcoming games for the next N days
        Uses UTC for consistent date calculation across timezones
        """
        # Use UTC to ensure consistency (important for GitHub Actions which runs in UTC)
        today = datetime.now(timezone.utc).date()
        end_date = today + timedelta(days=days_ahead)
        
        return self.get_schedule(
            start_date=today.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
    
    def get_daily_scores(self, date: str = None) -> List[Dict]:
        """
        Get daily scores for a specific date or current day
        Args:
            date: Date in YYYY-MM-DD format (optional, defaults to today)
        Returns:
            List of game score dictionaries
        """
        if date:
            endpoint = f"v1/score/{date}"
        else:
            endpoint = "v1/score/now"
        
        data = self._make_request(endpoint, follow_redirects=True)
        
        games = []
        if "games" in data:
            games = data["games"]
        elif "gameWeek" in data:
            for day in data["gameWeek"]:
                if "games" in day:
                    games.extend(day["games"])
        
        return games
    
    def get_game_boxscore(self, game_id: int) -> Dict:
        """
        Get boxscore information for a specific game
        Args:
            game_id: Game ID (e.g., 2023020204)
        Returns:
            Boxscore dictionary
        """
        endpoint = f"v1/gamecenter/{game_id}/boxscore"
        return self._make_request(endpoint)
    
    def get_game_landing(self, game_id: int) -> Dict:
        """
        Get landing information for a specific game
        Args:
            game_id: Game ID
        Returns:
            Game landing dictionary
        """
        endpoint = f"v1/gamecenter/{game_id}/landing"
        return self._make_request(endpoint)
    
    def get_standings(self, date: str = None) -> Dict:
        """
        Get league standings for a specific date or current standings
        Args:
            date: Date in YYYY-MM-DD format (optional)
        Returns:
            Standings dictionary
        """
        if date:
            endpoint = f"v1/standings/{date}"
        else:
            endpoint = "v1/standings/now"
        
        return self._make_request(endpoint, follow_redirects=True)
    
    def get_team_stats(self, team_code: str, season: Optional[str] = None, game_type: int = 2) -> Dict:
        """
        Get team statistics
        Args:
            team_code: Team ID
            season: Season in YYYY format
            game_type: Game type 
        Returns:
            Team stats dictionary
        """
        if season:
            endpoint = f"v1/club-stats/{team_code}/{season}/{game_type}"
        else:
            endpoint = f"v1/club-stats/{team_code}/now"
        
        return self._make_request(endpoint, follow_redirects=True)
    
    def get_team_roster(self, team_code: str, season: Optional[str] = None) -> Dict:
        """
        Get team roster
        Args:
            team_code: Three-letter team code
            season: Season in YYYYMMYY format (optional)
        Returns:
            Roster dictionary
        """
        if season:
            endpoint = f"v1/roster/{team_code}/{season}"
        else:
            endpoint = f"v1/roster/{team_code}/current"
        
        return self._make_request(endpoint, follow_redirects=True)
    
    def get_scoreboard(self) -> Dict:
        """Get current scoreboard"""
        endpoint = "v1/scoreboard/now"
        return self._make_request(endpoint, follow_redirects=True)

