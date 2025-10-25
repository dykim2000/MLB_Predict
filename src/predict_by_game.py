import requests
import pandas as pd
import os
import re

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")

pitching_2024 = pd.read_csv(os.path.join(data_path, "pitching_stats_2024.csv"))
pitching_2025 = pd.read_csv(os.path.join(data_path, "pitching_stats_2025.csv"))
team_batting_2024 = pd.read_csv(os.path.join(data_path, "team_batting_stats_2024.csv"))
team_batting_2025 = pd.read_csv(os.path.join(data_path, "team_batting_stats_2024.csv"))
game_logs_2024 = pd.read_csv(os.path.join(data_path, "game_logs_2024.csv"))
game_logs_2025 = pd.read_csv(os.path.join(data_path, "game_logs_2025.csv"))

def get_pitching_stats(year, pitcher_name):
    if year == 2024:
        df = pitching_2024
    else:
        df = pitching_2025
    
    df["Player_clean"] = df["Player"].str.replace(r"[*#]", "", regex=True).str.strip() # Getting Rid of * after player name
    pitcher_row = df[df['Player_clean'] == pitcher_name]
    if pitcher_row.empty:
        return None
    stats = pitcher_row.iloc[0]
    
    return {
        'ERA' : stats.get('ERA'),
        'ER' : stats.get('ER'),
        'R' : stats.get('R'),
        'SO' : stats.get('SO'),
        'BB' : stats.get('BB'),
        'SO/BB' : stats.get('SO/BB'),
        'WHIP' : stats.get('WHIP')
    }
    
def get_batting_stats(year, team_name):
    if year == 2024:
        df = team_batting_2024
    else:
        df = team_batting_2025
    
    batting_row = df[df['Tm'] == team_name]
    stats = batting_row.iloc[0]
    
    return {
        'OBP' : stats.get('OBP'),
        'SLG' : stats.get('SLG'),
        'HR' : stats.get('HR'),
        'R/G' : stats.get('R/G'),
        'BB_batting' : stats.get('BB'),
        'SO_batting' : stats.get('SO'),
        'IBB' : stats.get('IBB')
    }
def safe_float(v):
    try:
        return float(v)
    except:
        return None
    
def extract_ops(player_obj):
    batting = player_obj.get("seasonStats", {}).get("batting", {})
    ops = safe_float(batting.get("ops"))
    if ops is not None:
        return ops
    return None
    
def get_starters(players_dict):
    starters = []
    for p in players_dict.values():
        if "battingOrder" in p:
            starters.append(p)
    starters.sort(key=lambda x : int(x.get("battingOrder", "999"))) #999 = Fail Code
    return starters

def lineup_score(gameid):
    url = f"https://statsapi.mlb.com/api/v1/game/{gameid}/boxscore"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return None
    data = r.json()

    teams = data.get("teams", {})
    home_players = teams.get("home", {}).get("players", {})
    away_players = teams.get("away", {}).get("players", {})

    home_ops = []
    for p in get_starters(home_players):
        ops_val = extract_ops(p)
        if ops_val is not None:
            home_ops.append(ops_val)

    away_ops = []
    for p in get_starters(away_players):
        ops_val = extract_ops(p)
        if ops_val is not None:
            away_ops.append(ops_val)

    return {
        "game_id" : gameid,
        "home_lineup_score" : sum(home_ops)/len(home_ops) if home_ops else None,
        "away_lineup_score" : sum(away_ops)/len(away_ops) if home_ops else None,
        "home_n_starters" : len(home_ops),
        "away_n_starters" : len(away_ops)
    }
    
def get_last_7(game_id, year):
    if year == 2024:
        df = game_logs_2024.copy()
    else:
        df = game_logs_2025.copy()
    
    df["game_date"] = pd.to_datetime(df["game_date"])
    this_game = df[df["game_id"] == game_id]
    
    if this_game.empty:
        return None
    
    home_team = this_game.iloc[0]["home_name"]
    away_team = this_game.iloc[0]["away_name"]
    game_date = this_game.iloc[0]["game_date"]
    def calc_last_7(team, game_date):
        team_games =df[((df["home_name"] == team) | (df["away_name"] == team)) & (df["game_date"] < game_date)] .sort_values("game_date", ascending=False)
        if team_games.empty:
            return None
        team_games["team_win"] = (
            ((team_games["home_name"] == team) & (team_games["home_score"] > team_games["away_score"])) |
            ((team_games["away_name"] == team) & (team_games["away_score"] > team_games["home_score"]))
        ).astype(int)
        #print(team_games[["game_id", "game_date", "home_name", "away_name", "home_score", "away_score", "team_win"]])
        win = team_games.iloc[0]["team_win"]
        streak = -1 if win == 0 else 1
        for i, row in team_games.iloc[1:].iterrows():
            if row["team_win"] == win:
                if streak > 0:
                    streak += 1
                else:
                    streak -= 1
            else:
                break
        return team_games["team_win"].head(7).mean(), streak
    
    home_last7, home_streak = calc_last_7(home_team, game_date)
    away_last7, away_streak = calc_last_7(away_team, game_date)
    
    return {
        "home_last7_win_pct" : home_last7,
        "home_streak" : home_streak,
        "away_last7_win_pct" : away_last7,
        "away_streak" : away_streak
    }

def fetch_game_info(game_id):
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
    response =requests.get(url)
    
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch data for game ID {game_id}. Status code: {response.status_code}")
    
    data = response.json()
    game_data = data.get("gameData", {})
    linescore = data.get("liveData", {}).get("linescore", {}).get("teams", {})
    home_team = game_data["teams"]["home"]["name"]
    away_team = game_data["teams"]["away"]["name"]
    year = int(game_data.get("game", {}).get("season"))
    weather = data.get("gameData", {}).get("weather", {})
    
    home_pitcher = game_data.get("probablePitchers", {}).get("home", {}).get("fullName")
    away_pitcher = game_data.get("probablePitchers", {}).get("away", {}).get("fullName")
    
    
    home_pitching_stats = get_pitching_stats(year, home_pitcher)
    away_pitching_stats = get_pitching_stats(year, away_pitcher)
    
    home_batting_stats = get_batting_stats(year, home_team)
    away_batting_stats = get_batting_stats(year, away_team)
    
        
    #Game Info
    row = {
        "game_id" : game_id,
        "home_team" : home_team,
        "away_team" : away_team,
        "home_probable_pitcher" : home_pitcher,
        "away_probable_pitcher" : away_pitcher,
        "home_score" : linescore.get("home", {}).get("runs"),
        "away_score" : linescore.get("away", {}).get("runs"),
    }
    
    #Pitching Stats
    if home_pitching_stats:
        for key, value in home_pitching_stats.items():
           row[f"home_{key}"] = value
    else:
        for key in ['ERA', 'ER', 'R', 'SO', 'BB', 'SO/BB', 'WHIP']:
            row[f"home_{key}"] = None
    if away_pitching_stats:
        for key, value in away_pitching_stats.items():
           row[f"away_{key}"] = value
    else:
        for key in ['ERA', 'ER', 'R', 'SO', 'BB', 'SO/BB', 'WHIP']:
            row[f"away_{key}"] = None
            
    #Batting Stats
    for key, value in home_batting_stats.items():
        row[f"home_{key}"] = value 
    for key, value in away_batting_stats.items():
        row[f"away_{key}"] = value
        
    #Weather Data
    row["temp_x"] = weather.get("temp")
    row["condition_x"] = weather.get("condition")
    match = re.search(r"(\d+)", weather.get("wind", ""))
    row["wind_x"] = float(match.group(1)) if match else None
    
    #Lineup Score
    row.update(lineup_score(game_id))
    
    #Last 7 Win%
    row.update(get_last_7(game_id, year))
        
    #Last Games
    return pd.DataFrame([row])
    
if __name__ == "__main__":
    #778800
    p = fetch_game_info(778810)
    print(p.transpose())