import requests
import pandas as pd
import os
import re
import tensorflow as tf

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
        shifted_wins = team_games["team_win"].shift(1).dropna().astype(int)
        if shifted_wins.empty:
            return None
        win = shifted_wins.iloc[0]
        streak = -1 if win == 0 else 1
        for result in shifted_wins.iloc[1:]:
            if result == win:
                if streak > 0:
                    streak += 1
                else:
                    streak -= 1
            else:
                break
        return shifted_wins.head(7).mean(), streak
    
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
        "home_name" : home_team,
        "away_name" : away_team,
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
    row["temp"] = weather.get("temp")
    row["condition"] = weather.get("condition")
    condition_mapping = {
    "Partly Cloudy": 0,
    "Clear": 1,
    "Sunny": 2,
    "Cloudy": 3,
    "Roof Closed": 4,
    "Overcast": 5,
    "Dome": 6,
    "Drizzle": 7,
    "Unknown": 8,
    "Rain": 9
    }
    row["condition"] = condition_mapping.get(row.get("condition"), 8)
    match = re.search(r"(\d+)", weather.get("wind", ""))
    row["wind"] = float(match.group(1)) if match else None
    
    #Lineup Score
    row.update(lineup_score(game_id))
    
    #Last 7 Win%
    row.update(get_last_7(game_id, year))
        
    # Weather condition one-hot encoding
    for condition_name, condition_code in condition_mapping.items():
        row[condition_name] = 1 if row["condition"] == condition_code else 0
    
    
    return pd.DataFrame([row])
    
if __name__ == "__main__":
    
    model = tf.keras.models.load_model("./saved_models/model_16-8_drop0.5_20251025_200449.h5")
    
    p = fetch_game_info(777175)
    
    p.loc[:, 'home_win'] = (p['home_score'] > p['away_score']).astype(int)
    X = p.drop(columns=['game_id', 'home_score', 'away_score', 'home_win',
        'home_probable_pitcher', 'away_probable_pitcher',
        'home_name', 'away_name', 'condition'])
    y = p['home_win']
    pred = model.predict(X)
    
    print(f"Game ID : {p['game_id'].iloc[0]}")
    print(f"Home Team : {p['home_name'].iloc[0]}  Score : {p['home_score'].iloc[0]}")
    print(f"Away Team : {p['away_name'].iloc[0]}  Score : {p['away_score'].iloc[0]}")
    print(f"Actual Value : {y.iloc[0]}")
    print("Predicted Value : ", pred)