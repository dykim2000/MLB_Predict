import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

def load_data_2024(path=None):
    if path is None:
        base_dir = os.path.dirname(__file__)
        path = os.path.join(base_dir, "..", "data", "final_game_logs_2024.csv")
    abs_path = os.path.abspath(path)
    
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Data File not found")
    df = pd.read_csv(path)
    return df

def preprocess_2024(df):
    
    # Dropping NA Values
    df = df.dropna()
    
    # Target Variable (Home win = 1 / Home lose = 0)
    if 'home_win' not in df.columns:
        df.loc[:, 'home_win'] = (df['home_score'] > df['away_score']).astype(int)
    else:
        df.loc[:, 'home_win'] = df['home_win'].astype(int)
    
    # Columns
    # 게임 기본 정보 : game_id, home_name, away_name, home_probable_pitcher, away_probable_pitcher, away_score,home_score
    # 홈/원정 선발투수 지표 : home_ERA, home_ER, home_R, home_SO, home_BB, home_SO/BB, home_WHIP, away_ERA,away_ER,away_R,away_SO,away_BB,away_SO/BB,away_WHIP
    # 홈/원정 팀단위 타율 지표 : home_OBP,home_SLG,home_HR,home_R/G,home_BB_batting,home_SO_batting,home_IBB,away_home_OBP,away_home_SLG,away_home_HR,away_home_R/G,away_home_BB_batting,away_home_SO_batting,away_home_IBB
    # 날씨 지표 : temp_x,condition_x,wind_x, condition_0,condition_1,condition_2,condition_3,condition_4,condition_5,condition_6,condition_7,condition_8,condition_9
    # 홈/원정 선발 타자 지표 : home_lineup_score,away_lineup_score,home_n_starters,away_n_starters, 
    # 연승/연패, 최근전적 지표 : home_last7_win_pct,home_streak,away_last7_win_pct,away_streak,
    X = df.drop(columns=[
        'game_id', 'home_score', 'away_score', 'home_win',
        'home_probable_pitcher', 'away_probable_pitcher',
        'home_name', 'away_name', 'condition'
    ], errors='ignore')
    y = df['home_win']
    
    # Scaling with the StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def load_data_2025(path=None):
    if path is None:
        base_dir = os.path.dirname(__file__)
        path = os.path.join(base_dir, "..", "data", "final_game_logs_2025.csv")
    abs_path = os.path.abspath(path)
    
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Data File not found")
    df = pd.read_csv(path)
    return df

def preprocess_2025(df):
    
    # Dropping NA Values
    df = df.dropna()
    
    # Create target variable if it doesn't exist
    df['home_win'] = (df['home_score'] > df['away_score']).astype(int)

    X = df.drop(columns=[
        'game_id', 'home_score', 'away_score', 'home_win',
        'home_probable_pitcher', 'away_probable_pitcher',
        'home_name', 'away_name', 'condition'
    ])
    y = df['home_win']
    
    # Scaling with the StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y


if __name__ == "__main__":
    
    # Testing out preprocessing.py
    df_2024 = load_data_2024()
    df_2025 = load_data_2025()
    print(df_2024.head(), df_2024.shape)
    print("-----------------------")
    print(df_2025.head(), df_2025.shape)
    X_2024, y_2024 = preprocess_2024(df_2024)
    X_2025, y_2025 = preprocess_2025(df_2025)
    print(X_2024.shape, y_2024.shape)
    print(X_2025.shape, y_2025.shape)