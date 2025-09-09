import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path="\\data\final_game_logs.csv"):
    df = pd.read_csv("final_game_logs.csv")
    return df

def preprocess(df):
    
    # Dropping NA Values
    df = df.dropna()
    
    # Target Variable (Home win = 1 / Home lose = 0)
    df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
    
    X = df.drop(columns=['home_score', 'away_score', 'home_win','home_probable_pitcher', 'away_probable_pitcher','home_name', 'away_name'])
    y = df['home_win']
    
    # Scaling with the StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y
