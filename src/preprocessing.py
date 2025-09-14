import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(path="\\data\final_game_logs.csv"):
    df = pd.read_csv(path)
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

df = load_data()
X, y = preprocess(df)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False) # Disable shuffling