import os
import pandas as pd
from tensorflow.keras.models import load_model
from preprocessing import preprocess_2025

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Safe paths
data_path = os.path.join(project_root, "data", "final_game_logs_2025.csv")
model_path = os.path.join(project_root, "saved_models", "model_16-8_drop0.5_20251025_200449.h5")
results_path = os.path.join(project_root, "results", "predictions_2025.csv")

df = pd.read_csv(data_path)
X_2025, y_2025 = preprocess_2025(df)
df = df.dropna()

model = load_model(model_path)
y_pred_prob = model.predict(X_2025)
y_pred = (y_pred_prob > 0.5).astype(int)

import numpy as np
y_shuffled = np.random.permutation(y_2025)
model.fit(X_2025, y_shuffled)

results = pd.DataFrame({
    "game_id" : df["game_id"],
    "home_name" : df["home_name"],
    "home_score" : df["home_score"],
    "away_name" : df["away_name"],
    "away_score" : df["away_score"],
    "actual_result" : y_2025,
    "prediction" : y_pred.flatten(),
    "prediction value " : y_pred_prob.flatten()
})

results["correct"] = results["actual_result"] == results["prediction"]

results.to_csv(results_path, index=False)