from preprocessing import load_data_2025, preprocess_2025
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import os

# Load and preprocess 2025 data
X_2025, y_2025 = preprocess_2025()

# Load the trained model
models_dir = "./saved_models"
results = []

for fname in os.listdir(models_dir):
    if fname.endswith(".h5"):
        model_path = os.path.join(models_dir, fname)
        model = tf.keras.models.load_model(model_path)
        
        loss, acc = model.evaluate(X_2025, y_2025)
        y_pred = (model.predict(X_2025) > 0.5).astype("int32")
        results.append({
            "Model" : fname,
            "2025 Test Accuracy" : acc,
            "2025 Test Loss" : loss
        })
        
df_results = pd.DataFrame(results)

summary_path = os.path.join("./results", "eval_summary_2025.csv")
os.makedirs(os.path.dirname(summary_path), exist_ok=True)
df_results.to_csv(summary_path, index=False)

print(df_results) 