from preprocessing import load_data_2025, preprocess_2025
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import os

# Load and preprocess 2025 data
df = load_data_2025()
X_2025, y_2025 = preprocess_2025(df)

# Load the trained model
models_dir = "./saved_models"
results = []
cm_dir = "./results/2025_confusion_matrices"
os.makedirs(cm_dir, exist_ok=True)

for fname in os.listdir(models_dir):
    if fname.endswith(".h5"):
        model_path = os.path.join(models_dir, fname)
        model = tf.keras.models.load_model(model_path)
        
        loss, acc = model.evaluate(X_2025, y_2025)
        y_pred = (model.predict(X_2025) > 0.5).astype("int32")
        
        #Confusion Matrix
        cm = confusion_matrix(y_2025, y_pred)
        cm_df = pd.DataFrame(cm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"])
        cm_path = os.path.join(cm_dir, f"cm_{fname.replace('.h5', '.csv').replace('.keras', '.csv')}")
        cm_df.to_csv(cm_path, index=True)
        
        results.append({
            "Model" : fname,
            "2025 Test Accuracy" : acc,
            "2025 Test Loss" : loss
        })
        
df_results = pd.DataFrame(results)
df_results.to_csv("./results/eval_summary_2025.csv", index=False)
print(df_results)
    