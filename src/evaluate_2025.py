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

for fname in os.listdir(models_dir):
    if fname.endswith(".h5"):
        model_path = os.path.join(models_dir, fname)
        model = tf.keras.models.load_model(model_path)
        
        loss, acc = model.evaluate(X_2025, y_2025)
        y_pred = (model.predict(X_2025) > 0.5).astype("int32")
        
        #Confusion Matrix
        cm = confusion_matrix(y_2025, y_pred)
        
        results.append({
            "Model" : fname,
            "2025 Test Accuracy" : acc,
            "2025 Test Loss" : loss
        })
        
df_results = pd.DataFrame(results)

summary_path = os.path.join("./results", "eval_summary_2025.csv")
os.makedirs(os.path.dirname(summary_path), exist_ok=True)
df_results.to_csv(summary_path, index=False)

# Append confusion matrices below the summary table
with open(summary_path, "a") as f:
    for result in results:
        model_desc = result["Model"].replace(".h5", "").replace("_", " ")
        f.write(f"\nModel : {model_desc}\n")
        cm = confusion_matrix(y_2025, (tf.keras.models.load_model(os.path.join(models_dir, result["Model"])).predict(X_2025) > 0.5).astype("int32"))
        cm_df = pd.DataFrame(cm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"])
        cm_df.to_csv(f)
        f.write("\n")

print(df_results)