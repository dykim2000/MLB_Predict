import os
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocessing import load_data, preprocess
from sklearn.model_selection import train_test_split

df = load_data()
X, y = preprocess(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

models_dir = "./saved_models"
results = []

output_path = os.path.join(os.path.dirname(__file__), "..", "results", "eval_summary.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
output_path = os.path.abspath(output_path)

for fname in os.listdir(models_dir):
    if fname.endswith(".h5"):
        model_path = os.path.join(models_dir, fname)
        model = tf.keras.models.load_model(model_path)
        
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        
        #Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"])
        model_desc = fname.replace(".h5", "").replace("_", " ")
        
        results.append({
            "Model" : model_desc,
            "Test Accuracy" : acc,
            "Test Loss" : loss
        })

df_results = pd.DataFrame(results)
df_results.to_csv(output_path, index=False)

for fname in os.listdir(models_dir):
    if fname.endswith(".h5"):
        model_desc = fname.replace(".h5", "").replace("_", " ")
        model_path = os.path.join(models_dir, fname)
        model = tf.keras.models.load_model(model_path)

        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"])

        with open(output_path, "a") as f:
            f.write(f"\nModel : {model_desc}\n")
            cm_df.to_csv(f)
            f.write("\n")