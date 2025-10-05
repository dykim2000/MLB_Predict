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

for fname in os.listdir(models_dir):
    if fname.endswith(".h5"):
        model_path = os.path.join(models_dir, fname)
        model = tf.keras.models.load_model(model_path)
        
        loss, acc = model.evaluate(X_test, y_test)
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        
        #Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        cm = pd.DataFrame(cm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"])
        cm_path = os.path.join("./saved_models",f"confusion_matrix_{fname.replace('.h5', '.csv')}")
        cm.to_csv(cm_path, index=True)
        
        results.append({
            "Model" : fname,
            "Test Accuracy" : acc,
            "Test Loss" : loss
        })
        
        
df_results = pd.DataFrame(results)
print(df_results)
df_results.to_csv("../eval_summary.csv", index=False)
        