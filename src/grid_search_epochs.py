import pandas as pd
from sklearn.model_selection import train_test_split
from model import build_model
from preprocessing import load_data, preprocess
from keras.callbacks import EarlyStopping
import tensorflow as tf

# Grid search setup
layer_configs = [
    [32, 16],
    [32, 16, 8],
    [16, 8, 4],
    [16, 8]
]
epoch_candidates = [10, 20, 30, 40, 50, 60, 70, 80]

results = []

# Load and preprocess
df = load_data()
X, y = preprocess(df)

# Split dataset
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.125, shuffle=False)

for layers in layer_configs:
    for epochs in epoch_candidates:
        print(f"\n🔍 Training {layers} for {epochs} epochs...")
        
        model = build_model(X_train.shape[1], hidden_layers=layers, dropout_rate=0.5)
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            verbose=0,
            callbacks=[early_stop]
        )
        
        best_val_acc = max(history.history['val_accuracy'])
        best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
        
        results.append({
            "Layers": str(layers),
            "Epochs_Trained": epochs,
            "Best_Epoch": best_epoch,
            "Best_Val_Accuracy": round(best_val_acc, 4)
        })
        print(f"✅ Layers {layers}, Epoch {best_epoch}, Val Acc {best_val_acc:.4f}")

# Save results
results_df = pd.DataFrame(results)
results_df.sort_values(by="Best_Val_Accuracy", ascending=False, inplace=True)
results_df.to_csv("results/gridsearch_results.csv", index=False)

print("\nResults saved to results/gridsearch_results.csv")