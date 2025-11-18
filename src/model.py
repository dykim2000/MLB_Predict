import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from preprocessing import load_data_2024, preprocess_2024
from datetime import datetime
import os

#Model Function
def build_model(input_dim, hidden_layers=[64, 32], learning_rate=0.001, dropout_rate=0.0):
    model = Sequential([])
    
    for i, nodes in enumerate(hidden_layers):
        if i == 0: #input layer
            model.add(Dense(
                nodes,
                activation='relu',
                input_shape=(input_dim,)
            ))
        else:
            model.add(Dense(
                nodes,
                activation='relu'
            ))
        if dropout_rate > 0: #add dropouts
            model.add(Dropout(dropout_rate))
            
    #output layer
    model.add(Dense(1, activation='sigmoid'))
            
    
    model.compile(
        optimizer = 'adam',
        loss = 'binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

#Visualization (and saving) Function
def visualize(history, hyperparams=None, save_dir="figures", save_model_flag=False, model=None, test_metrics=None):
    
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    layer_str = "-".join(map(str, hyperparams.get("Layers", [])))
    dropout = hyperparams.get("Dropout Rate", 0.0)
    
    plt.figure(figsize=(18,6))
    # Accuracy Plot (Train vs Validation)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    if test_metrics:
        _, test_acc = test_metrics
        plt.axhline(y=test_acc, color='green', linestyle='--', linewidth=1.2, label=f"Test Acc: {test_acc:.4f}")
        plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss Plot (Train vs Validation)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    if test_metrics:
        test_loss, _ = test_metrics
        plt.axhline(y=test_loss, color='red', linestyle='--', linewidth=1.2, label=f"Test Loss: {test_loss:.4f}")
        plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    if hyperparams:
        textstr = "\n".join([f"{k}: {v}" for k, v in hyperparams.items()])
        plt.figtext(0.82, 0.5, textstr, fontsize=10,
            verticalalignment='center', ha='left', bbox=dict(facecolor='white', alpha=0.6))

    if test_metrics:
        test_loss, test_acc = test_metrics
    else:
        test_loss, test_acc = (None, None)
    
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]

    metrics_str = (
        f"Final Train Acc: {final_train_acc:.4f}\n"
        f"Final Val Acc: {final_val_acc:.4f}\n"
        f"Final Train Loss: {final_train_loss:.4f}\n"
        f"Final Val Loss: {final_val_loss:.4f}"
    )

    if test_acc is not None:
        metrics_str += f"\nFinal Test Acc: {test_acc:.4f}\nFinal Test Loss: {test_loss:.4f}"

    plt.figtext(0.82, 0.25, metrics_str, fontsize=10, ha='left', bbox=dict(facecolor='white', alpha=0.6))
    
    filename = f"model_{layer_str}_drop{dropout}_epochs{epochs}_{timestamp}.png"
    save_path = os.path.join(save_dir, filename)
    
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.savefig(save_path, dpi=300)
    #plt.show()

    print(f"Visualization saved at: {save_path}")
    
    #Saving the model
    if save_model_flag:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs("saved_models", exist_ok=True)
        
        layer_str = "-".join(map(str, hyperparams.get("Layers", [])))
        dropout = hyperparams.get("Dropout Rate", 0.0)
        
        model_path = os.path.join("saved_models", f"model_{layer_str}_drop{dropout}_epochs{epochs}_{timestamp}.h5")
        model.save(model_path)
        print(f"Model Saved at {model_path}")
    

if __name__ == "__main__" :
    X, y = preprocess_2024()
    
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.1)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1)
    
    # Layer configurations with their optimal epochs
    model_configs = [
        {"layers": [16,8], "epochs" : 200},
        {"layers": [16,8,4], "epochs" : 200}
    ]
    
    for config in model_configs:
        layers = config["layers"]
        epochs = config["epochs"]
        
        model = build_model(X_train.shape[1], layers, dropout_rate=0.5)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32
        )
        
        hyperparams = {
            "Layers": layers,
            "Activation": "ReLU, ReLU, Sigmoid",
            "Optimizer": "ADAM",
            "Epochs": epochs,
            "Batch-Size": 32,
            "Dropout Rate": 0.5
        }
        test_loss, test_acc = model.evaluate(X_test, y_test)
        visualize(history, hyperparams, save_dir="figures", save_model_flag=True, model=model, test_metrics=(test_loss, test_acc))

    print(f"Final Test Accuracy: {test_acc:.4f}")