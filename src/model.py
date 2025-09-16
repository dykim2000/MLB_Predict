import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from preprocessing import load_data, preprocess
from datetime import datetime
import os

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

def visualize(history, hyperparams=None, save_dir="figures"):
    
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plt.figure(figsize=(18,6))
    # Accuracy Plot (Train vs Validation)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss Plot (Train vs Validation)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    if hyperparams:
        textstr = "\n".join([f"{k}: {v}" for k, v in hyperparams.items()])
        plt.figtext(0.82, 0.5, textstr, fontsize=10,
            verticalalignment='center', ha='left', bbox=dict(facecolor='white', alpha=0.6))

    
    filename = f"training_plot_{timestamp}.png"
    save_path = os.path.join(save_dir, filename)
    
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.savefig(save_path, dpi=300)
    #plt.show()

    print(f"Visualization saved at: {save_path}")
    
    

if __name__ == "__main__" :
    df = load_data()
    X, y = preprocess(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    #model = build_model(X_train.shape[1])
    hidden_layers = [[64,32], [32,16], [16, 8]]
    for i,layers in enumerate(hidden_layers):
        #Without Dropout
        model = build_model(X_train.shape[1], layers, dropout_rate=0.0)
        history = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=50, batch_size=32)
        
        hyperparams = {
        "Layers" : layers,
        "Activation" : "ReLU, ReLU, Sigmoid",
        "Optimizer" : "ADAM",
        "Epochs" : 50,
        "Batch-Size" : 32,
        "Dropout Rate" : 0.0
        }
        visualize(history, hyperparams, save_dir="figures")
        
    for i,layers in enumerate(hidden_layers):
        #With Dropout
        model = build_model(X_train.shape[1], layers, dropout_rate=0.5)
        history = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=50, batch_size=32)
        
        hyperparams = {
        "Layers" : layers,
        "Activation" : "ReLU, ReLU, Sigmoid",
        "Optimizer" : "ADAM",
        "Epochs" : 50,
        "Batch-Size" : 32,
        "Dropout Rate" : 0.5
        }
        visualize(history, hyperparams, save_dir="figures")