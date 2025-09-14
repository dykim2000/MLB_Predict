import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from preprocessing import load_data, preprocess

def build_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim, )),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer = 'adam',
        loss = 'binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__" :
    df = load_data()
    X, y = preprocess(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = build_model(X_train.shape[1])
    model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=50, batch_size=32)
    
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy :  {accuracy:.4f}")