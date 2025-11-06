"""train_dl.py
Optional: train a small deep learning (Keras) MLP on the tabular CSV.

Note: this requires TensorFlow to be installed. To keep the main pipeline lightweight
we don't install TensorFlow by default. If you want to run this script, install
`tensorflow` or `tensorflow-cpu` in your environment.

Usage:
    python train_dl.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data.csv"
MODEL_OUT = ROOT / "dl_model.h5"

def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    # basic cleaning
    df = df[[c for c in df.columns if not c.startswith('Unnamed')]]
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    if df['diagnosis'].dtype == object:
        df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})
    X = df.drop(columns=['diagnosis']).values
    y = df['diagnosis'].values
    return X, y

def build_and_train(X_train, y_train, X_val, y_val):
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=32,
        callbacks=[es],
        verbose=2
    )
    model.save(MODEL_OUT)
    return model, history

def main():
    assert DATA_PATH.exists(), f"{DATA_PATH} not found"
    X, y = load_and_prepare()
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model, history = build_and_train(X_train, y_train, X_test, y_test)
    print('Saved DL model to', MODEL_OUT)

if __name__ == '__main__':
    main()
