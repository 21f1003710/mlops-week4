import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger

# Paths
DATA_PATH = "data/iris.csv"
MODEL_DIR = "models"
SAMPLE_DIR = "sample"
MODEL_PATH = os.path.join(MODEL_DIR, "model.keras")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.csv")
SAMPLE_PATH = os.path.join(SAMPLE_DIR, "sample.csv")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

def train_and_save():
    df = pd.read_csv(DATA_PATH)
    X = df.drop("species", axis=1).values
    le = LabelEncoder()
    y = le.fit_transform(df["species"])
    y_cat = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

    # Build model
    model = Sequential([
        Input(shape=(4,)),
        Dense(10, activation='relu'),
        Dense(3, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.01),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Save training metrics
    csv_logger = CSVLogger(METRICS_PATH)

    # Train
    model.fit(X_train, y_train,
              epochs=20,
              batch_size=8,
              validation_data=(X_test, y_test),
              callbacks=[csv_logger],
              verbose=1)

    # Save complete model (structure + weights + config)
    model.save(MODEL_PATH)
    model.save_weights(os.path.join(MODEL_DIR, "model.weights.h5"))

    # Save test sample
    y_test_labels = le.inverse_transform(np.argmax(y_test, axis=1))
    sample_df = pd.DataFrame(X_test, columns=df.columns[:-1])
    sample_df["species"] = y_test_labels
    sample_df.to_csv(SAMPLE_PATH, index=False)

    print(f"✅ Model saved to {MODEL_PATH}")
    print(f"📈 Metrics saved to {METRICS_PATH}")

if __name__ == "__main__":
    train_and_save()
