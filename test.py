import unittest
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Paths
MODEL_PATH = "models/model.keras"
SAMPLE_PATH = "sample/sample.csv"

class TestIrisKerasModel(unittest.TestCase):
    def setUp(self):
        # Load Keras model
        self.model = tf.keras.models.load_model(MODEL_PATH)

        # Load sample data
        df = pd.read_csv(SAMPLE_PATH)
        self.X = df.drop("species", axis=1).values
        self.y_labels = df["species"].values

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(["setosa", "versicolor", "virginica"])
        self.y_true = self.label_encoder.transform(self.y_labels)

    def test_prediction_accuracy(self):
        y_pred_prob = self.model.predict(self.X)
        y_pred = np.argmax(y_pred_prob, axis=1)
        accuracy = np.mean(y_pred == self.y_true)
        print(f"✅ Test accuracy: {accuracy:.2%}")
        self.assertGreaterEqual(accuracy, 0.9, "Model accuracy should be >= 90%")

    def test_prediction_label_consistency(self):
        y_pred_prob = self.model.predict(self.X)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)

        for label in y_pred_labels:
            self.assertIn(label, self.label_encoder.classes_)

if __name__ == "__main__":
    unittest.main()
