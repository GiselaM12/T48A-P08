import unittest
import pandas as pd
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import model, y_pred, y_test, df


class TestTextClassificationModel(unittest.TestCase):

    def test_model_exists(self):
        """Verifica que el modelo se haya creado correctamente."""
        self.assertIsNotNone(model, "El modelo no fue creado correctamente.")

    def test_prediction_output(self):
        """Verifica que el número de predicciones coincida con el número de ejemplos de prueba."""
        self.assertEqual(len(y_pred), len(y_test), "El número de predicciones no coincide con el número de ejemplos de prueba.")

    def test_prediction_labels(self):
        """Verifica que las etiquetas predichas sean válidas ('positivo' o 'negativo')."""
        for label in y_pred:
            self.assertIn(label, ["positivo", "negativo"], f"Etiqueta no válida: {label}")

    def test_dataset_length(self):
        """Verifica que el dataset tenga al menos 10 registros."""
        self.assertGreaterEqual(len(df), 10, "El archivo dataset.csv debe contener al menos 10 registros.")


if __name__ == "__main__":
    unittest.main()
