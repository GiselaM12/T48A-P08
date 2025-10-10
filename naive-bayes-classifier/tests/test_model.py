import unittest
from src.model import model, y_pred, y_test


class TestTextClassificationModel(unittest.TestCase):
    def test_model_training(self):
        """Verifica que el modelo se haya creado correctamente."""
        self.assertIsNotNone(model, "El modelo no fue creado correctamente.")

    def test_prediction_output(self):
        """Verifica que el número de predicciones coincida con el número de ejemplos de prueba."""
        self.assertEqual(len(y_pred), len(y_test), "El número de predicciones no coincide con el número de ejemplos de prueba.")

    def test_prediction_labels(self):
        """Verifica que las etiquetas predichas sean válidas ('positivo' o 'negativo')."""
        for label in y_pred:
            self.assertIn(label, ["positivo", "negativo"], "Etiqueta de predicción no válida.")
