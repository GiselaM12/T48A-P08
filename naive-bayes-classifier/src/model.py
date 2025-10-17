import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Cargar datos
df = pd.read_csv("naive-bayes-classifier/data/dataset.csv")

# Separar características y etiquetas
X = df["text"]
y = df["label"]

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear pipeline (vectorización + modelo Naive Bayes)
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Mostrar reporte de clasificación (solo para ver resultados en consola)
print(classification_report(y_test, y_pred, zero_division=0))

# Exportar variables clave para los tests
__all__ = ["model", "y_pred", "y_test", "df"]
