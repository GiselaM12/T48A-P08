import unittest
import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import model, y_pred, y_test, df



y = df["label"]

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Crear pipeline
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Entrenar
model.fit(X_train, y_train)

#elementos de entrenamiento
print("Textos de entrenamiento:")
print(X_train)

#elementos de prueba
print("\n\nTextos de prueba:")
print(X_test)

# Evaluar
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

