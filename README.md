# T48A-P08

## 📊 Métricas de Evaluación de Clasificación

Este proyecto utiliza las siguientes métricas para evaluar el desempeño del modelo Naive Bayes:

### 1. Precisión (Precision)
Indica qué proporción de las predicciones positivas fueron realmente correctas.

$$
\text{Precisión} = \frac{TP}{TP + FP}
$$

- **TP**: Verdaderos positivos  
- **FP**: Falsos positivos

---

### 2. Recall (Sensibilidad o Exhaustividad)
Indica qué proporción de los casos positivos reales fueron correctamente identificados.

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

- **FN**: Falsos negativos

---

### 3. F1-score
Es la media armónica entre precisión y recall. Se usa cuando se necesita un balance entre ambas métricas.

$$
\text{F1-score} = 2 \cdot \frac{\text{Precisión} \cdot \text{Recall}}{\text{Precisión} + \text{Recall}}
$$

---

### 4. Accuracy (Exactitud)
Es la proporción de predicciones correctas sobre el total de predicciones.

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

- **TN**: Verdaderos negativos

---

### 5. Promedios: Macro vs Weighted

- **Macro Average**: Promedio simple entre clases (no considera el tamaño de cada clase).

$$
\text{Macro Avg} = \frac{1}{N} \sum_{i=1}^{N} \text{Métrica}_i
$$

- **Weighted Average**: Promedio ponderado por el número de ejemplos en cada clase.

$$
\text{Weighted Avg} = \frac{1}{\sum \text{support}_i} \sum_{i=1}^{N} \text{Métrica}_i \cdot \text{support}_i
$$

---

> Estas métricas se calculan automáticamente al ejecutar el script `train.py` y se imprimen en consola como parte del reporte de clasificación.
