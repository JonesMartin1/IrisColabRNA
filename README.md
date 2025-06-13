Clasificación de Flores Iris con Redes Neuronales Artificiales (RNA)

Este proyecto explora el uso de redes neuronales artificiales (MLPClassifier) para clasificar flores del conjunto de datos **Iris** mediante distintas arquitecturas, funciones de activación y parámetros de entrenamiento. Se evalúa el rendimiento de tres modelos diferentes aplicando métricas como precisión, recall y F1-score.

---

Contenido

- ✅ Preprocesamiento del dataset Iris (limpieza, normalización y estandarización)
- ✅ División del conjunto en entrenamiento y prueba
- ✅ Entrenamiento de 3 modelos de RNA con distintas configuraciones:
  - **Modelo ReLU**
  - **Modelo Softmax**
  - **Modelo Perfecto**
- ✅ Predicción sobre conjunto de test
- ✅ Evaluación de rendimiento con métricas
- ✅ Comparación tabular de resultados
- ✅ Visualización final de métricas promedio y desviaciones

---

Dataset utilizado

- **Nombre:** Iris Dataset  
- **Fuente:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris) / [Kaggle](https://www.kaggle.com/datasets/uciml/iris)
- **Características:** 150 muestras, 3 clases, 4 atributos numéricos  
- **Objetivo:** Clasificar una flor en una de las tres especies: *Setosa*, *Versicolor*, *Virginica*

---

Modelos Entrenados

| Modelo           | Capas Ocultas    | Activación | Optimizador |
|------------------|------------------|------------|-------------|
| Modelo ReLU      | (50, 25)         | ReLU       | SGD         |
| Modelo Softmax   | (30,)            | Logistic   | LBFGS       |
| Modelo Perfecto  | (20, 10)         | ReLU       | Adam        |

---

Tecnologías Utilizadas

- Python 3
- scikit-learn (MLPClassifier, métricas)
- pandas, numpy
- Google Colab (como entorno de ejecución)

---

Resultados (resumen)

| Métrica      | ReLU  | Softmax | Perfecto | Media | Desvío Est. |
|--------------|-------|---------|----------|-------|-------------|
| Accuracy     | 0.87  | 1.00    | 1.00     | 0.96  | 0.06        |
| F1-Score     | 0.86  | 1.00    | 1.00     | 0.95  | 0.07        |
| Recall       | 0.83  | 1.00    | 1.00     | 0.94  | 0.08        |
| Precision    | 0.87  | 1.00    | 1.00     | 0.96  | 0.06        |

---

Conclusiones

- El **Modelo Perfecto** y el **Modelo Softmax** lograron un desempeño óptimo con 100% de precisión, recall y F1-score.
- La arquitectura, función de activación y algoritmo de optimización afectan significativamente el rendimiento.
- Las RNA son una herramienta potente para tareas de clasificación multiclase cuando se ajustan correctamente sus hiperparámetros.

---

Cómo ejecutar

```python
# Entrenar y evaluar un modelo
modelo = MLPClassifier(...)
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
accuracy = modelo.score(X_test, y_test)
