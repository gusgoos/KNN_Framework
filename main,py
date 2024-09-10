# Importar las bibliotecas necesarias de sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np

# Cargar el dataset Wine desde sklearn
wine = load_wine()
X = wine.data
y = wine.target

# Dividir los datos en conjuntos de entrenamiento y prueba (75% entrenamiento, 25% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=19)

# Estandarizar las características usando StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Implementar K-Nearest Neighbors usando el framework proporcionado por sklearn
k = 2  # Número de vecinos
knn = KNeighborsClassifier(n_neighbors=k) 
knn.fit(X_train, y_train)  # Ajustar el modelo con los datos de entrenamiento

# Predecir las etiquetas para los datos de prueba
y_pred = knn.predict(X_test)

# Evaluar el modelo utilizando matriz de confusión y otros métricos
print("Evaluación del modelo utilizando sklearn:")
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred)) # Calcular Matriz de Confusión

# Calcular Precisión, Recall y F1Score
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
print(f"Precisión: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Grid Search para ajuste de hiperparámetros
from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors': np.arange(1, 50)}

# Implementar Grid Search usando validación cruzada para encontrar el mejor k, cv = 5 para validación cruzada de 5 folds
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Imprimir el mejor valor de k encontrado por GridSearch
print(f"Mejor valor de k encontrado por GridSearch: {grid_search.best_params_['n_neighbors']}")

# Reajustar el modelo con el mejor valor de k encontrado
best_knn = grid_search.best_estimator_
best_knn.fit(X_train, y_train)
y_best_pred = best_knn.predict(X_test)

# Evaluar el nuevo modelo con el hiperparámetro k optimizado
print("Evaluación del modelo después de Grid Search:")
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_best_pred))
precision = precision_score(y_test, y_best_pred, average='macro')
recall = recall_score(y_test, y_best_pred, average='macro')
f1 = f1_score(y_test, y_best_pred, average='macro')
print(f"Precisión: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Conclusiones:
# Usar un framework como sklearn para implementar algoritmos como K-Nearest Neighbors ofrece varias ventajas:
# - Proporciona una implementación más rápida y eficiente.
# - Permite una fácil integración con otros métodos y herramientas de aprendizaje automático.
# - Ofrece optimizaciones internas y simplifica la estructura del código.
