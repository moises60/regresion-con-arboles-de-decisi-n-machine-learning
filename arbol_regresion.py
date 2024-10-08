
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar estilo de los gráficos
sns.set(style="whitegrid")

# Cargar el dataset
datos = pd.read_csv('datos_precio_casa.csv')

# Visualizar los datos
plt.figure(figsize=(10, 6))
plt.scatter(datos['Edad_Casa'], datos['Precio_Casa'], color='blue', label='Datos')
plt.title('Relación entre Edad de la Casa y Precio')
plt.xlabel('Edad de la Casa (años)')
plt.ylabel('Precio de la Casa (en miles de dólares)')
plt.legend()
plt.show()

# Variables independientes y dependiente
X = datos[['Edad_Casa']]
y = datos['Precio_Casa']

# Dividir el dataset en conjunto de entrenamiento y prueba
from sklearn.model_selection import train_test_split

X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Entrenar el árbol de regresión
from sklearn.tree import DecisionTreeRegressor

# Crear el modelo
arbol_regresion = DecisionTreeRegressor(random_state=42, max_depth=4)

# Entrenar el modelo
arbol_regresion.fit(X_entrenamiento, y_entrenamiento)

# Predecir en el conjunto de prueba
y_pred = arbol_regresion.predict(X_prueba)

# Evaluar el modelo
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_prueba, y_pred)
mse = mean_squared_error(y_prueba, y_pred)
r2 = r2_score(y_prueba, y_pred)

print("Evaluación del Árbol de Regresión:")
print(f"Error Absoluto Medio (MAE): {mae:.2f}")
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
print(f"R^2 Score: {r2:.4f}")

# Visualizar las predicciones
# Para visualización, ordenamos los datos
X_prueba_sorted = X_prueba.sort_values(by='Edad_Casa')
y_pred_sorted = arbol_regresion.predict(X_prueba_sorted)

plt.figure(figsize=(10, 6))
plt.scatter(X_prueba['Edad_Casa'], y_prueba, color='blue', label='Datos Reales')
plt.plot(X_prueba_sorted['Edad_Casa'], y_pred_sorted, color='red', label='Predicciones del Árbol')
plt.title('Árbol de Regresión - Predicciones vs Datos Reales')
plt.xlabel('Edad de la Casa (años)')
plt.ylabel('Precio de la Casa (en miles de dólares)')
plt.legend()
plt.show()

# Comparación con Regresión Lineal
from sklearn.linear_model import LinearRegression

# Entrenar el modelo lineal
modelo_lineal = LinearRegression()
modelo_lineal.fit(X_entrenamiento, y_entrenamiento)

# Predecir en el conjunto de prueba
y_pred_lineal = modelo_lineal.predict(X_prueba)

# Evaluar el modelo lineal
mae_lineal = mean_absolute_error(y_prueba, y_pred_lineal)
mse_lineal = mean_squared_error(y_prueba, y_pred_lineal)
r2_lineal = r2_score(y_prueba, y_pred_lineal)

print("\nEvaluación del Modelo Lineal:")
print(f"Error Absoluto Medio (MAE): {mae_lineal:.2f}")
print(f"Error Cuadrático Medio (MSE): {mse_lineal:.2f}")
print(f"R^2 Score: {r2_lineal:.4f}")

# Visualizar las predicciones del modelo lineal
plt.figure(figsize=(10, 6))
plt.scatter(X_prueba['Edad_Casa'], y_prueba, color='blue', label='Datos Reales')
plt.plot(X_prueba['Edad_Casa'], y_pred_lineal, color='green', label='Predicciones Lineales')
plt.title('Regresión Lineal - Predicciones vs Datos Reales')
plt.xlabel('Edad de la Casa (años)')
plt.ylabel('Precio de la Casa (en miles de dólares)')
plt.legend()
plt.show()

# Comparación de modelos
comparativa = pd.DataFrame({
    'Modelo': ['Árbol de Regresión', 'Regresión Lineal'],
    'MAE': [mae, mae_lineal],
    'MSE': [mse, mse_lineal],
    'R^2 Score': [r2, r2_lineal]
})

print("\nComparativa de Métricas entre Modelos:")
print(comparativa)

# Visualizar la comparación
comparativa_melted = comparativa.melt(id_vars='Modelo', var_name='Métrica', value_name='Valor')

plt.figure(figsize=(10, 6))
sns.barplot(x='Métrica', y='Valor', hue='Modelo', data=comparativa_melted)
plt.title('Comparación de Métricas entre Modelos')
plt.ylabel('Valor')
plt.show()
