
import numpy as np
import pandas as pd

# Establecer la semilla para reproducibilidad
np.random.seed(42)

# Número de muestras
n_samples = 250

# Generar la edad de las casas entre 0 y 50 años
edad_casa = np.random.rand(n_samples) * 70  # Valores entre 0 y 50 años

# Definir una función no lineal para generar el precio
# Por ejemplo, suponemos que las casas nuevas y muy antiguas tienen precios más altos debido a renovaciones
precio_casa = -0.05 * (edad_casa - 25)**2 + 200 + np.random.normal(0, 5, n_samples)

# Asegurarse de que los precios sean positivos
precio_casa = np.clip(precio_casa, a_min=50, a_max=None)

# Crear un DataFrame
datos = pd.DataFrame({
    'Edad_Casa': edad_casa,
    'Precio_Casa': precio_casa
})

datos.to_csv('datos_precio_casa.csv', index=False)

print("Dataset 'datos_precio_casa.csv' creado exitosamente.")
