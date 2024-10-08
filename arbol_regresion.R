# install.packages("rpart")
# install.packages("rpart.plot")
# install.packages("ggplot2")
# install.packages("caret")
# install.packages("reshape2")

# Cargar las librerías
library(rpart)        # Para el árbol de regresión
library(rpart.plot)   # Para visualizar el árbol de regresión
library(ggplot2)      # Para visualizaciones
library(caret)        # Para funciones de partición y evaluación
library(reshape2)     # Para transformar datos para visualización

# ---------------------------------------------------
# 2. Cargar y Visualizar el Dataset
# ---------------------------------------------------

# Cargar el dataset
datos <- read.csv('datos_precio_casa.csv')

# Mostrar las primeras filas del dataset
cat("Primeras filas del dataset:\n")
print(head(datos))

# Descripción estadística del dataset
cat("\nDescripción Estadística del Dataset:\n")
print(summary(datos))

# Visualizar la relación entre Edad_Casa y Precio_Casa
ggplot(datos, aes(x = Edad_Casa, y = Precio_Casa)) +
  geom_point(color = 'blue') +
  ggtitle('Relación entre Edad de la Casa y Precio') +
  xlab('Edad de la Casa (años)') +
  ylab('Precio de la Casa (en miles de dólares)') +
  theme_minimal()

# ---------------------------------------------------
# 3. Preparación de los Datos
# ---------------------------------------------------

# Variables independientes y dependiente
X <- datos$Edad_Casa
y <- datos$Precio_Casa

# Dividir el dataset en conjunto de entrenamiento y prueba (80% - 20%)
set.seed(42)  # Para reproducibilidad
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[trainIndex]
y_train <- y[trainIndex]
X_test <- X[-trainIndex]
y_test <- y[-trainIndex]

# Crear data frames para entrenamiento y prueba
train_data <- data.frame(Edad_Casa = X_train, Precio_Casa = y_train)
test_data <- data.frame(Edad_Casa = X_test, Precio_Casa = y_test)

# ---------------------------------------------------
# 4. Entrenamiento del Árbol de Regresión
# ---------------------------------------------------

# Crear el modelo de árbol de regresión con una profundidad máxima de 4
arbol_regresion <- rpart(Precio_Casa ~ Edad_Casa, data = train_data, method = "anova",
                         control = rpart.control(maxdepth = 4))

# Visualizar el árbol de regresión
rpart.plot(arbol_regresion, type = 3, fallen.leaves = TRUE,
           main = "Árbol de Regresión de Precio de Casa")

# Predecir en el conjunto de prueba
y_pred <- predict(arbol_regresion, newdata = test_data)

# ---------------------------------------------------
# 5. Evaluación del Árbol de Regresión
# ---------------------------------------------------

# Calcular métricas de evaluación
mae <- mean(abs(y_test - y_pred))
mse <- mean((y_test - y_pred)^2)
r2 <- 1 - sum((y_test - y_pred)^2) / sum((y_test - mean(y_test))^2)

# Mostrar los resultados
cat("\nEvaluación del Árbol de Regresión:\n")
cat(sprintf("Error Absoluto Medio (MAE): %.2f\n", mae))
cat(sprintf("Error Cuadrático Medio (MSE): %.2f\n", mse))
cat(sprintf("R² Score: %.4f\n", r2))

# Visualizar las predicciones del árbol de regresión
ggplot() +
  geom_point(aes(x = test_data$Edad_Casa, y = test_data$Precio_Casa),
             color = 'blue', alpha = 0.6, label = 'Datos Reales') +
  geom_line(aes(x = test_data$Edad_Casa, y = y_pred),
            color = 'red', size = 1, label = 'Predicciones del Árbol') +
  ggtitle('Árbol de Regresión - Predicciones vs Datos Reales') +
  xlab('Edad de la Casa (años)') +
  ylab('Precio de la Casa (en miles de dólares)') +
  theme_minimal()

# ---------------------------------------------------
# 6. Entrenamiento del Modelo de Regresión Lineal
# ---------------------------------------------------

# Crear y entrenar el modelo de regresión lineal
modelo_lineal <- lm(Precio_Casa ~ Edad_Casa, data = train_data)

# Predecir en el conjunto de prueba con el modelo lineal
y_pred_lineal <- predict(modelo_lineal, newdata = test_data)

# ---------------------------------------------------
# 7. Evaluación del Modelo de Regresión Lineal
# ---------------------------------------------------

# Calcular métricas de evaluación para el modelo lineal
mae_lineal <- mean(abs(y_test - y_pred_lineal))
mse_lineal <- mean((y_test - y_pred_lineal)^2)
r2_lineal <- 1 - sum((y_test - y_pred_lineal)^2) / sum((y_test - mean(y_test))^2)

# Mostrar los resultados
cat("\nEvaluación del Modelo Lineal:\n")
cat(sprintf("Error Absoluto Medio (MAE): %.2f\n", mae_lineal))
cat(sprintf("Error Cuadrático Medio (MSE): %.2f\n", mse_lineal))
cat(sprintf("R² Score: %.4f\n", r2_lineal))

# Visualizar las predicciones del modelo lineal
ggplot() +
  geom_point(aes(x = test_data$Edad_Casa, y = test_data$Precio_Casa),
             color = 'blue', alpha = 0.6, label = 'Datos Reales') +
  geom_smooth(aes(x = test_data$Edad_Casa, y = y_pred_lineal),
              method = 'lm', se = FALSE, color = 'green', size = 1, label = 'Regresión Lineal') +
  ggtitle('Regresión Lineal - Predicciones vs Datos Reales') +
  xlab('Edad de la Casa (años)') +
  ylab('Precio de la Casa (en miles de dólares)') +
  theme_minimal()

# ---------------------------------------------------
# 8. Comparación de Modelos
# ---------------------------------------------------

# Crear un DataFrame para comparar las métricas
comparativa <- data.frame(
  Modelo = c("Árbol de Regresión", "Regresión Lineal"),
  MAE = c(mae, mae_lineal),
  MSE = c(mse, mse_lineal),
  R2_Score = c(r2, r2_lineal)
)

# Mostrar la comparativa
cat("\nComparativa de Métricas entre Modelos:\n")
print(comparativa)

# Visualizar la comparativa de métricas
comparativa_melted <- melt(comparativa, id.vars = "Modelo")

ggplot(comparativa_melted, aes(x = variable, y = value, fill = Modelo)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  ggtitle('Comparación de Métricas entre Modelos') +
  xlab('Métrica') +
  ylab('Valor') +
  theme_minimal() +
  scale_fill_brewer(palette = "Set1")
