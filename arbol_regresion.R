# install.packages("rpart")
# install.packages("rpart.plot")
# install.packages("ggplot2")
# install.packages("caret")
# install.packages("reshape2")

# Cargar las librer�as
library(rpart)        # Para el �rbol de regresi�n
library(rpart.plot)   # Para visualizar el �rbol de regresi�n
library(ggplot2)      # Para visualizaciones
library(caret)        # Para funciones de partici�n y evaluaci�n
library(reshape2)     # Para transformar datos para visualizaci�n

# ---------------------------------------------------
# 2. Cargar y Visualizar el Dataset
# ---------------------------------------------------

# Cargar el dataset
datos <- read.csv('datos_precio_casa.csv')

# Mostrar las primeras filas del dataset
cat("Primeras filas del dataset:\n")
print(head(datos))

# Descripci�n estad�stica del dataset
cat("\nDescripci�n Estad�stica del Dataset:\n")
print(summary(datos))

# Visualizar la relaci�n entre Edad_Casa y Precio_Casa
ggplot(datos, aes(x = Edad_Casa, y = Precio_Casa)) +
  geom_point(color = 'blue') +
  ggtitle('Relaci�n entre Edad de la Casa y Precio') +
  xlab('Edad de la Casa (a�os)') +
  ylab('Precio de la Casa (en miles de d�lares)') +
  theme_minimal()

# ---------------------------------------------------
# 3. Preparaci�n de los Datos
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
# 4. Entrenamiento del �rbol de Regresi�n
# ---------------------------------------------------

# Crear el modelo de �rbol de regresi�n con una profundidad m�xima de 4
arbol_regresion <- rpart(Precio_Casa ~ Edad_Casa, data = train_data, method = "anova",
                         control = rpart.control(maxdepth = 4))

# Visualizar el �rbol de regresi�n
rpart.plot(arbol_regresion, type = 3, fallen.leaves = TRUE,
           main = "�rbol de Regresi�n de Precio de Casa")

# Predecir en el conjunto de prueba
y_pred <- predict(arbol_regresion, newdata = test_data)

# ---------------------------------------------------
# 5. Evaluaci�n del �rbol de Regresi�n
# ---------------------------------------------------

# Calcular m�tricas de evaluaci�n
mae <- mean(abs(y_test - y_pred))
mse <- mean((y_test - y_pred)^2)
r2 <- 1 - sum((y_test - y_pred)^2) / sum((y_test - mean(y_test))^2)

# Mostrar los resultados
cat("\nEvaluaci�n del �rbol de Regresi�n:\n")
cat(sprintf("Error Absoluto Medio (MAE): %.2f\n", mae))
cat(sprintf("Error Cuadr�tico Medio (MSE): %.2f\n", mse))
cat(sprintf("R� Score: %.4f\n", r2))

# Visualizar las predicciones del �rbol de regresi�n
ggplot() +
  geom_point(aes(x = test_data$Edad_Casa, y = test_data$Precio_Casa),
             color = 'blue', alpha = 0.6, label = 'Datos Reales') +
  geom_line(aes(x = test_data$Edad_Casa, y = y_pred),
            color = 'red', size = 1, label = 'Predicciones del �rbol') +
  ggtitle('�rbol de Regresi�n - Predicciones vs Datos Reales') +
  xlab('Edad de la Casa (a�os)') +
  ylab('Precio de la Casa (en miles de d�lares)') +
  theme_minimal()

# ---------------------------------------------------
# 6. Entrenamiento del Modelo de Regresi�n Lineal
# ---------------------------------------------------

# Crear y entrenar el modelo de regresi�n lineal
modelo_lineal <- lm(Precio_Casa ~ Edad_Casa, data = train_data)

# Predecir en el conjunto de prueba con el modelo lineal
y_pred_lineal <- predict(modelo_lineal, newdata = test_data)

# ---------------------------------------------------
# 7. Evaluaci�n del Modelo de Regresi�n Lineal
# ---------------------------------------------------

# Calcular m�tricas de evaluaci�n para el modelo lineal
mae_lineal <- mean(abs(y_test - y_pred_lineal))
mse_lineal <- mean((y_test - y_pred_lineal)^2)
r2_lineal <- 1 - sum((y_test - y_pred_lineal)^2) / sum((y_test - mean(y_test))^2)

# Mostrar los resultados
cat("\nEvaluaci�n del Modelo Lineal:\n")
cat(sprintf("Error Absoluto Medio (MAE): %.2f\n", mae_lineal))
cat(sprintf("Error Cuadr�tico Medio (MSE): %.2f\n", mse_lineal))
cat(sprintf("R� Score: %.4f\n", r2_lineal))

# Visualizar las predicciones del modelo lineal
ggplot() +
  geom_point(aes(x = test_data$Edad_Casa, y = test_data$Precio_Casa),
             color = 'blue', alpha = 0.6, label = 'Datos Reales') +
  geom_smooth(aes(x = test_data$Edad_Casa, y = y_pred_lineal),
              method = 'lm', se = FALSE, color = 'green', size = 1, label = 'Regresi�n Lineal') +
  ggtitle('Regresi�n Lineal - Predicciones vs Datos Reales') +
  xlab('Edad de la Casa (a�os)') +
  ylab('Precio de la Casa (en miles de d�lares)') +
  theme_minimal()

# ---------------------------------------------------
# 8. Comparaci�n de Modelos
# ---------------------------------------------------

# Crear un DataFrame para comparar las m�tricas
comparativa <- data.frame(
  Modelo = c("�rbol de Regresi�n", "Regresi�n Lineal"),
  MAE = c(mae, mae_lineal),
  MSE = c(mse, mse_lineal),
  R2_Score = c(r2, r2_lineal)
)

# Mostrar la comparativa
cat("\nComparativa de M�tricas entre Modelos:\n")
print(comparativa)

# Visualizar la comparativa de m�tricas
comparativa_melted <- melt(comparativa, id.vars = "Modelo")

ggplot(comparativa_melted, aes(x = variable, y = value, fill = Modelo)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  ggtitle('Comparaci�n de M�tricas entre Modelos') +
  xlab('M�trica') +
  ylab('Valor') +
  theme_minimal() +
  scale_fill_brewer(palette = "Set1")
