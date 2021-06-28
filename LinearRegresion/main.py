from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

boston = datasets.load_boston()

#features / labels

X = boston.data

y = boston.target

print("X")
print(X)
print(X.shape) # dimensiones de la matriz
print('y')
print(y)

# algoritmo de regresion lineal
l_reg = linear_model.LinearRegression()

# mostrar grafico feature 1 de X vs y
plt.scatter(X.T[0],y)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

# entrenar
model = l_reg.fit(X_train, y_train) # modelo es una recta y = m * X + b
predictions = model.predict(X_test)
print('Prediccioines',predictions)
print('R*2 value:', l_reg.score(X, y)) # mientras mas grande es mejor --> mas cerca estan los puntos de la prediccion
print("coedd:", l_reg.coef_) # m --> pendiente --> es la derivada de y respecto a x
print("intercept:", l_reg.intercept_) # es la ordenada al origen (b)

# cuando no genera una buena prediccion el modelo porque es dificil modelarlo con
# una recta, se usa logistic regresion, se predice con una funcion signum ( arecida a Heavise)
# y(x) = 1/(1 + e*-x) ( * = elevado a la  )