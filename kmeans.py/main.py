from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import pandas as pd

bc = load_breast_cancer()
print(bc)
# escala los datos (features) para ayudar al algoritmo
X = scale(bc.data)
print(X)

y = bc.target
print(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
## nro clusters = nro labels
model = KMeans(n_clusters=2, random_state=0)

model.fit(x_train)

predictions = model.predict(x_test)

labels = model.labels_

print('labels',labels)
print ('predictions', predictions)
print('accuracy', accuracy_score(y_test,predictions))
print('actual',y_test)
## accuracy = 0.12 --> se equivoco la etiqueta 0 con la 1, hay que cambiarlas
## prediction es la prediccion y actual es el valor real
## hay que cambiar las etiquetas de la prediccion donde dice 1 va 0 y al reves
print(pd.crosstab(y_train, labels))
