import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('car.data')
# print(data.head())

# me quedo con 3 features de data
X = data[[
    'buying',
    'maint',
    'safety'
]].values

# me quedo con la label
y = data[['class']]

# convertir data(X) de string a int
Le = LabelEncoder()
for i in range(len(X[0])):
    # len(X[0]) = 3
    X[:, i] = Le.fit_transform(X[:, i])

# mapeo labels de string a int
label_mapping = {
    'unacc': 0,
    'acc': 1,
    'good': 2,
    'vgood': 3
}
y['class'] = y['class'].map(label_mapping)
y = np.array(y)

# create model
knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')

# separar data en data trainning(80%) y data testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train model
knn.fit(X_train, y_train)

#  Hacer predicciones del modelo con test data
prediction = knn.predict(X_test)

accuracy = metrics.accuracy_score(y_test, prediction)
print("predictions:", prediction)
print("accuracy:", accuracy)
