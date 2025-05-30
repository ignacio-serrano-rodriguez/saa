import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler

# Extraemos matriz de características y target
dataset = load_breast_cancer()
X, y = dataset.data, dataset.target

# MNB es bastante sensible al rango de las características, mejor normalizar
# Para que cada característica esté en [0, 1], a cada columna se le resta el
# mínimo y se divide entre el rango. MinMaxScaler hace justo eso si le dejáis
# los parámetros por defecto
# Se usa en aquellos modelos que dependan de la distancia
scaler = MinMaxScaler()

# Ojo. Los clasificadores tienen fit+predict. Los transformadores (manipulación
# de datos, tienen fit y luego transform. fit_transform es ambas cosas. En
# producción, usaréis fit_transform para entrenar y transform para predecir.
# Si os llega una instancia con valores fuera del rango inicial de entrenamiento
# se quedará por debajo de cero o por encima de uno. It"s not a bug, just a
# feature.
# fit y transform sólo se usan en entrenamiento
# transform sólo con los datos reales
X_scaled = scaler.fit_transform(X)

# A partir de aquí es todo lo mismo que en el script de kNN, revisad los
# comentarios
metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

clf = MultinomialNB()
scores = cross_validate(clf, X_scaled, y, cv=kf, scoring=metrics)

plt.figure()
for metric, value in scores.items():
    if metric.startswith("test_"):
        metric = metric.removeprefix("test_")
        plt.bar(metric, value, label=metric)

plt.title("Multinomial Naive Bayes Performance Metrics")
plt.ylabel("Score")
plt.legend()
plt.grid(axis="y")
plt.show()
