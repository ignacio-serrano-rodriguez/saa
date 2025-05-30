import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier

# Extraemos matriz de características y target
data = load_breast_cancer()
X, y = data.data, data.target

metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# Esto lo podéis poner abajo en el for, en el ejemplo del otro día lo hice a
# mano con una lista [1,3,5,7......]. Lo he subido arriba como una variable para
# poder configurarlo rápidamente sin tener que buscarlo en el código
# ir probando valores de k haciendo análisis exploratorio a no ser que el conocimiento experto nos lo diga
k_values = range(1, 21, 2)

# Lo vimos el otro día, crea la lista de índices para los folds. Cuando estoy
# haciendo pruebas, me gusta fijar la semilla del aleatorio para obtener los
# mismos resultados entre ejecuciones. Esto no siempre funciona, especialmente
# cuando las bibliotecas que usa sklearn tienen su propio generador de
# aleatorios o cuando se utiliza multiprocessing.
# el random_state es para que la gráficia sea igual en cada ejecución
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Creación de diccionario por comprensión. Yo prefiero defaultdict
results = {metric: [] for metric in metrics}

# Cross_validate entrenará clf tantas veces como folds tengamos y usará los
# scorers para evaluar test. Guardo las medias en una lista.
# OJO. Uso una lista porque quiero comprobar las métricas vs el valor de k.
# Si no estáis optimizando hiperparámetros os sirve el diccionario pelao
for k in k_values:
    clf = KNeighborsClassifier(n_neighbors=k)
    scores = cross_validate(clf, X, y, cv=kf, scoring=metrics)

    for metric in metrics:
        results[metric].append(scores[f"test_{metric}"].mean())

# Gráfica simple
plt.figure()
for metric, values in results.items():
    plt.plot(k_values, values, label=metric)

plt.title("KNN Performance Metrics vs. k")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Score")
plt.legend()
plt.xticks(ticks=k_values)
plt.grid()
plt.show()
