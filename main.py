import ssl

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
def getDataFromFile():
    ssl._create_default_https_context = ssl._create_unverified_context
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    headers = ["sepal_length", "sepal_width", "petal_length", "petal_width",
    "class"]
    df = pd.read_csv(url,names=headers)
    return df
df = getDataFromFile()
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2023)
bayes = GaussianNB()
bayes.fit(X_train, y_train)
kfold = KFold(n_splits=5,random_state=2023,shuffle=True)
scores = cross_val_score(bayes, X_train, y_train, cv=kfold, scoring="accuracy")
print(scores)
print(f"Średnia dokładność: {scores.mean()}")
param_grid = {
 'var_smoothing': np.logspace(0,-9, num=100)
}
grid_search = GridSearchCV(bayes, param_grid, cv=KFold(n_splits=5,
random_state=2023,shuffle=True))
grid_search.fit(X_train, y_train)
results = pd.DataFrame(grid_search.cv_results_)
print(results)
results.to_csv("results.csv")
print(results["mean_test_score"])
print("Najlepsze parametry: ", grid_search.best_params_)
print("Najlepszy wynik: ", grid_search.best_score_)
best_model = grid_search.best_estimator_
best_model = grid_search.best_estimator_
best_predict = best_model.predict(X_test)
print("Dokładność modelu na zbiorze testowym: ", accuracy_score(y_test,
best_predict))
best_predict_train = best_model.predict(X_train)
print("Dokładność na zbiorze treningowym: ", accuracy_score(y_train,
best_predict_train ))
best_predict = best_model.predict(X_test)
print("Dokładność na zbiorze testowym: ", accuracy_score(y_test, best_predict))
cm_train = confusion_matrix(y_train, best_predict_train)
print("Macierz pomyłek dla zbioru treningowego:")
print(cm_train)
report = classification_report(y_train, best_predict_train)
print(report)