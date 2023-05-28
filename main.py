import ssl
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def getDataFromFile():
    ssl._create_default_https_context = ssl._create_unverified_context
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    headers = ["sepal_length", "sepal_width", "petal_length", "petal_width",
    "class"]
    df = pd.read_csv(url,names=headers)
    encoder = LabelEncoder()
    df['class'] = encoder.fit_transform(df['class'])
    return df

df = getDataFromFile()
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2023)
kmeans = KMeans(n_clusters=3, n_init=10)
kmeans.fit(X_train, y_train)
kfold = KFold(n_splits=5, random_state=2023, shuffle=True)
scores = cross_val_score(kmeans, X_train, y_train, cv=kfold, scoring="accuracy")
print(scores)
print(f"Średnia dokładność: {scores.mean()}")
best_param = -1
best_param_val = 0
best_accuracy_train = 0.0
best_accuracy_test = 0.0
for i in range(1, 20):
    test = KMeans(n_clusters=i, n_init=10)
    test.fit(X_train, y_train)
    train_labels = test.predict(X_train)
    test_labels = test.predict(X_test)
    accuracy_train = accuracy_score(y_train, train_labels)
    accuracy_test = accuracy_score(y_test, test_labels)
    accuracy = (accuracy_train+accuracy_test)*100/2
    if(accuracy > best_param_val):
        best_param = i
        best_param_val = accuracy
        best_accuracy_train = accuracy_train
        best_accuracy_test = accuracy_test
kmeans = KMeans(n_clusters=best_param, n_init=10)
kmeans.fit(X_train, y_train)
train_labels = kmeans.predict(X_train)
test_labels = kmeans.predict(X_test)
print("Najlepszy parametr: ", best_param)
print("Najlepszy wynik: ", best_param_val)
print("Dokładność na zbiorze treningowym: ", best_accuracy_train)
print("Dokładność na zbiorze testowym: ", best_accuracy_test)
print(classification_report(y_train, train_labels))
