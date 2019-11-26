import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


dataset = load_iris()
X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler().fit(X_train)
X_train_mvn = scaler.transform(X_train)
X_test_mvn = scaler.transform(X_test)

logreg = LogisticRegression(multi_class="auto").fit(X_train_mvn, y_train)
y_test_hat = logreg.predict_proba(X_test_mvn)
print(accuracy_score(np.argmax(y_test_hat, axis=1), y_test))
