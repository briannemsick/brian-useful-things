import numpy as np
from sklearn.datasets import fetch_20newsgroups as load_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

dataset = load_data(
    categories=["alt.atheism", "soc.religion.christian", "talk.politics.guns"],
    shuffle=True,
)
X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

bow = TfidfVectorizer().fit(X_train)
X_train_bow = bow.transform(X_train)
X_test_bow = bow.transform(X_test)

logreg = LogisticRegression(multi_class="auto").fit(X_train_bow, y_train)
y_test_hat = logreg.predict_proba(X_test_bow)
print(accuracy_score(np.argmax(y_test_hat, axis=1), y_test))
