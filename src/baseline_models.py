import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/processed/train.csv")
df["label"] = df["sentiment"].map({"negative": 0, "positive": 1})

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df["review"])
y = df["label"]

lr = LogisticRegression()
lr.fit(X, y)

nb = MultinomialNB()
nb.fit(X, y)

joblib.dump(tfidf, "models/baseline/tfidf_vectorizer.pkl")
joblib.dump(lr, "models/baseline/logistic_regression.pkl")
joblib.dump(nb, "models/baseline/naive_bayes.pkl")

print("Logistic Regression Accuracy:", accuracy_score(y, lr.predict(X)))
print("Naive Bayes Accuracy:", accuracy_score(y, nb.predict(X)))
