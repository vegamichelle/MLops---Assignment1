import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# load the dataset
df = pd.read_csv("IMDB Dataset.csv")

# the two columns we need are review and sentiment
X = df["review"]
y = df["sentiment"]

# TfidfVectorizer turns the text into numbers the model can understand
# MultinomialNB is the classifier we used in class, works well for text
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("nb", MultinomialNB())
])

# train the model on all the data
model.fit(X, y)
print("done training!")

# save it so the app can use it later without retraining
joblib.dump(model, "sentiment_model.pkl")
print("model saved as sentiment_model.pkl")
