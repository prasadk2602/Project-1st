import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("data/hate_speech.csv")

# Features and labels
X = df["text"]
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', LogisticRegression())
])

# Train model
model.fit(X_train, y_train)

# Save model
with open("models/model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained successfully!")