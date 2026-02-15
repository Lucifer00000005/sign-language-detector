import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Loading dataset...")

data = pd.read_csv("dataset.csv")

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Model accuracy:", accuracy)

joblib.dump(model, "sign_model.pkl")
print("Model saved as sign_model.pkl")
