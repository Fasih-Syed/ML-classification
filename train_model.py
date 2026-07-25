import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv("placement.csv")

# Features
X = data[["CGPA", "Internships", "Aptitude", "Communication", "Projects"]]

# Target
y = data["Placed"]

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("Model Trained Successfully!")