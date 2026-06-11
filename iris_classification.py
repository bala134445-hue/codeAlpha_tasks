import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
df = pd.read_csv("Iris.csv")

# Display first 5 rows
print("Dataset Preview:")
print(df.head())

# Remove Id column if present
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Features and Target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature Importance Graph
importance = model.feature_importances_

plt.figure(figsize=(8, 5))
plt.bar(X.columns, importance)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()

# Custom Prediction
sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample)

print("\nPredicted Species:")
print(prediction[0])

