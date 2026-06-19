"""
Task 1 - Iris Flower Classification
-------------------------------------
Loads the Iris dataset, trains multiple classifiers, evaluates them,
and prints a comparison along with a detailed report for the best model.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------
# 1. Load Data
# ---------------------------------------------------
DATA_PATH = "Iris.csv"   # change path if needed
df = pd.read_csv(r"C:\Users\ADMIN\Downloads\Iris.csv")

# Drop the Id column if present (not a useful feature)
if "Id" in df.columns:
    df = df.drop(columns=["Id"])

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nClass distribution:\n", df["Species"].value_counts())

# ---------------------------------------------------
# 2. Prepare Features & Labels
# ---------------------------------------------------
X = df.drop(columns=["Species"])
y = df["Species"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)   # Iris-setosa -> 0, Iris-versicolor -> 1, Iris-virginica -> 2

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features (helps KNN, SVM, Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------
# 3. Train & Compare Multiple Models
# ---------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (RBF Kernel)": SVC(kernel="rbf", C=1.0),
}

results = {}
print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    results[name] = acc
    print(f"{name:22s} | Test Accuracy: {acc*100:6.2f}% | "
          f"CV Mean: {cv_scores.mean()*100:6.2f}%")

# ---------------------------------------------------
# 4. Detailed Report for Best Model
# ---------------------------------------------------
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
best_preds = best_model.predict(X_test_scaled)

print("\n" + "=" * 50)
print(f"BEST MODEL: {best_model_name}")
print("=" * 50)
print("\nClassification Report:\n",
      classification_report(y_test, best_preds, target_names=le.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, best_preds))



