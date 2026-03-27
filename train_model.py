import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# 1. Load the dataset
# -----------------------------
df = pd.read_csv("diabetes.csv")

print("First 5 rows:")
print(df.head())
print("\nColumns:")
print(df.columns.tolist())

# -----------------------------
# 2. Clean obvious invalid values
# In this dataset, 0 may mean missing for some columns
# -----------------------------
columns_with_invalid_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in columns_with_invalid_zeros:
    df[col] = df[col].replace(0, pd.NA)

# -----------------------------
# 3. Split features and target
# -----------------------------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# -----------------------------
# 4. Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 5. Build pipeline
# - impute missing values with median
# - train logistic regression
# -----------------------------
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", LogisticRegression(max_iter=1000))
])

# -----------------------------
# 6. Train model
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# 7. Evaluate model
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel accuracy: {accuracy:.3f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# 8. Save model
# -----------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nSaved trained model as model.pkl")