import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = "/content/drive/MyDrive/AI_Final_Project/backend"
CSV_PATH = os.path.join(BASE_DIR, "dataset", "pose_features.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "behavior_rf_model.pkl")

df = pd.read_csv(CSV_PATH).dropna()
feature_cols = [c for c in df.columns if c != "label"]
X = df[feature_cols]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=clf.classes_))

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump({
    "model": clf,
    "feature_columns": feature_cols,
    "classes": list(clf.classes_)
}, MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")
