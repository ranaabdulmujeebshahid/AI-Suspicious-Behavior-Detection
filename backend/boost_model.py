import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = "/content/drive/MyDrive/AI_Final_Project/backend"
CLEAN_CSV = os.path.join(BASE_DIR, "dataset", "pose_features_cctv_clean.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "behavior_rf_model.pkl")

df = pd.read_csv(CLEAN_CSV).dropna()
vis_cols = [f"lm_{i}_vis" for i in range(33)]
df = df[df[vis_cols].mean(axis=1) >= 0.45].reset_index(drop=True)
print(f"High-confidence skeleton samples kept: {len(df)}")

feature_cols = [c for c in df.columns if c != "label"]
X = df[feature_cols]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

et = ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1)
hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=8, random_state=42)

clf = VotingClassifier(estimators=[("et", et), ("hgb", hgb)], voting="soft", n_jobs=-1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"\nBoosted Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print(classification_report(y_test, y_pred))

joblib.dump({"model": clf, "feature_columns": feature_cols, "classes": list(clf.classes_)}, MODEL_PATH)
print(f"Saved boosted model to {MODEL_PATH}")
