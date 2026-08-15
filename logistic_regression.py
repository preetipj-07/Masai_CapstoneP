import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score

# --- Load dataset ---
df = pd.read_csv("orders_dataset.csv")
X = df.drop(columns=["returned", "order_id"])
y = df["returned"]

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- Column groups ---
numeric_features = [
    "price_inr", "discount_pct", "customer_tenure_days",
    "num_previous_orders", "num_previous_returns",
    "delivery_distance_km", "delivery_days", "rating_given"
]
categorical_features = ["product_category", "payment_method", "is_weekend_order"]

# --- Preprocessing ---
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# --- Logistic Regression pipeline ---
logreg_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(class_weight="balanced", max_iter=500, random_state=42))
])

# --- Fit model ---
logreg_pipeline.fit(X_train, y_train)

# --- Evaluate at default threshold (0.5) ---
y_proba = logreg_pipeline.predict_proba(X_test)[:, 1]
y_pred_default = (y_proba >= 0.5).astype(int)

print("\nLogistic Regression (default threshold = 0.5)")
print("Accuracy:", round(accuracy_score(y_test, y_pred_default), 4))
print("F1 (returned=1):", round(f1_score(y_test, y_pred_default), 4))
print("Recall:", round(recall_score(y_test, y_pred_default), 4))
print("Precision:", round(precision_score(y_test, y_pred_default), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))

# --- Threshold sweep ---
def sweep_thresholds(y_true, y_proba, start=0.1, end=0.9, step=0.02):
    results = []
    for t in np.arange(start, end + step, step):
        y_pred_t = (y_proba >= t).astype(int)
        results.append({
            "threshold": round(t, 2),
            "f1": f1_score(y_true, y_pred_t),
            "recall": recall_score(y_true, y_pred_t),
            "precision": precision_score(y_true, y_pred_t)
        })
    return results

results = sweep_thresholds(y_test, y_proba)
best = max(results, key=lambda r: r["f1"])

print("\nThreshold sweep summary (showing best F1):")
print("Best threshold:", best["threshold"])
print("F1:", round(best["f1"], 3))
print("Recall:", round(best["recall"], 3))
print("Precision:", round(best["precision"], 3))

print("\nBusiness trade-off:")
print("Lowering the threshold increases recall (catching more risky orders), "
      "but precision drops (more false alarms). In support workflows, "
      "missing a true return is more costly than flagging a few safe orders.")
