import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load dataset
df = pd.read_csv("orders_dataset.csv")

# Features and target
X = df.drop(columns=["returned", "order_id"])
y = df["returned"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Identify column types
numeric_features = [
    "price_inr", "discount_pct", "customer_tenure_days",
    "num_previous_orders", "num_previous_returns",
    "delivery_distance_km", "delivery_days", "rating_given"
]
categorical_features = ["product_category", "payment_method", "is_weekend_order"]

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Baseline model pipeline
baseline_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DummyClassifier(strategy="most_frequent"))
])

# Fit and evaluate
baseline_pipeline.fit(X_train, y_train)
y_pred = baseline_pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, pos_label=1)

print("Baseline DummyClassifier Results")
print("Accuracy:", round(acc, 4))
print("F1 (returned=1):", round(f1, 4))
print("\nExplanation: High accuracy here is misleading because the model "
      "always predicts the majority class (no return). This gives zero recall "
      "for returned orders, which is the real business problem.")
