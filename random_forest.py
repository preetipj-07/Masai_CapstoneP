import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")

# ── Load dataset ──
df = pd.read_csv("orders_dataset.csv")
X = df.drop(columns=["returned", "order_id"])
y = df["returned"]

# ── Train/test split ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ── Column groups ──
numeric_features = [
    "price_inr", "discount_pct", "customer_tenure_days",
    "num_previous_orders", "num_previous_returns",
    "delivery_distance_km", "delivery_days", "rating_given"
]
categorical_features = ["product_category", "payment_method", "is_weekend_order"]

# ── Preprocessing pipeline ──
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

# ═══════════════════════════════════════════════════════════════
# TASK 6: Random Forest + GridSearchCV
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("TASK 6: Random Forest + GridSearchCV")
print("=" * 60)

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(class_weight="balanced", random_state=42))
])

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [6, 10, None]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    rf_pipeline, param_grid,
    scoring="roc_auc", cv=cv, n_jobs=-1, verbose=0
)
grid_search.fit(X_train, y_train)

print("\nBest parameters:", grid_search.best_params_)
print("Best cross-validated ROC-AUC:", round(grid_search.best_score_, 4))

# ── Held-out test evaluation ──
best_rf = grid_search.best_estimator_
y_proba_rf = best_rf.predict_proba(X_test)[:, 1]
y_pred_rf = (y_proba_rf >= 0.5).astype(int)

test_roc_auc = roc_auc_score(y_test, y_proba_rf)
print("Test-set ROC-AUC:", round(test_roc_auc, 4))
print("Cross-val vs Test gap:", round(abs(grid_search.best_score_ - test_roc_auc), 4))
print("(Gap < 0.05 indicates no severe overfitting)")

print("\nFull classification report (threshold=0.5):")
print(classification_report(y_test, y_pred_rf, target_names=["No Return", "Return"]))

# ═══════════════════════════════════════════════════════════════
# TASK 7: Feature Importance + Permutation Importance
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TASK 7: Feature Importance Analysis")
print("=" * 60)

# Get feature names after preprocessing
ohe_features = best_rf.named_steps["preprocessor"].named_transformers_["cat"] \
    .named_steps["onehot"].get_feature_names_out(categorical_features).tolist()
all_feature_names = numeric_features + ohe_features

# Impurity-based importance
rf_model = best_rf.named_steps["classifier"]
importances = rf_model.feature_importances_
feat_imp_df = pd.DataFrame({
    "feature": all_feature_names,
    "impurity_importance": importances
}).sort_values("impurity_importance", ascending=False)

print("\nTop 10 features (impurity-based, one-hot encoded):")
print(feat_imp_df.head(10).to_string(index=False))

# Aggregate impurity importance back to raw feature level for fair comparison
raw_imp = {}
for feat_name, imp_val in zip(all_feature_names, importances):
    # Map one-hot features back to their parent
    if feat_name in numeric_features:
        parent = feat_name
    else:
        parent = feat_name.rsplit("_", 1)[0] if "_" in feat_name else feat_name
        # Handle cases like "payment_method_COD" -> "payment_method"
        for cat_feat in categorical_features:
            if feat_name.startswith(cat_feat):
                parent = cat_feat
                break
    raw_imp[parent] = raw_imp.get(parent, 0) + imp_val

raw_imp_df = pd.DataFrame([
    {"feature": k, "impurity_importance": v} for k, v in raw_imp.items()
]).sort_values("impurity_importance", ascending=False)

print("\nImpurity importance aggregated to raw feature level:")
print(raw_imp_df.to_string(index=False))

top5_raw = raw_imp_df.head(5)["feature"].tolist()
print("\nTop 5 raw features (impurity-aggregated):", top5_raw)

# Permutation importance on test set (runs on raw features before preprocessing)
print("\nComputing permutation importance on test set...")
perm_result = permutation_importance(
    best_rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
)

raw_feature_names = X_test.columns.tolist()
perm_imp_df = pd.DataFrame({
    "feature": raw_feature_names,
    "perm_mean": perm_result.importances_mean,
    "perm_std": perm_result.importances_std
}).sort_values("perm_mean", ascending=False)

print("\nTop 10 features (permutation-based):")
print(perm_imp_df.head(10).to_string(index=False))

# ── Side-by-side comparison ──
print("\n--- Side-by-side comparison (top 5 impurity vs permutation, raw feature level) ---")
for feat in top5_raw:
    imp_row = raw_imp_df[raw_imp_df["feature"] == feat]
    imp_val = imp_row["impurity_importance"].values[0]
    imp_rank = raw_imp_df.index.get_loc(imp_row.index[0]) + 1
    perm_row = perm_imp_df[perm_imp_df["feature"] == feat]
    if len(perm_row) > 0:
        perm_rank = perm_imp_df.index.get_loc(perm_row.index[0]) + 1
        perm_val = perm_row["perm_mean"].values[0]
    else:
        perm_rank = "N/A"
        perm_val = 0.0
    print(f"  {feat:35s}  impurity={imp_val:.4f} (rank {imp_rank})  permutation={perm_val:.4f} (rank {perm_rank})")

# Identify features that drop under permutation
print("\nFeature importance drop analysis:")
for feat in top5_raw:
    imp_val = raw_imp_df[raw_imp_df["feature"] == feat]["impurity_importance"].values[0]
    perm_row = perm_imp_df[perm_imp_df["feature"] == feat]
    if len(perm_row) > 0:
        perm_val = perm_row["perm_mean"].values[0]
        drop_pct = ((imp_val - perm_val) / imp_val * 100) if imp_val > 0 else 0
        marker = " *** DROPS SUBSTANTIALLY" if drop_pct > 50 else ""
        print(f"  {feat:35s}  drop: {drop_pct:.1f}%{marker}")

print("\nWhy impurity-based importance overrates noisy continuous features:")
print("Impurity importance measures how much a feature reduces Gini impurity across all")
print("splits in all trees. A high-cardinality continuous feature (like delivery_distance_km)")
print("offers many possible split points, so the algorithm can find 'useful-looking' splits")
print("by chance even when the feature carries no real signal. Permutation importance, by contrast,")
print("measures the actual drop in model performance when a feature is shuffled, so it is not")
print("inflated by spurious splits on noisy continuous columns.")

# ═══════════════════════════════════════════════════════════════
# TASK 8: Subgroup / Root-Cause Analysis
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TASK 8: Subgroup / Root-Cause Analysis")
print("=" * 60)

# Get test set predictions
test_df = X_test.copy()
test_df["y_true"] = y_test.values
test_df["y_pred"] = y_pred_rf
test_df["y_proba"] = y_proba_rf

overall_recall = recall_score(y_test, y_pred_rf)
overall_precision = precision_score(y_test, y_pred_rf)
print(f"\nOverall recall: {overall_recall:.4f}")
print(f"Overall precision: {overall_precision:.4f}")

# ── By product_category ──
print("\n--- Recall and Precision by product_category ---")
cat_results = []
for cat in sorted(test_df["product_category"].unique()):
    mask = test_df["product_category"] == cat
    y_true_cat = test_df.loc[mask, "y_true"]
    y_pred_cat = test_df.loc[mask, "y_pred"]
    n = mask.sum()
    pos = y_true_cat.sum()
    if pos > 0:
        rec = recall_score(y_true_cat, y_pred_cat)
        prec = precision_score(y_true_cat, y_pred_cat, zero_division=0)
        cat_results.append({"category": cat, "n": n, "positives": pos, "recall": rec, "precision": prec})
        print(f"  {cat:15s}  n={n:4d}  pos={pos:3d}  recall={rec:.4f}  precision={prec:.4f}")
    else:
        print(f"  {cat:15s}  n={n:4d}  pos=0  (no positives to evaluate)")

# ── By payment_method ──
print("\n--- Recall and Precision by payment_method ---")
pay_results = []
for pay in sorted(test_df["payment_method"].unique()):
    mask = test_df["payment_method"] == pay
    y_true_pay = test_df.loc[mask, "y_true"]
    y_pred_pay = test_df.loc[mask, "y_pred"]
    n = mask.sum()
    pos = y_true_pay.sum()
    if pos > 0:
        rec = recall_score(y_true_pay, y_pred_pay)
        prec = precision_score(y_true_pay, y_pred_pay, zero_division=0)
        pay_results.append({"payment": pay, "n": n, "positives": pos, "recall": rec, "precision": prec})
        print(f"  {pay:15s}  n={n:4d}  pos={pos:3d}  recall={rec:.4f}  precision={prec:.4f}")
    else:
        print(f"  {pay:15s}  n={n:4d}  pos=0  (no positives to evaluate)")

# ── Identify weakest subgroup ──
all_results = []
for r in cat_results:
    all_results.append({"group": f"cat:{r['category']}", "recall": r["recall"], "precision": r["precision"]})
for r in pay_results:
    all_results.append({"group": f"pay:{r['payment']}", "recall": r["recall"], "precision": r["precision"]})

worst_f1 = 0
worst_group = None
for r in all_results:
    if r["recall"] + r["precision"] > 0:
        f1 = 2 * r["recall"] * r["precision"] / (r["recall"] + r["precision"])
        if f1 < worst_f1 or worst_group is None:
            worst_f1 = f1
            worst_group = r

print(f"\nWeakest subgroup: {worst_group['group']} (F1={worst_f1:.4f})")
print(f"  Overall recall={overall_recall:.4f}, precision={overall_precision:.4f}")
print(f"  Subgroup recall={worst_group['recall']:.4f}, precision={worst_group['precision']:.4f}")
print(f"\nProposed next step: Add a category-specific threshold adjustment for {worst_group['group']}")
print(f"  -- use a lower decision threshold for this subgroup to boost recall where it is weakest,")
print(f"  or engineer a subgroup-specific feature (e.g. category_x_COD interaction term) to help")
print(f"  the model distinguish within this subgroup more effectively.")

# ═══════════════════════════════════════════════════════════════
# TASK 9: Save Model + Compute t*_rf
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TASK 9: Save Model + Compute t*_rf")
print("=" * 60)

# ── Threshold sweep on the Random Forest's own predict_proba ──
def sweep_thresholds(y_true, y_proba, start=0.1, end=0.9, step=0.02):
    results = []
    for t in np.arange(start, end + step, step):
        y_pred_t = (y_proba >= t).astype(int)
        rec = recall_score(y_true, y_pred_t)
        prec = precision_score(y_true, y_pred_t, zero_division=0)
        f1 = f1_score(y_true, y_pred_t)
        results.append({"threshold": round(t, 2), "f1": f1, "recall": rec, "precision": prec})
    return results

results_rf = sweep_thresholds(y_test, y_proba_rf)
best_rf_thresh = max(results_rf, key=lambda r: r["f1"])

t_rf = best_rf_thresh["threshold"]
print(f"\nt*_rf (F1-maximising threshold on RF predict_proba): {t_rf}")
print(f"  F1: {best_rf_thresh['f1']:.4f}")
print(f"  Recall: {best_rf_thresh['recall']:.4f}")
print(f"  Precision: {best_rf_thresh['precision']:.4f}")

# ── Compare with default threshold recall ──
default_recall = recall_score(y_test, (y_proba_rf >= 0.5).astype(int))
recall_improvement = best_rf_thresh["recall"] - default_recall
print(f"\nDefault threshold (0.5) recall: {default_recall:.4f}")
print(f"Optimal threshold ({t_rf}) recall: {best_rf_thresh['recall']:.4f}")
print(f"Recall improvement: {recall_improvement:.4f} ({recall_improvement*100:.1f} percentage points)")
prec_drop = precision_score(y_test, (y_proba_rf >= 0.5).astype(int)) - best_rf_thresh["precision"]
print(f"Precision drop: {prec_drop:.4f} ({prec_drop*100:.1f} percentage points)")

# ── Define risk buckets anchored to t*_rf ──
low_bucket = t_rf
high_bucket = t_rf + 0.15
print(f"\nRisk bucket cut points (anchored to t*_rf={t_rf}):")
print(f"  Low:    probability < {low_bucket:.2f}")
print(f"  Medium: {low_bucket:.2f} <= probability < {high_bucket:.2f}")
print(f"  High:   probability >= {high_bucket:.2f}")

# ── Save the final pipeline ──
joblib.dump(best_rf, "models/return_risk_model.pkl")
print(f"\nModel saved to models/return_risk_model.pkl")

# Verify it loads correctly
loaded_model = joblib.load("models/return_risk_model.pkl")
y_verify = loaded_model.predict_proba(X_test)[:, 1]
assert np.allclose(y_proba_rf, y_verify), "Loaded model produces different predictions!"
print("Verification: loaded model produces identical predictions.")

# ── Save metadata for Part 3 ──
metadata = {
    "t_rf": t_rf,
    "low_threshold": low_bucket,
    "high_threshold": high_bucket,
    "feature_names": all_feature_names,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "test_roc_auc": test_roc_auc,
    "cv_roc_auc": grid_search.best_score_,
    "best_params": grid_search.best_params_
}
joblib.dump(metadata, "models/return_risk_metadata.pkl")
print("Metadata saved to models/return_risk_metadata.pkl")
print(f"\nFinal summary: CV ROC-AUC={grid_search.best_score_:.4f}, Test ROC-AUC={test_roc_auc:.4f}, t*={t_rf}")
