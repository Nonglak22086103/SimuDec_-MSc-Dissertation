import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# =========================
# CONFIG
# =========================
DATA_PATH = "survey_with_scenarios.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

BASE_FEATURES = [
    "coolant_flag",
    "speed_drill_rpm",
    "speed_ream_rpm",
    "feed_mm_rev",
    "tool_wear_fraction",
    "pilot_hole_diam_mm",
    "runout_mm",
    "tool_overhang_mm",
]

OPTIONAL_FEATURES = ["defect_present", "defect_type"]

# 3 levels each (81 configs)
PARAM_GRID = {
    "n_estimators": [200, 500, 800],          # number of trees
    "max_features": ["sqrt", "log2", 0.8],    # mtry
    "min_samples_leaf": [1, 5, 10],           # minimum node size
    "max_samples": [0.6, 0.8, 1.0],           # sample size per tree
}


def extract_code_prefix(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    m = re.match(r"^([A-Z]\d+)\s*[:\-]?\s*", s)
    return m.group(1) if m else s


def build_X(df):
    features = BASE_FEATURES.copy()
    for f in OPTIONAL_FEATURES:
        if f in df.columns:
            features.append(f)

    X = df[features].copy()

    # numeric coercion
    for c in BASE_FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    if "defect_present" in X.columns:
        X["defect_present"] = pd.to_numeric(X["defect_present"], errors="coerce")

    # normalize + one-hot defect_type
    if "defect_type" in X.columns:
        X["defect_type"] = X["defect_type"].astype(str).str.strip().str.lower()
        X = pd.get_dummies(X, columns=["defect_type"], drop_first=False)

    # impute
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna("UNKNOWN")

    return X


def gridsearch_fit_predict(X_train, y_train, X_test, target_name):
    rf = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
        bootstrap=True,
    )

    grid = GridSearchCV(
        estimator=rf,
        param_grid=PARAM_GRID,
        scoring="accuracy",
        cv=CV_FOLDS,
        n_jobs=-1,
        verbose=1,
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    return best_model, grid.best_params_, grid.best_score_, y_pred


def main():
    df = pd.read_csv(DATA_PATH)

    # -------- Build X once --------
    X = build_X(df)

    # -------- Prepare BOTH targets on the SAME rows --------
    y1_raw = df["primary_root_cause"].replace("", np.nan).apply(extract_code_prefix)
    y2_raw = df["primary_corrective_action"].replace("", np.nan).apply(extract_code_prefix)

    # Keep only rows where BOTH targets exist (so the split is identical)
    mask = y1_raw.notna() & y2_raw.notna()
    X = X.loc[mask].reset_index(drop=True)
    y1_raw = y1_raw.loc[mask].reset_index(drop=True)
    y2_raw = y2_raw.loc[mask].reset_index(drop=True)

    # Encode targets
    le1 = LabelEncoder()
    le2 = LabelEncoder()
    y1 = le1.fit_transform(y1_raw)
    y2 = le2.fit_transform(y2_raw)

    # -------- ONE train/test split used for BOTH models --------
    # Stratify: choose y1 (root cause) so class distribution is preserved there
    X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
        X, y1, y2,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y1
    )

    # -------- Generate ONE random 5 indices from the test set --------
    rng = np.random.default_rng(RANDOM_STATE)
    idx5 = rng.choice(len(X_test), size=5, replace=False)

    # =========================
    # MODEL 1: Root Cause
    # =========================
    model1, best_params1, best_cv1, y1_pred = gridsearch_fit_predict(
        X_train, y1_train, X_test, "Model 1"
    )

    acc1 = accuracy_score(y1_test, y1_pred)
    print("\n" + "=" * 90)
    print("MODEL 1 (Root Cause) - Grid Search Results")
    print("Best params:", best_params1)
    print(f"Best CV score: {best_cv1:.4f}")
    print(f"TEST accuracy: {acc1:.4f}\n")
    print(classification_report(y1_test, y1_pred, target_names=le1.classes_, zero_division=0))

    # Print SAME random 5 samples for Model 1
    samp1 = X_test.iloc[idx5].copy()
    samp1["True_Label"] = le1.inverse_transform(y1_test[idx5])
    samp1["Predicted_Label"] = le1.inverse_transform(y1_pred[idx5])
    print("\n--- SAME RANDOM 5 TEST SAMPLES (MODEL 1) ---")
    print(samp1)

    # =========================
    # MODEL 2: Corrective Action
    # =========================
    model2, best_params2, best_cv2, y2_pred = gridsearch_fit_predict(
        X_train, y2_train, X_test, "Model 2"
    )

    acc2 = accuracy_score(y2_test, y2_pred)
    print("\n" + "=" * 90)
    print("MODEL 2 (Corrective Action) - Grid Search Results")
    print("Best params:", best_params2)
    print(f"Best CV score: {best_cv2:.4f}")
    print(f"TEST accuracy: {acc2:.4f}\n")
    print(classification_report(y2_test, y2_pred, target_names=le2.classes_, zero_division=0))

    # Print SAME random 5 samples for Model 2 (same X_test rows!)
    samp2 = X_test.iloc[idx5].copy()
    samp2["True_Label"] = le2.inverse_transform(y2_test[idx5])
    samp2["Predicted_Label"] = le2.inverse_transform(y2_pred[idx5])
    print("\n--- SAME RANDOM 5 TEST SAMPLES (MODEL 2) ---")
    print(samp2)


if __name__ == "__main__":
    main()
