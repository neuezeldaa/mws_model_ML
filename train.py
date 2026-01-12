# train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
import joblib
from catboost import CatBoostClassifier

print("Загрузка подготовленных данных...")

X = pd.read_csv('data/gitleaks_dataset_sklearn_X.csv')
y = pd.read_csv('data/gitleaks_dataset_y.csv').iloc[:, 0]

X_cb = pd.read_csv('data/gitleaks_dataset_catboost_X.csv')

numeric_features = [
    'secret_length',
    'secret_special_chars',
    'secret_has_url',
    'StartLine',
    'EndLine',
    'StartColumn',
    'EndColumn'
]
categorical_features = ['RuleID', 'file_extension']

# train/test для sklearn‑моделей
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# train/test для CatBoost
X_cb_train, X_cb_test, y_cb_train, y_cb_test = train_test_split(
    X_cb, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

cat_features = [
    X_cb_train.columns.get_loc('RuleID'),
    X_cb_train.columns.get_loc('file_extension'),
]


# ---------- KNN ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

k = 11
knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
knn.fit(X_train_scaled, y_train)
y_pred_train = knn.predict(X_train_scaled)
y_pred_test = knn.predict(X_test_scaled)
f1_knn = f1_score(y_test, y_pred_test)

print(f"\nKNN F1: {f1_knn:.4f}")

# ---------- Logistic Regression ----------
lr = LogisticRegression(
    max_iter=1000,
    random_state=42,
    solver='lbfgs',
    class_weight='balanced'
)
lr.fit(X_train_scaled, y_train)
y_pred_lr_test = lr.predict(X_test_scaled)
y_proba_lr = lr.predict_proba(X_test_scaled)
f1_lr = f1_score(y_test, y_pred_lr_test)
roc_auc_lr = roc_auc_score(y_test, y_proba_lr[:, 1])

print(f"\nLogReg F1: {f1_lr:.4f}, ROC-AUC: {roc_auc_lr:.4f}")

# ---------- Random Forest ----------
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf_test = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)
f1_rf = f1_score(y_test, y_pred_rf_test)
roc_auc_rf = roc_auc_score(y_test, y_proba_rf[:, 1])

print(f"\nRandomForest F1: {f1_rf:.4f}, ROC-AUC: {roc_auc_rf:.4f}")

# ---------- Gradient Boosting (sklearn) ----------
gb = GradientBoostingClassifier(
    n_estimators=250,
    learning_rate=0.05,
    max_depth=3,
    min_samples_split=40,
    min_samples_leaf=20,
    subsample=0.8,
)
gb.fit(X_train, y_train)

y_pred_gb_test = gb.predict(X_test)
y_proba_gb = gb.predict_proba(X_test)
f1_gb = f1_score(y_test, y_pred_gb_test)
roc_auc_gb = roc_auc_score(y_test, y_proba_gb[:, 1])

print(f"\nGradientBoosting F1: {f1_gb:.4f}, ROC-AUC: {roc_auc_gb:.4f}")

# ---------- CatBoost ----------
print("\nОбучение CatBoost...")

n_negative = (y_cb_train == 0).sum()
n_positive = (y_cb_train == 1).sum()

cb = CatBoostClassifier(
    iterations=200,
    learning_rate=0.2521,
    depth=10,
    l2_leaf_reg=8.28,
    random_strength=6.842,
    min_data_in_leaf=23,
    loss_function='Logloss',
    eval_metric='F1',
    class_weights=[1.0, n_negative / n_positive],
    random_seed=42,
    verbose=False
)

cb.fit(
    X_cb_train,
    y_cb_train,
    cat_features=cat_features,
    eval_set=(X_cb_test, y_cb_test),
    verbose=False
)

y_pred_cb_test = cb.predict(X_cb_test)
y_proba_cb = cb.predict_proba(X_cb_test)[:, 1]
f1_cb = f1_score(y_cb_test, y_pred_cb_test)
roc_auc_cb = roc_auc_score(y_cb_test, y_proba_cb)

print(f"\nCatBoost F1: {f1_cb:.4f}, ROC-AUC: {roc_auc_cb:.4f}")



print("\nF1 по моделям:")
print(f"KNN:              {f1_knn:.4f}")
print(f"LogisticRegression: {f1_lr:.4f}")
print(f"RandomForest:     {f1_rf:.4f}")
print(f"GradientBoosting: {f1_gb:.4f}")
print(f"CatBoost:         {f1_cb:.4f}")

# Сохранение лучших моделей/скейлера
joblib.dump(gb, 'models/model_gb.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
cb.save_model('models/model_cb.cbm')

print("\nМодели сохранены.")


print("\n=== Создание ансамбля ===")

# Предсказания всех моделей на тестовой выборке
models_predictions = {
    'gb': y_proba_gb[:, 1],
    'cb': y_proba_cb,
    'rf': y_proba_rf[:, 1],
}

# Простое взвешенное ансамблирование (можно оптимизировать)
weights = {
    'cb': 0.5,    # CatBoost - лучшая модель
    'gb': 0.3,    # Gradient Boosting
    'rf': 0.2,    # Random Forest
}

ensemble_proba = (
    weights['cb'] * models_predictions['cb'] +
    weights['gb'] * models_predictions['gb'] +
    weights['rf'] * models_predictions['rf']
)

ensemble_pred = (ensemble_proba >= 0.5).astype(int)
f1_ensemble = f1_score(y_test, ensemble_pred)
roc_auc_ensemble = roc_auc_score(y_test, ensemble_proba)

print(f"\nEnsemble F1: {f1_ensemble:.4f}, ROC-AUC: {roc_auc_ensemble:.4f}")

# Сравнение результатов
print("\n=== Итоговое сравнение ===")
results = {
    'KNN': f1_knn,
    'LogReg': f1_lr,
    'RandomForest': f1_rf,
    'GradientBoosting': f1_gb,
    'CatBoost': f1_cb,
}

for model, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{model:20s}: {score:.4f}")

# Сохранение всех необходимых моделей
print("\n=== Сохранение моделей ===")
joblib.dump(gb, 'models/model_gb.pkl')
joblib.dump(rf, 'models/model_rf.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(weights, 'models/ensemble_weights.pkl')
cb.save_model('models/model_cb.cbm')




