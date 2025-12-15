import pandas as pd
import numpy as np
import math
import joblib
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

print("=" * 70)
print("ТЕСТИРОВАНИЕ МОДЕЛЕЙ С АНСАМБЛИРОВАНИЕМ")
print("=" * 70)

# ================== 1. ЗАГРУЗКА ДАННЫХ ==================

print("\n[1/6] Загрузка тестового датасета...")
df = pd.read_csv("data/gitleaks_test_100.csv")

# Целевая переменная
y_true = (df["IsRealLeak"] == True).astype(int) if df["IsRealLeak"].dtype != int else df["IsRealLeak"]
print(f"Загружено {len(df)} образцов")
print(f"Распределение классов: {y_true.value_counts().to_dict()}")

# ================== 2. ОБРАБОТКА ДАННЫХ ==================

print("\n[2/6] Извлечение признаков...")

# Базовые признаки
df["secret_length"] = df["Secret"].str.len()

SPECIAL = "!@#$%^&*()-_=+[]{}|;:',.<>?/~`"


def count_special_chars(text):
    return sum(1 for c in str(text) if c in SPECIAL)


df["secret_special_chars"] = df["Secret"].apply(count_special_chars)
df["secret_has_url"] = df["Secret"].str.contains("http", case=False, na=False).astype(int)
df["file_extension"] = df["File"].str.split(".").str[-1]


# Энтропия
def shannon_entropy(text: str) -> float:
    s = str(text)
    if not s:
        return 0.0
    from collections import Counter
    counts = Counter(s)
    total = len(s)
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent


df["secret_entropy"] = df["Secret"].apply(shannon_entropy)


# Доли типов символов
def char_stats(text: str):
    s = str(text)
    if not s:
        return 0, 0, 0, 0
    letters = sum(c.isalpha() for c in s)
    digits = sum(c.isdigit() for c in s)
    uppers = sum(c.isupper() for c in s)
    specials = sum(c in SPECIAL for c in s)
    n = len(s)
    return (letters / n, digits / n, uppers / n, specials / n)


stats = df["Secret"].apply(char_stats)
df["secret_letter_ratio"] = stats.apply(lambda t: t[0])
df["secret_digit_ratio"] = stats.apply(lambda t: t[1])
df["secret_upper_ratio"] = stats.apply(lambda t: t[2])
df["secret_special_ratio"] = stats.apply(lambda t: t[3])

# Base64‑подобность
BASE64_ALPH = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="


def base64_ratio(text: str) -> float:
    s = str(text)
    if not s:
        return 0.0
    cnt = sum(c in BASE64_ALPH for c in s)
    return cnt / len(s)


df["secret_base64_ratio"] = df["Secret"].apply(base64_ratio)


# Префиксы
def secret_prefix_flags(text: str):
    s = str(text)
    s_lower = s.lower()
    return {
        "has_prefix_aws_akid": int(s.startswith("AKIA") or s.startswith("ASIA")),
        "has_prefix_github": int(s.startswith("ghp_") or s.startswith("gho_") or s.startswith("ghs_")),
        "has_prefix_slack": int(s_lower.startswith("xoxb-") or s_lower.startswith("xoxp-")),
    }


pref_df = df["Secret"].apply(secret_prefix_flags).apply(pd.Series)
df = pd.concat([df, pref_df], axis=1)


# Контекст
def context_flags(row):
    ctx = f"{row.get('RuleID', '')} {row.get('File', '')}"
    ctx = str(ctx).lower()
    return {
        "ctx_has_pass": int(any(k in ctx for k in ["pass", "pwd"])),
        "ctx_has_token": int("token" in ctx),
        "ctx_has_key": int("key" in ctx),
        "ctx_has_secret": int("secret" in ctx),
        "ctx_has_example": int("example" in ctx or "sample" in ctx or "test" in ctx),
    }


ctx_df = df.apply(context_flags, axis=1, result_type="expand")
df = pd.concat([df, ctx_df], axis=1)

numeric_features = [
    "secret_length", "secret_special_chars", "secret_has_url",
    "StartLine", "EndLine", "StartColumn", "EndColumn",
    "secret_entropy", "secret_letter_ratio", "secret_digit_ratio",
    "secret_upper_ratio", "secret_special_ratio", "secret_base64_ratio",
    "has_prefix_aws_akid", "has_prefix_github", "has_prefix_slack",
    "ctx_has_pass", "ctx_has_token", "ctx_has_key",
    "ctx_has_secret", "ctx_has_example",
]
categorical_features = ["RuleID", "file_extension"]
all_features = numeric_features + categorical_features

print(f"Извлечено {len(all_features)} признаков")

# ================== 3. ЗАГРУЗКА МОДЕЛЕЙ ==================

print("\n[3/6] Загрузка моделей...")

# CatBoost
cb_model = CatBoostClassifier()
cb_model.load_model("models/model_cb.cbm")
print("✓ CatBoost загружен")

# Gradient Boosting
gb_model = joblib.load("models/model_gb.pkl")
print("✓ Gradient Boosting загружен")

# Random Forest
rf_model = joblib.load("models/model_rf.pkl")
print("✓ Random Forest загружен")

# Label Encoder для sklearn моделей
le_dict = joblib.load("models/le_dict.pkl")
print("✓ Label Encoders загружены")

# Веса ансамбля
try:
    ensemble_weights = joblib.load("models/ensemble_weights.pkl")
    print(f"✓ Веса ансамбля загружены: {ensemble_weights}")
except FileNotFoundError:
    ensemble_weights = {'cb': 0.5, 'gb': 0.3, 'rf': 0.2}
    print(f"⚠ Файл весов не найден, используются дефолтные: {ensemble_weights}")

# ================== 4. ПРЕДСКАЗАНИЯ ==================

print("\n[4/6] Получение предсказаний...")

# Подготовка данных для CatBoost
X_cb = df[all_features].copy()
cat_features_idx = [
    X_cb.columns.get_loc("RuleID"),
    X_cb.columns.get_loc("file_extension"),
]
cb_pool = Pool(X_cb, cat_features=cat_features_idx)

# CatBoost
y_pred_cb = np.array(cb_model.predict(cb_pool)).astype(int).ravel()
y_proba_cb = cb_model.predict_proba(cb_pool)[:, 1]
print("✓ CatBoost: предсказания получены")

# Подготовка данных для sklearn моделей (с label encoding)
X_sklearn = df[all_features].copy()
for col in categorical_features:
    if col in le_dict:
        le = le_dict[col]
        X_sklearn[col] = X_sklearn[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        X_sklearn[col] = le.transform(X_sklearn[col])

# Gradient Boosting
y_pred_gb = gb_model.predict(X_sklearn)
y_proba_gb = gb_model.predict_proba(X_sklearn)[:, 1]
print("✓ Gradient Boosting: предсказания получены")

# Random Forest
y_pred_rf = rf_model.predict(X_sklearn)
y_proba_rf = rf_model.predict_proba(X_sklearn)[:, 1]
print("✓ Random Forest: предсказания получены")

# ================== 5. АНСАМБЛИРОВАНИЕ ==================

print("\n[5/6] Создание ансамбля...")

# Взвешенное ансамблирование
ensemble_proba = (
        ensemble_weights['cb'] * y_proba_cb +
        ensemble_weights['gb'] * y_proba_gb +
        ensemble_weights['rf'] * y_proba_rf
)

# Финальные предсказания
y_pred_ensemble = (ensemble_proba >= 0.5).astype(int)

# Confidence для всех моделей
conf_cb = np.maximum(y_proba_cb, 1 - y_proba_cb) * 100.0
conf_gb = np.maximum(y_proba_gb, 1 - y_proba_gb) * 100.0
conf_rf = np.maximum(y_proba_rf, 1 - y_proba_rf) * 100.0
conf_ensemble = np.maximum(ensemble_proba, 1 - ensemble_proba) * 100.0

print("✓ Ансамбль создан")

# ================== 6. МЕТРИКИ И СОХРАНЕНИЕ ==================

print("\n[6/6] Вычисление метрик...")


def print_metrics(y_true, y_pred, y_proba, model_name):
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)
    print(f"\n{model_name}:")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  Avg Confidence: {np.maximum(y_proba, 1 - y_proba).mean() * 100:.2f}%")
    return f1, roc_auc


# Метрики для каждой модели
print("\n" + "=" * 70)
print("РЕЗУЛЬТАТЫ МОДЕЛЕЙ")
print("=" * 70)

f1_cb, auc_cb = print_metrics(y_true, y_pred_cb, y_proba_cb, "CatBoost")
f1_gb, auc_gb = print_metrics(y_true, y_pred_gb, y_proba_gb, "Gradient Boosting")
f1_rf, auc_rf = print_metrics(y_true, y_pred_rf, y_proba_rf, "Random Forest")
f1_ens, auc_ens = print_metrics(y_true, y_pred_ensemble, ensemble_proba, "Ensemble")

# Сравнительная таблица
print("\n" + "=" * 70)
print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА")
print("=" * 70)
comparison = pd.DataFrame({
    'Model': ['CatBoost', 'Gradient Boosting', 'Random Forest', 'Ensemble'],
    'F1-Score': [f1_cb, f1_gb, f1_rf, f1_ens],
    'ROC-AUC': [auc_cb, auc_gb, auc_rf, auc_ens],
    'Avg Confidence': [
        conf_cb.mean(),
        conf_gb.mean(),
        conf_rf.mean(),
        conf_ensemble.mean()
    ]
})
comparison = comparison.sort_values('F1-Score', ascending=False)
print(comparison.to_string(index=False))

# Confusion Matrix для ансамбля
print("\n" + "=" * 70)
print("CONFUSION MATRIX (Ensemble)")
print("=" * 70)
cm = confusion_matrix(y_true, y_pred_ensemble)
print(cm)
print("\nClassification Report (Ensemble):")
print(classification_report(y_true, y_pred_ensemble, target_names=['False Positive', 'Real Leak']))

# Сохранение результатов
results = pd.DataFrame({
    "Secret": df["Secret"],
    "RuleID": df["RuleID"],
    "File": df["File"],
    "IsRealLeak_true": y_true,

    # CatBoost
    "CB_Predicted": y_pred_cb,
    "CB_Probability": y_proba_cb * 100.0,
    "CB_Confidence": conf_cb,

    # Gradient Boosting
    "GB_Predicted": y_pred_gb,
    "GB_Probability": y_proba_gb * 100.0,
    "GB_Confidence": conf_gb,

    # Random Forest
    "RF_Predicted": y_pred_rf,
    "RF_Probability": y_proba_rf * 100.0,
    "RF_Confidence": conf_rf,

    # Ensemble
    "Ensemble_Predicted": y_pred_ensemble,
    "Ensemble_Probability": ensemble_proba * 100.0,
    "Ensemble_Confidence": conf_ensemble,
})

output_file = "gitleaks_test_100_results_ensemble.csv"
results.to_csv(output_file, index=False)

print("\n" + "=" * 70)
print(f"✓ Результаты сохранены в {output_file}")
print("=" * 70)

# Анализ улучшения confidence
print("\n" + "=" * 70)
print("АНАЛИЗ УЛУЧШЕНИЯ CONFIDENCE")
print("=" * 70)

low_conf_cb = (conf_cb < 60).sum()
low_conf_ensemble = (conf_ensemble < 60).sum()

print(f"Образцы с confidence < 60% (CatBoost):  {low_conf_cb} ({low_conf_cb / len(df) * 100:.1f}%)")
print(f"Образцы с confidence < 60% (Ensemble):  {low_conf_ensemble} ({low_conf_ensemble / len(df) * 100:.1f}%)")
print(f"Улучшение: -{low_conf_cb - low_conf_ensemble} образцов")

print(f"\nСредняя confidence CatBoost:  {conf_cb.mean():.2f}%")
print(f"Средняя confidence Ensemble:  {conf_ensemble.mean():.2f}%")
print(f"Прирост: +{conf_ensemble.mean() - conf_cb.mean():.2f}%")



