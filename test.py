import pandas as pd
import numpy as np
import math
from catboost import CatBoostClassifier, Pool

# 1. Загружаем тестовый датасет
df = pd.read_csv("data/gitleaks_test_100.csv")

# 2. Целевая переменная
y_true = (df["IsRealLeak"] == True).astype(int) if df["IsRealLeak"].dtype != int else df["IsRealLeak"]

# ================== ОБРАБОТКА ДАННЫХ ==================

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
    return (
        letters / n,
        digits / n,
        uppers / n,
        specials / n,
    )

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
        "has_prefix_github":   int(s.startswith("ghp_") or s.startswith("gho_") or s.startswith("ghs_")),
        "has_prefix_slack":    int(s_lower.startswith("xoxb-") or s_lower.startswith("xoxp-")),
    }

pref_df = df["Secret"].apply(secret_prefix_flags).apply(pd.Series)
df = pd.concat([df, pref_df], axis=1)

# Контекст (RuleID + File)
def context_flags(row):
    ctx = f"{row.get('RuleID', '')} {row.get('File', '')}"
    ctx = str(ctx).lower()
    return {
        "ctx_has_pass":    int(any(k in ctx for k in ["pass", "pwd"])),
        "ctx_has_token":   int("token" in ctx),
        "ctx_has_key":     int("key" in ctx),
        "ctx_has_secret":  int("secret" in ctx),
        "ctx_has_example": int("example" in ctx or "sample" in ctx or "test" in ctx),
    }

ctx_df = df.apply(context_flags, axis=1, result_type="expand")
df = pd.concat([df, ctx_df], axis=1)

numeric_features = [
    "secret_length",
    "secret_special_chars",
    "secret_has_url",
    "StartLine",
    "EndLine",
    "StartColumn",
    "EndColumn",
    "secret_entropy",
    "secret_letter_ratio",
    "secret_digit_ratio",
    "secret_upper_ratio",
    "secret_special_ratio",
    "secret_base64_ratio",
    "has_prefix_aws_akid",
    "has_prefix_github",
    "has_prefix_slack",
    "ctx_has_pass",
    "ctx_has_token",
    "ctx_has_key",
    "ctx_has_secret",
    "ctx_has_example",
]
categorical_features = ["RuleID", "file_extension"]
all_features = numeric_features + categorical_features


# ================== МОДЕЛЬ CatBoost ==================

cb_model = CatBoostClassifier()
cb_model.load_model("models/model_cb.cbm")

X_cb = df[all_features]

cat_features_idx = [
    X_cb.columns.get_loc("RuleID"),
    X_cb.columns.get_loc("file_extension"),
]

cb_pool = Pool(X_cb, cat_features=cat_features_idx)

y_pred_cb = cb_model.predict(cb_pool)
y_proba_cb = cb_model.predict_proba(cb_pool)[:, 1]
conf_cb = np.maximum(y_proba_cb, 1 - y_proba_cb) * 100.0

# CatBoost возвращает (n,1) или строки, приведём к int
y_pred_cb = np.array(y_pred_cb).astype(int).ravel()

# ================== СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ==================

results = pd.DataFrame({
    "Secret": df["Secret"],
    "RuleID": df["RuleID"],
    "File": df["File"],
    "IsRealLeak_true": y_true,
    "CB_PredictedIsRealLeak": y_pred_cb,
    "CB_Confidence": conf_cb,
})

results.to_csv("data/gitleaks_test_100_results.csv", index=False)
print("Результаты сохранены в data/gitleaks_test_100_results.csv")
