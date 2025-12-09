import pandas as pd
import numpy as np
import joblib


# 1. Загружаем тестовый датасет
df = pd.read_csv("gitleaks_test_100.csv")

# 2. Целевая переменная
y_true = (df["IsRealLeak"] == True).astype(int) if df["IsRealLeak"].dtype != int else df["IsRealLeak"]

# 3. Создаём признаки (как при обучении)

# secret_length
df["secret_length"] = df["Secret"].str.len()

# secret_special_chars
special_chars = "!@#$%^&*()-_=+[]{}|;:',.<>?/~`"
df["secret_special_chars"] = df["Secret"].apply(
    lambda x: sum(1 for c in str(x) if c in special_chars)
)

# secret_has_url
df["secret_has_url"] = df["Secret"].str.contains("http", case=False, na=False).astype(int)

# file_extension
df["file_extension"] = df["File"].str.split(".").str[-1]

numeric_features = [
    "secret_length",
    "secret_special_chars",
    "secret_has_url",
    "StartLine",
    "EndLine",
    "StartColumn",
    "EndColumn",
]
categorical_features = ["RuleID", "file_extension"]

# 4. Загружаем энкодеры, скейлер, модель
le_dict = joblib.load("le_dict.pkl")
scaler = joblib.load("scaler.pkl")
model = joblib.load("model_gb.pkl")

# 5. Кодируем категориальные признаки
for col in categorical_features:
    le = le_dict[col]
    # значения, которых не было при обучении, отправляем в первый класс
    mask_unknown = ~df[col].astype(str).isin(le.classes_)
    if mask_unknown.any():
        df.loc[mask_unknown, col] = le.classes_[0]
    df[col] = le.transform(df[col].astype(str))

# 6. Собираем матрицу признаков в правильном порядке
all_features = numeric_features + categorical_features
X = df[all_features]

# 7. Масштабируем
X_scaled = scaler.transform(X)

# 8. Предсказания и уверенность
y_pred = model.predict(X_scaled)              # 0/1
y_proba = model.predict_proba(X_scaled)       # [:,1] — P(RealLeak)

proba_real = y_proba[:, 1]
confidence = np.maximum(proba_real, 1 - proba_real) * 100.0

# 9. Формируем результаты
results = pd.DataFrame({
    "Secret": df["Secret"],
    "RuleID": df["RuleID"],
    "File": df["File"],
    "IsRealLeak_true": y_true,
    "PredictedIsRealLeak": y_pred,
    "IsRealLeak_bool": y_pred.astype(bool),
    "Prob_FalsePositive": (1 - proba_real) * 100.0,
    "Prob_RealLeak": proba_real * 100.0,
    "Confidence": confidence,
})

# 10. Сохраняем
results.to_csv("gitleaks_test_100_results.csv", index=False)
print("Результаты сохранены в gitleaks_test_100_results.csv")
