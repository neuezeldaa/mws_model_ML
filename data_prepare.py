# prepare_data.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import math

print("Очистка и подготовка данных...")

data = pd.read_csv('gitleaks_detection_dataset.csv')

data_filtered = data.copy()
data_filtered = data_filtered.drop(columns=['Match', 'Email', 'Date'])
data_filtered['IsRealLeak'] = (data_filtered['IsRealLeak'] == True).astype(int)


# Длина секрета
data_filtered['secret_length'] = data_filtered['Secret'].str.len()

# Кол-во спецсимволов
SPECIAL = "!@#$%^&*()-_=+[]{}|;:',.<>?/~`"
def count_special_chars(text):
    return sum(1 for c in str(text) if c in SPECIAL)

data_filtered['secret_special_chars'] = data_filtered['Secret'].apply(count_special_chars)

# Есть ли url
data_filtered['secret_has_url'] = data_filtered['Secret'].str.contains('http', case=False, na=False).astype(int)

# Расширение файла
data_filtered['file_extension'] = data_filtered['File'].str.split('.').str[-1]


# Энтропия секрета
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

data_filtered['secret_entropy'] = data_filtered['Secret'].apply(shannon_entropy)

# Доли разных типов символов
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

stats = data_filtered['Secret'].apply(char_stats)
data_filtered['secret_letter_ratio'] = stats.apply(lambda t: t[0])
data_filtered['secret_digit_ratio'] = stats.apply(lambda t: t[1])
data_filtered['secret_upper_ratio'] = stats.apply(lambda t: t[2])
data_filtered['secret_special_ratio'] = stats.apply(lambda t: t[3])

# Base64‑подобность (доля символов из алфавита base64)
BASE64_ALPH = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
def base64_ratio(text: str) -> float:
    s = str(text)
    if not s:
        return 0.0
    cnt = sum(c in BASE64_ALPH for c in s)
    return cnt / len(s)

data_filtered['secret_base64_ratio'] = data_filtered['Secret'].apply(base64_ratio)

# Префиксы распространённых секретов (AWS, GitHub, Slack и т.п.)
def secret_prefix_flags(text: str):
    s = str(text)
    s_lower = s.lower()
    return {
        'has_prefix_aws_akid': int(s.startswith('AKIA') or s.startswith('ASIA')),
        'has_prefix_github':   int(s.startswith('ghp_') or s.startswith('gho_') or s.startswith('ghs_')),
        'has_prefix_slack':    int(s_lower.startswith('xoxb-') or s_lower.startswith('xoxp-')),
    }

pref_df = data_filtered['Secret'].apply(secret_prefix_flags).apply(pd.Series)
data_filtered = pd.concat([data_filtered, pref_df], axis=1)

# Контекст имени переменной: наличие ключевых слов в Match или в строке целиком
def context_flags(row):
    # берем RuleID и File как прокси-контекст
    ctx = f"{row.get('RuleID', '')} {row.get('File', '')}"
    ctx = str(ctx).lower()
    return {
        'ctx_has_pass':   int(any(k in ctx for k in ['pass', 'pwd'])),
        'ctx_has_token':  int('token' in ctx),
        'ctx_has_key':    int('key' in ctx),
        'ctx_has_secret': int('secret' in ctx),
        'ctx_has_example':int('example' in ctx or 'sample' in ctx or 'test' in ctx),
    }

ctx_df = data_filtered.apply(context_flags, axis=1, result_type='expand')
data_filtered = pd.concat([data_filtered, ctx_df], axis=1)

numeric_features = [
    'secret_length',
    'secret_special_chars',
    'secret_has_url',
    'StartLine',
    'EndLine',
    'StartColumn',
    'EndColumn',
    'secret_entropy',
    'secret_letter_ratio',
    'secret_digit_ratio',
    'secret_upper_ratio',
    'secret_special_ratio',
    'secret_base64_ratio',
    'has_prefix_aws_akid',
    'has_prefix_github',
    'has_prefix_slack',
    'ctx_has_pass',
    'ctx_has_token',
    'ctx_has_key',
    'ctx_has_secret',
    'ctx_has_example',
]

categorical_features = ['RuleID', 'file_extension']

X = data_filtered.copy()
y = X['IsRealLeak']

# Лейбл‑энкодинг для моделей sklearn
le_dict = {}
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    le_dict[col] = le

joblib.dump(le_dict, 'models/le_dict.pkl')

all_features = numeric_features + categorical_features
X_sklearn = X[all_features]

# Сохраняем датасеты
X_sklearn.to_csv('gitleaks_dataset_sklearn_X.csv', index=False)
y.to_csv('gitleaks_dataset_y.csv', index=False)

X_catboost = data_filtered[all_features]
X_catboost.to_csv('gitleaks_dataset_catboost_X.csv', index=False)

print("Файлы сохранены:")
print("  gitleaks_dataset_sklearn_X.csv")
print("  gitleaks_dataset_catboost_X.csv")
print("  gitleaks_dataset_y.csv")
print("  le_dict.pkl")
