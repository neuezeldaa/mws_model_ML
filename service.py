import joblib
import pandas as pd
import numpy as np
import warnings


warnings.filterwarnings('ignore')
le_dict = joblib.load('le_dict.pkl')

df = pd.DataFrame([
    {
        'Secret': 'ghp_jtIZ2pjtlY6UlsNag1eDkLzkUWN3ZglxAM7O',
        'RuleID': 'github-pat',
        'File': 'utils/auth.ts',
        'StartLine': 28,
        'EndLine': 28,
        'StartColumn': 9,
        'EndColumn': 28
    },
])

# Создаем признаки
df['secret_length'] = df['Secret'].str.len()

special_chars = "!@#$%^&*()-_=+[]{}|;:',.<>?/~`"
df['secret_special_chars'] = df['Secret'].apply(
    lambda x: sum(1 for c in str(x) if c in special_chars)
)

df['secret_has_url'] = df['Secret'].str.contains('http', case=False, na=False).astype(int)

df['file_extension'] = df['File'].str.split('.').str[-1]

# Кодируем
df['RuleID'] = le_dict['RuleID'].transform(df['RuleID'].astype(str))
df['file_extension'] = le_dict['file_extension'].transform(df['file_extension'].astype(str))


feature_names = [
    'secret_length',
    'secret_special_chars',
    'secret_has_url',
    'StartLine',
    'EndLine',
    'StartColumn',
    'EndColumn',
    'RuleID',
    'file_extension'
]

X = df[feature_names]

scaler = joblib.load('scaler.pkl')
X_scaled = scaler.transform(X)

model = joblib.load('model_gb.pkl')

y_pred = model.predict(X_scaled)
y_proba = model.predict_proba(X_scaled)

prediction = y_pred[0]
confidence = np.max(y_proba[0]) * 100

prediction_label = 'TP' if prediction == 1 else 'FP'

print(f"Результат: {prediction_label}")
print(f"Уверенность: {confidence:.2f}%")
