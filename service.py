from flask import Flask, request, jsonify
import pandas as pd
import joblib
import math
from catboost import CatBoostClassifier
import io

app = Flask(__name__)

# Загрузка моделей при старте
print("Загрузка моделей...")
model_gb = joblib.load('models/model_gb.pkl')
model_cb = CatBoostClassifier()
model_cb.load_model('models/model_cb.cbm')
scaler = joblib.load('models/scaler.pkl')
le_dict = joblib.load('models/le_dict.pkl')

SPECIAL = "!@#$%^&*()-_=+[]{}|;:',.<>?/~`"
BASE64_ALPH = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="


def extract_features_from_sarif(sarif_json):
    """Извлекает признаки из SARIF JSON формата Gitleaks"""
    results = []

    for run in sarif_json.get('runs', []):
        for result in run.get('results', []):
            # Извлекаем данные
            rule_id = result.get('ruleId', 'unknown')
            location = result['locations'][0]['physicalLocation']
            region = location['region']
            props = result.get('properties', {})

            # Получаем secret из fingerprint или создаем mock
            secret = props.get('fingerprint', 'unknown_secret')

            # Извлекаем путь к файлу
            file_path = location['artifactLocation']['uri']

            row = {
                'RuleID': rule_id,
                'Secret': secret,
                'File': file_path,
                'StartLine': region.get('startLine', 0),
                'EndLine': region.get('endLine', 0),
                'StartColumn': region.get('startColumn', 0),
                'EndColumn': region.get('endColumn', 0),
            }
            results.append(row)

    return pd.DataFrame(results)


def engineer_features(df):
    """Feature engineering для предсказания"""
    # Длина секрета
    df['secret_length'] = df['Secret'].str.len()

    # Спецсимволы
    df['secret_special_chars'] = df['Secret'].apply(lambda x: sum(1 for c in str(x) if c in SPECIAL))

    # URL
    df['secret_has_url'] = df['Secret'].str.contains('http', case=False, na=False).astype(int)

    # Расширение файла
    df['file_extension'] = df['File'].str.split('.').str[-1]

    # Энтропия
    def shannon_entropy(text):
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

    df['secret_entropy'] = df['Secret'].apply(shannon_entropy)

    # Статистика символов
    def char_stats(text):
        s = str(text)
        if not s:
            return 0, 0, 0, 0
        letters = sum(c.isalpha() for c in s)
        digits = sum(c.isdigit() for c in s)
        uppers = sum(c.isupper() for c in s)
        specials = sum(c in SPECIAL for c in s)
        n = len(s)
        return (letters / n, digits / n, uppers / n, specials / n)

    stats = df['Secret'].apply(char_stats)
    df['secret_letter_ratio'] = stats.apply(lambda t: t[0])
    df['secret_digit_ratio'] = stats.apply(lambda t: t[1])
    df['secret_upper_ratio'] = stats.apply(lambda t: t[2])
    df['secret_special_ratio'] = stats.apply(lambda t: t[3])

    # Base64 ratio
    def base64_ratio(text):
        s = str(text)
        if not s:
            return 0.0
        cnt = sum(c in BASE64_ALPH for c in s)
        return cnt / len(s)

    df['secret_base64_ratio'] = df['Secret'].apply(base64_ratio)

    # Префиксы
    def secret_prefix_flags(text):
        s = str(text)
        s_lower = s.lower()
        return {
            'has_prefix_aws_akid': int(s.startswith('AKIA') or s.startswith('ASIA')),
            'has_prefix_github': int(s.startswith('ghp_') or s.startswith('gho_') or s.startswith('ghs_')),
            'has_prefix_slack': int(s_lower.startswith('xoxb-') or s_lower.startswith('xoxp-')),
        }

    pref_df = df['Secret'].apply(secret_prefix_flags).apply(pd.Series)
    df = pd.concat([df, pref_df], axis=1)

    # Контекст
    def context_flags(row):
        ctx = f"{row.get('RuleID', '')} {row.get('File', '')}"
        ctx = str(ctx).lower()
        return {
            'ctx_has_pass': int(any(k in ctx for k in ['pass', 'pwd'])),
            'ctx_has_token': int('token' in ctx),
            'ctx_has_key': int('key' in ctx),
            'ctx_has_secret': int('secret' in ctx),
            'ctx_has_example': int('example' in ctx or 'sample' in ctx or 'test' in ctx),
        }

    ctx_df = df.apply(context_flags, axis=1, result_type='expand')
    df = pd.concat([df, ctx_df], axis=1)

    return df


def prepare_for_prediction(df):
    """Подготовка данных для предсказания"""
    # Label encoding для категориальных признаков
    for col in ['RuleID', 'file_extension']:
        if col in le_dict:
            le = le_dict[col]
            # Обработка неизвестных значений
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])

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

    # Все признаки в правильном порядке
    all_features = numeric_features + categorical_features

    return df[all_features]



@app.route('/predict', methods=['POST'])
def predict():
    """
    Принимает SARIF JSON от Gitleaks и возвращает предсказания
    """
    try:
        # Получаем JSON
        sarif_data = request.get_json()

        if not sarif_data:
            return jsonify({'error': 'No JSON data provided'}), 400

        # Извлекаем признаки из SARIF
        df = extract_features_from_sarif(sarif_data)

        if df.empty:
            return jsonify({'error': 'No results found in SARIF data'}), 400

        # Feature engineering
        df = engineer_features(df)

        # Подготовка для предсказания
        X = prepare_for_prediction(df)

        # Предсказание (используем CatBoost как основную модель)
        predictions = model_cb.predict(X)
        probabilities = model_cb.predict_proba(X)[:, 1]

        # Формирование ответа - ИСПРАВЛЕНО
        results = []
        for idx, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                'index': int(idx),  # Конвертируем в int
                'is_real_leak': bool(int(pred)),  # Конвертируем в bool
                'confidence': float(prob),  # Конвертируем в float
                'rule_id': str(df.iloc[idx]['RuleID']),  # Конвертируем в str
                'file': str(df.iloc[idx]['File'])  # Конвертируем в str
            })

        return jsonify({
            'predictions': results,
            'total_analyzed': int(len(results)),  # Конвертируем в int
            'real_leaks_detected': int(sum(predictions))  # Конвертируем в int
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
