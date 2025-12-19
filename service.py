from flask import Flask, request, jsonify
import pandas as pd
import joblib
import math
import numpy as np
from catboost import CatBoostClassifier, Pool

app = Flask(__name__)

print("Загрузка моделей...")
model_cb = CatBoostClassifier()
model_cb.load_model('models/model_cb.cbm')
cat_features_idx = joblib.load('models/cat_features_idx.pkl')

# Загружаем конфигурацию порога
try:
    threshold_config = joblib.load('models/threshold.pkl')
    CLASSIFICATION_THRESHOLD = threshold_config['threshold']
    print(f"✓ Загружен оптимальный порог: {CLASSIFICATION_THRESHOLD:.3f}")
    print(f"  Режим: {threshold_config.get('mode', 'unknown')}")
    print(f"  Precision: {threshold_config.get('precision', 0):.4f}")
    print(f"  Recall: {threshold_config.get('recall', 0):.4f}")
    print(f"  F1: {threshold_config.get('f1_score', 0):.4f}")
except FileNotFoundError:
    CLASSIFICATION_THRESHOLD = 0.7  # Дефолтный высокий порог для безопасности
    print(f"⚠ threshold.pkl не найден, используем дефолтный порог: {CLASSIFICATION_THRESHOLD}")

SPECIAL = "!@#$%^&*()-_=+[]{}|;:',.<>?/~`"
BASE64_ALPH = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="


def extract_features_from_json(json_data):
    """Извлекает признаки из упрощенного JSON формата"""
    results = []
    for item in json_data:
        row = {
            'RuleID': item.get('rule_id', 'unknown'),
            'Secret': item.get('value', 'unknown_secret'),
            'File': item.get('file_path', 'unknown'),
            'StartLine': item.get('line', 0),
            'EndLine': item.get('line', 0),
            'StartColumn': 0,
            'EndColumn': len(str(item.get('value', ''))),
        }
        results.append(row)
    return pd.DataFrame(results)


def engineer_features(df):
    """Feature engineering для предсказания"""
    df['secret_length'] = df['Secret'].str.len()
    df['secret_special_chars'] = df['Secret'].apply(lambda x: sum(1 for c in str(x) if c in SPECIAL))
    df['secret_has_url'] = df['Secret'].str.contains('http', case=False, na=False).astype(int)
    df['file_extension'] = df['File'].str.split('.').str[-1]

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

    def base64_ratio(text):
        s = str(text)
        if not s:
            return 0.0
        cnt = sum(c in BASE64_ALPH for c in s)
        return cnt / len(s)

    df['secret_base64_ratio'] = df['Secret'].apply(base64_ratio)

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
    numeric_features = [
        'secret_length', 'secret_special_chars', 'secret_has_url',
        'StartLine', 'EndLine', 'StartColumn', 'EndColumn',
        'secret_entropy', 'secret_letter_ratio', 'secret_digit_ratio',
        'secret_upper_ratio', 'secret_special_ratio', 'secret_base64_ratio',
        'has_prefix_aws_akid', 'has_prefix_github', 'has_prefix_slack',
        'ctx_has_pass', 'ctx_has_token', 'ctx_has_key',
        'ctx_has_secret', 'ctx_has_example',
    ]
    categorical_features = ['RuleID', 'file_extension']
    all_features = numeric_features + categorical_features
    return df[all_features]


@app.route('/predict', methods=['POST'])
def predict():
    """Принимает JSON массив секретов и возвращает предсказания"""
    try:
        json_data = request.get_json()
        if not json_data or not isinstance(json_data, list):
            return jsonify({'error': 'Expected JSON array'}), 400

        df = extract_features_from_json(json_data)
        if df.empty:
            return jsonify({'error': 'No results found'}), 400

        ids = [item.get('id') for item in json_data]
        df = engineer_features(df)
        X = prepare_for_prediction(df)

        # Предсказание с оптимизированным порогом (≥ 0.5)
        pool = Pool(X, cat_features=cat_features_idx)
        probabilities = model_cb.predict_proba(pool)[:, 1]
        predictions = (probabilities >= CLASSIFICATION_THRESHOLD).astype(int)

        results = []
        for idx, (pred, prob) in enumerate(zip(predictions, probabilities)):
            result = {
                'id': ids[idx],
                'MLPredict': bool(int(pred)),
                'MLConfidence': float(prob)
            }
            results.append(result)

        return jsonify({'results': results})

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model': 'CatBoost',
        'threshold': float(CLASSIFICATION_THRESHOLD),
        'mode': 'high_precision'
    }), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
