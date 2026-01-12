import json
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

api_response_text = """
{
   "results": [
    {
      "MLConfidence": 0.19345142914878619,
      "MLPredict": false,
      "id": 1
    },
    {
      "MLConfidence": 0.5497978568689518,
      "MLPredict": false,
      "id": 2
    },
    {
      "MLConfidence": 0.041474879823256296,
      "MLPredict": false,
      "id": 3
    },
    {
      "MLConfidence": 0.07088916641407444,
      "MLPredict": false,
      "id": 4
    },
    {
      "MLConfidence": 0.7600104313245634,
      "MLPredict": false,
      "id": 5
    },
    {
      "MLConfidence": 0.04660178528920643,
      "MLPredict": false,
      "id": 6
    },
    {
      "MLConfidence": 0.1480200873071504,
      "MLPredict": false,
      "id": 7
    },
    {
      "MLConfidence": 0.16071772166347895,
      "MLPredict": false,
      "id": 8
    },
    {
      "MLConfidence": 0.19618958010890641,
      "MLPredict": false,
      "id": 9
    },
    {
      "MLConfidence": 0.9405075211788374,
      "MLPredict": true,
      "id": 10
    },
    {
      "MLConfidence": 0.7138773048286685,
      "MLPredict": false,
      "id": 11
    },
    {
      "MLConfidence": 0.025001009577521858,
      "MLPredict": false,
      "id": 12
    },
    {
      "MLConfidence": 0.8578895609827306,
      "MLPredict": false,
      "id": 13
    },
    {
      "MLConfidence": 0.16071772166347895,
      "MLPredict": false,
      "id": 14
    },
    {
      "MLConfidence": 0.027608408957490966,
      "MLPredict": false,
      "id": 15
    },
    {
      "MLConfidence": 0.05501364403916014,
      "MLPredict": false,
      "id": 16
    },
    {
      "MLConfidence": 0.7419340459707889,
      "MLPredict": false,
      "id": 17
    },
    {
      "MLConfidence": 0.04123648686080465,
      "MLPredict": false,
      "id": 18
    },
    {
      "MLConfidence": 0.05587079867467658,
      "MLPredict": false,
      "id": 19
    },
    {
      "MLConfidence": 0.1578192856326435,
      "MLPredict": false,
      "id": 20
    },
    {
      "MLConfidence": 0.04042828581764348,
      "MLPredict": false,
      "id": 21
    },
    {
      "MLConfidence": 0.028491663944154057,
      "MLPredict": false,
      "id": 22
    },
    {
      "MLConfidence": 0.8578895609827306,
      "MLPredict": false,
      "id": 23
    },
    {
      "MLConfidence": 0.02360975741112603,
      "MLPredict": false,
      "id": 24
    },
    {
      "MLConfidence": 0.40513179297245744,
      "MLPredict": false,
      "id": 25
    },
    {
      "MLConfidence": 0.8748429094136481,
      "MLPredict": false,
      "id": 26
    },
    {
      "MLConfidence": 0.028491663944154057,
      "MLPredict": false,
      "id": 27
    },
    {
      "MLConfidence": 0.12039175150384292,
      "MLPredict": false,
      "id": 28
    },
    {
      "MLConfidence": 0.07693910452093844,
      "MLPredict": false,
      "id": 29
    },
    {
      "MLConfidence": 0.8627948027419415,
      "MLPredict": false,
      "id": 30
    },
    {
      "MLConfidence": 0.11987902153773304,
      "MLPredict": false,
      "id": 31
    },
    {
      "MLConfidence": 0.06105745742154869,
      "MLPredict": false,
      "id": 32
    },
    {
      "MLConfidence": 0.05429963786064143,
      "MLPredict": false,
      "id": 33
    },
    {
      "MLConfidence": 0.13154955544325792,
      "MLPredict": false,
      "id": 34
    },
    {
      "MLConfidence": 0.812659960712061,
      "MLPredict": false,
      "id": 35
    },
    {
      "MLConfidence": 0.041474879823256296,
      "MLPredict": false,
      "id": 36
    },
    {
      "MLConfidence": 0.8182016554443043,
      "MLPredict": false,
      "id": 37
    },
    {
      "MLConfidence": 0.7419340459707889,
      "MLPredict": false,
      "id": 38
    },
    {
      "MLConfidence": 0.041474879823256296,
      "MLPredict": false,
      "id": 39
    },
    {
      "MLConfidence": 0.22490482018152025,
      "MLPredict": false,
      "id": 40
    },
    {
      "MLConfidence": 0.033571775262841275,
      "MLPredict": false,
      "id": 41
    },
    {
      "MLConfidence": 0.5296034030268495,
      "MLPredict": false,
      "id": 42
    },
    {
      "MLConfidence": 0.05916884299219391,
      "MLPredict": false,
      "id": 43
    },
    {
      "MLConfidence": 0.8537854737967748,
      "MLPredict": false,
      "id": 44
    },
    {
      "MLConfidence": 0.148455065738022,
      "MLPredict": false,
      "id": 45
    },
    {
      "MLConfidence": 0.03647052827093467,
      "MLPredict": false,
      "id": 46
    },
    {
      "MLConfidence": 0.8277955464218193,
      "MLPredict": false,
      "id": 47
    },
    {
      "MLConfidence": 0.8923162882032283,
      "MLPredict": false,
      "id": 48
    },
    {
      "MLConfidence": 0.041474879823256296,
      "MLPredict": false,
      "id": 49
    },
    {
      "MLConfidence": 0.05255881068672515,
      "MLPredict": false,
      "id": 50
    },
    {
      "MLConfidence": 0.05587079867467658,
      "MLPredict": false,
      "id": 51
    },
    {
      "MLConfidence": 0.028491663944154057,
      "MLPredict": false,
      "id": 52
    },
    {
      "MLConfidence": 0.7419340459707889,
      "MLPredict": false,
      "id": 53
    },
    {
      "MLConfidence": 0.40513179297245744,
      "MLPredict": false,
      "id": 54
    },
    {
      "MLConfidence": 0.07816831237022577,
      "MLPredict": false,
      "id": 55
    },
    {
      "MLConfidence": 0.6849073952011544,
      "MLPredict": false,
      "id": 56
    },
    {
      "MLConfidence": 0.03266631194889295,
      "MLPredict": false,
      "id": 57
    },
    {
      "MLConfidence": 0.8520810422720936,
      "MLPredict": false,
      "id": 58
    },
    {
      "MLConfidence": 0.9109227776949765,
      "MLPredict": true,
      "id": 59
    },
    {
      "MLConfidence": 0.08247158450874978,
      "MLPredict": false,
      "id": 60
    },
    {
      "MLConfidence": 0.025001009577521858,
      "MLPredict": false,
      "id": 61
    },
    {
      "MLConfidence": 0.7138773048286685,
      "MLPredict": false,
      "id": 62
    },
    {
      "MLConfidence": 0.027608408957490966,
      "MLPredict": false,
      "id": 63
    },
    {
      "MLConfidence": 0.8748429094136481,
      "MLPredict": false,
      "id": 64
    },
    {
      "MLConfidence": 0.03291799040930094,
      "MLPredict": false,
      "id": 65
    },
    {
      "MLConfidence": 0.04123648686080465,
      "MLPredict": false,
      "id": 66
    },
    {
      "MLConfidence": 0.8748429094136481,
      "MLPredict": false,
      "id": 67
    },
    {
      "MLConfidence": 0.02360975741112603,
      "MLPredict": false,
      "id": 68
    },
    {
      "MLConfidence": 0.9405075211788374,
      "MLPredict": true,
      "id": 69
    },
    {
      "MLConfidence": 0.04313739779458486,
      "MLPredict": false,
      "id": 70
    },
    {
      "MLConfidence": 0.027608408957490966,
      "MLPredict": false,
      "id": 71
    },
    {
      "MLConfidence": 0.9252483896003205,
      "MLPredict": true,
      "id": 72
    },
    {
      "MLConfidence": 0.35050987299911923,
      "MLPredict": false,
      "id": 73
    },
    {
      "MLConfidence": 0.40513179297245744,
      "MLPredict": false,
      "id": 74
    },
    {
      "MLConfidence": 0.7600104313245634,
      "MLPredict": false,
      "id": 75
    },
    {
      "MLConfidence": 0.06194303704953786,
      "MLPredict": false,
      "id": 76
    },
    {
      "MLConfidence": 0.35050987299911923,
      "MLPredict": false,
      "id": 77
    },
    {
      "MLConfidence": 0.06194303704953786,
      "MLPredict": false,
      "id": 78
    },
    {
      "MLConfidence": 0.05429963786064143,
      "MLPredict": false,
      "id": 79
    },
    {
      "MLConfidence": 0.028491663944154057,
      "MLPredict": false,
      "id": 80
    },
    {
      "MLConfidence": 0.40513179297245744,
      "MLPredict": false,
      "id": 81
    },
    {
      "MLConfidence": 0.03647052827093467,
      "MLPredict": false,
      "id": 82
    },
    {
      "MLConfidence": 0.8584839752231718,
      "MLPredict": false,
      "id": 83
    },
    {
      "MLConfidence": 0.8452299582519573,
      "MLPredict": false,
      "id": 84
    },
    {
      "MLConfidence": 0.35050987299911923,
      "MLPredict": false,
      "id": 85
    },
    {
      "MLConfidence": 0.917636649926597,
      "MLPredict": true,
      "id": 86
    },
    {
      "MLConfidence": 0.7419340459707889,
      "MLPredict": false,
      "id": 87
    },
    {
      "MLConfidence": 0.03291799040930094,
      "MLPredict": false,
      "id": 88
    },
    {
      "MLConfidence": 0.5936190044997027,
      "MLPredict": false,
      "id": 89
    },
    {
      "MLConfidence": 0.22490482018152025,
      "MLPredict": false,
      "id": 90
    },
    {
      "MLConfidence": 0.9112092252279048,
      "MLPredict": true,
      "id": 91
    },
    {
      "MLConfidence": 0.04814837642900727,
      "MLPredict": false,
      "id": 92
    },
    {
      "MLConfidence": 0.5988404634494005,
      "MLPredict": false,
      "id": 93
    },
    {
      "MLConfidence": 0.029226334198264196,
      "MLPredict": false,
      "id": 94
    },
    {
      "MLConfidence": 0.09716564318077472,
      "MLPredict": false,
      "id": 95
    },
    {
      "MLConfidence": 0.11987902153773304,
      "MLPredict": false,
      "id": 96
    },
    {
      "MLConfidence": 0.04931032637462592,
      "MLPredict": false,
      "id": 97
    },
    {
      "MLConfidence": 0.03266631194889295,
      "MLPredict": false,
      "id": 98
    },
    {
      "MLConfidence": 0.9252483896003205,
      "MLPredict": true,
      "id": 99
    },
    {
      "MLConfidence": 0.8584839752231718,
      "MLPredict": false,
      "id": 100
    }
  ]

}
"""

api_data = json.loads(api_response_text)
df_pred = pd.DataFrame(api_data['results'])

# Загружаем истинные метки
df_true = pd.read_csv('data/gitleaks_test_100.csv')

# Объединяем по ID
df = df_true.merge(df_pred, left_on='ID', right_on='id')

# Извлекаем истинные и предсказанные значения
y_true = df['IsRealLeak'].astype(bool)
y_pred = df['MLPredict'].astype(bool)

# Вычисляем метрики
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score:  {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"\nTrue Positives:  {tp}")
print(f"True Negatives:  {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")

# Детальный отчёт
print(classification_report(y_true, y_pred, target_names=['Fake', 'Real Leak']))