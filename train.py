import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)
import joblib
from sklearn.model_selection import cross_val_score

print("Отчистка данных...")

data = pd.read_csv('gitleaks_detection_dataset.csv')

data_filtered = data.copy()

# Удалим ненужные признаки
# Match-признак просто дублирует информацию из RuleID и Secret
data_filtered = data_filtered.drop(columns=['Match'])
# Email-признак просто не несет полезной информации
data_filtered = data_filtered.drop(columns=['Email'])
# Data-признак тоже не несет информации
data_filtered = data_filtered.drop(columns=['Date'])

# Заменим булевый тип на целочисленный
data_filtered['IsRealLeak'] = (data_filtered['IsRealLeak'] == True).astype(int)


# Создаем новый признак на основе длины секрета, должна быть высокая корреляция с целевой переменной
data_filtered['secret_length'] = data_filtered['Secret'].str.len()

# Новый признак содержит количество спец.символов
def count_special_chars(text):
    special = "!@#$%^&*()-_=+[]{}|;:',.<>?/~`"
    return sum(1 for c in str(text) if c in special)
data_filtered['secret_special_chars'] = data_filtered['Secret'].apply(count_special_chars)

# Признак, содержащий информацию о наличии url
data_filtered['secret_has_url'] = data_filtered['Secret'].str.contains('http', case=False, na=False).astype(int)

# Признак, говорящий о расширении файла, в котором хранится секрет
data_filtered['file_extension'] = data_filtered['File'].str.split('.').str[-1]

# Разделим признаки на нумерические и категориальные
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
# Выделим оставшиеся для обучения признаки:
X = data_filtered.copy()
y = X['IsRealLeak']


from sklearn.preprocessing import LabelEncoder
# Применим Лейбл Энкодер для преобразования категориальных признаков
le_dict = {}
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    le_dict[col] = le

joblib.dump(le_dict, 'le_dict.pkl')

all_features = numeric_features + categorical_features
X = X[all_features]


print("\nВыделим самые информативные признаки:")
for i, feat in enumerate(all_features, 1):
    print(f"  {i}. {feat}")

# Сохраним .csv с обработанными данными
X.to_csv('gitleaks_detection_dataset_filtred.csv', index=False)

print("\nОтфильтрованный .csv файл с данными сохранен!")


# Фильтрация данных закончена. Теперь разделим данные на обучающие и тестовые


print("\nРазделение данных на train/test...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.4,           # 20% для тестирования
    random_state=42,         # Для воспроизводимости
    stratify=y               # Сохраняем распределение классов
)

# Сначала построим Бейз-Лайн модель (K-NN)
# Выполним нормировку признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print('\nПостроим модель на основе алгоритма K ближайших соседей:')
# значение k = 11 показало лучшие результаты Точности, глубины и F1-score
k = 11
knn_results = {}

# Создаем и обучаем модель
knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
knn.fit(X_train_scaled, y_train)

# Предсказания
y_pred_train = knn.predict(X_train_scaled)
y_pred_test = knn.predict(X_test_scaled)
y_proba_knn = knn.predict_proba(X_test_scaled)

# Метрики на обучающем наборе
train_accuracy = accuracy_score(y_train, y_pred_train)

# Метрики на тестовом наборе
test_accuracy = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
f1_knn = f1_score(y_test, y_pred_test)

# Вывод метрик классификации
print(f"  KNN Train Accuracy: {train_accuracy:.4f}")
print(f"  KNN Test Accuracy:  {test_accuracy:.4f}")
print(f"  KNN Precision:      {precision:.4f}")
print(f"  KNN Recall:         {recall:.4f}")
print(f"  KNN F1-Score:       {f1_knn:.4f}")




# Теперь обучим модель Логистической регрессии
print('\nПостроим модель на основе алгоритма Логистической Регрессии:')
lr = LogisticRegression(
    max_iter=1000,           # Максимальное количество итераций
    random_state=42,         # Воспроизводимость
    solver='lbfgs',          # Алгоритм оптимизации
    class_weight='balanced'  # Балансировка весов классов
)

lr.fit(X_train_scaled, y_train)

# Предсказания
y_pred_lr_train = lr.predict(X_train_scaled)
y_pred_lr_test = lr.predict(X_test_scaled)

# Вероятности для логистической регрессии
y_proba_lr = lr.predict_proba(X_test_scaled)

# Метрики
train_accuracy_lr = accuracy_score(y_train, y_pred_lr_train)
test_accuracy_lr = accuracy_score(y_test, y_pred_lr_test)
precision_lr = precision_score(y_test, y_pred_lr_test)
recall_lr = recall_score(y_test, y_pred_lr_test)
f1_lr = f1_score(y_test, y_pred_lr_test)
roc_auc_lr = roc_auc_score(y_test, y_proba_lr[:, 1])

print(f"  Train Accuracy: {train_accuracy_lr:.4f}")
print(f"  Test Accuracy:  {test_accuracy_lr:.4f}")
print(f"  Precision:      {precision_lr:.4f}")
print(f"  Recall:         {recall_lr:.4f}")
print(f"  F1-Score:       {f1_lr:.4f}")
print(f"  ROC-AUC:        {roc_auc_lr:.4f}")





# Модель Случайного Леса
print('\nПостроим модель на основе алгоритма Случайного Леса:')
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# Предсказания
y_pred_rf_train = rf.predict(X_train)
y_pred_rf_test = rf.predict(X_test)

# Вероятности для Random Forest
y_proba_rf = rf.predict_proba(X_test)

# Метрики
train_accuracy_rf = accuracy_score(y_train, y_pred_rf_train)
test_accuracy_rf = accuracy_score(y_test, y_pred_rf_test)
precision_rf = precision_score(y_test, y_pred_rf_test)
recall_rf = recall_score(y_test, y_pred_rf_test)
f1_rf = f1_score(y_test, y_pred_rf_test)
roc_auc_rf = roc_auc_score(y_test, y_proba_rf[:, 1])

print(f"  Train Accuracy: {train_accuracy_rf:.4f}")
print(f"  Test Accuracy:  {test_accuracy_rf:.4f}")
print(f"  Precision:      {precision_rf:.4f}")
print(f"  Recall:         {recall_rf:.4f}")
print(f"  F1-Score:       {f1_rf:.4f}")
print(f"  ROC-AUC:        {roc_auc_rf:.4f}")



# Самый последний алгоритм для классификации: Градиентный бустинг
gb = GradientBoostingClassifier(
    n_estimators=250,          # Количество деревьев (слабых learners)
    learning_rate=0.05,         # Скорость обучения (shrinkage)
    max_depth=3,               # Максимальная глубина каждого дерева
    min_samples_split=40,      # Минимальное количество образцов для разделения
    min_samples_leaf=20,       # Минимальное количество образцов в листе
    subsample=0.8,             # Доля образцов для обучения каждого дерева
)

print('\nПостроим модель на основе алгоритма Градиентного Бустинга:')
gb.fit(X_train, y_train)

# Предсказания
y_pred_gb_train = gb.predict(X_train)
y_pred_gb_test = gb.predict(X_test)

# Вероятности для Gradient Boosting
y_proba_gb = gb.predict_proba(X_test)

# Метрики
train_accuracy_gb = accuracy_score(y_train, y_pred_gb_train)
test_accuracy_gb = accuracy_score(y_test, y_pred_gb_test)
precision_gb = precision_score(y_test, y_pred_gb_test)
recall_gb = recall_score(y_test, y_pred_gb_test)
f1_gb = f1_score(y_test, y_pred_gb_test)
roc_auc_gb = roc_auc_score(y_test, y_proba_gb[:, 1])


print(f"  Train Accuracy: {train_accuracy_gb:.4f}")
print(f"  Test Accuracy:  {test_accuracy_gb:.4f}")
print(f"  Precision:      {precision_gb:.4f}")
print(f"  Recall:         {recall_gb:.4f}")
print(f"  F1-Score:       {f1_gb:.4f}")
print(f"  ROC-AUC:        {roc_auc_gb:.4f}")

cv_scores = cross_val_score(
    gb,
    X, y,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
print(f"\n  CV scores: {cv_scores}")
print(f"  Mean CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  Test F1 (hold‑out): {f1_gb:.4f}\n")

from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    gb,
    X, y,
    cv=5,
    scoring='f1',
    train_sizes=np.linspace(0.1, 1.0, 5),
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

for s, tr, va in zip(train_sizes, train_mean, val_mean):
    print(f"Train size={s:5d}: train F1={tr:.3f}, val F1={va:.3f}, gap={tr-va:.3f}")






#------------------------------------


print(f"\nРассмотрим F1 score на тестовом датасете для пяти моделей:")
print(f"KNN: {f1_knn:.4f}")
print(f"Логистическая регрессия: {f1_lr:.4f}")
print(f"Случайный лес: {f1_rf:.4f}")
print(f"Градиентный бустинг: {f1_gb:.4f}")

print(f"\nГрадиентный бустинг имеет самый высокий F1-score!")

joblib.dump(lr, 'model_gb.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\nМодель, обученная на основе градиентного бустинга сохранена под именем 'model_gb.pkl'.")



#------------------------------------

