FROM python:3.11-slim

WORKDIR /app

# Обновляем pip
RUN pip install --upgrade pip

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Создаем директории для данных и моделей
RUN mkdir -p data models

# Команда по умолчанию
CMD ["python", "train.py"]
