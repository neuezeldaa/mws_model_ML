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

# Создаем директории
RUN mkdir -p data models

# Открываем порт для API
EXPOSE 5000

# Команда запуска API (ИЗМЕНЕНО)
CMD ["python", "service.py"]
