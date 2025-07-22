# Используем официальный образ Python 3.10
FROM python:3.10

# Устанавливаем рабочую директорию
WORKDIR /tioi_pz1

# Копируем файлы проекта в контейнер
COPY . .

# Устанавливаем зависимости (если есть requirements.txt)
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Запуск программы
CMD ["python", "pz1.py"]