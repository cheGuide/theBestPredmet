# 🚀 TIOI Projects: Modern Machine Learning Portfolio

Комплексное портфолио проектов машинного обучения с AutoML и MLflow.

## 📂 Проекты

### 🧠 [TIOI PZ1 - Логический движок](./tioi_pz1/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/[YOUR_USERNAME]/tioi/blob/main/tioi_pz1/TIOI_PZ1_Colab.ipynb)

**Возможности:** Логический вывод, бенчмаркинг, градиентный спуск с нуля
- Собственная реализация ML алгоритмов
- Сравнение с scikit-learn
- Docker поддержка

### 🔢 [TIOI PZ2 - MNIST Классификация](./tioi_pz2/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/[YOUR_USERNAME]/tioi/blob/main/tioi_pz2/TIOI_PZ2_MNIST_Colab.ipynb)

**Возможности:** Определение чётности цифр, MLflow tracking
- Нейросети с нуля (forward/backward pass)
- Интерактивное рисование цифр
- Точность: 97.66%

### 🎙️ [Digit Recognition - AutoML](./digit_recognition/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/[YOUR_USERNAME]/tioi/blob/main/digit_recognition/Digit_Recognition_AutoML_Colab.ipynb)

**Революционный подход:** Голос → Computer Vision → Классификация
- AutoML с AutoKeras
- MLflow для tracking экспериментов
- Transfer Learning (ImageNet → Audio)

## 🚀 Быстрый старт

### Google Colab (1 клик):
Просто нажмите на любую кнопку [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)] выше!

### Локальный запуск:
```bash
# Клонировать репозиторий
git clone https://github.com/[YOUR_USERNAME]/tioi.git
cd tioi

# Запустить проекты
cd tioi_pz1 && python pz1.py
cd ../tioi_pz2 && python main.py --model nn --load weights/model_nn.npz
cd ../digit_recognition && python test_prediction.py audio/1_nicolas_18.wav
```

## 📊 Результаты

| Проект | Технологии | Результат |
|--------|------------|-----------|
| TIOI PZ1 | Python, NumPy, Sklearn | Логический вывод: `[1,2] → [1,2,10,50]` |
| TIOI PZ2 | TensorFlow, MLflow | Точность: **97.66%** на MNIST |
| Digit Recognition | AutoKeras, MLflow | **AutoML** нашёл лучшую архитектуру |

## 🎓 Образовательная ценность

### Освоенные технологии:
- **Classical ML**: Градиентный спуск, логистическая регрессия
- **Deep Learning**: Нейронные сети, CNN, backpropagation
- **AutoML**: AutoKeras, гиперпараметр-оптимизация
- **MLOps**: MLflow, эксперимент-трекинг
- **Signal Processing**: Спектрограммы, FFT
- **DevOps**: Docker, контейнеризация

### Инновационные подходы:
- 🎙️ **Междисциплинарность**: Акустика → Computer Vision
- 🤖 **AutoML Pipeline**: Автоматический поиск архитектуры
- 📈 **MLOps**: Полный жизненный цикл ML
- 🔄 **Transfer Learning**: ImageNet → Audio Classification

## 📁 Структура репозитория

```
tioi/
├── 📄 README.md                           # Этот файл
├── 📄 Google_Colab_Instructions.md        # Инструкции для Colab
├── 📄 COMPLETE_PROJECT_OVERVIEW.md        # Полный обзор
├── 🧠 tioi_pz1/                          # Логический движок
│   ├── TIOI_PZ1_Colab.ipynb             # Google Colab
│   ├── pz1.py                           # Основная программа
│   ├── rules.json & facts.json          # Данные
│   └── docker-compose.yml               # Docker
├── 🔢 tioi_pz2/                          # MNIST классификация
│   ├── TIOI_PZ2_MNIST_Colab.ipynb       # Google Colab
│   ├── main.py                          # Основная программа
│   ├── weights/                         # Модели
│   └── mlruns/                          # MLflow эксперименты
└── 🎙️ digit_recognition/                 # Распознавание голоса
    ├── Digit_Recognition_AutoML_Colab.ipynb  # Google Colab
    ├── automl_experiment.py                 # AutoML + MLflow
    ├── audio/                               # WAV файлы
    └── spectrogram_dataset/                 # Спектрограммы
```

## 🏆 Ключевые достижения

1. **🎙️ Голос через зрение**: Первый проект, который "слышит" через computer vision
2. **🔬 Полная реализация**: Градиентный спуск и нейросети без готовых решений  
3. **🤖 AutoML интеграция**: Современный MLOps в образовательном проекте
4. **📊 Production-ready**: Docker, MLflow, версионирование

## 📞 Контакты

- **Автор**: [Ваше имя]
- **Email**: [ваш email]
- **Курс**: TIOI
- **Год**: 2025

## 🎓 Инструкции для преподавателя

### 🚀 Вариант 1: Быстрая проверка (5 минут на проект)
**Не требует установки ничего локально**

1. Нажмите любую кнопку [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)] выше
2. В Google Colab: **Runtime** → **Run all** 
3. Ожидайте результаты (5-15 минут на проект)

**Что увидите:**
- **TIOI PZ1**: Логический вывод `[1,2] → [1,2,10,50]` + графики бенчмарка
- **TIOI PZ2**: Точность 97.66% на MNIST + confusion matrix + MLflow метрики  
- **Digit Recognition**: AutoML результаты + спектрограммы + детальная аналитика

### 🔍 Вариант 2: Детальное изучение (30 минут)
**Для углубленного анализа кода**

```bash
# Клонирование
git clone https://github.com/[YOUR_USERNAME]/tioi.git
cd tioi

# Запуск проектов
cd tioi_pz1 && python pz1.py                    # Логический движок
cd ../tioi_pz2 && python main.py --model nn     # MNIST классификация  
cd ../digit_recognition && python test_prediction.py audio/1_nicolas_18.wav

# MLflow UI (полный интерфейс)
cd tioi_pz2  # или digit_recognition 
mlflow ui
# Откройте: http://localhost:5000
```

### 📋 Что оценивать:
- **Техническая реализация**: Алгоритмы с нуля, правильность кода
- **Инновационность**: Междисциплинарный подход (голос → зрение)
- **MLOps практики**: MLflow tracking, версионирование  
- **Документация**: README, комментарии, структура проекта

---

**🚀 Все проекты готовы к запуску одним кликом в Google Colab!**
