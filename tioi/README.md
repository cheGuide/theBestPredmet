# 🚀 TIOI Projects: Modern Machine Learning Portfolio

Комплексное портфолио проектов машинного обучения с AutoML и MLflow.

## 📂 Проекты

### 🧠 [TIOI PZ1 - Логический движок](./tioi_pz1/)
[![Open In Colab](https://colab.research.google.com/drive/1L0axdWmna_ukPVVUhGfuOm3FFHrplHk2#scrollTo=N-eZl7XCamST)

**Возможности:** Логический вывод, бенчмаркинг, градиентный спуск с нуля
- Собственная реализация ML алгоритмов
- Сравнение с scikit-learn
- Docker поддержка

### 🔢 [TIOI PZ2 - MNIST Классификация](./tioi_pz2/)
[![Open In Colab](https://colab.research.google.com/drive/1-BdXMtZcvwDeFvQcSlCfM0X_bc9jS_Z7#scrollTo=ZEwsC2csbRCm)

**Возможности:** Определение чётности цифр, MLflow tracking
- Нейросети с нуля (forward/backward pass)
- Интерактивное рисование цифр
- Точность: 97.66%

### 🎙️ [Digit Recognition - AutoML](./digit_recognition/)
[![Open In Colab](https://colab.research.google.com/drive/1JfwS2D_MANB2eBAn_mMbo4dVN1a-pmXh#scrollTo=UZ8HJrKbVW_I)

**Подход:** Голос → Computer Vision → Классификация
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

##  Результаты

| Проект | Технологии | Результат |
|--------|------------|-----------|
| TIOI PZ1 | Python, NumPy, Sklearn | Логический вывод: `[1,2] → [1,2,10,50]` |
| TIOI PZ2 | TensorFlow, MLflow | Точность: **97.66%** на MNIST |
| Digit Recognition | AutoKeras, MLflow | **AutoML** нашёл лучшую архитектуру |


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

