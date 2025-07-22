
# 🎙️ Распознавание цифр по звуковым файлам (спектрограммы)

## 📦 Описание проекта

Проект выполняет распознавание произносимых цифр (0, 1, 2) по звуковым файлам `.wav`, преобразуя их в спектрограммы и обучая две модели:

- 🧠 **Simple CNN** — собственная сверточная нейросеть
- 🧠 **MobileNetV2** — мощная предобученная модель из Keras

Результаты обучения сравниваются, выводится график точности (`comparison.png`) и матрицы ошибок (`confusion_matrix.png`, `confusion_matrix_mobilenet.png`).

---

## 📁 Структура проекта

```
digit_recognition/
├── audio/                         # .wav файлы для обучения
├── spectrogram_dataset/          # Генерируются автоматически
│   ├── train/                    # Спектрограммы для обучения
│   └── val/                      # Спектрограммы для валидации
├── generate_spectrograms.py      # Генерация спектрограмм из .wav
├── split_train_val.py            # Разделение на train/val
├── spectrogram_digit_classifier.py  # Обучение 2 моделей
├── predict_one.py                # Предсказание по спектрограмме
├── predict_from_wav.py           # Предсказание по .wav файлу
├── setup_and_run.sh              # Автоматический запуск всех этапов
├── comparison.png                # График точности моделей
├── confusion_matrix.png          # Матрица ошибок Simple CNN
├── confusion_matrix_mobilenet.png # Матрица ошибок MobileNetV2
└── prediction_preview.png        # Предсказание по .wav (с визуализацией)
```

---

## ✅ Установка зависимостей

Активируй виртуальное окружение, если нужно:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Установи зависимости:

```bash
pip install -r requirements.txt
```

---

## 🚀 Полный запуск (всё сразу)

```bash
bash setup_and_run.sh
```

Этот скрипт:
- генерирует спектрограммы
- делит на train/val
- обучает обе модели
- строит график сравнения
- сохраняет confusion matrix

---

## 📊 Предсказания

### 🔍 Предсказание по уже готовой спектрограмме `.png`:

```bash
python predict_one.py spectrogram_dataset/val/1/recordings_1_lucas_0.png
```

### 🔊 Предсказание по звуковому файлу `.wav`:

```bash
python predict_from_wav.py audio/0_george_1.wav
```

После запуска появится:
- текст с предсказанием
- файл `prediction_preview.png` — визуализация + текст

---

## 📈 Результаты

- `comparison.png` — точность Simple CNN и MobileNetV2
- `confusion_matrix.png` — матрица ошибок для Simple CNN
- `confusion_matrix_mobilenet.png` — матрица ошибок для MobileNetV2
- `prediction_preview.png` — результат по одному .wav файлу

---

## 🛠️ Подготовка новых данных

1. Добавь `.wav` файлы в папку `audio/`
2. Пропиши нужные метки (`labels`) в `generate_spectrograms.py`
3. Запусти:

```bash
python generate_spectrograms.py
python split_train_val.py
python spectrogram_digit_classifier.py
```

Или одной командой:
```bash
bash setup_and_run.sh
```

---

## 📎 Требования

- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- matplotlib
- scipy

---



