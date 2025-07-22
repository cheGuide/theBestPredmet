# 🚀 Загрузка проектов TIOI на Google Colab

Руководство по запуску всех проектов в облачной среде Google Colab.

## 📂 Обзор проектов

### 1. 🧠 **TIOI PZ1** - Логический движок
- **Файл**: `TIOI_PZ1_Colab.ipynb`
- **Возможности**: Логический вывод, бенчмаркинг, градиентный спуск
- **Особенности**: Собственная реализация ML алгоритмов

### 2. 🔢 **TIOI PZ2** - MNIST классификация 
- **Файл**: `TIOI_PZ2_MNIST_Colab.ipynb`
- **Возможности**: Определение чётности цифр, MLflow tracking
- **Особенности**: Нейросети с нуля, интерактивное рисование

### 3. 🎙️ **Digit Recognition** - Распознавание голоса
- **Файл**: `Digit_Recognition_AutoML_Colab.ipynb`
- **Возможности**: AutoML, MLflow, спектрограммы
- **Особенности**: Голос → Computer Vision

---

## 🌐 Пошаговая инструкция загрузки на Colab

### Шаг 1: Открытие Google Colab
1. Перейдите на [colab.research.google.com](https://colab.research.google.com)
2. Войдите в свой Google аккаунт

### Шаг 2: Загрузка notebook'ов

**Способ A: Прямая загрузка файлов**
1. Нажмите **"File"** → **"Upload notebook"**
2. Выберите один из `.ipynb` файлов:
   - `tioi_pz1/TIOI_PZ1_Colab.ipynb`
   - `tioi_pz2/TIOI_PZ2_MNIST_Colab.ipynb`
   - `digit_recognition/Digit_Recognition_AutoML_Colab.ipynb`

**Способ B: Через GitHub**
1. Загрузите проекты на GitHub
2. В Colab: **"File"** → **"Open notebook"** → **"GitHub"**
3. Введите URL репозитория

### Шаг 3: Настройка среды выполнения

**Рекомендуемые настройки для каждого проекта:**

#### TIOI PZ1 (логический движок):
- **Runtime**: Python 3
- **Hardware**: CPU (достаточно)
- **RAM**: Стандартная

#### TIOI PZ2 (MNIST):
- **Runtime**: Python 3  
- **Hardware**: GPU (рекомендуется)
- **RAM**: Высокая RAM

#### Digit Recognition (AutoML):
- **Runtime**: Python 3
- **Hardware**: GPU (обязательно)
- **RAM**: Высокая RAM

**Изменение настроек:**
```
Runtime → Change runtime type → 
Hardware accelerator: GPU → Save
```

---

## 🔧 Автоматическая настройка

Каждый notebook автоматически:
- ✅ Установит все зависимости
- ✅ Создаст необходимые данные
- ✅ Настроит MLflow (где нужно)
- ✅ Запустит демонстрацию

---

## 🎯 Порядок изучения (рекомендуется)

### 1. **Начните с TIOI PZ1** (30 мин)
```python
# Простой логический движок
# Хорошо для понимания основ
```

### 2. **Затем TIOI PZ2** (45 мин)  
```python
# Классическое машинное обучение
# Нейросети с нуля
```

### 3. **Завершите Digit Recognition** (60 мин)
```python
# Продвинутые техники: AutoML + MLflow
# Междисциплинарный подход
```

---

## 🚨 Возможные проблемы и решения

### Проблема: "ModuleNotFoundError"
**Решение**: 
```python
!pip install --upgrade [package_name]
```

### Проблема: "CUDA out of memory"
**Решение**:
1. Runtime → Factory reset runtime
2. Уменьшите `BATCH_SIZE` в настройках
3. Используйте CPU вместо GPU

### Проблема: "Session timeout"
**Решение**:
```python
# Сохраните промежуточные результаты
import pickle
pickle.dump(model, open('model.pkl', 'wb'))
```

### Проблема: MLflow не работает
**Решение**:
```python
# MLflow в Colab имеет ограничения
# Все метрики будут выведены в консоль
```

---

## 📊 Ожидаемые результаты

### TIOI PZ1:
- ✅ Логический вывод: `[1, 2, 10, 50]`
- ✅ Графики сравнения алгоритмов
- ✅ Бенчмарк производительности

### TIOI PZ2:
- ✅ Точность ~97%+ на MNIST чёт/нечёт
- ✅ Confusion matrix
- ✅ Графики обучения

### Digit Recognition:
- ✅ AutoML нашёл лучшую архитектуру  
- ✅ Спектрограммы созданы
- ✅ Классификация 0/1/2 работает

---

## 🎓 Образовательная ценность

### Освоите технологии:
- **Python**: NumPy, Matplotlib, Scikit-learn
- **Deep Learning**: TensorFlow, Keras, AutoKeras
- **MLOps**: MLflow, эксперимент-трекинг
- **Computer Vision**: CNN, Transfer Learning
- **Signal Processing**: Спектрограммы, FFT

### Изучите концепции:
- Логические системы и вывод
- Градиентный спуск с нуля
- Нейронные сети (forward/backward prop)
- AutoML и гиперпараметр-оптимизация
- Междисциплинарные подходы (Audio → Vision)

---

## 🔗 Полезные ссылки

- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [AutoKeras Tutorials](https://autokeras.com/tutorial/overview/)
- [TensorFlow Guides](https://www.tensorflow.org/guide)

---

## 🎉 Заключение

Эти notebooks демонстрируют полный спектр современных технологий машинного обучения:

1. **Классические методы** (логический вывод)
2. **Фундаментальные техники** (нейросети с нуля)  
3. **Современные подходы** (AutoML, MLOps)

**Готовы начать? Выберите любой notebook и запустите в Colab!** 🚀 