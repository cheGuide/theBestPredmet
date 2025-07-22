import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
# === Распаковка архива вручную ===
# Распакуйте вручную архив spectrogram_dataset.zip в эту же папку перед запуском

# === Настройки ===
IMG_SIZE = (224, 224)
BATCH_SIZE = 4
EPOCHS = 10

train_dir = "spectrogram_dataset/train"
val_dir = "spectrogram_dataset/val"

# === Подготовка данных ===
train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

# === Первая модель: простая CNN ===
model_simple = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

model_simple.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_simple.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# === Вторая модель: MobileNetV2 ===
base_model = MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
predictions = Dense(train_gen.num_classes, activation='softmax')(x)

model_mobilenet = Model(inputs=base_model.input, outputs=predictions)
model_mobilenet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_mobilenet.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# === Сравнение ===
plt.plot(model_simple.history.history['val_accuracy'], label='Simple CNN')
plt.plot(model_mobilenet.history.history['val_accuracy'], label='MobileNetV2')
plt.title('Сравнение точности')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()
plt.grid(True)
plt.savefig("comparison.png")
print("[✓] Готово: comparison.png")



# Получаем предсказания на валидационной выборке
val_gen.reset()
y_true = val_gen.classes
y_pred_simple = model_simple.predict(val_gen, verbose=0)
y_pred_classes = np.argmax(y_pred_simple, axis=1)

# Создаём матрицу ошибок
cm = confusion_matrix(y_true, y_pred_classes)
labels = list(val_gen.class_indices.keys())

# Рисуем и сохраняем
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues')
plt.title("Матрица ошибок: Simple CNN")
plt.savefig("confusion_matrix.png")
plt.close()

print("[✓] Матрица ошибок сохранена в confusion_matrix.png")

