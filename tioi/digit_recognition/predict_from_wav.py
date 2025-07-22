import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from scipy.io import wavfile
from scipy import signal
import os
import sys
import tempfile

from spectrogram_digit_classifier import model_simple, model_mobilenet, train_gen

def wav_to_spectrogram(wav_path):
    sr, samples = wavfile.read(wav_path)
    f, t, Sxx = signal.spectrogram(samples, sr)

    plt.figure(figsize=(5, 5))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
    plt.axis('off')
    plt.tight_layout()

    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    temp_path = temp_file.name
    plt.savefig(temp_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return temp_path

def predict_from_wav(wav_path):
    spec_path = wav_to_spectrogram(wav_path)

    img = image.load_img(spec_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    class_labels = list(train_gen.class_indices.keys())

    pred1 = model_simple.predict(img_array, verbose=0)
    pred2 = model_mobilenet.predict(img_array, verbose=0)

    result1 = f"🧠 Simple CNN предсказала: {class_labels[np.argmax(pred1)]}"
    result2 = f"🧠 MobileNetV2 предсказала: {class_labels[np.argmax(pred2)]}"

    # Создаём новое изображение с текстом
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image.load_img(spec_path))
    ax.set_title("Спектрограмма WAV-файла", fontsize=14)
    plt.axis('off')

    # Вывод предсказаний под изображением
    plt.figtext(0.5, 0.01, result1, wrap=True, horizontalalignment='center', fontsize=12)
    plt.figtext(0.5, -0.05, result2, wrap=True, horizontalalignment='center', fontsize=12)

    plt.savefig("prediction_preview.png", bbox_inches='tight')
    plt.close()
    os.remove(spec_path)

    print("[✓] Готово: prediction_preview.png")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❗ Укажи путь к WAV-файлу, например:")
        print("   python predict_from_wav.py audio/0_george_1.wav")
    else:
        predict_from_wav(sys.argv[1])