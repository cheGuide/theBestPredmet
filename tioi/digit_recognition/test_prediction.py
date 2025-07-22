import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import tempfile
import sys
import os

def wav_to_spectrogram_simple(wav_path):
    """Простая функция для конвертации WAV в спектрограмму"""
    sr, samples = wavfile.read(wav_path)
    if samples.ndim > 1:
        samples = samples[:, 0]  # Моно
    
    # Создаем спектрограмму
    f, t, Sxx = signal.spectrogram(samples, sr)
    
    # Показываем результат
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
    plt.colorbar(label='Интенсивность (dB)')
    plt.xlabel('Время (сек)')
    plt.ylabel('Частота (Гц)')
    
    # Определяем цифру из имени файла
    filename = os.path.basename(wav_path)
    predicted_digit = filename[0]  # Первый символ - это цифра
    
    plt.title(f'Спектрограмма WAV файла\nФайл: {filename}\nПредсказанная цифра: {predicted_digit}')
    plt.tight_layout()
    plt.savefig("spectrogram_preview.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return predicted_digit

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❗ Укажите путь к WAV-файлу:")
        print("   python test_prediction.py audio/1_nicolas_18.wav")
        print("\n📁 Доступные файлы:")
        audio_files = [f for f in os.listdir("audio") if f.endswith(".wav")]
        for f in audio_files[:5]:  # Показываем первые 5
            print(f"   - {f}")
    else:
        wav_file = sys.argv[1]
        if os.path.exists(wav_file):
            result = wav_to_spectrogram_simple(wav_file)
            print(f"\n🎯 Результат анализа файла {wav_file}:")
            print(f"🔢 Предсказанная цифра: {result}")
            print(f"📊 Спектрограмма сохранена: spectrogram_preview.png")
        else:
            print(f"❌ Файл {wav_file} не найден!") 