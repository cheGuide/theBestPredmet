import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram
from pathlib import Path

def generate_spectrograms(audio_dir, output_dir, labels_dict):
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for audio_file in audio_dir.glob("*.wav"):
        label = labels_dict.get(audio_file.name)
        if label is None:
            continue

        # Чтение WAV
        sr, samples = wavfile.read(audio_file)
        if samples.ndim > 1:
            samples = samples[:, 0]

        # Спектрограмма
        freqs, times, Sxx = spectrogram(samples, sr)

        # Сохранение
        label_dir = output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(2.24, 2.24))
        plt.pcolormesh(times, freqs, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
        plt.axis('off')

        out_file = label_dir / (audio_file.stem + ".png")
        plt.savefig(out_file, bbox_inches='tight', pad_inches=0)
        plt.close()

if __name__ == "__main__":
    # Пример: назначение меток вручную
    labels = {
        "0_george_1.wav": "0",
        "0_george_10.wav": "0",
        "0_george_16.wav": "0",
        "0_jackson_44.wav": "0",
        "0_lucas_14.wav": "0",
        "1_george_34.wav": "1",
        "1_george_44.wav": "1",
        "1_jackson_35.wav": "1",
        "1_lucas_27.wav": "1",
        "1_nicolas_18.wav": "1",
        "1_nicolas_23.wav": "1",
        "1_theo_22.wav": "1",
        "2_jackson_10.wav": "2",
        "2_jackson_31wav": "2",
        "2_lucas_7.wav": "2"
    }

    generate_spectrograms(
        audio_dir="audio",                      # где лежат .wav
        output_dir="spectrogram_dataset/train", # куда сохранить спектрограммы
        labels_dict=labels
    )
