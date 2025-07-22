import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import sys

from spectrogram_digit_classifier import model_simple, model_mobilenet, train_gen

def predict_from_path(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred1 = model_simple.predict(img_array, verbose=0)
    pred2 = model_mobilenet.predict(img_array, verbose=0)

    class_labels = list(train_gen.class_indices.keys())
    predicted_class_1 = class_labels[np.argmax(pred1)]
    predicted_class_2 = class_labels[np.argmax(pred2)]

    print("=========================================")
    print("🧠 Simple CNN предсказала:", predicted_class_1, "→", pred1)
    print("🧠 MobileNetV2 предсказала:", predicted_class_2, "→", pred2)
    print("=========================================")

    plt.imshow(img)
    plt.title("Изображение спектрограммы")
    plt.axis('off')
    plt.savefig("prediction_preview.png")

    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❗ Укажи путь к изображению спектрограммы, например:")
        print("   python predict_one.py spectrogram_dataset/val/1/myfile.png")
    else:
        predict_from_path(sys.argv[1])
