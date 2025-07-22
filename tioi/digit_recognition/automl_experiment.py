"""
AutoML эксперимент для распознавания цифр из аудио
Использует Auto-Keras для автоматического поиска архитектуры
И MLflow для отслеживания экспериментов
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import autokeras as ak
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from datetime import datetime
import json

# Настройки
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
MAX_TRIALS = 5  # Количество архитектур для AutoML
EPOCHS = 10

# Пути к данным
train_dir = "spectrogram_dataset/train"
val_dir = "spectrogram_dataset/val"

def setup_mlflow():
    """Настройка MLflow эксперимента"""
    experiment_name = "Digit_Recognition_AutoML"
    
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
    
    mlflow.set_experiment(experiment_name)
    return experiment_id

def prepare_data():
    """Подготовка данных для AutoML"""
    print("🔄 Загружаем данные...")
    
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False
    ).flow_from_directory(
        train_dir, 
        target_size=IMG_SIZE, 
        batch_size=BATCH_SIZE, 
        class_mode='categorical',
        shuffle=True
    )
    
    val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        val_dir, 
        target_size=IMG_SIZE, 
        batch_size=BATCH_SIZE, 
        class_mode='categorical',
        shuffle=False
    )
    
    return train_gen, val_gen

def run_automl_experiment():
    """Запуск AutoML эксперимента с MLflow tracking"""
    
    # Настраиваем MLflow
    experiment_id = setup_mlflow()
    
    with mlflow.start_run(run_name=f"AutoML_Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Логируем параметры эксперимента
        mlflow.log_param("max_trials", MAX_TRIALS)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("img_size", IMG_SIZE)
        mlflow.log_param("automl_type", "ImageClassifier")
        
        # Подготавливаем данные
        train_gen, val_gen = prepare_data()
        
        print(f"📊 Найдено классов: {train_gen.num_classes}")
        print(f"🔢 Обучающих примеров: {train_gen.samples}")
        print(f"🎯 Валидационных примеров: {val_gen.samples}")
        
        # Логируем информацию о данных
        mlflow.log_param("num_classes", train_gen.num_classes)
        mlflow.log_param("train_samples", train_gen.samples)
        mlflow.log_param("val_samples", val_gen.samples)
        mlflow.log_param("class_names", list(train_gen.class_indices.keys()))
        
        # Создаём AutoML классификатор
        print("🤖 Запускаем AutoML поиск архитектуры...")
        
        clf = ak.ImageClassifier(
            max_trials=MAX_TRIALS,
            overwrite=True,
            project_name="digit_recognition_automl"
        )
        
        # Обучаем AutoML модель
        history = clf.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            verbose=1
        )
        
        # Получаем лучшую модель
        best_model = clf.export_model()
        
        print("✅ AutoML обучение завершено!")
        
        # Оценка модели
        print("📈 Оцениваем модель...")
        
        val_loss, val_accuracy = best_model.evaluate(val_gen, verbose=0)
        
        # Предсказания для confusion matrix
        val_gen.reset()
        predictions = best_model.predict(val_gen, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        y_true = val_gen.classes
        
        # Логируем метрики
        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.log_metric("val_loss", val_loss)
        
        # Classification report
        class_names = list(val_gen.class_indices.keys())
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        # Логируем метрики по классам
        for class_name in class_names:
            if class_name in report:
                mlflow.log_metric(f"precision_{class_name}", report[class_name]['precision'])
                mlflow.log_metric(f"recall_{class_name}", report[class_name]['recall'])
                mlflow.log_metric(f"f1_{class_name}", report[class_name]['f1-score'])
        
        # Общие метрики
        mlflow.log_metric("macro_avg_f1", report['macro avg']['f1-score'])
        mlflow.log_metric("weighted_avg_f1", report['weighted avg']['f1-score'])
        
        # Создаём confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('AutoML Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig("automl_confusion_matrix.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Логируем артефакты
        mlflow.log_artifact("automl_confusion_matrix.png")
        
        # Сохраняем модель
        model_path = "automl_best_model"
        best_model.save(model_path)
        mlflow.tensorflow.log_model(best_model, "model")
        
        # Логируем summary модели
        with open("model_summary.txt", "w") as f:
            best_model.summary(print_fn=lambda x: f.write(x + '\n'))
        mlflow.log_artifact("model_summary.txt")
        
        # Сохраняем подробный отчёт
        report_data = {
            "experiment_info": {
                "timestamp": datetime.now().isoformat(),
                "max_trials": MAX_TRIALS,
                "epochs": EPOCHS,
                "final_accuracy": float(val_accuracy),
                "final_loss": float(val_loss)
            },
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "class_names": class_names
        }
        
        with open("automl_report.json", "w") as f:
            json.dump(report_data, f, indent=2)
        mlflow.log_artifact("automl_report.json")
        
        print(f"\n🎯 Результаты AutoML:")
        print(f"   Точность: {val_accuracy:.4f}")
        print(f"   Потери: {val_loss:.4f}")
        print(f"   Макро F1: {report['macro avg']['f1-score']:.4f}")
        print(f"\n📊 Артефакты сохранены в MLflow")
        print(f"🚀 Запустите 'mlflow ui' для просмотра результатов")
        
        return best_model, val_accuracy, report

if __name__ == "__main__":
    print("🤖 Запуск AutoML эксперимента для распознавания цифр")
    print("=" * 60)
    
    try:
        model, accuracy, report = run_automl_experiment()
        print(f"\n✅ Эксперимент завершён успешно!")
        print(f"🏆 Лучшая точность: {accuracy:.4f}")
        
    except Exception as e:
        print(f"❌ Ошибка в эксперименте: {e}")
        print("💡 Проверьте наличие данных в spectrogram_dataset/") 