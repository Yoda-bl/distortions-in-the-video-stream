"""
Сравнительное тестирование архитектур ResNet18 и MobileNetV3-Small на тестовой выборке
Вычисляет и выводит метрики качества классификации для каждой модели, а также формирует сравнительную таблицу
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 7


def load_resnet(path='models/best_model_v2.pth'):
    """Загружает контрольную точку обученной модели ResNet18 и восстанавливает ее архитектуру"""
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model.load_state_dict(checkpoint['model_state_dict'])
    classes = checkpoint.get('classes', None)
    if classes is None:
        raise ValueError("В контрольной точке отсутствует список классов")
    model = model.to(DEVICE)
    model.eval()
    return model, classes


def load_mobilenetv3(path='models/best_model_mobilenetv3_v2.pth'):
    """Загружает контрольную точку обученной модели MobileNetV3-Small и восстанавливает ее архитектуру"""
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
    model = models.mobilenet_v3_small(weights=None)
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, NUM_CLASSES)
    model.load_state_dict(checkpoint['model_state_dict'])
    classes = checkpoint.get('classes', None)
    if classes is None:
        raise ValueError("В контрольной точке отсутствует список классов.")
    model = model.to(DEVICE)
    model.eval()
    return model, classes


def evaluate_model(model, model_name, test_loader, classes):
    """
    Проводит оценку модели на тестовой выборке.
    Возвращает точность, макро-усредненную F1-меру, 
    массивы предсказаний и истинных меток, а также среднее время инференса.
    """
    print()
    print(f"Тестирование: {model_name}")
    print()

    all_preds = []
    all_labels = []
    total_inference_time = 0.0
    total_samples = 0
    correct = 0

    print("Запуск процедуры тестирования")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # Замер времени прямого прохода
            start_time = time.perf_counter()
            outputs = model(inputs)
            inference_time = (time.perf_counter() - start_time) * 1000  # мс
            total_inference_time += inference_time

            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_samples += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total_samples
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    avg_inference_time = total_inference_time / total_samples

    print(f"Test Accuracy:  {accuracy:.2f}%")
    print(f"Macro F1:       {macro_f1:.4f}")
    print(f"Время инференса: {avg_inference_time:.3f} мс/кадр")

    print("\nОтчет о классификации (Classification Report):")
    report = classification_report(
        all_labels, all_preds,
        target_names=classes,
        digits=4,
        zero_division=0
    )
    print(report)

    # Построение и сохранение матрицы ошибок
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Предсказанный класс (Predicted)')
    plt.ylabel('Истинный класс (True)')
    plt.title(f'Матрица ошибок — {model_name}')
    plt.tight_layout()
    cm_filename = f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(cm_filename, dpi=150)
    print(f"Матрица ошибок сохранена: {cm_filename}")
    plt.close()

    return {
        'name': model_name,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'avg_inference_time': avg_inference_time,
        'report': report,
        'cm': cm,
    }


def print_comparison_table(results):
    """Формирует и выводит в консоль сравнительную таблицу метрик исследуемых моделей"""
    print()
    print("Сравнительная таблица")
    print("="*70)
    header = f"{'Метрика':<25} | {results[0]['name']:<20} | {results[1]['name']:<20}"
    print(header)
    print("-"*70)

    for metric_key, metric_name in [
        ('accuracy', 'Точность (%)'),
        ('macro_f1', 'Макро-усредненная F1-мера'),
        ('avg_inference_time', 'Время инференса (мс/кадр)')
    ]:
        val1 = results[0][metric_key]
        val2 = results[1][metric_key]

        if metric_key == 'accuracy':
            row = f"{metric_name:<25} | {val1:<20.2f} | {val2:<20.2f}"
        elif metric_key == 'macro_f1':
            row = f"{metric_name:<25} | {val1:<20.4f} | {val2:<20.4f}"
        else:
            row = f"{metric_name:<25} | {val1:<20.3f} | {val2:<20.3f}"
        print(row)

    print("-"*70)

    # Вычисление абсолютной разницы метрик
    acc_diff = results[0]['accuracy'] - results[1]['accuracy']
    f1_diff = results[0]['macro_f1'] - results[1]['macro_f1']
    time_diff = results[0]['avg_inference_time'] - results[1]['avg_inference_time']

    print(f"\nРазница (ResNet18 − MobileNetV3-Small):")
    print(f"Точность:   {acc_diff:+.2f}%")
    print(f"Macro F1:   {f1_diff:+.4f}")
    print(f"Время:      {time_diff:+.3f} мс/кадр")

    if time_diff > 0:
        speedup = results[0]['avg_inference_time'] / results[1]['avg_inference_time']
        print(f"MobileNetV3-Small быстрее в {speedup:.2f}× раз")
    else:
        speedup = results[1]['avg_inference_time'] / results[0]['avg_inference_time']
        print(f"ResNet18 быстрее в {speedup:.2f}× раз")


def test():
    print(f"Устройство: {DEVICE}")

    # Инициализация и загрузка обеих моделей
    print("\nЗагрузка моделей")
    try:
        model_resnet, classes_resnet = load_resnet()
        print("ResNet18 успешно загружена")
    except FileNotFoundError as e:
        print(f" ResNet18 не найдена: {e}")
        return
    except Exception as e:
        print(f"Ошибка загрузки ResNet18: {e}")
        return

    try:
        model_mobilenet, classes_mobilenet = load_mobilenetv3()
        print("MobileNetV3-Small успешно загружена")
    except FileNotFoundError as e:
        print(f" MobileNetV3-Small не найдена: {e}")
        return
    except Exception as e:
        print(f"Ошибка загрузки MobileNetV3-Small: {e}")
        return

    # Контроль совпадения списков классов между моделями
    if classes_resnet != classes_mobilenet:
        print("Предупреждение: списки классов в моделях различаются")
        print(f"\tResNet18:           {classes_resnet}")
        print(f"\tMobileNetV3-Small:  {classes_mobilenet}")
    classes = classes_resnet

    # Загрузка тестовой выборки и определение преобразований данных
    print("\nЗагрузка тестовых данных")
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder('data/dataset_v2/test', transform=test_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False,
        num_workers=2, persistent_workers=True,
        pin_memory=(DEVICE.type == "cuda")
    )

    print(f"Тестовая выборка: {len(test_dataset)} изображений")
    print(f"Классы: {classes}")

    # Запуск процедуры оценки для ResNet18
    print("\nОценка ResNet18")
    resnet_results = evaluate_model(model_resnet, "ResNet18", test_loader, classes)

    # Запуск процедуры оценки для MobileNetV3-Small
    print("\nОценка MobileNetV3-Small")
    mobilenet_results = evaluate_model(model_mobilenet, "MobileNetV3-Small", test_loader, classes)

    # Формирование сравнительной таблицы
    print("\nСравнение моделей")
    print_comparison_table([resnet_results, mobilenet_results])

    # Экспорт сводных результатов в текстовый файл
    os.makedirs('results', exist_ok=True)
    with open('results/test_comparison_v2.txt', 'w', encoding='utf-8') as f:
        f.write("Сравнение моделей\n")
        f.write(f"Устройство: {DEVICE}\n")
        f.write(f"Тестовая выборка: {len(test_dataset)} изображений\n")
        f.write(f"Классы: {classes}\n\n")

        for res in [resnet_results, mobilenet_results]:
            f.write(f"{'='*60}\n")
            f.write(f"Модель: {res['name']}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Точность:              {res['accuracy']:.2f}%\n")
            f.write(f"Макро-усредненная F1:  {res['macro_f1']:.4f}\n")
            f.write(f"Время инференса:       {res['avg_inference_time']:.3f} мс/кадр\n\n")
            f.write("Отчет о классификации:\n")
            f.write(res['report'])
            f.write("\n\n")

        f.write("\n")
        f.write("Таблица\n")
        f.write("\n")
        f.write(f"{'Метрика':<25} | {resnet_results['name']:<20} | {mobilenet_results['name']:<20}\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Точность (%)':<25} | {resnet_results['accuracy']:<20.2f} | {mobilenet_results['accuracy']:<20.2f}\n")
        f.write(f"{'Макро-усредненная F1':<25} | {resnet_results['macro_f1']:<20.4f} | {mobilenet_results['macro_f1']:<20.4f}\n")
        f.write(f"{'Время инференса (мс)':<25} | {resnet_results['avg_inference_time']:<20.3f} | {mobilenet_results['avg_inference_time']:<20.3f}\n")

    print(f"\nРезультаты сохранены: results/test_comparison_v2.txt")


if __name__ == "__main__":
    test()