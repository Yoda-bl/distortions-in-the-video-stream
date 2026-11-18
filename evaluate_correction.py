"""
Оценка качества компенсации геометрических искажений для обеих моделей

Для каждого изображения тестовой выборки:
1. Нейросеть предсказывает класс геометрического искажения
2. На основе предсказанного класса подбираются коэффициенты коррекции
3. Применяется обратное геометрическое преобразование
4. Скорректированное изображение сравнивается с исходным по общей валидной области
5. Результаты разделяются для случаев "верная классификация" и "ошибочная классификация"
"""
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
import os
import csv
import time
from skimage.metrics import structural_similarity as ssim_skimage

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 7

# Коэффициенты коррекции
CORRECTION_COEFFS = {
    'clean':              ( 0.00,  0.000),
    'barrel_weak':        ( 0.10,  0.015),
    'barrel_medium':      ( 0.25,  0.0375),
    'barrel_strong':      ( 0.50,  0.075),
    'pincushion_weak':    (-0.10, -0.015),
    'pincushion_medium':  (-0.25, -0.0375),
    'pincushion_strong':  (-0.50, -0.075),
}


# Расчет метрик качества

def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Вычисляет пиковое отношение сигнал/шум между двумя изображениями одинакового размера"""
    assert img1.shape == img2.shape, "Размеры изображений должны совпадать"
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Вычисляет индекс структурного сходства для изображений в градациях серого с использованием библиотеки skimage"""
    assert img1.shape == img2.shape, "Размеры изображений должны совпадать"
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return ssim_skimage(
        gray1, gray2,
        data_range=255,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False
    )


def get_common_roi(clean_img: np.ndarray,
                   corrected_img: np.ndarray,
                   threshold: int = 10,
                   margin: int = 16):
    """
    Определяет общую ограничивающую область для обоих изображений.
    Исключает черные артефакты на границах, возникающие после геометрической коррекции
    """
    assert clean_img.shape == corrected_img.shape
    clean_gray = cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY)
    corrected_gray = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2GRAY)
    clean_mask = clean_gray > threshold
    corrected_mask = corrected_gray > threshold
    common_mask = clean_mask & corrected_mask

    ys, xs = np.where(common_mask)
    if len(ys) == 0:
        raise ValueError("Отсутствует общая валидная область")

    y1 = max(int(ys.min()) + margin, 0)
    y2 = min(int(ys.max()), clean_img.shape[0])
    x1 = max(int(xs.min()) + margin, 0)
    x2 = min(int(xs.max()), clean_img.shape[1])

    if (y2 - y1) < 64 or (x2 - x1) < 64:
        raise ValueError("Общая валидная область имеет недостаточный размер")

    return y1, y2, x1, x2

# Загрузка моделей

def load_resnet(path='models/best_model_v2.pth'):
    """Загружает контрольную точку модели ResNet18, восстанавливает архитектуру и извлекает список классов"""
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model.load_state_dict(checkpoint['model_state_dict'])
    classes = checkpoint.get('classes', None)
    if classes is None:
        raise ValueError("В контрольной точке ResNet18 отсутствует список классов")
    model.to(DEVICE).eval()
    return model, classes


def load_mobilenetv3(path='models/best_model_mobilenetv3_v2.pth'):
    """Загружает контрольную точку модели MobileNetV3-Small, восстанавливает архитектуру и извлекает список классов"""
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
    model = models.mobilenet_v3_small(weights=None)
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, NUM_CLASSES)
    model.load_state_dict(checkpoint['model_state_dict'])
    classes = checkpoint.get('classes', None)
    if classes is None:
        raise ValueError("В контрольной точке MobileNetV3 отсутствует список классов")
    model.to(DEVICE).eval()
    return model, classes


def apply_correction(image: np.ndarray, k1: float, k2: float) -> np.ndarray:
    """Применяет обратное геометрическое преобразование для компенсации искажения на основе заданных коэффициентов"""
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    camera_matrix = np.array([
        [w, 0, center_x],
        [0, w, center_y],
        [0, 0, 1]
    ], dtype=float)
    dist_coeffs = np.array([k1, k2, 0.0, 0.0, 0.0])
    return cv2.undistort(image, camera_matrix, dist_coeffs)


def predict_class(model, transform, image_path: str):
    """Выполняет инференс модели для предсказания класса геометрического искажения входного изображения"""
    img = Image.open(image_path)
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        pred_class_idx = torch.argmax(probs).item()
    return pred_class_idx

# Процедура оценки одной модели

def evaluate_model(model, model_name, classes, transform, test_rows):
    """
    Проводит оценку качества геометрической коррекции для заданной модели
    Возвращает словарь с агрегированными метриками и детальными результатами
    """
    print()
    print(f"Оценка коррекции: {model_name}")
    print()

    results_correct = {'psnr': [], 'ssim': [], 'inference_time': [], 'correction_time': []}
    results_wrong = {'psnr': [], 'ssim': [], 'inference_time': [], 'correction_time': []}
    all_results = []
    skipped = 0

    for i, row in enumerate(test_rows):
        true_class = row['class_name']
        distorted_path = row['distorted_path']
        clean_path = row['clean_path']

        if not os.path.exists(distorted_path) or not os.path.exists(clean_path):
            continue

        # Предсказание класса с замером времени инференса
        start = time.perf_counter()
        pred_idx = predict_class(model, transform, distorted_path)
        inference_time = (time.perf_counter() - start) * 1000

        pred_class = classes[pred_idx]

        # Применение коррекции с замером времени
        k1, k2 = CORRECTION_COEFFS[pred_class]
        distorted_img = cv2.imread(distorted_path)

        start = time.perf_counter()
        corrected_img = apply_correction(distorted_img, k1, k2)
        correction_time = (time.perf_counter() - start) * 1000

        # Загрузка эталонного изображения и определение общей валидной области
        clean_img = cv2.imread(clean_path)
        try:
            y1, y2, x1, x2 = get_common_roi(clean_img, corrected_img)
        except ValueError as e:
            skipped += 1
            if skipped <= 3:
                print(f"  [skip] {row['source_file']}: {e}")
            continue

        clean_roi = clean_img[y1:y2, x1:x2]
        corrected_roi = corrected_img[y1:y2, x1:x2]

        psnr = calculate_psnr(clean_roi, corrected_roi)
        ssim_val = calculate_ssim(clean_roi, corrected_roi)

        is_correct = (pred_class == true_class)
        entry = {
            'source': row['source_file'],
            'true_class': true_class,
            'pred_class': pred_class,
            'correct': is_correct,
            'psnr': psnr,
            'ssim': ssim_val,
            'inference_time_ms': inference_time,
            'correction_time_ms': correction_time,
            'total_time_ms': inference_time + correction_time,
        }

        if is_correct:
            results_correct['psnr'].append(psnr)
            results_correct['ssim'].append(ssim_val)
            results_correct['inference_time'].append(inference_time)
            results_correct['correction_time'].append(correction_time)
        else:
            results_wrong['psnr'].append(psnr)
            results_wrong['ssim'].append(ssim_val)
            results_wrong['inference_time'].append(inference_time)
            results_wrong['correction_time'].append(correction_time)

        all_results.append(entry)

        if (i + 1) % 1000 == 0:
            print(f"\tОбработано {i+1}/{len(test_rows)} (пропущено: {skipped})")

    # Агрегация итоговых метрик
    print(f"\nВсего обработано: {len(all_results)}")
    print(f"Пропущено (малая общая область): {skipped}")

    summary = {'name': model_name, 'total': len(all_results), 'skipped': skipped}

    if results_correct['psnr']:
        n = len(results_correct['psnr'])
        summary['correct_count'] = n
        summary['correct_psnr'] = np.mean(results_correct['psnr'])
        summary['correct_ssim'] = np.mean(results_correct['ssim'])
        summary['correct_inference'] = np.mean(results_correct['inference_time'])
        summary['correct_correction'] = np.mean(results_correct['correction_time'])
        print(f"\n[Правильная классификация] ({n} изображений)")
        print(f"\tСредний PSNR: {summary['correct_psnr']:.2f} дБ")
        print(f"\tСредний SSIM: {summary['correct_ssim']:.4f}")
        print(f"\tВремя инференса: {summary['correct_inference']:.2f} мс")
        print(f"\tВремя коррекции: {summary['correct_correction']:.2f} мс")
    else:
        summary['correct_count'] = 0

    if results_wrong['psnr']:
        n = len(results_wrong['psnr'])
        summary['wrong_count'] = n
        summary['wrong_psnr'] = np.mean(results_wrong['psnr'])
        summary['wrong_ssim'] = np.mean(results_wrong['ssim'])
        print(f"\n[Ошибочная классификация] ({n} изображений)")
        print(f"\tСредний PSNR: {summary['wrong_psnr']:.2f} дБ")
        print(f"\tСредний SSIM: {summary['wrong_ssim']:.4f}")
    else:
        summary['wrong_count'] = 0

    overall_psnr = [r['psnr'] for r in all_results]
    overall_ssim = [r['ssim'] for r in all_results]
    overall_inf = [r['inference_time_ms'] for r in all_results]
    overall_corr = [r['correction_time_ms'] for r in all_results]
    accuracy = sum(1 for r in all_results if r['correct']) / len(all_results) * 100

    summary['accuracy'] = accuracy
    summary['overall_psnr'] = np.mean(overall_psnr)
    summary['overall_ssim'] = np.mean(overall_ssim)
    summary['overall_inference'] = np.mean(overall_inf)
    summary['overall_correction'] = np.mean(overall_corr)
    summary['overall_total'] = summary['overall_inference'] + summary['overall_correction']

    print(f"\n[Общий результат] ({len(all_results)} изображений)")
    print(f"\tТочность классификации: {accuracy:.2f}%")
    print(f"\tСредний PSNR: {summary['overall_psnr']:.2f} дБ")
    print(f"\tСредний SSIM: {summary['overall_ssim']:.4f}")
    print(f"\tВремя инференса: {summary['overall_inference']:.2f} мс")
    print(f"\tВремя коррекции: {summary['overall_correction']:.2f} мс")
    print(f"\tОбщее время: {summary['overall_total']:.2f} мс")

    return summary, all_results

# Создание таблицы

def print_comparison_table(resnet_summary, mobilenet_summary):
    print("\n" + "="*90)
    print("Сравнительная таблица")
    print("="*90)
    header = (f"{'Метрика':<30} | "
              f"{'ResNet18':<25} | "
              f"{'MobileNetV3-Small':<25}")
    print(header)
    print("-"*90)

    rows = [
        ("Точность классификации, %",
         f"{resnet_summary['accuracy']:.2f}",
         f"{mobilenet_summary['accuracy']:.2f}"),
        ("Средний PSNR (все), дБ",
         f"{resnet_summary['overall_psnr']:.2f}",
         f"{mobilenet_summary['overall_psnr']:.2f}"),
        ("Средний SSIM (все)",
         f"{resnet_summary['overall_ssim']:.4f}",
         f"{mobilenet_summary['overall_ssim']:.4f}"),
        ("PSNR при правильной классиф., дБ",
         f"{resnet_summary.get('correct_psnr', 0):.2f}",
         f"{mobilenet_summary.get('correct_psnr', 0):.2f}"),
        ("SSIM при правильной классиф.",
         f"{resnet_summary.get('correct_ssim', 0):.4f}",
         f"{mobilenet_summary.get('correct_ssim', 0):.4f}"),
        ("PSNR при ошибочной классиф., дБ",
         f"{resnet_summary.get('wrong_psnr', 0):.2f}",
         f"{mobilenet_summary.get('wrong_psnr', 0):.2f}"),
        ("SSIM при ошибочной классиф.",
         f"{resnet_summary.get('wrong_ssim', 0):.4f}",
         f"{mobilenet_summary.get('wrong_ssim', 0):.4f}"),
        ("Время инференса, мс",
         f"{resnet_summary['overall_inference']:.2f}",
         f"{mobilenet_summary['overall_inference']:.2f}"),
        ("Время коррекции, мс",
         f"{resnet_summary['overall_correction']:.2f}",
         f"{mobilenet_summary['overall_correction']:.2f}"),
        ("Общее время, мс",
         f"{resnet_summary['overall_total']:.2f}",
         f"{mobilenet_summary['overall_total']:.2f}"),
    ]

    for name, v1, v2 in rows:
        print(f"{name:<30} | {v1:<25} | {v2:<25}")

    print("-"*90)

    # Вычисление коэффициента ускорения
    speedup = resnet_summary['overall_inference'] / mobilenet_summary['overall_inference']
    print(f"\nMobileNetV3-Small быстрее ResNet18 в инференсе в {speedup:.2f}× раз")
    print(f"Разница в общем PSNR: {resnet_summary['overall_psnr'] - mobilenet_summary['overall_psnr']:+.2f} дБ")
    print(f"Разница в общем SSIM: {resnet_summary['overall_ssim'] - mobilenet_summary['overall_ssim']:+.4f}")


def evaluate():
    print(f"Устройство: {DEVICE}")

    # Инициализация и загрузка обученных моделей
    print("\nЗагрузка моделей")
    model_resnet, classes_resnet = load_resnet()
    print("\tResNet18 загружена")
    model_mobilenet, classes_mobilenet = load_mobilenetv3()
    print("\tMobileNetV3-Small загружена")

    if classes_resnet != classes_mobilenet:
        print("Списки классов различаются. Использую ResNet18.")
    classes = classes_resnet
    print(f"  Классы: {classes}")

    # Определение преобразований данных
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Чтение файла манифеста тестовой выборки
    manifest_path = "data/dataset_manifest.csv"
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Манифест не найден: {manifest_path}. "
            "Сначала запустите create_dataset.py"
        )

    test_rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['split'] == 'test' and row['class_name'] != 'clean':
                test_rows.append(row)

    print(f"\nНайдено {len(test_rows)} изображений тестовой выборки для оценки коррекции")

    # Запуск процедуры оценки для ResNet18
    print("\nОценка ResNet18")
    resnet_summary, resnet_results = evaluate_model(
        model_resnet, "ResNet18", classes, transform, test_rows
    )

    # Запуск процедуры оценки для MobileNetV3-Small
    print("\nОценка MobileNetV3-Small")
    mobilenet_summary, mobilenet_results = evaluate_model(
        model_mobilenet, "MobileNetV3-Small", classes, transform, test_rows
    )

    # Вывод сравнительной таблицы
    print("\nСравнение моделей")
    print_comparison_table(resnet_summary, mobilenet_summary)

    # Экспорт результатов в файлы
    os.makedirs('results', exist_ok=True)

    # Сохранение детальных результатов в формате CSV
    for name, results in [("resnet18", resnet_results), ("mobilenetv3", mobilenet_results)]:
        csv_path = f"results/correction_{name}.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['source', 'true_class', 'pred_class', 'correct',
                          'psnr', 'ssim',
                          'inference_time_ms', 'correction_time_ms', 'total_time_ms']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Детальные результаты {name}: {csv_path}")

    # Сохранение сводного отчtта в текстовом формате
    with open('results/correction_comparison.txt', 'w', encoding='utf-8') as f:
        f.write("Сравнение качества коррекции\n")
        f.write(f"Устройство: {DEVICE}\n")
        f.write(f"Тестовых изображений: {len(test_rows)}\n\n")

        for summary in [resnet_summary, mobilenet_summary]:
            f.write(f"{'='*60}\n")
            f.write(f"Модель: {summary['name']}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Точность классификации: {summary['accuracy']:.2f}%\n")
            f.write(f"Общий PSNR: {summary['overall_psnr']:.2f} дБ\n")
            f.write(f"Общий SSIM: {summary['overall_ssim']:.4f}\n")
            f.write(f"PSNR (правильная классиф.): {summary.get('correct_psnr', 0):.2f} дБ\n")
            f.write(f"SSIM (правильная классиф.): {summary.get('correct_ssim', 0):.4f}\n")
            f.write(f"PSNR (ошибочная классиф.): {summary.get('wrong_psnr', 0):.2f} дБ\n")
            f.write(f"SSIM (ошибочная классиф.): {summary.get('wrong_ssim', 0):.4f}\n")
            f.write(f"Время инференса: {summary['overall_inference']:.2f} мс\n")
            f.write(f"Время коррекции: {summary['overall_correction']:.2f} мс\n")
            f.write(f"Общее время: {summary['overall_total']:.2f} мс\n\n")

        f.write("="*90 + "\n")
        f.write("Таблица сравнения\n")
        f.write("="*90 + "\n")
        f.write(f"{'Метрика':<35} | {'ResNet18':<25} | {'MobileNetV3-Small':<25}\n")
        f.write("-"*90 + "\n")
        rows = [
            ("Точность классификации, %", f"{resnet_summary['accuracy']:.2f}", f"{mobilenet_summary['accuracy']:.2f}"),
            ("Средний PSNR (все), дБ", f"{resnet_summary['overall_psnr']:.2f}", f"{mobilenet_summary['overall_psnr']:.2f}"),
            ("Средний SSIM (все)", f"{resnet_summary['overall_ssim']:.4f}", f"{mobilenet_summary['overall_ssim']:.4f}"),
            ("PSNR при правильной классиф., дБ", f"{resnet_summary.get('correct_psnr', 0):.2f}", f"{mobilenet_summary.get('correct_psnr', 0):.2f}"),
            ("SSIM при правильной классиф.", f"{resnet_summary.get('correct_ssim', 0):.4f}", f"{mobilenet_summary.get('correct_ssim', 0):.4f}"),
            ("Время инференса, мс", f"{resnet_summary['overall_inference']:.2f}", f"{mobilenet_summary['overall_inference']:.2f}"),
            ("Время коррекции, мс", f"{resnet_summary['overall_correction']:.2f}", f"{mobilenet_summary['overall_correction']:.2f}"),
            ("Общее время, мс", f"{resnet_summary['overall_total']:.2f}", f"{mobilenet_summary['overall_total']:.2f}"),
        ]
        for name, v1, v2 in rows:
            f.write(f"{name:<35} | {v1:<25} | {v2:<25}\n")

    print(f"\nИтоговый отчет: results/correction_comparison.txt")


if __name__ == "__main__":
    evaluate()