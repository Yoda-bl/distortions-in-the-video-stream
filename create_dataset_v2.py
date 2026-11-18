"""
Генерация синтетического набора данных с геометрическими искажениями.
"""
import cv2
import numpy as np
import os
from pathlib import Path
import random
import csv

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Минимальные требования к количеству исходных сцен
MIN_TRAIN_SCENES = 1000
MIN_VAL_SCENES = 200
MIN_TEST_SCENES = 200

# Конфигурация классов искажений
CLASS_CONFIGS = {
    'clean':                {'k1':  0.00, 'k2':  0.000},
    'barrel_weak':          {'k1': -0.10, 'k2': -0.015},
    'barrel_medium':        {'k1': -0.25, 'k2': -0.0375},
    'barrel_strong':        {'k1': -0.50, 'k2': -0.075},
    'pincushion_weak':      {'k1':  0.10, 'k2':  0.015},
    'pincushion_medium':    {'k1':  0.25, 'k2':  0.0375},
    'pincushion_strong':    {'k1':  0.50, 'k2':  0.075},
}


def add_distortion(image, k1, k2):
    """Применяет геометрическое искажение к изображению с заданными коэффициентами k1 и k2"""
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    camera_matrix = np.array([
        [w, 0, center_x],
        [0, w, center_y],
        [0, 0, 1]
    ], dtype=float)
    dist_coeffs = np.array([k1, k2, 0.0, 0.0, 0.0])
    distorted = cv2.undistort(image, camera_matrix, dist_coeffs)
    return distorted


def load_images(source_dir):
    """Загружает список имен файлов изображений из указанной директории"""
    files = [f for f in os.listdir(source_dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(files)
    return files


def process_scene(image, class_name, config, output_path):
    """Обрабатывает одно изображение для заданного класса искажений"""
    # Сохраняем все изображения в формате PNG для обеспечения чистоты метрик и отсутствия артефактов сжатия
    if class_name == 'clean':
        cv2.imwrite(output_path, image)
    else:
        distorted = add_distortion(image, config['k1'], config['k2'])
        cv2.imwrite(output_path, distorted)


def create_split_from_files(files, source_dir, output_dir, split_name, manifest_rows=None):
    """
    Формирует выборку данных из переданного списка файлов
    
    Аргументы:
        files: список имен файлов для данной выборки
        source_dir: путь к исходным изображениям
        output_dir: базовая выходная директория
        split_name: название выборки - 'train', 'val' или 'test' 
        manifest_rows: список для накопления строк манифеста
    """
    # Создаем необходимую структуру каталогов
    for class_name in CLASS_CONFIGS.keys():
        Path(f"{output_dir}/{split_name}/{class_name}").mkdir(parents=True, exist_ok=True)

    count = 0
    for filename in files:
        img_path = os.path.join(source_dir, filename)
        image = cv2.imread(img_path)
        
        if image is None:
            continue

        base_name = os.path.splitext(filename)[0]

        for class_name, config in CLASS_CONFIGS.items():
            out_filename = f"{base_name}_{class_name}.png"
            output_path = f"{output_dir}/{split_name}/{class_name}/{out_filename}"

            process_scene(image, class_name, config, output_path)

            if manifest_rows is not None:
                manifest_rows.append([
                    filename,
                    split_name,
                    class_name,
                    config['k1'],
                    config['k2'],
                    output_path,
                    f"{output_dir}/{split_name}/clean/{base_name}_clean.png"
                ])

            count += 1

    return count


if __name__ == "__main__":
    # Пути к исходным данным
    source_train_dir = "data/raw/test2017"   # 41 тыс. изображений
    source_test_dir = "data/raw/val2017"     # 5 тыс. изображений
    output_dir = "data/dataset_v2"

    # Пропорции распределения данных по выборкам
    TRAIN_RATIO = 0.80   # 80% от source_train_dir пойдет в обучающую выборку
    VAL_RATIO = 0.20     # 20% от source_train_dir пойдет в валидационную выборку
    TEST_RATIO = 0.20    # 20% от source_test_dir пойдет в тестовую выборку

    # Загружаем и случайным образом перемешиваем списки файлов
    train_files_all = load_images(source_train_dir)
    test_files_all = load_images(source_test_dir)

    print(f"Найдено {len(train_files_all)} изображений в {source_train_dir}")
    print(f"Найдено {len(test_files_all)} изображений в {source_test_dir}")

    # Разделяем данные на обучающую и валидационную выборки
    n_train = int(len(train_files_all) * TRAIN_RATIO)
    n_val = len(train_files_all) - n_train

    train_files = train_files_all[:n_train]
    val_files = train_files_all[n_train:n_train + n_val]

    print(f"\nРазделение train/val:")
    print(f"\tTrain scenes: {len(train_files)}")
    print(f"\tVal scenes:   {len(val_files)}")

    # Проверка соблюдения минимальных требований к объему выборок
    if n_train < MIN_TRAIN_SCENES:
        raise ValueError(f"Слишком мало сцен в обучении: {n_train} (минимум {MIN_TRAIN_SCENES})")
    if n_val < MIN_VAL_SCENES:
        raise ValueError(f"Слишком мало сцен в валидации: {n_val} (минимум {MIN_VAL_SCENES})")

    # Формирование тестовой выборки
    n_test = int(len(test_files_all) * TEST_RATIO)
    test_files = test_files_all[:n_test]

    print(f"Test scenes:  {len(test_files)}")

    if n_test < MIN_TEST_SCENES:
        raise ValueError(f"Слишком мало тестовых сцен: {n_test} (минимум {MIN_TEST_SCENES})")

    # Непосредственная генерация искаженных изображений
    manifest_rows = []

    print("\n[1/3] Генерация train")
    train_count = create_split_from_files(
        train_files, source_train_dir, output_dir, 'train', manifest_rows
    )

    print("\n[2/3] Генерация val")
    val_count = create_split_from_files(
        val_files, source_train_dir, output_dir, 'val', manifest_rows
    )

    print("\n[3/3] Генерация test")
    test_count = create_split_from_files(
        test_files, source_test_dir, output_dir, 'test', manifest_rows
    )

    # Сохранение манифеста
    with open("data/dataset_manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "source_file", "split", "class_name",
            "k1", "k2", "distorted_path", "clean_path"
        ])
        writer.writerows(manifest_rows)
    print(f"\Манифест сохранен: data/dataset_manifest.csv ({len(manifest_rows)} строк)")

    print()
    print()
    print("Создание завершено успешно")
    print()
    print(f"Обучающих изображений:       {train_count} ({len(train_files)} сцен × 7 классов)")
    print(f"Валидационных изображений:   {val_count} ({len(val_files)} сцен × 7 классов)")
    print(f"Тестирующих изображений:     {test_count} ({len(test_files)} сцен × 7 классов)")
    print(f"Всего:    {train_count + val_count + test_count}")
    print(f"Классов:  {len(CLASS_CONFIGS)}")