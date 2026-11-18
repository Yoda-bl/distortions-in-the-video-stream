"""
Обучение нейронной сети MobileNetV3-Small на наборе данных с 7 классами геометрических искажений
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import csv
from collections import defaultdict, Counter
from pathlib import Path

# Гиперпараметры и настройки обучения
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 7
BATCH_SIZE = 32
NUM_EPOCHS = 30
PATIENCE = 5

# Дифференцированная скорость обучения для различных слоев модели
LR_CLASSIFIER = 1e-3
LR_BACKBONE = 1e-4

print(f"Используемое устройство: {DEVICE}")

# Преобразования данных (идентичны ResNet18 для обеспечения корректности сравнения)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10, interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_balanced_indices(manifest_path, split, max_scenes, dataset):
    """
    Формирует сбалансированное подмножество индексов изображений на основе 
    исходных сцен для предотвращения утечки данных между выборками
    """
    # Сопоставление абсолютного пути с индексом в dataset.samples
    path_to_index = {
        str(Path(sample[0]).resolve()): idx
        for idx, sample in enumerate(dataset.samples)
    }

    # Группировка индексов по исходным сценам
    scenes = defaultdict(list)
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] != split:
                continue
            # Нормализация пути из файла манифеста
            image_path = str(Path(row["distorted_path"]).resolve())
            if image_path in path_to_index:
                scenes[row["source_file"]].append(path_to_index[image_path])

    # Сортировка имен сцен и выбор первых max_scenes сцен
    scene_names = sorted(scenes.keys())
    
    print(f"  [debug] Найдено {len(scene_names)} уникальных сцен для split='{split}'")
    
    if len(scene_names) < max_scenes:
        raise ValueError(
            f"В манифесте недостаточно сцен для split='{split}': "
            f"найдено {len(scene_names)}, требуется {max_scenes}"
        )

    selected_scenes = scene_names[:max_scenes]

    indices = []
    for scene in selected_scenes:
        indices.extend(scenes[scene])

    return indices, len(selected_scenes)

def print_class_balance(dataset, name):
    """Выводит в консоль распределение количества изображений по классам"""
    if isinstance(dataset, torch.utils.data.Subset):
        targets = [dataset.dataset.targets[i] for i in dataset.indices]
    else:
        targets = dataset.targets

    counts = Counter(targets)
    classes = dataset.dataset.classes

    print(f"\nБаланс классов в {name}:")
    for idx, class_name in enumerate(classes):
        print(f"  {class_name}: {counts.get(idx, 0)}")


def load_data(max_train_scenes=None, max_val_scenes=None):
    print("Загрузка набора данных")

    train_dataset = datasets.ImageFolder(
        "data/dataset_v2/train", transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        "data/dataset_v2/val", transform=val_transform
    )

    print(f"Полный train: {len(train_dataset)} изображений")
    print(f"Полный val: {len(val_dataset)} изображений")

    manifest_path = "data/dataset_manifest.csv"

    if max_train_scenes is not None:
        train_indices, n_train_scenes = load_balanced_indices(
            manifest_path, "train", max_train_scenes, train_dataset
        )
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        print(f"Ограниченный train: {len(train_dataset)} изображений "
              f"({n_train_scenes} сцен)")

    if max_val_scenes is not None:
        val_indices, n_val_scenes = load_balanced_indices(
            manifest_path, "val", max_val_scenes, val_dataset
        )
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
        print(f"Ограниченный val: {len(val_dataset)} изображений "
              f"({n_val_scenes} сцен)")

    # Контроль сбалансированности классов в выборках
    print_class_balance(train_dataset, "train")
    print_class_balance(val_dataset, "val")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, persistent_workers=True,
        pin_memory=(DEVICE.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, persistent_workers=True,
        pin_memory=(DEVICE.type == "cuda")
    )

    return train_loader, val_loader, train_dataset.dataset.classes


def build_model():
    print("\nЗагрузка архитектуры MobileNetV3-Small")
    model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.DEFAULT
    )
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, NUM_CLASSES)
    return model.to(DEVICE)


def build_optimizer(model):
    return optim.AdamW([
        {"params": model.classifier.parameters(), "lr": LR_CLASSIFIER},
        {"params": [p for n, p in model.named_parameters()
                    if not n.startswith("classifier")],
         "lr": LR_BACKBONE},
    ])


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return val_loss / len(val_loader), 100. * correct / total


def train():
    # Используются те же ограничения на количество сцен, что и для ResNet18
    train_loader, val_loader, classes = load_data(max_train_scenes=600, max_val_scenes=200)
    model = build_model()

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    print("\nНачало процесса обучения")
    train_losses, val_losses, val_accs = [], [], []
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0

    os.makedirs('models', exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step(val_loss)

        print(f"Эпоха {epoch+1}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        # Проверка значения функции потерь на валидации для реализации механизма ранней остановки
        if val_loss < best_val_loss - 0.0001:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            # Сохранение контрольной точки модели и состояния оптимизатора
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'classes': classes,
            }, 'models/best_model_mobilenetv3_v2.pth')
            print(f"\tНовая лучшая модель (Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nРаннее прекращение обучения на эпохе {epoch+1}")
                break

    print(f"\nОбучение завершено. Лучшая точность: {best_val_acc:.2f}% (Val Loss: {best_val_loss:.4f})")

    # Визуализация и сохранение графиков метрик обучения
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.plot(val_losses, label='Val Loss', marker='s')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Функция потерь (Loss)')
    ax1.set_title('Динамика функции потерь (MobileNetV3-Small)')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(val_accs, label='Val Accuracy', marker='o', color='green')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Точность (Accuracy, %)')
    ax2.set_title('Точность на валидационной выборке (MobileNetV3-Small)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_results_mobilenetv3_v2.png', dpi=150)
    print("График сохранён: training_results_mobilenetv3_v2.png")


if __name__ == "__main__":
    train()