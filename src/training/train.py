# src/training/train.py

import os
import csv
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from collections import Counter
import numpy as np
import random

from src.dataset.loader import SingleInputDataset
from src.models.resnet import get_model
from src.evaluation.evaluate import evaluate_model

def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def compute_class_weights(dataset, label_map):
    labels = [label_map[sample['label']] for sample in dataset]
    class_counts = Counter(labels)
    print("Class distribution:", class_counts)

    total = sum(class_counts.values())
    weights = [total / class_counts[i] for i in range(len(label_map))]
    print("Class weights:", weights)
    return torch.tensor(weights, dtype=torch.float32).cuda()

def train_model(data_dir, num_classes, label_map, save_path):
    seed_everything()
    os.makedirs(save_path, exist_ok=True)

    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    aug_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])

    full_dataset = SingleInputDataset(data_dir, transform=base_transform)
    train_len = int(0.7 * len(full_dataset))
    test_len = len(full_dataset) - train_len
    train_set, test_set = random_split(full_dataset, [train_len, test_len])

    # Apply augmentation to training set always
    train_set.dataset.transform = aug_transform

    class_weights = compute_class_weights(train_set, label_map)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, pin_memory=True)

    model = get_model(model_name='resnet18', num_classes=num_classes).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(10):
        model.train()
        total_loss = 0
        correct, total = 0, 0

        for batch in train_loader:
            inputs = batch['segmented'].cuda(non_blocking=True)
            labels = torch.tensor(
                [label_map[label] for label in batch['label']],
                dtype=torch.long
            ).cuda(non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            total_loss += loss.item()

        acc = correct / total
        print(f"[Epoch {epoch+1}] Loss: {total_loss / len(train_loader):.4f} | Train Accuracy: {acc:.4f}")

    torch.save(model.state_dict(), os.path.join(save_path, 'model.pt'))
    evaluate_model(model, test_loader, label_map, os.path.join(save_path, 'report.json'))



def load_label_map(dataset_name):
    label_csv_path = os.path.join("datasets", dataset_name, "labels.csv")
    label_map = {}

    with open(label_csv_path, mode='r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            label_id = int(row["label_id"])
            label_name = row["label_name"]
            label_map[label_id] = label_name

    return label_map



if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()

    dataset_name = 'crack'  # Adjust as per need
    label_map = load_label_map(dataset_name)
    data_dir = os.path.join("..", "..", "datasets", dataset_name)
    save_path = os.path.join("..", "..", "outputs", "models", dataset_name)

    train_model(
        data_dir=data_dir,
        num_classes=len(label_map),
        label_map=label_map,
        save_path=save_path
    )

# src/training/train.py

# import os
# import torch
# from torch.utils.data import DataLoader, random_split
# from torchvision import transforms
# from collections import Counter
# from src.dataset.loader import SingleInputDataset
# from src.models.resnet import get_model
# from src.evaluation.evaluate import evaluate_model
#
# def compute_class_weights(dataset, label_map):
#     labels = [label_map[sample['label']] for sample in dataset]
#     class_counts = Counter(labels)
#     total = sum(class_counts.values())
#     weights = [total / class_counts[i] for i in range(len(label_map))]
#     return torch.tensor(weights, dtype=torch.float32).cuda()
#
# def train_model(data_dir, num_classes, label_map, save_path):
#     os.makedirs(save_path, exist_ok=True)
#
#     base_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor()
#     ])
#
#     aug_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(15),
#         transforms.ColorJitter(),
#         transforms.ToTensor()
#     ])
#     print(2)
#     full_dataset = SingleInputDataset(data_dir, transform=base_transform)
#     train_len = int(0.7 * len(full_dataset))
#     test_len = len(full_dataset) - train_len
#     train_set, test_set = random_split(full_dataset, [train_len, test_len])
#     print(3)
#     if len(train_set) < 200:
#         print("Augmentation applied.")
#         train_set.dataset.transform = aug_transform
#
#     class_weights = compute_class_weights(train_set, label_map)
#     criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
#
#     train_loader = DataLoader(
#         train_set, batch_size=32, shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True
#     )
#     test_loader = DataLoader(
#         test_set, batch_size=32, shuffle=False, pin_memory=True, num_workers=2, persistent_workers=True
#     )
#
#     print(4)
#     model = get_model(model_name='resnet18', num_classes=num_classes).cuda()
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#     scaler = torch.cuda.amp.GradScaler()
#     print(5)
#     for epoch in range(10):
#         model.train()
#         total_loss = 0
#
#         for batch in train_loader:
#             inputs = batch['segmented'].cuda(non_blocking=True)
#             labels = torch.tensor([label_map[label] for label in batch['label']], dtype=torch.long).cuda()
#
#             with torch.cuda.amp.autocast():
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#             optimizer.zero_grad()
#             total_loss += loss.item()
#
#         print(f"[Epoch {epoch+1}] Avg Loss: {total_loss / len(train_loader):.4f}")
#
#     torch.save(model.state_dict(), os.path.join(save_path, 'model.pt'))
#     evaluate_model(model, test_loader, label_map, os.path.join(save_path, 'report.json'))
#
# if __name__ == "__main__":
#     print(0)
#     import torch.multiprocessing
#     torch.multiprocessing.freeze_support()
#     print(1)
#     dataset_name = 'bt'
#     label_map = {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}
#     train_model(
#         data_dir=r"..\..\datasets\bt",
#         num_classes=len(label_map),
#         label_map=label_map,
#         save_path=r"..\..\outputs\models\bt"
#     )
#
