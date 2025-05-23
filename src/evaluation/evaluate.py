# src/evaluation/evaluate.py

import torch
import json
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, dataloader, label_map, report_path):
    model.eval()
    y_true = []
    y_pred = []
    class_names = list(label_map.keys())
    inv_label_map = {v: k for k, v in label_map.items()}

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['segmented'].cuda()
            labels = torch.tensor([label_map[label] for label in batch['label']], dtype=torch.long).cuda()

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred).tolist()

    output = {
        "classification_report": report,
        "confusion_matrix": cm,
        "labels": class_names
    }

    with open(report_path, 'w') as f:
        json.dump(output, f, indent=4)

    print("✔ Classification report saved.")
