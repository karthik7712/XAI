import numpy as np
from PIL import Image
import os
from torchvision import transforms, models
import torch

def save_reference_features(reference_folder, output_file):
    # Load pretrained model (e.g., ResNet18) and remove classification head
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Identity()
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    features = []
    filenames = []

    for fname in os.listdir(reference_folder):
        if fname.lower().endswith((".jpg", ".png")):
            img_path = os.path.join(reference_folder, fname)
            try:
                img = Image.open(img_path).convert("RGB")
                tensor = transform(img).unsqueeze(0)  # shape: [1, 3, 224, 224]

                with torch.no_grad():
                    feat = model(tensor).squeeze().numpy().astype('float32')  # [512]

                features.append(feat)
                filenames.append(fname)
            except Exception as e:
                print(f"⚠️ Skipped {fname}: {e}")

    np.savez(output_file, features=np.array(features), filenames=np.array(filenames))
    print(f"✅ Saved {len(features)} CNN features to {output_file}")

if __name__ == "__main__":
    datasets = ["crack"]
    base_path = "datasets"

    for dataset in datasets:
        ref_folder = os.path.join(base_path, dataset, "smoothed_images")
        output_path = os.path.join(base_path, dataset, "reference_features.npz")

        if not os.path.exists(ref_folder):
            print(f"❌ Skipping {dataset}: folder not found at {ref_folder}")
            continue

        print(f"📦 Processing {dataset}...")
        save_reference_features(ref_folder, output_path)
