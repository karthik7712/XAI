import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms, models
from PIL import Image
import os
import glob
import pandas as pd

def generate_gradcam(image_path, model, output_dir, segmented=False, device="cpu"):
    os.makedirs(output_dir, exist_ok=True)
    model.to(device)
    model.eval()

    target_layer = model.layer4[1].conv2  # for ResNet18
    activations, gradients = [], []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_backward_hook(backward_hook)

    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = transforms.ToTensor()(rgb_image).unsqueeze(0).to(device)

    output = model(input_tensor)
    class_idx = output.argmax().item()

    model.zero_grad()
    loss = output[0, class_idx]
    loss.backward()

    grads = gradients[0]
    acts = activations[0]
    pooled_grads = grads.mean(dim=[0, 2, 3])

    for i in range(acts.shape[1]):
        acts[:, i, :, :] *= pooled_grads[i]

    heatmap = acts.mean(dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= heatmap.max()
    heatmap = heatmap.detach().cpu().numpy()
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    filename = os.path.splitext(os.path.basename(image_path))[0]
    suffix = "_seg" if segmented else "_raw"
    cv2.imwrite(os.path.join(output_dir, f"{filename}_gradcam{suffix}.jpg"), superimposed_img)

    handle_fw.remove()
    handle_bw.remove()
    print(f"✅ Grad-CAM done for {filename}")


def run_for_dataset(dataset_name):
    base_dir = os.path.join("datasets", dataset_name)
    smoothed_dir = os.path.join(base_dir, "smoothed_images")
    segmented_dir = os.path.join(base_dir, "segmented_images")
    labels_path = os.path.join(base_dir, "labels.csv")
    model_path = os.path.join("outputs", "models", dataset_name, "model.pt")

    labels_df = pd.read_csv(labels_path)
    image_files = glob.glob(os.path.join(smoothed_dir, "*.jpg")) + glob.glob(os.path.join(smoothed_dir, "*.png"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"🔍 Found {len(image_files)} images in {dataset_name.upper()}")

    for img_path in image_files:
        img_name = os.path.basename(img_path)
        label_row = labels_df[labels_df["image"] == img_name]

        if label_row.empty:
            print(f"⚠️ Label not found for: {img_name}")
            continue

        label = label_row["class"].values[0]
        segmented_path = os.path.join(segmented_dir, img_name) if os.path.exists(os.path.join(segmented_dir, img_name)) else None
        output_dir = os.path.join("outputs", "xai", dataset_name, os.path.splitext(img_name)[0])

        print(f"🧠 Running XAI for {img_name} → Label: {label}")
        # if (int)((img_name.split("_")[1]).split(".")[0]) >= 5000:
        try:
            # Grad-CAM (raw + segmented)
            generate_gradcam(img_path, model, os.path.join(output_dir, "gradcam"), segmented=False, device=device)
            if segmented_path:
                generate_gradcam(segmented_path, model, os.path.join(output_dir, "gradcam"), segmented=True, device=device)
            # Add other methods here (counterfactuals, etc.) if needed
        except Exception as e:
            print(f"❌ Error processing {img_name}: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name folder in datasets/")
    args = parser.parse_args()
    run_for_dataset(args.dataset)

# python generate_gradcam.py --dataset mias
