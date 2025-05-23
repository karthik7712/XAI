import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from src.models.load_model import load_model
import os
CLASS_TO_INDEX = {
    "glioma": 0,
    "meningioma": 1,
    "notumor":2,
    "pituitary":3
    # "NORMAL":0,
    # "BENIGN":1,
    # "MALIGNANT":2,
    # "Thin":0,
    # "Thick":1

    # Add others if needed
}

def generate_counterfactuals(image_path, target_class, output_dir, model_path, step_size=0.01, max_iter=100):
    try:
        # Load model
        model = load_model(model_path,len(CLASS_TO_INDEX))
        model.eval()
        # print("counterstart")
        # print(f"Model type: {type(model)}")

        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        input_tensor = transform(image).unsqueeze(0).requires_grad_(True)
        target_class_idx = CLASS_TO_INDEX.get(str(target_class))
        if target_class_idx is None:
            raise ValueError(f"Unknown target class: {target_class}")

        # Generate counterfactual
        for _ in range(max_iter):
            output = model(input_tensor)
            pred_class = output.argmax().item()
            if pred_class == target_class_idx:
                break
            loss = -output[0, target_class_idx]
            model.zero_grad()
            loss.backward()
            input_tensor.data -= step_size * input_tensor.grad.data
            input_tensor.grad.zero_()

        # Convert to image
        counterfactual_img = input_tensor.detach().squeeze().permute(1, 2, 0).numpy()
        counterfactual_img = (counterfactual_img * 255).clip(0, 255).astype(np.uint8)

        # Ensure output directory is writable
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        if not os.access(output_dir, os.W_OK):
            print(f"❌ Cannot write to directory: {output_dir}")
            return

        # Save image
        output_path = os.path.join(output_dir, "counterfactual.png")
        Image.fromarray(counterfactual_img).save(output_path)
        print("counterend ", output_path)

    except PermissionError as pe:
        print(f"❌ PermissionError: {pe}")
    except FileNotFoundError as fnf:
        print(f"❌ FileNotFoundError: {fnf}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

