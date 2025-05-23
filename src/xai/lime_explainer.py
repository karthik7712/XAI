import os
import numpy as np
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
from torchvision import transforms
import torch


def generate_lime(model_path,image_path, output_dir ):
    os.makedirs(output_dir, exist_ok=True)
    print("lime starts")

    from src.models.load_model import load_model  # assume this exists

    # Pass the model_path to load_model
    model = load_model(model_path)
    model.eval()

    image = Image.open(image_path).convert("RGB").resize((224, 224))
    image_np = np.array(image)

    def batch_predict(images):
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        batch = torch.stack([transform(Image.fromarray(img)) for img in images], dim=0)
        with torch.no_grad():
            outputs = model(batch)
        return outputs.numpy()

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image_np, batch_predict, top_labels=1, hide_color=0, num_samples=1000)

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10,
                                                hide_rest=False)
    output_img = mark_boundaries(temp / 255.0, mask)

    filename = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"{filename}_lime.jpg")
    Image.fromarray((output_img * 255).astype(np.uint8)).save(out_path)
    print("lime ends")
