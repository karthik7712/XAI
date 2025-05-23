import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms, models
from PIL import Image
import os

def generate_gradcam(image_path, model_path, output_dir, segmented=False):
    os.makedirs(output_dir, exist_ok=True)

    # Load the model architecture (ResNet in this case)
    model = models.resnet18(pretrained=False)  # Use the architecture you are working with
    # Modify the final fully connected layer to match your dataset (2 classes in this case)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    # Load the weights into the model
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    model.eval()  # Now you can call eval()

    # Define the target layer for Grad-CAM
    target_layer = model.layer4[1].conv2  # Example: for ResNet18, you might use layer4[1].conv2

    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(image).unsqueeze(0)

    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_backward_hook(backward_hook)

    output = model(input_tensor)
    class_idx = output.argmax().item()
    loss = output[0, class_idx]
    loss.backward()

    grads = gradients[0]
    acts = activations[0]

    # print("Grads shape:", grads.shape)  # Debug print for gradients shape
    # print("Activations shape:", acts.shape)  # Debug print for activations shape

    # Pool gradients along batch, height, and width dimensions
    pooled_grads = grads.mean(dim=[0, 2, 3])  # Adjust this if necessary

    for i in range(acts.shape[1]):  # Iterate over the channels
        acts[:, i, :, :] *= pooled_grads[i]

    heatmap = acts.mean(dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= heatmap.max()
    heatmap = heatmap.detach().numpy()
    heatmap = cv2.resize(heatmap, (224, 224))

    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    filename = os.path.splitext(os.path.basename(image_path))[0]
    suffix = "_seg" if segmented else "_raw"
    cv2.imwrite(os.path.join(output_dir, f"{filename}_gradcam{suffix}.jpg"), superimposed_img)

    handle_fw.remove()
    handle_bw.remove()
    print("graddone")

## FOR DATASET
