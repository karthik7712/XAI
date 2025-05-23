# import torch
# import numpy as np
# import faiss
# from PIL import Image
# import os
# from torchvision import transforms, models
#
# def generate_case_similarity(image_path, label, output_dir, feature_file):
#     os.makedirs(output_dir, exist_ok=True)
#     print("similaritystart")
#
#     # Load reference features
#     data = np.load(feature_file, allow_pickle=True)
#     ref_features = data['features'].astype('float32')
#     ref_filenames = data['filenames']
#
#     # Load model for feature extraction
#     model = models.resnet18(pretrained=True)
#     model.fc = torch.nn.Identity()  # Remove classification layer
#     model.eval()
#
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#     ])
#
#     image = Image.open(image_path).convert("RGB")
#     input_tensor = transform(image).unsqueeze(0)
#
#     with torch.no_grad():
#         feat = model(input_tensor).squeeze().numpy().astype('float32')
#
#     # Build FAISS index
#     index = faiss.IndexFlatL2(ref_features.shape[1])
#     index.add(ref_features)
#
#     D, I = index.search(np.array([feat]), 3)  # top 3
#     filename = os.path.splitext(os.path.basename(image_path))[0]
#     with open(os.path.join(output_dir, f"{filename}_similarity.txt"), 'w') as f:
#         for idx in I[0]:
#             f.write(f"{ref_filenames[idx]}\n")
#
#     print("similarityend ✅")


# generate_case_similarity.py
import numpy as np
from PIL import Image
import os
import faiss
import torch
from torchvision import transforms, models

def generate_case_similarity(image_path, label, output_dir, feature_file):
    os.makedirs(output_dir, exist_ok=True)
    print("similaritystart")

    # Load precomputed reference features from npz
    data = np.load(feature_file, allow_pickle=True)
    ref_features = data['features'].astype('float32')  # shape: [N, D]
    ref_filenames = data['filenames']

    # Load CNN model (ResNet18) for feature extraction
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Identity()
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load and preprocess input image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

    with torch.no_grad():
        feature = model(input_tensor).squeeze().numpy().astype('float32')  # [512]

    # Use FAISS for fast similarity search
    index = faiss.IndexFlatL2(ref_features.shape[1])  # L2 norm (or use IndexFlatIP for cosine)
    index.add(ref_features)
    D, I = index.search(np.array([feature]), 3)  # top 3 similar

    # Save results
    filename = os.path.splitext(os.path.basename(image_path))[0]
    with open(os.path.join(output_dir, f"{filename}_similarity.txt"), 'w') as f:
        for idx in I[0]:
            f.write(f"{ref_filenames[idx]}\t{D[0][list(I[0]).index(idx)]:.4f}\n")

    print("similarityend ✅")
