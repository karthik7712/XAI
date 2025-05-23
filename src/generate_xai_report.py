import os
import glob
import pandas as pd
from xai.gradcam import generate_gradcam
from xai.superpixel import generate_superpixels
from xai.lime_explainer import generate_lime
from xai.counterfactuals import generate_counterfactuals
from xai.case_similarity import generate_case_similarity

def run_xai_pipeline(img_path, segmented_path, label, model_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "counterfactuals"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "gradcam"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "superpixels"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "case_similarities"), exist_ok=True)
    # Grad-CAM
    # generate_gradcam(img_path, model_path, os.path.join(output_dir, "gradcam"), segmented=False)
    # if segmented_path:
    #     generate_gradcam(segmented_path, model_path, os.path.join(output_dir, "gradcam"), segmented=True)

    # Superpixels
    generate_superpixels(img_path, os.path.join(output_dir, "superpixels"), segmented=False)
    if segmented_path:
        generate_superpixels(segmented_path, os.path.join(output_dir, "superpixels"), segmented=True)

    # LIME (only on segmented image if available)
    # if segmented_path:
    #     generate_lime(model_path, segmented_path, os.path.join(output_dir, "lime"))

    # Counterfactual Explanations
    generate_counterfactuals(img_path,label,os.path.join(output_dir, "counterfactuals"),model_path)

    # Case Similarity
    generate_case_similarity(img_path, label, os.path.join(output_dir, "case_similarity"),os.path.join("..", "datasets", "crack","reference_features.npz"))

def run_for_dataset(dataset_name):
    base_dir = os.path.join("..", "datasets", dataset_name)
    smoothed_dir = os.path.join(base_dir, "smoothed_images")
    segmented_dir = os.path.join(base_dir, "segmented_images")
    labels_path = os.path.join(base_dir, "labels.csv")
    model_path = os.path.join("..", "outputs", "models", dataset_name, "model.pt")

    labels_df = pd.read_csv(labels_path)
    image_files = glob.glob(os.path.join(smoothed_dir, "*.jpg")) + glob.glob(os.path.join(smoothed_dir, "*.png"))

    print(f"🔍 Found {len(image_files)} images in {dataset_name.upper()}")

    for img_path in image_files:
        img_name = os.path.basename(img_path)
        label_row = labels_df[labels_df["image"] == img_name]

        if label_row.empty:
            print(f"⚠️ Label not found for: {img_name}")
            continue

        label = label_row["class"].values[0]
        segmented_path = os.path.join(segmented_dir, img_name) if os.path.exists(os.path.join(segmented_dir, img_name)) else None
        output_dir = os.path.join("..", "outputs", "xai", dataset_name, os.path.splitext(img_name)[0])

        print(f"✅ Running XAI for {img_name} → Label: {label}")
        try:
            # if (int)((img_name.split("k")[1]).split(".")[0]) > 1 and (int)((img_name.split("k")[1]).split(".")[0]) < 100:      when the code stops mid-run and this conditional helps not to start from beginning
            #     run_xai_pipeline(img_path, segmented_path, label, model_path, output_dir)
            run_xai_pipeline(img_path, segmented_path, label, model_path, output_dir)
        except Exception as e:
            print(f"❌ Error processing {img_name}: {e}")

if __name__ == "__main__":
    datasets = ['crack'] # Adjust based on need
    for dataset in datasets:
        print(f"\n🚀 Starting XAI generation for dataset: {dataset.upper()}")
        run_for_dataset(dataset)
