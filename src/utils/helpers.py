def map_labels(label_str):
    label_dict = {
        "Normal": 0, "Benign": 1, "Malignant": 2,
        "glioma": 0, "meningioma": 1, "notumor": 2, "pituitary": 3,
        "Thin": 0, "Thick": 1
    }
    return label_dict[label_str]
