# -*- coding: utf-8 -*-
"""
t-SNE feature-space visualization (adapted to a nested model structure).
-----------------------------------------------------------------------------------
Extracts the pre-softmax features for test-set samples from the saved (.h5)
fine-tuned model, without retraining, and reduces them to two dimensions with
t-SNE.

IMPORTANT: the model contains the preprocessing and augmentation layers
internally, so images are passed as RAW (0-255); no external preprocessing is
applied.

Usage (from the project root):
    python analysis/tsne_analysis.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.manifold import TSNE

# =========================================================================
# CONFIG
# =========================================================================
DATA_DIR = "data"
SPLIT    = "test"
IMG_SIZE = 224
OUT_DIR  = "figures"

MODELS = [
    ("ResNet-50",
     "weights/seed42/ResNet50_FINETUNED_best_model.h5"),
    # ("EfficientNet-B0",
    #  "weights/seed42/EfficientNetB0_FINETUNED_best_model.h5"),
]

SAMPLES_PER_CLASS = 150
TSNE_PERPLEXITY   = 30
RANDOM_SEED       = 42
# =========================================================================


def list_classes(data_dir, split):
    p = os.path.join(data_dir, split)
    return sorted([d for d in os.listdir(p)
                   if os.path.isdir(os.path.join(p, d))])


def feature_model(model):
    """Model that returns the pre-softmax embedding (input to the final Dense layer).
    These are top-level layers, so the graph connection is straightforward."""
    try:
        return tf.keras.Model(model.inputs, model.layers[-1].input)
    except Exception:
        return tf.keras.Model(model.inputs, model.layers[-2].output)


def collect_samples(data_dir, split, classes, n_per):
    paths, labels = [], []
    for idx, cls in enumerate(classes):
        class_dir = os.path.join(data_dir, split, cls)
        files = sorted(os.listdir(class_dir))[:n_per]
        for d in files:
            paths.append(os.path.join(class_dir, d))
            labels.append(idx)
    return paths, np.array(labels)


def load_image(path, size):
    img = tf.keras.utils.load_img(path, target_size=(size, size))
    return tf.keras.utils.img_to_array(img)        # raw 0-255


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    classes = list_classes(DATA_DIR, SPLIT)
    print("Class order:", classes)
    paths, y = collect_samples(DATA_DIR, SPLIT, classes, SAMPLES_PER_CLASS)
    print(f"Collected {len(paths)} samples.")

    for label, path in MODELS:
        if not os.path.exists(path):
            print(f"[SKIP] model not found: {path}")
            continue
        print(f"\n=== {label} ===")
        model = tf.keras.models.load_model(path, compile=False)
        fmodel = feature_model(model)

        feats = []
        B = 32
        for i in range(0, len(paths), B):
            group = paths[i:i + B]
            arr = np.stack([load_image(p, IMG_SIZE) for p in group])   # raw 0-255
            feats.append(fmodel.predict(arr, verbose=0))
            print(f"  {min(i + B, len(paths))}/{len(paths)} features extracted", end="\r")
        feats = np.concatenate(feats, axis=0)
        print("\nFeature shape:", feats.shape)

        tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY,
                    init="pca", random_state=RANDOM_SEED)
        emb = tsne.fit_transform(feats)

        plt.figure(figsize=(7, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
        for idx, cls in enumerate(classes):
            m = y == idx
            plt.scatter(emb[m, 0], emb[m, 1], s=12, color=colors[idx],
                        label=cls, alpha=0.7, edgecolors="none")
        plt.legend(markerscale=1.6, fontsize=9, loc="best", frameon=True)
        plt.title(f"{label} - feature space (t-SNE)", fontsize=12)
        plt.xticks([]); plt.yticks([])
        plt.tight_layout()

        name = f"tsne_{label.replace('-', '').replace(' ', '')}"
        plt.savefig(os.path.join(OUT_DIR, name + ".png"), dpi=200, bbox_inches="tight")
        plt.savefig(os.path.join(OUT_DIR, name + ".pdf"), bbox_inches="tight")
        plt.close()
        print(f"saved: {name}.png / .pdf")

    print(f"\nDone. Outputs: {OUT_DIR}/")


if __name__ == "__main__":
    main()
