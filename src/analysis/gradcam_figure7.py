# -*- coding: utf-8 -*-
"""
Figure 7 - composite Grad-CAM figure.
----------------------------------------------
Produces a single publication-ready Grad-CAM figure over the fine-tuned
ResNet-50 model for a set of representative examples.

Rows (5):
  1) A correctly classified LDPE example
  2) A uv-PE example misclassified as LDPE (illustrates the confusion)
  3) A correctly classified oxo-PE example (same family, yet separated)
  4) A correctly classified PVC example (distinct class)
  5) A correctly classified PP example (distinct class)

Usage (from the project root):
    python analysis/gradcam_figure7.py

Output: figure7.png and figure7.pdf (in the working directory)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

DATA_DIR = "data"
SPLIT    = "test"
IMG_SIZE = 224
MODEL_PATH = "weights/seed42/ResNet50_FINETUNED_best_model.h5"
SCAN_LIMIT = 80   # max number of images scanned per class

# Examples to find: (true class, desired prediction, row label)
SELECTIONS = [
    ("LDPE",   "LDPE",   "LDPE  ->  LDPE  (correct)"),
    ("uv-PE",  "LDPE",   "uv-PE  ->  LDPE  (error)"),
    ("oxo-PE", "oxo-PE", "oxo-PE  ->  oxo-PE  (correct)"),
    ("PVC",    "PVC",    "PVC  ->  PVC  (correct)"),
    ("PP",     "PP",     "PP  ->  PP  (correct)"),
]


def list_classes():
    p = os.path.join(DATA_DIR, SPLIT)
    return sorted([d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))])


def load_image(path):
    img = tf.keras.utils.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
    return tf.keras.utils.img_to_array(img)


def find_backbone(model):
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.Model):
            if any(isinstance(l, tf.keras.layers.Conv2D) for l in layer.layers):
                return i, layer
    return None, None


def last_conv_layer(backbone):
    for layer in reversed(backbone.layers):
        try:
            shp = layer.output_shape
        except Exception:
            continue
        if isinstance(shp, tuple) and len(shp) == 4:
            return layer
    return None


def build_gradcam(model):
    idx, backbone = find_backbone(model)
    last_conv = last_conv_layer(backbone)
    grad_backbone = tf.keras.Model(backbone.input, [last_conv.output, backbone.output])
    return grad_backbone, model.layers[:idx], model.layers[idx + 1:]


def heatmap(raw, grad_backbone, pre_layers, post_layers):
    x = raw
    for L in pre_layers:
        x = L(x, training=False)
    with tf.GradientTape() as tape:
        conv_out, backbone_out = grad_backbone(x)
        tape.watch(conv_out)
        y = backbone_out
        for L in post_layers:
            y = L(y, training=False)
        cls = tf.argmax(y[0])
        score = y[:, cls]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)
    cam = tf.maximum(cam, 0)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], (IMG_SIZE, IMG_SIZE))[..., 0]
    return cam.numpy(), int(cls.numpy())


def overlay(raw_img, cam):
    base = raw_img / 255.0
    colored = plt.get_cmap("jet")(cam)[..., :3]
    return np.clip(0.55 * base + 0.45 * colored, 0, 1)


def find_example(true_cls, target, classes, grad_backbone, pre_layers, post_layers):
    class_dir = os.path.join(DATA_DIR, SPLIT, true_cls)
    for fname in sorted(os.listdir(class_dir))[:SCAN_LIMIT]:
        raw_img = load_image(os.path.join(class_dir, fname))
        cam, idx = heatmap(np.expand_dims(raw_img.copy(), 0), grad_backbone, pre_layers, post_layers)
        if classes[idx] == target:
            return raw_img, cam
    # fall back to the first image if no match is found
    raw_img = load_image(os.path.join(class_dir, sorted(os.listdir(class_dir))[0]))
    cam, idx = heatmap(np.expand_dims(raw_img.copy(), 0), grad_backbone, pre_layers, post_layers)
    print(f"  WARNING: no {true_cls}->{target} example found, using the first image (prediction: {classes[idx]})")
    return raw_img, cam


def main():
    classes = list_classes()
    print("Class order:", classes)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    grad_backbone, pre_layers, post_layers = build_gradcam(model)

    n = len(SELECTIONS)
    fig, ax = plt.subplots(n, 2, figsize=(4.6, 1.85 * n))
    for r, (true_cls, target, label) in enumerate(SELECTIONS):
        raw_img, cam = find_example(true_cls, target, classes, grad_backbone, pre_layers, post_layers)
        ax[r, 0].imshow(raw_img.astype("uint8")); ax[r, 0].axis("off")
        ax[r, 1].imshow(overlay(raw_img, cam));   ax[r, 1].axis("off")
        ax[r, 0].set_ylabel(label, fontsize=9)
        # row label on the left panel (top-left instead of a centered title)
        ax[r, 0].set_title(label, fontsize=9, loc="left")
        if r == 0:
            ax[r, 0].annotate("Input", xy=(0.5, 1.18), xycoords="axes fraction",
                              ha="center", fontsize=9)
            ax[r, 1].annotate("Grad-CAM", xy=(0.5, 1.18), xycoords="axes fraction",
                              ha="center", fontsize=9)
    fig.suptitle("ResNet-50 — Grad-CAM", fontsize=11, y=0.995)
    fig.tight_layout()
    fig.savefig("figure7.png", dpi=300, bbox_inches="tight")
    fig.savefig("figure7.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved: figure7.png / figure7.pdf")


if __name__ == "__main__":
    main()
