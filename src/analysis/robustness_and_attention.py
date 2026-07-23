# -*- coding: utf-8 -*-
"""
Quantitative shortcut / robustness analysis (inference only, no retraining).
---------------------------------------------------------------------------
Produces two diagnostics:

  (A) Accuracy under test-time image corruptions:
        - corner/vignette masking
        - illumination (brightness) normalization
        - contrast reduction
        - center crop (removes the outer band / vignette)
      If accuracy is robust to these corruptions, the decision rests on the
      surface itself rather than on an acquisition/vignette shortcut.

  (B) Fraction of Grad-CAM mass in the border/vignette region:
      For each test image, how much of the Grad-CAM heatmap falls inside the
      border band. If this fraction is clearly below the band's area fraction,
      attention is not locked onto the border. Reported separately for correct
      and incorrect predictions and for the LDPE/uv-PE subset.

Usage (from the project root):
    python robustness_and_attention.py

Output: console summary tables + robustness_summary.csv, attention_border_summary.csv
"""

import os
import numpy as np
import tensorflow as tf

# ------------------------- CONFIG -------------------------
DATA_DIR   = "data"     # root containing test/<class>/*.jpeg
SPLIT      = "test"
IMG_SIZE   = 224
MODEL_PATH = "weights/seed42/ResNet50_FINETUNED_best_model.h5"   # seed-42 fine-tuned model

BATCH      = 32
CAM_PER_CLASS = 80          # images per class for the Grad-CAM analysis (for speed)
BORDER_FRAC   = 0.18        # width of the border band (fraction of the image size)
RANDOM_SEED   = 42
# ----------------------------------------------------------

np.random.seed(RANDOM_SEED)


def list_classes():
    p = os.path.join(DATA_DIR, SPLIT)
    return sorted(d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d)))


def load_image(path):
    img = tf.keras.utils.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
    return tf.keras.utils.img_to_array(img)          # 0..255 float32


def all_paths(classes):
    paths, labels = [], []
    for i, s in enumerate(classes):
        class_dir = os.path.join(DATA_DIR, SPLIT, s)
        for f in sorted(os.listdir(class_dir)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(class_dir, f)); labels.append(i)
    return paths, np.array(labels)


# ------------------------- CORRUPTIONS (on raw 0..255) -------------------------
def _vignette_mask():
    """Circular mask around the center; the outside (corners) is True."""
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    cx = cy = (IMG_SIZE - 1) / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return r > (IMG_SIZE / 2.0)      # corner regions


_VMASK = _vignette_mask()


def no_perturbation(x):
    return x


def mask_corners(x):
    """Fill the corner/vignette region with the image mean (remove information)."""
    out = x.copy()
    m = float(out.mean())
    out[_VMASK] = m
    return out


def normalize_illumination(x, target=128.0):
    """Fix the global brightness by shifting the mean brightness to the target."""
    out = x.astype(np.float32)
    out = out - out.mean() + target
    return np.clip(out, 0, 255)


def reduce_contrast(x, factor=0.5):
    m = float(x.mean())
    return np.clip(m + factor * (x - m), 0, 255)


def center_crop(x, keep=0.80):
    """Take the central `keep` fraction as a square and resize back to 224 (drops the border/vignette)."""
    k = int(IMG_SIZE * keep)
    b = (IMG_SIZE - k) // 2
    crop = x[b:b + k, b:b + k, :]
    crop = tf.image.resize(crop, (IMG_SIZE, IMG_SIZE)).numpy()
    return np.clip(crop, 0, 255)


PERTURBATIONS = [
    ("Baseline (no perturbation)", no_perturbation),
    ("Corner/vignette masking",    mask_corners),
    ("Illumination normalization", normalize_illumination),
    ("Contrast reduction (0.5x)",  reduce_contrast),
    ("Center crop (80%)",          center_crop),
]


# ------------------------- (A) ROBUSTNESS -------------------------
def compute_accuracy(model, paths, labels, perturb):
    correct = 0
    buf_x, buf_y = [], []

    def flush():
        nonlocal correct
        if not buf_x:
            return
        arr = np.stack(buf_x, 0)
        pr = model.predict(arr, verbose=0)
        pred = np.argmax(pr, axis=1)
        correct += int(np.sum(pred == np.array(buf_y)))
        buf_x.clear(); buf_y.clear()

    for p, y in zip(paths, labels):
        x = load_image(p)
        buf_x.append(perturb(x)); buf_y.append(y)
        if len(buf_x) >= BATCH:
            flush()
    flush()
    return correct / len(paths)


# ------------------------- (B) GRAD-CAM (supports a nested base model) -------------------------
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


def _border_mask():
    b = int(IMG_SIZE * BORDER_FRAC)
    m = np.zeros((IMG_SIZE, IMG_SIZE), bool)
    m[:b, :] = True; m[-b:, :] = True; m[:, :b] = True; m[:, -b:] = True
    return m


def attention_border_fraction(model, classes):
    grad_backbone, pre_layers, post_layers = build_gradcam(model)
    bmask = _border_mask()
    area_fraction = bmask.mean()

    records = []   # (class_name, is_correct, border_fraction)
    for i, s in enumerate(classes):
        class_dir = os.path.join(DATA_DIR, SPLIT, s)
        files = sorted(f for f in os.listdir(class_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))[:CAM_PER_CLASS]
        for f in files:
            raw_img = load_image(os.path.join(class_dir, f))
            cam, pred = heatmap(np.expand_dims(raw_img.copy(), 0), grad_backbone, pre_layers, post_layers)
            total = cam.sum() + 1e-8
            border = cam[bmask].sum() / total
            records.append((s, pred == i, float(border)))
    return records, area_fraction


# ------------------------- MAIN -------------------------
def main():
    classes = list_classes()
    print("Class order:", classes)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    paths, labels = all_paths(classes)
    print(f"Number of test images: {len(paths)}")

    # (A) robustness
    print("\n=== (A) Accuracy under test-time corruptions ===")
    rows = []
    baseline = None
    for name, fn in PERTURBATIONS:
        acc = compute_accuracy(model, paths, labels, fn)
        if baseline is None:
            baseline = acc
        delta = acc - baseline
        rows.append((name, acc, delta))
        print(f"  {name:32s}  acc={acc*100:6.2f}%   delta={delta*100:+.2f} pt")
    with open("robustness_summary.csv", "w") as fh:
        fh.write("perturbation,accuracy,delta_vs_baseline\n")
        for name, acc, d in rows:
            fh.write(f"{name},{acc:.4f},{d:.4f}\n")
    print("  -> robustness_summary.csv written")

    # (B) attention border fraction
    print("\n=== (B) Fraction of Grad-CAM mass in the border region ===")
    records, area_fraction = attention_border_fraction(model, classes)
    arr = np.array([k[2] for k in records])
    correct = np.array([k[1] for k in records])
    pe = np.array([k[0] in ("LDPE", "uv-PE") for k in records])
    print(f"  Border-band AREA fraction (reference)      : {area_fraction*100:5.1f}%")
    print(f"  CAM border fraction - all images           : {arr.mean()*100:5.1f}%  (std {arr.std()*100:.1f})")
    print(f"  CAM border fraction - correctly classified : {arr[correct].mean()*100:5.1f}%")
    print(f"  CAM border fraction - misclassified        : {arr[~correct].mean()*100:5.1f}%")
    print(f"  CAM border fraction - LDPE + uv-PE subset  : {arr[pe].mean()*100:5.1f}%")
    print("  (If the CAM border fraction is clearly below the AREA fraction, attention is not locked onto the border.)")
    with open("attention_border_summary.csv", "w") as fh:
        fh.write("group,mean_cam_border_fraction,n\n")
        fh.write(f"border_area_reference,{area_fraction:.4f},0\n")
        fh.write(f"all,{arr.mean():.4f},{len(arr)}\n")
        fh.write(f"correct,{arr[correct].mean():.4f},{int(correct.sum())}\n")
        fh.write(f"incorrect,{arr[~correct].mean():.4f},{int((~correct).sum())}\n")
        fh.write(f"LDPE_uvPE,{arr[pe].mean():.4f},{int(pe.sum())}\n")
    print("  -> attention_border_summary.csv written")


if __name__ == "__main__":
    main()
