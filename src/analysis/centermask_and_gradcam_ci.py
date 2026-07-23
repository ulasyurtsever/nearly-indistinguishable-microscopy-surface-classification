# -*- coding: utf-8 -*-
"""
Additional robustness / attention analysis (inference only, no retraining).
===========================================================================
Uses the SAME loading/preprocessing as robustness_and_attention.py, so the
numbers are directly comparable to the corruption table in the paper.

Adds two checks:

  (1) EQUAL-AREA CENTER-MASK control:
      Masks exactly the same number of pixels as the corner/vignette mask, but
      in the CENTER of the image. Purpose: to tell whether the large drop under
      corner masking reflects reliance on the border, or simply the removal of
      about 21% of the pixels. If the center mask produces a comparable drop,
      the loss is due to the removed area / out-of-distribution input, not the
      border region.

  (2) CONFIDENCE INTERVAL + UNIFORM-ATTENTION NULL for the Grad-CAM border-mass
      metric: computes a 95% bootstrap CI for the border fraction and compares
      it against the border band's AREA fraction (the value expected if
      attention were spatially uniform). If the CI upper bound is below the area
      fraction, attention statistically avoids the border (no border shortcut).

Usage (from the project root, in the TF environment):
    python analysis/centermask_and_gradcam_ci.py

Output: console summary + robustness_centermask_summary.csv, attention_border_CI.csv
"""

import os
import numpy as np
import tensorflow as tf

# ============================ CONFIG (edit paths as needed) ============================
DATA_DIR   = "data"                                    # root containing test/<class>/*.jpeg
SPLIT      = "test"
IMG_SIZE   = 224
MODEL_PATH = "weights/seed42/ResNet50_FINETUNED_best_model.h5"
OUT_DIR    = "results"                                 # output directory
BATCH      = 32
CAM_PER_CLASS = 80        # images per class for Grad-CAM (same as the main analysis)
BORDER_FRAC   = 0.18      # border-band width (same as the main analysis -> area fraction ~58.7%)
N_BOOT        = 2000      # number of bootstrap replicates
RANDOM_SEED   = 42
# ======================================================================================

rng = np.random.default_rng(RANDOM_SEED)


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


# ------------------- masks -------------------
def _vignette_mask():
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    c = (IMG_SIZE - 1) / 2.0
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    return r > (IMG_SIZE / 2.0)      # corners

_VMASK = _vignette_mask()

def _center_mask_equal_area(n_pixels):
    """Mask the same number of pixels as the corner mask, as a square in the center."""
    side = int(round(np.sqrt(n_pixels)))
    b = (IMG_SIZE - side) // 2
    m = np.zeros((IMG_SIZE, IMG_SIZE), bool)
    m[b:b + side, b:b + side] = True
    return m

_CMASK = _center_mask_equal_area(int(_VMASK.sum()))

def no_perturbation(x):
    return x

def mask_corners(x):
    out = x.copy(); out[_VMASK] = float(out.mean()); return out

def mask_center(x):
    out = x.copy(); out[_CMASK] = float(out.mean()); return out


PERTURBATIONS = [
    ("Baseline (none)",                no_perturbation),
    ("Corner/vignette mask",           mask_corners),
    ("Center mask (equal area)",       mask_center),
]

def compute_accuracy(model, paths, labels, perturb):
    correct = 0; buf_x, buf_y = [], []
    def flush():
        nonlocal correct
        if not buf_x: return
        pr = model.predict(np.stack(buf_x, 0), verbose=0)
        correct += int(np.sum(np.argmax(pr, axis=1) == np.array(buf_y)))
        buf_x.clear(); buf_y.clear()
    for p, y in zip(paths, labels):
        buf_x.append(perturb(load_image(p))); buf_y.append(y)
        if len(buf_x) >= BATCH: flush()
    flush()
    return correct / len(paths)


# ------------------- Grad-CAM (supports a nested base model) -------------------
def find_backbone(model):
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.Model) and any(isinstance(l, tf.keras.layers.Conv2D) for l in layer.layers):
            return i, layer
    return None, None

def last_conv_layer(backbone):
    for layer in reversed(backbone.layers):
        try: shp = layer.output_shape
        except Exception: continue
        if isinstance(shp, tuple) and len(shp) == 4: return layer
    return None

def build_gradcam(model):
    idx, backbone = find_backbone(model)
    last_conv = last_conv_layer(backbone)
    grad_backbone = tf.keras.Model(backbone.input, [last_conv.output, backbone.output])
    return grad_backbone, model.layers[:idx], model.layers[idx + 1:]

def heatmap(raw, grad_backbone, pre_layers, post_layers):
    x = raw
    for L in pre_layers: x = L(x, training=False)
    with tf.GradientTape() as tape:
        conv_out, backbone_out = grad_backbone(x); tape.watch(conv_out)
        y = backbone_out
        for L in post_layers: y = L(y, training=False)
        cls = tf.argmax(y[0]); score = y[:, cls]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)
    cam = tf.maximum(cam, 0); cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], (IMG_SIZE, IMG_SIZE))[..., 0]
    return cam.numpy(), int(cls.numpy())

def _border_mask():
    b = int(IMG_SIZE * BORDER_FRAC)
    m = np.zeros((IMG_SIZE, IMG_SIZE), bool)
    m[:b, :] = True; m[-b:, :] = True; m[:, :b] = True; m[:, -b:] = True
    return m


def main():
    classes = list_classes()
    print("Class order:", classes)
    print(f"Corner mask area fraction : {_VMASK.mean()*100:.1f}%  ({int(_VMASK.sum())} pixels)")
    print(f"Center mask area fraction : {_CMASK.mean()*100:.1f}%  ({int(_CMASK.sum())} pixels)  <- equal area")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    paths, labels = all_paths(classes)
    print(f"Number of test images: {len(paths)}")
    os.makedirs(OUT_DIR, exist_ok=True)

    # (1) center vs corner mask
    print("\n=== (1) Corner mask vs equal-area CENTER mask ===")
    rows = []; baseline = None
    for name, fn in PERTURBATIONS:
        acc = compute_accuracy(model, paths, labels, fn)
        if baseline is None: baseline = acc
        d = acc - baseline; rows.append((name, acc, d))
        print(f"  {name:28s} acc={acc*100:6.2f}%  delta={d*100:+.2f} pt")
    with open(os.path.join(OUT_DIR, "robustness_centermask_summary.csv"), "w") as fh:
        fh.write("perturbation,accuracy,delta_vs_baseline\n")
        for name, acc, d in rows: fh.write(f"{name},{acc:.4f},{d:.4f}\n")
    print("  -> robustness_centermask_summary.csv")
    print("  NOTE: if the center-mask drop is close to the corner-mask drop, the loss comes from the")
    print("        hidden AREA / out-of-distribution input, not the border (not evidence for a shortcut).")

    # (2) Grad-CAM border fraction + bootstrap CI + uniform null
    print("\n=== (2) Grad-CAM border-mass fraction: 95% bootstrap CI + uniform null ===")
    grad_backbone, pre_layers, post_layers = build_gradcam(model)
    bmask = _border_mask(); area_ref = bmask.mean()
    fr_all, fr_pe = [], []
    for i, s in enumerate(classes):
        class_dir = os.path.join(DATA_DIR, SPLIT, s)
        files = sorted(f for f in os.listdir(class_dir) if f.lower().endswith((".jpg",".jpeg",".png")))[:CAM_PER_CLASS]
        for f in files:
            raw_img = load_image(os.path.join(class_dir, f))
            cam, _ = heatmap(np.expand_dims(raw_img.copy(), 0), grad_backbone, pre_layers, post_layers)
            fr = float(cam[bmask].sum() / (cam.sum() + 1e-8))
            fr_all.append(fr)
            if s in ("LDPE", "uv-PE"): fr_pe.append(fr)

    def boot_ci(vals):
        vals = np.asarray(vals); n = len(vals)
        means = np.array([vals[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
        return vals.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)

    m, lo, hi = boot_ci(fr_all)
    mpe, lope, hipe = boot_ci(fr_pe)
    print(f"  Border-band AREA fraction (uniform null)   : {area_ref*100:5.2f}%")
    print(f"  CAM border fraction - all      : {m*100:5.2f}%  (95% CI {lo*100:.2f}-{hi*100:.2f})  n={len(fr_all)}")
    print(f"  CAM border fraction - LDPE+uvPE: {mpe*100:5.2f}%  (95% CI {lope*100:.2f}-{hipe*100:.2f})  n={len(fr_pe)}")
    verdict = "AVOIDS THE BORDER (CI upper bound < area fraction)" if hi < area_ref else "inconclusive"
    print(f"  Uniform-null test: {verdict}")
    with open(os.path.join(OUT_DIR, "attention_border_CI.csv"), "w") as fh:
        fh.write("group,mean,ci_low,ci_high,uniform_null_area,n\n")
        fh.write(f"all,{m:.4f},{lo:.4f},{hi:.4f},{area_ref:.4f},{len(fr_all)}\n")
        fh.write(f"LDPE_uvPE,{mpe:.4f},{lope:.4f},{hipe:.4f},{area_ref:.4f},{len(fr_pe)}\n")
    print("  -> attention_border_CI.csv")


if __name__ == "__main__":
    main()
