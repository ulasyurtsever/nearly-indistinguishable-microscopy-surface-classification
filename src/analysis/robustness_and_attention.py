# -*- coding: utf-8 -*-
"""
Robustness and Grad-CAM border analysis of the fine-tuned ResNet-50 (inference only).

For each seed (42, 7, 123) the script evaluates the fine-tuned ResNet-50 on the test
set under nine conditions and measures the share of Grad-CAM mass that falls in the
border band of the image.

Conditions:
    baseline, illumination normalization, center crop (80%), contrast reduction (0.5x),
    corner mask with global-mean fill, center mask of equal area, edge-midpoint mask of
    equal area (four patches), corner mask with local-mean fill, corners only (inverse
    mask). All masks cover about 21% of the image area.

Test images are read with tf.keras.utils.image_dataset_from_directory (224 x 224,
bilinear), the same input pipeline as training. As a consistency check, the baseline
predictions are compared with the predictions saved during training
(results/seed{S}/ResNet50_FINETUNED_predictions.csv) and the agreement rate is printed.

Usage (from the repository root, in a TensorFlow environment):
    python src/analysis/robustness_and_attention.py
Paths can be overridden:
    python src/analysis/robustness_and_attention.py --data data --weights weights \
        --results results --out results/robustness

Expected layout:
    data/test/<class>/*.jpeg
    weights/seed{42,7,123}/ResNet50_FINETUNED_best_model.h5
    results/seed{42,7,123}/ResNet50_FINETUNED_predictions.csv

Memory: the test set (4811 x 224 x 224 x 3 float32) is kept in RAM, about 2.9 GB.

Outputs ({out}/):
    seed{S}/robustness_summary.csv        perturbation, accuracy, delta_vs_baseline, n_images
    seed{S}/baseline_predictions.csv      filename, y_true, y_pred
    seed{S}/attention_border_summary.csv  Grad-CAM border share by group
    seed{S}/attention_border_CI.csv       bootstrap 95% interval of the border share
    robustness_summary_allseeds.csv       perturbation, acc_seed42, acc_seed7, acc_seed123, mean, std, delta_mean
    attention_border_allseeds.csv         group, seed, mean, ci_low, ci_high, uniform_null_area, n
    run_info.json
"""

import os, json, csv, argparse, datetime, platform
import numpy as np
import tensorflow as tf

# ------------------------------------------------------------------ arguments
ap = argparse.ArgumentParser()
ap.add_argument("--data",    default="data", help="root containing test/<class>/")
ap.add_argument("--weights", default="weights", help="root containing seed{S}/ResNet50_FINETUNED_best_model.h5")
ap.add_argument("--results", default="results", help="root containing seed{S}/ResNet50_FINETUNED_predictions.csv")
ap.add_argument("--out",     default="results/robustness")
ap.add_argument("--seeds",   default="42,7,123")
ap.add_argument("--cam-per-class", type=int, default=80)
ap.add_argument("--n-boot",  type=int, default=2000)
ap.add_argument("--skip-cam", action="store_true", help="skip the Grad-CAM part (robustness only)")
args = ap.parse_args()

IMG_SIZE    = 224
BATCH       = 32
BORDER_FRAC = 0.18          # border band width as a fraction of the image size (area fraction about 58.7%)
RNG_SEED    = 42
SEEDS       = [int(s) for s in args.seeds.split(",")]
rng         = np.random.default_rng(RNG_SEED)
os.makedirs(args.out, exist_ok=True)

# ------------------------------------------------------------------ data (same input pipeline as training)
test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(args.data, "test"),
    image_size=(IMG_SIZE, IMG_SIZE),   # default interpolation='bilinear', as in training
    batch_size=BATCH,
    label_mode="int",
    shuffle=False,
)
CLASSES = list(test_ds.class_names)
FILES   = [os.path.basename(p) for p in test_ds.file_paths]
print("Class order:", CLASSES)

X_list, y_list = [], []
for xb, yb in test_ds:
    X_list.append(xb.numpy()); y_list.append(yb.numpy())
X = np.concatenate(X_list, 0).astype(np.float32)   # (N,224,224,3), 0..255, bilinear
Y = np.concatenate(y_list, 0)
del X_list, y_list
N = len(Y)
print(f"Test images: {N}   array: {X.shape}  {X.nbytes/1e9:.2f} GB")

# ------------------------------------------------------------------ masks
def _vignette_mask():
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    c = (IMG_SIZE - 1) / 2.0
    return np.sqrt((xx - c) ** 2 + (yy - c) ** 2) > (IMG_SIZE / 2.0)

VMASK  = _vignette_mask()                       # corners (about 21.4%)
N_PIX  = int(VMASK.sum())

def _center_mask(n):
    side = int(round(np.sqrt(n))); b = (IMG_SIZE - side) // 2
    m = np.zeros((IMG_SIZE, IMG_SIZE), bool); m[b:b+side, b:b+side] = True
    return m

def _edge_midpoint_mask(n):
    """Four squares touching the border at the midpoints of the four edges; total area = n."""
    side = int(round(np.sqrt(n / 4.0))); c0 = (IMG_SIZE - side) // 2
    m = np.zeros((IMG_SIZE, IMG_SIZE), bool)
    m[:side, c0:c0+side] = True            # top
    m[-side:, c0:c0+side] = True           # bottom
    m[c0:c0+side, :side] = True            # left
    m[c0:c0+side, -side:] = True           # right
    return m

CMASK = _center_mask(N_PIX)
EMASK = _edge_midpoint_mask(N_PIX)
h = IMG_SIZE // 2
QUADS = [VMASK.copy() for _ in range(4)]     # the four pieces of the corner mask
QUADS[0][h:, :] = False; QUADS[0][:, h:] = False   # top-left
QUADS[1][h:, :] = False; QUADS[1][:, :h] = False   # top-right
QUADS[2][:h, :] = False; QUADS[2][:, h:] = False   # bottom-left
QUADS[3][:h, :] = False; QUADS[3][:, :h] = False   # bottom-right

print(f"Corner mask        : {VMASK.mean()*100:.1f}%  ({N_PIX} px)")
print(f"Center mask        : {CMASK.mean()*100:.1f}%  ({int(CMASK.sum())} px)")
print(f"Edge-midpoint mask : {EMASK.mean()*100:.1f}%  ({int(EMASK.sum())} px)")

# ------------------------------------------------------------------ perturbations (batch, float32 0..255)
def gmean(xb):                                   # per-image global mean over (H,W,C)
    return xb.mean(axis=(1, 2, 3), keepdims=True)

def p_baseline(xb): return xb

def p_illum(xb, target=128.0):
    return np.clip(xb - gmean(xb) + target, 0, 255)

def p_center_crop(xb, frac=0.80):
    k = int(IMG_SIZE * frac); b = (IMG_SIZE - k) // 2
    crop = xb[:, b:b+k, b:b+k, :]
    return np.clip(tf.image.resize(crop, (IMG_SIZE, IMG_SIZE)).numpy(), 0, 255)

def p_contrast(xb, f=0.5):
    m = gmean(xb); return np.clip(m + f * (xb - m), 0, 255)

def _fill(xb, mask, value):
    out = xb.copy(); out[:, mask, :] = np.broadcast_to(value, out[:, mask, :].shape); return out

def p_corner_global(xb):
    return _fill(xb, VMASK, gmean(xb)[:, :, 0, :])

def p_center_global(xb):
    return _fill(xb, CMASK, gmean(xb)[:, :, 0, :])

def p_edge_mid_global(xb):
    return _fill(xb, EMASK, gmean(xb)[:, :, 0, :])

def p_corner_local(xb):
    out = xb.copy()
    for q in QUADS:
        m = xb[:, q, :].mean(axis=(1, 2), keepdims=True)    # (N,1,1): mean of that corner
        out[:, q, :] = np.broadcast_to(m, out[:, q, :].shape)
    return out

def p_corners_only(xb):
    return _fill(xb, ~VMASK, gmean(xb)[:, :, 0, :])

PERTURBATIONS = [
    ("Baseline (none)",                            p_baseline),
    ("Illumination normalization",                 p_illum),
    ("Center crop (80%)",                          p_center_crop),
    ("Contrast reduction (0.5x)",                  p_contrast),
    ("Corner/vignette mask (global-mean fill)",    p_corner_global),
    ("Center mask (equal area)",                   p_center_global),
    ("Edge-midpoint mask (equal area, 4 patches)", p_edge_mid_global),
    ("Corner mask (local-mean fill)",              p_corner_local),
    ("Corners only (inverse mask)",                p_corners_only),
]

def predict_all(model, fn):
    preds = np.empty(N, dtype=np.int64)
    for i in range(0, N, BATCH):
        xb = fn(X[i:i+BATCH])
        preds[i:i+BATCH] = np.argmax(model.predict(xb, verbose=0), axis=1)
    return preds

# ------------------------------------------------------------------ Grad-CAM
def find_backbone(model):
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.Model) and any(isinstance(l, tf.keras.layers.Conv2D) for l in layer.layers):
            return i, layer
    return None, None

def last_4d(backbone):
    for layer in reversed(backbone.layers):
        try: shp = layer.output_shape
        except Exception: continue
        if isinstance(shp, tuple) and len(shp) == 4: return layer
    return None

def build_gradcam(model):
    idx, backbone = find_backbone(model)
    grad_model = tf.keras.Model(backbone.input, [last_4d(backbone).output, backbone.output])
    return grad_model, model.layers[:idx], model.layers[idx+1:]

def heatmap(raw, grad_model, before, after):
    x = raw
    for L in before: x = L(x, training=False)
    with tf.GradientTape() as tape:
        conv_out, backbone_out = grad_model(x); tape.watch(conv_out)
        y = backbone_out
        for L in after: y = L(y, training=False)
        cls = tf.argmax(y[0]); score = y[:, cls]
    grads = tape.gradient(score, conv_out)
    w = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(conv_out[0] * w, axis=-1)
    cam = tf.maximum(cam, 0); cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], (IMG_SIZE, IMG_SIZE))[..., 0]
    return cam.numpy(), int(cls.numpy())

def _border_mask():
    b = int(IMG_SIZE * BORDER_FRAC)
    m = np.zeros((IMG_SIZE, IMG_SIZE), bool)
    m[:b, :] = True; m[-b:, :] = True; m[:, :b] = True; m[:, -b:] = True
    return m

BMASK = _border_mask(); AREA_REF = float(BMASK.mean())

def boot_ci(vals):
    vals = np.asarray(vals); n = len(vals)
    means = np.array([vals[rng.integers(0, n, n)].mean() for _ in range(args.n_boot)])
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def write_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(rows)

# ------------------------------------------------------------------ main loop
all_rows = {}          # perturbation -> {seed: acc}
cam_rows = []          # (group, seed, mean, lo, hi, area, n)

for seed in SEEDS:
    print(f"\n==================== SEED {seed} ====================")
    mpath = os.path.join(args.weights, f"seed{seed}", "ResNet50_FINETUNED_best_model.h5")
    model = tf.keras.models.load_model(mpath, compile=False)
    sdir = os.path.join(args.out, f"seed{seed}"); os.makedirs(sdir, exist_ok=True)

    # (A) robustness
    print("--- (A) accuracy under perturbations ---")
    rows = []; base_pred = None
    for name, fn in PERTURBATIONS:
        pred = predict_all(model, fn)
        acc = float((pred == Y).mean())
        if base_pred is None:
            base_pred = pred; base_acc = acc
        rows.append((name, acc, acc - base_acc))
        all_rows.setdefault(name, {})[seed] = acc
        print(f"  {name:44s} acc={acc*100:6.2f}%  delta={(acc-base_acc)*100:+6.2f} pt")
    write_csv(os.path.join(sdir, "robustness_summary.csv"),
              ["perturbation", "accuracy", "delta_vs_baseline", "n_images"],
              [(name, f"{acc:.4f}", f"{d:.4f}", N) for name, acc, d in rows])
    write_csv(os.path.join(sdir, "baseline_predictions.csv"), ["filename", "y_true", "y_pred"],
              [(f, CLASSES[t], CLASSES[p]) for f, t, p in zip(FILES, Y, base_pred)])

    # agreement check against the predictions saved during training
    ref = os.path.join(args.results, f"seed{seed}", "ResNet50_FINETUNED_predictions.csv")
    if os.path.isfile(ref):
        refrows = list(csv.DictReader(open(ref)))
        if len(refrows) == N:
            ref_pred = np.array([CLASSES.index(r["y_pred"]) if r["y_pred"] in CLASSES else int(r["y_pred"]) for r in refrows])
            ref_true = np.array([CLASSES.index(r["y_true"]) if r["y_true"] in CLASSES else int(r["y_true"]) for r in refrows])
            print(f"  CHECK: agreement with training-time predictions = {(ref_pred==base_pred).mean()*100:.2f}%  "
                  f"(saved acc={(ref_pred==ref_true).mean()*100:.2f}%, here={base_acc*100:.2f}%)")
        else:
            print(f"  CHECK: reference CSV has a different number of rows ({len(refrows)} vs {N}), skipped")
    else:
        print("  CHECK: reference prediction CSV not found, skipped")

    # (B) Grad-CAM border share
    if not args.skip_cam:
        print("--- (B) Grad-CAM border mass fraction ---")
        grad_model, before, after = build_gradcam(model)
        counter = {c: 0 for c in range(len(CLASSES))}
        recs = []                                  # (class, correct, fraction)
        for i in range(N):
            c = int(Y[i])
            if counter[c] >= args.cam_per_class: continue
            counter[c] += 1
            cam, pred = heatmap(X[i:i+1].copy(), grad_model, before, after)
            fr = float(cam[BMASK].sum() / (cam.sum() + 1e-8))
            recs.append((CLASSES[c], pred == c, fr))
        arr = np.array([r[2] for r in recs]); ok = np.array([r[1] for r in recs])
        pe = np.array([r[0] in ("LDPE", "uv-PE") for r in recs])
        print(f"  Border band AREA fraction (uniform null): {AREA_REF*100:5.1f}%")
        print(f"  CAM border fraction - all         : {arr.mean()*100:5.1f}%  (std {arr.std()*100:.1f})  n={len(arr)}")
        print(f"  CAM border fraction - correct     : {arr[ok].mean()*100:5.1f}%  n={int(ok.sum())}")
        print(f"  CAM border fraction - incorrect   : {arr[~ok].mean()*100:5.1f}%  n={int((~ok).sum())}")
        print(f"  CAM border fraction - LDPE+uv-PE  : {arr[pe].mean()*100:5.1f}%  n={int(pe.sum())}")
        write_csv(os.path.join(sdir, "attention_border_summary.csv"), ["group", "mean_cam_border_fraction", "n"],
                  [("border_area_reference", f"{AREA_REF:.4f}", 0), ("all", f"{arr.mean():.4f}", len(arr)),
                   ("correct", f"{arr[ok].mean():.4f}", int(ok.sum())), ("incorrect", f"{arr[~ok].mean():.4f}", int((~ok).sum())),
                   ("LDPE_uvPE", f"{arr[pe].mean():.4f}", int(pe.sum()))])
        ci_rows = []
        for g, v in (("all", arr), ("LDPE_uvPE", arr[pe])):
            m, lo, hi = boot_ci(v)
            ci_rows.append((g, f"{m:.4f}", f"{lo:.4f}", f"{hi:.4f}", f"{AREA_REF:.4f}", len(v)))
            cam_rows.append((g, seed, f"{m:.4f}", f"{lo:.4f}", f"{hi:.4f}", f"{AREA_REF:.4f}", len(v)))
            print(f"  95% CI [{g}]: {m*100:.2f}%  ({lo*100:.2f}-{hi*100:.2f})  "
                  f"{'< area fraction' if hi < AREA_REF else 'covers/exceeds area fraction'}")
        write_csv(os.path.join(sdir, "attention_border_CI.csv"),
                  ["group", "mean", "ci_low", "ci_high", "uniform_null_area", "n"], ci_rows)
    del model
    tf.keras.backend.clear_session()

# ------------------------------------------------------------------ combined summary
print("\n==================== THREE-SEED SUMMARY ====================")
base_mean = np.mean([all_rows["Baseline (none)"][s] for s in SEEDS])
out_rows = []
for name, _ in PERTURBATIONS:
    v = np.array([all_rows[name][s] for s in SEEDS])
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    out_rows.append([name] + [f"{x:.4f}" for x in v] + [f"{v.mean():.4f}", f"{sd:.4f}", f"{v.mean()-base_mean:.4f}"])
    print(f"  {name:44s} " + "  ".join(f"{x*100:6.2f}" for x in v) +
          f"  | {v.mean()*100:6.2f} +/- {sd*100:4.2f}  (delta {(v.mean()-base_mean)*100:+6.2f})")
write_csv(os.path.join(args.out, "robustness_summary_allseeds.csv"),
          ["perturbation"] + [f"acc_seed{s}" for s in SEEDS] + ["mean", "std", "delta_mean"], out_rows)
if cam_rows:
    write_csv(os.path.join(args.out, "attention_border_allseeds.csv"),
              ["group", "seed", "mean", "ci_low", "ci_high", "uniform_null_area", "n"], cam_rows)

json.dump({
    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    "tensorflow": tf.__version__, "python": platform.python_version(),
    "gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
    "loader": "tf.keras.utils.image_dataset_from_directory(image_size=(224,224), bilinear, shuffle=False) -- same as training",
    "seeds": SEEDS, "n_test_images": int(N), "classes": CLASSES,
    "perturbations": [p[0] for p in PERTURBATIONS],
    "corner_mask_pixels": N_PIX, "border_frac": BORDER_FRAC, "border_area_ref": AREA_REF,
    "cam_per_class": args.cam_per_class, "n_boot": args.n_boot, "rng_seed": RNG_SEED,
}, open(os.path.join(args.out, "run_info.json"), "w"), indent=2)
print(f"\nDone. Outputs: {args.out}/")
