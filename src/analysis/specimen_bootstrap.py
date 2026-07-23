# -*- coding: utf-8 -*-
"""
Specimen-level (cluster) bootstrap of accuracy.
================================================================================
Motivation: the test set has only five physical specimens per class (identified
by the D#_N# filename prefix). The hundreds of images from one specimen are
strongly correlated, so the dominant source of uncertainty is which specimens
fall in the test fold, not the random seed. The seed-to-seed standard deviation
does not capture this. This script produces a specimen-clustered bootstrap 95%
CI for accuracy.

Steps:
  1) Runs the model through the SAME test pipeline as training
     (image_dataset_from_directory, shuffle=False), reproducing the reported
     accuracy exactly.
  2) Derives a specimen id from each filename: 'D1_N1_img5.jpeg' -> 'D1_N1'.
  3) Writes predictions_with_specimen_seed42.csv (filename, specimen, y_true, y_pred).
  4) Specimen-clustered bootstrap: for each class it resamples the five specimens
     with replacement, pools all their images, and computes overall accuracy and
     per-class recall; repeats N_BOOT times and reports the mean and 95% CI.

Note: the same script can be run for each seed model (change MODEL_PATH and
SEED_LABEL) and the intervals compared; reporting all three seeds is ideal.

Usage (from the project root, in the TF environment):
    python analysis/specimen_bootstrap.py

Output: console + predictions_with_specimen_seed42.csv + specimen_bootstrap_summary_seed42.csv
"""

import os, re
import numpy as np
import tensorflow as tf

# ============================ CONFIG ============================
DATA_DIR   = "data"
SPLIT      = "test"
IMG_SIZE   = (224, 224)
BATCH      = 32
MODEL_PATH = "weights/seed42/ResNet50_FINETUNED_best_model.h5"
OUT_DIR    = "results"
SEED_LABEL = "seed42"          # used in the output file names
N_BOOT     = 2000
RANDOM_SEED = 42
# ===============================================================

rng = np.random.default_rng(RANDOM_SEED)

def to_rgb(x, y):
    x = tf.image.grayscale_to_rgb(x) if x.shape[-1] == 1 else x
    return x, y

def specimen_of(path):
    """'.../LDPE/D1_N1_img5.jpeg' -> 'LDPE/D1_N1' (class + prefix)."""
    fn = os.path.basename(path)
    m = re.match(r"(.+?)_img\d+", fn)
    prefix = m.group(1) if m else fn
    cls = os.path.basename(os.path.dirname(path))
    return f"{cls}/{prefix}"

def main():
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, SPLIT), image_size=IMG_SIZE, batch_size=BATCH,
        label_mode="int", shuffle=False)
    class_names = test_ds.class_names
    file_paths = list(test_ds.file_paths)          # same order as predict (shuffle=False)
    print("Class order:", class_names)
    print(f"Number of test images: {len(file_paths)}")
    test_ds = test_ds.map(to_rgb).prefetch(tf.data.AUTOTUNE)

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    y_true, y_pred = [], []
    for images, labels in test_ds:
        p = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(p, axis=1)); y_true.extend(labels.numpy())
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    assert len(y_true) == len(file_paths), "order/length mismatch!"

    acc = float((y_true == y_pred).mean())
    print(f"\nOverall accuracy (reproduction): {acc*100:.2f}%   "
          f"(should match the reported seed-42 value, ~91.8%)")

    specimens = np.array([specimen_of(p) for p in file_paths])
    # sanity check:
    print("Example specimen ids:", sorted(set(specimens))[:6], "...")

    # write predictions_with_specimen
    pcsv = os.path.join(OUT_DIR, f"predictions_with_specimen_{SEED_LABEL}.csv")
    with open(pcsv, "w") as fh:
        fh.write("filename,specimen,y_true,y_pred\n")
        for fp, sp, yt, yp in zip(file_paths, specimens, y_true, y_pred):
            fh.write(f"{os.path.basename(fp)},{sp},{int(yt)},{int(yp)}\n")
    print(f"-> {pcsv}")

    # class -> its specimens
    cls_of_spec = {sp: sp.split("/")[0] for sp in set(specimens)}
    by_class = {}
    for sp in set(specimens):
        by_class.setdefault(cls_of_spec[sp], []).append(sp)
    # specimen -> indices
    idx_of = {sp: np.where(specimens == sp)[0] for sp in set(specimens)}

    def one_bootstrap():
        sel = []
        for c, sps in by_class.items():
            sps = np.array(sps)
            pick = sps[rng.integers(0, len(sps), len(sps))]   # 5 out of 5, with replacement
            for sp in pick: sel.append(idx_of[sp])
        sel = np.concatenate(sel)
        yt, yp = y_true[sel], y_pred[sel]
        overall = (yt == yp).mean()
        rec = {}
        for ci_, c in enumerate(class_names):
            mask = yt == ci_
            rec[c] = float((yp[mask] == ci_).mean()) if mask.sum() else np.nan
        return overall, rec

    boots_overall = np.empty(N_BOOT)
    boots_rec = {c: np.empty(N_BOOT) for c in class_names}
    for b in range(N_BOOT):
        ov, rec = one_bootstrap()
        boots_overall[b] = ov
        for c in class_names: boots_rec[c][b] = rec[c]

    def ci(a):
        a = a[~np.isnan(a)]
        return a.mean(), np.percentile(a, 2.5), np.percentile(a, 97.5)

    print(f"\n=== Specimen-clustered bootstrap ({N_BOOT} replicates) ===")
    m, lo, hi = ci(boots_overall)
    print(f"Overall accuracy : {m*100:5.2f}%  (95% CI {lo*100:.2f} - {hi*100:.2f})")
    print("Per-class recall (specimen-level CI):")
    rows = [("__overall__", m, lo, hi)]
    for c in class_names:
        mc, loc, hic = ci(boots_rec[c])
        print(f"  {c:8s}: {mc*100:5.2f}%  (95% CI {loc*100:.2f} - {hic*100:.2f})")
        rows.append((c, mc, loc, hic))
    scsv = os.path.join(OUT_DIR, f"specimen_bootstrap_summary_{SEED_LABEL}.csv")
    with open(scsv, "w") as fh:
        fh.write("group,mean,ci_low,ci_high\n")
        for g, mm, l, h in rows: fh.write(f"{g},{mm:.4f},{l:.4f},{h:.4f}\n")
    print(f"-> {scsv}")
    print("\nNOTE: because each class has only five specimens, these intervals are")
    print("      considerably wider than the seed-based standard deviation; this is")
    print("      the true (specimen-level) uncertainty.")


if __name__ == "__main__":
    main()
