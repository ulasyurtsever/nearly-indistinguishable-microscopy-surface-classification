# -*- coding: utf-8 -*-
"""
Generate manuscript figures 3-6 from the multi-seed (seed-42) results.

Reads the seed-42 result folder and writes, into the LaTeX folder:
    figure3.pdf   - feature-extraction accuracy bars (8 models)
    figure4a.pdf  - training/validation dynamics, feature extraction (ResNet50, EfficientNetB0)
    figure4b.pdf  - training/validation dynamics, fine-tuning
    figure5.pdf   - confusion matrices (ResNet50-FT, EfficientNetB0-FT)
    figure6a.pdf  - class-wise ROC, ResNet50-FT
    figure6b.pdf  - class-wise ROC, EfficientNetB0-FT

Run from the project root:
    python analiz_scriptleri/figures3_6_from_seed42.py
Dependencies: pandas, numpy, matplotlib, scikit-learn
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# ------------------------- CONFIG -------------------------
SEED_DIR = "results/seed42"   # seed-42 results
OUT_DIR  = "figures"                             # where the .pdf figures go
CLASSES  = ["LDPE", "PLA", "PP", "PS", "PVC", "XPS", "oxo-PE", "uv-PE"]
# ----------------------------------------------------------


def figure3():
    p1 = pd.read_csv(f"{SEED_DIR}/tum_modeller_karsilastirma_zscore_PHASE1.csv").set_index("Model")
    order = [("LeNet-5", "LeNet-5"), ("Custom_Modern_Net", "CustomNet"),
             ("ResNet50-Scratch", "ResNet-50 (scr)"), ("VGG16", "VGG-16"),
             ("VGG19", "VGG-19"), ("InceptionV3", "InceptionV3"),
             ("EfficientNetB0", "EffNet-B0"), ("ResNet50", "ResNet-50")]
    labels = [l for _, l in order]
    tr = [p1.loc[k, "Train Acc"] for k, _ in order]
    va = [p1.loc[k, "Val Acc"] for k, _ in order]
    te = [p1.loc[k, "Test Acc"] for k, _ in order]
    x = np.arange(len(labels)); w = 0.26
    plt.figure(figsize=(14, 7)); ax = plt.gca()
    ax.bar(x - w, tr, w, label="Training Accuracy",   color="#4C78A8", edgecolor="black", linewidth=0.6)
    ax.bar(x,     va, w, label="Validation Accuracy", color="#F58518", edgecolor="black", linewidth=0.6)
    ax.bar(x + w, te, w, label="Test Accuracy",       color="#54A24B", edgecolor="black", linewidth=0.6)
    for i in range(len(labels)):
        for off, v in ((-w, tr[i]), (0, va[i]), (w, te[i])):
            ax.text(x[i] + off, v + 0.008, f"{v*100:.1f}", ha="center", va="bottom", fontsize=7.5, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20); ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("Accuracy"); ax.legend(loc="lower right"); ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/figure3.pdf"); plt.close()


def _dynamics(eff_file, res_file, out):
    e = pd.read_csv(eff_file); r = pd.read_csv(res_file)
    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=300); ax2 = ax1.twinx()
    ax1.plot(e["epoch"], e["accuracy"],     "-",  lw=2, color="#1f77b4", label="EffNetB0 Train Acc")
    ax1.plot(e["epoch"], e["val_accuracy"], "--", lw=2, color="#1f77b4", label="EffNetB0 Val Acc")
    ax1.plot(r["epoch"], r["accuracy"],     "-",  lw=2, color="#17becf", label="ResNet50 Train Acc")
    ax1.plot(r["epoch"], r["val_accuracy"], "--", lw=2, color="#17becf", label="ResNet50 Val Acc")
    ax2.plot(e["epoch"], e["loss"],     "-",  lw=1.5, color="#d62728", alpha=.7, label="EffNetB0 Train Loss")
    ax2.plot(e["epoch"], e["val_loss"], "--", lw=1.5, color="#d62728", alpha=.7, label="EffNetB0 Val Loss")
    ax2.plot(r["epoch"], r["loss"],     "-",  lw=1.5, color="#ff7f0e", alpha=.7, label="ResNet50 Train Loss")
    ax2.plot(r["epoch"], r["val_loss"], "--", lw=1.5, color="#ff7f0e", alpha=.7, label="ResNet50 Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy"); ax2.set_ylabel("Loss"); ax1.grid(True, alpha=.3)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right", fontsize=7)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/{out}"); plt.close()


def figure4():
    _dynamics(f"{SEED_DIR}/EfficientNetB0_training_log.csv",
              f"{SEED_DIR}/ResNet50_training_log.csv", "figure4a.pdf")
    _dynamics(f"{SEED_DIR}/EfficientNetB0_FINETUNED_training_log.csv",
              f"{SEED_DIR}/ResNet50_FINETUNED_training_log.csv", "figure4b.pdf")


def _cm(name):
    d = pd.read_csv(f"{SEED_DIR}/{name}_predictions.csv")
    return confusion_matrix(d["y_true"], d["y_pred"], labels=range(8))


def figure5():
    cmR, cmE = _cm("ResNet50_FINETUNED"), _cm("EfficientNetB0_FINETUNED")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), dpi=300); fig.subplots_adjust(wspace=0.2)
    vmax = max(cmR.max(), cmE.max())
    for ax, cm, title in ((axes[0], cmR, "(a) ResNet-50"), (axes[1], cmE, "(b) EfficientNet-B0")):
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(8)); ax.set_yticks(range(8))
        ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=7); ax.set_yticklabels(CLASSES, fontsize=7)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=10)
        th = cm.max() / 2
        for i in range(8):
            for j in range(8):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                        color="white" if cm[i, j] > th else "black")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/figure5.pdf"); plt.close()


def _roc(name, out, title):
    d = pd.read_csv(f"{SEED_DIR}/{name}_predictions.csv"); y = d["y_true"].values
    P = d[[f"prob_{c}" for c in CLASSES]].values
    yb = label_binarize(y, classes=range(8))
    plt.figure(figsize=(6.5, 6), dpi=300)
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    for i, c in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(yb[:, i], P[:, i]); a = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=1.6, label=f"{c} (AUC={a:.3f})")
    fpr, tpr, _ = roc_curve(yb.ravel(), P.ravel()); am = auc(fpr, tpr)
    plt.plot(fpr, tpr, "k:", lw=2.5, label=f"Micro-avg (AUC={am:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    plt.xlim(0, 1); plt.ylim(0, 1.02); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(title, fontsize=10); plt.legend(loc="lower right", fontsize=7); plt.grid(True, alpha=.3)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/{out}"); plt.close()


def figure6():
    _roc("ResNet50_FINETUNED", "figure6a.pdf", "(a) ResNet-50")
    _roc("EfficientNetB0_FINETUNED", "figure6b.pdf", "(b) EfficientNet-B0")


if __name__ == "__main__":
    figure3(); figure4(); figure5(); figure6()
    print("Done: figure3.pdf, figure4a.pdf, figure4b.pdf, figure5.pdf, figure6a.pdf, figure6b.pdf")
