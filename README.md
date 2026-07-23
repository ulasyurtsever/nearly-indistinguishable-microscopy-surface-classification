# Nearly Indistinguishable Microscopy Surface Classification

Code and per-run results for the paper **"Transfer Learning and Model Attention on
Nearly Indistinguishable Optical-Microscopy Surfaces"** (manuscript currently under peer review).

The study evaluates how well convolutional neural networks separate eight polymer
surfaces (LDPE, PLA, PP, PS, PVC, XPS, oxo-PE, uv-PE) imaged under an optical
microscope, where the classes are visually almost indistinguishable. It compares
ImageNet-pretrained architectures against from-scratch controls under a common
two-stage protocol, reports mean and standard deviation over three random seeds, and
audits where the models look with Grad-CAM, t-SNE, and a set of robustness and
attention measurements.

## Key result

The fine-tuned ResNet-50 reaches **91.85 ± 0.31%** accuracy and **0.996** AUC over
seeds 42, 7, and 123. For the same architecture, ImageNet pretraining adds 13.6
accuracy points at feature extraction and 16.6 after fine-tuning. The residual
confusion concentrates on the LDPE and uv-PE pair, which the attention analyses link
to genuine visual similarity rather than to acquisition shortcuts.

## Repository layout

```
nearly-indistinguishable-microscopy-surface-classification/
├── src/
│   ├── train_models.py            # two-stage, multi-seed training driver
│   ├── train_models.ipynb         # notebook version of the same pipeline
│   └── analysis/
│       ├── make_figures_3_6.py         # comparison / class-wise / confusion / ROC figures (CSV only)
│       ├── gradcam_figure7.py          # Grad-CAM heatmaps (needs model + data)
│       ├── tsne_analysis.py            # t-SNE of pre-softmax features (needs model + data)
│       ├── robustness_and_attention.py # corruption robustness + Grad-CAM border mass
│       ├── centermask_and_gradcam_ci.py# equal-area centre-mask control + bootstrap CI
│       └── specimen_bootstrap.py       # specimen-level (cluster) bootstrap of accuracy
└── results/
    ├── seed42/                     # seed7/ and seed123/ mirror this structure
    │   ├── <Model>_predictions.csv           # per-image true label, predicted label, class probabilities
    │   ├── <Model>_classification_report.csv # per-class precision, recall, F1
    │   ├── <Model>_training_log.csv          # per-epoch loss and accuracy
    │   ├── run_info_PHASE1.json              # feature-extraction run metadata (versions, seed, timing)
    │   ├── run_info_FINETUNED.json           # fine-tuning run metadata
    │   ├── tum_modeller_karsilastirma_zscore_PHASE1.csv    # all-models summary, feature-extraction stage
    │   └── tum_modeller_karsilastirma_zscore_FINETUNED.csv # all-models summary, fine-tuning stage
    ├── seed7/                      # same layout as seed42/
    ├── seed123/                    # same layout as seed42/
    ├── robustness_summary.csv
    ├── attention_border_summary.csv
    ├── robustness_centermask_summary.csv
    ├── attention_border_CI.csv
    ├── specimen_bootstrap_summary_seed42.csv
    └── predictions_with_specimen_seed42.csv
```

In each seed folder, `<Model>` covers the eight architectures evaluated at the feature-extraction stage (LeNet-5, Custom_Modern_Net, ResNet50-Scratch, VGG16, VGG19, InceptionV3, EfficientNetB0, ResNet50) plus the two fine-tuned models (ResNet50_FINETUNED and EfficientNetB0_FINETUNED).

## Data and trained weights

The microscopy dataset and the trained model weights (`*.h5`) are **not** included here
because of their size and access constraints, in line with the paper's availability
statement. They are available from the corresponding author on reasonable request.

To run the scripts that need them, place the files as follows (paths are configurable
at the top of each script):

```
data/                                   # test/<class>/*.jpeg  (and train/, val/ for training)
weights/seed42/ResNet50_FINETUNED_best_model.h5
```

## Environment

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.10
pip install -r requirements.txt
```

Runs were produced with Python 3.10 and TensorFlow 2.15.1 on an NVIDIA RTX A6000 GPU.
The random seeds are 42, 7, and 123.

## Reproducing the results

1. **Figures from the released CSVs (no GPU, no data needed).**
   ```bash
   python src/analysis/make_figures_3_6.py      # reads results/seed42, writes figures/
   ```

2. **Attention and robustness analyses (need model + data).**
   ```bash
   python src/analysis/robustness_and_attention.py
   python src/analysis/centermask_and_gradcam_ci.py
   python src/analysis/specimen_bootstrap.py
   python src/analysis/gradcam_figure7.py
   python src/analysis/tsne_analysis.py
   ```

3. **Full retraining (optional).** Set the dataset paths at the top of
   `src/train_models.py`, then run it. For each seed it runs Stage 1 (frozen base,
   feature extraction) followed by Stage 2 (partial fine-tuning of the top layers),
   and writes the per-model CSVs found under `results/`.

All scripts are inference-only except `train_models.py`. The per-image prediction CSVs
in `results/` are enough to recompute the tables, the McNemar tests, and the bootstrap
intervals without rerunning any model.

## Citation

```bibtex
@unpublished{yurtsever2026indistinguishable,
  title   = {Transfer Learning and Model Attention on Nearly Indistinguishable Optical-Microscopy Surfaces},
  author  = {Deliaslan, A. and Yurtsever, U.},
  note    = {Manuscript under review},
  year    = {2026}
}
```

## License

Released under the MIT License. See [LICENSE](LICENSE).

## Authors

- Adem Deliaslan
- Ulaş Yurtsever (corresponding author)

Department of Computer Engineering, Sakarya University, Türkiye.

This work was supported by TÜBİTAK under the 1001 programme (project 220M024).