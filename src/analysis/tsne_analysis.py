# -*- coding: utf-8 -*-
"""
t-SNE oznitelik uzayi gorsellestirmesi  (surum 2 - ic ice model yapisina uyarlandi)
-----------------------------------------------------------------------------------
Kayitli (.h5) ince ayarli model uzerinde, YENIDEN EGITIM YAPMADAN, test
kumesinden ornekler icin softmax oncesi oznitelikleri cikarir ve t-SNE ile
iki boyuta indirger.

ONEMLI: Model onisleme ve augmentation katmanlarini ICINDE barindirir.
Goruntuler HAM (0-255) verilir, disaridan preprocess UYGULANMAZ.

Calistirma (proje kok klasorunden):
    python analiz_scriptleri/tsne_analizi.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.manifold import TSNE

# =========================================================================
# AYARLAR
# =========================================================================
DATA_DIR = "data"
SPLIT    = "test"
IMG_SIZE = 224
OUT_DIR  = "figures"

MODELS = [
    ("ResNet-50",
     "weights/seed42/ResNet50_FINETUNED_best_model.h5"),
    # ("EfficientNet-B0",
    #  "Kaynak Kodlar/Aşama 2 loglar ve grafikler/2. aşama ağırlıklarının h5 formatındaki kaydı/EfficientNetB0_FINETUNED_best_model.h5"),
]

ORNEK_PER_SINIF = 150
TSNE_PERPLEXITY = 30
RASTGELE_TOHUM  = 42
# =========================================================================


def sinif_listesi(data_dir, split):
    p = os.path.join(data_dir, split)
    return sorted([d for d in os.listdir(p)
                   if os.path.isdir(os.path.join(p, d))])


def oznitelik_modeli(model):
    """Softmax oncesi (son Dense'in girdisi) gomme temsilini veren model.
    Bunlar ust duzey katmanlar oldugu icin grafik baglantisi sorunsuzdur."""
    try:
        return tf.keras.Model(model.inputs, model.layers[-1].input)
    except Exception:
        return tf.keras.Model(model.inputs, model.layers[-2].output)


def ornekleri_topla(data_dir, split, siniflar, n_per):
    yollar, etiketler = [], []
    for idx, sinif in enumerate(siniflar):
        kdir = os.path.join(data_dir, split, sinif)
        dosyalar = sorted(os.listdir(kdir))[:n_per]
        for d in dosyalar:
            yollar.append(os.path.join(kdir, d))
            etiketler.append(idx)
    return yollar, np.array(etiketler)


def goruntu_yukle(path, size):
    img = tf.keras.utils.load_img(path, target_size=(size, size))
    return tf.keras.utils.img_to_array(img)        # 0-255 ham


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    siniflar = sinif_listesi(DATA_DIR, SPLIT)
    print("Sinif sirasi:", siniflar)
    yollar, y = ornekleri_topla(DATA_DIR, SPLIT, siniflar, ORNEK_PER_SINIF)
    print(f"Toplam {len(yollar)} ornek toplandi.")

    for etiket, yol in MODELS:
        if not os.path.exists(yol):
            print(f"[ATLA] Model bulunamadi: {yol}")
            continue
        print(f"\n=== {etiket} ===")
        model = tf.keras.models.load_model(yol, compile=False)
        fmodel = oznitelik_modeli(model)

        oz = []
        B = 32
        for i in range(0, len(yollar), B):
            grup = yollar[i:i + B]
            arr = np.stack([goruntu_yukle(p, IMG_SIZE) for p in grup])   # ham 0-255
            oz.append(fmodel.predict(arr, verbose=0))
            print(f"  {min(i + B, len(yollar))}/{len(yollar)} oznitelik cikarildi", end="\r")
        oz = np.concatenate(oz, axis=0)
        print("\nOznitelik boyutu:", oz.shape)

        tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY,
                    init="pca", random_state=RASTGELE_TOHUM)
        gomme = tsne.fit_transform(oz)

        plt.figure(figsize=(7, 6))
        renkler = plt.cm.tab10(np.linspace(0, 1, len(siniflar)))
        for idx, sinif in enumerate(siniflar):
            m = y == idx
            plt.scatter(gomme[m, 0], gomme[m, 1], s=12, color=renkler[idx],
                        label=sinif, alpha=0.7, edgecolors="none")
        plt.legend(markerscale=1.6, fontsize=9, loc="best", frameon=True)
        plt.title(f"{etiket} - feature space (t-SNE)", fontsize=12)
        plt.xticks([]); plt.yticks([])
        plt.tight_layout()

        ad = f"tsne_{etiket.replace('-', '').replace(' ', '')}"
        plt.savefig(os.path.join(OUT_DIR, ad + ".png"), dpi=200, bbox_inches="tight")
        plt.savefig(os.path.join(OUT_DIR, ad + ".pdf"), bbox_inches="tight")
        plt.close()
        print(f"kaydedildi: {ad}.png / .pdf")

    print(f"\nBitti. Ciktilar: {OUT_DIR}/")


if __name__ == "__main__":
    main()
