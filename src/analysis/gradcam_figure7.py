# -*- coding: utf-8 -*-
"""
Figure 7 - derli toplu Grad-CAM bilesik figuru
----------------------------------------------
ResNet-50 ince ayarli model uzerinde, secili temsili ornekler icin tek bir
yayina hazir Grad-CAM figuru uretir (Ingilizce etiketli).

Satirlar (5):
  1) Dogru siniflanan bir LDPE ornegi
  2) LDPE olarak yanlis siniflanan bir uv-PE ornegi (karisikligi gosterir)
  3) Dogru siniflanan bir oxo-PE ornegi (ayni aile, ama ayrisiyor)
  4) Dogru siniflanan bir PVC ornegi (ayirt edici sinif)
  5) Dogru siniflanan bir PP ornegi (ayirt edici sinif)

Calistirma (proje kok klasorunden):
    python analiz_scriptleri/gradcam_figure7.py

Cikti: figure7.png ve figure7.pdf (calisilan klasorde)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

DATA_DIR = "data"
SPLIT    = "test"
IMG_SIZE = 224
MODEL_YOLU = "weights/seed42/ResNet50_FINETUNED_best_model.h5"
TARAMA_LIMITI = 80   # her sinifta en fazla kac ornek taransin

# Aranacak ornekler: (gercek sinif, istenen tahmin, satir etiketi)
SECIMLER = [
    ("LDPE",   "LDPE",   "LDPE  ->  LDPE  (correct)"),
    ("uv-PE",  "LDPE",   "uv-PE  ->  LDPE  (error)"),
    ("oxo-PE", "oxo-PE", "oxo-PE  ->  oxo-PE  (correct)"),
    ("PVC",    "PVC",    "PVC  ->  PVC  (correct)"),
    ("PP",     "PP",     "PP  ->  PP  (correct)"),
]


def sinif_listesi():
    p = os.path.join(DATA_DIR, SPLIT)
    return sorted([d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))])


def goruntu_yukle(path):
    img = tf.keras.utils.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
    return tf.keras.utils.img_to_array(img)


def govde_bul(model):
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.Model):
            if any(isinstance(l, tf.keras.layers.Conv2D) for l in layer.layers):
                return i, layer
    return None, None


def son_4b(govde):
    for layer in reversed(govde.layers):
        try:
            shp = layer.output_shape
        except Exception:
            continue
        if isinstance(shp, tuple) and len(shp) == 4:
            return layer
    return None


def gradcam_kur(model):
    idx, govde = govde_bul(model)
    son_conv = son_4b(govde)
    grad_govde = tf.keras.Model(govde.input, [son_conv.output, govde.output])
    return grad_govde, model.layers[:idx], model.layers[idx + 1:]


def isi_haritasi(raw, grad_govde, onceki, sonraki):
    x = raw
    for L in onceki:
        x = L(x, training=False)
    with tf.GradientTape() as tape:
        conv_out, govde_out = grad_govde(x)
        tape.watch(conv_out)
        y = govde_out
        for L in sonraki:
            y = L(y, training=False)
        sinif = tf.argmax(y[0])
        skor = y[:, sinif]
    grads = tape.gradient(skor, conv_out)
    agirlik = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(conv_out[0] * agirlik, axis=-1)
    cam = tf.maximum(cam, 0)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], (IMG_SIZE, IMG_SIZE))[..., 0]
    return cam.numpy(), int(sinif.numpy())


def bindir(ham, cam):
    taban = ham / 255.0
    renkli = plt.get_cmap("jet")(cam)[..., :3]
    return np.clip(0.55 * taban + 0.45 * renkli, 0, 1)


def ornek_bul(gercek, istenen, siniflar, grad_govde, onceki, sonraki):
    kdir = os.path.join(DATA_DIR, SPLIT, gercek)
    for dosya in sorted(os.listdir(kdir))[:TARAMA_LIMITI]:
        ham = goruntu_yukle(os.path.join(kdir, dosya))
        cam, idx = isi_haritasi(np.expand_dims(ham.copy(), 0), grad_govde, onceki, sonraki)
        if siniflar[idx] == istenen:
            return ham, cam
    # bulunamazsa ilk ornegi dondur
    ham = goruntu_yukle(os.path.join(kdir, sorted(os.listdir(kdir))[0]))
    cam, idx = isi_haritasi(np.expand_dims(ham.copy(), 0), grad_govde, onceki, sonraki)
    print(f"  UYARI: {gercek}->{istenen} ornegi bulunamadi, ilk ornek kullanildi (tahmin: {siniflar[idx]})")
    return ham, cam


def main():
    siniflar = sinif_listesi()
    print("Sinif sirasi:", siniflar)
    model = tf.keras.models.load_model(MODEL_YOLU, compile=False)
    grad_govde, onceki, sonraki = gradcam_kur(model)

    n = len(SECIMLER)
    fig, ax = plt.subplots(n, 2, figsize=(4.6, 1.85 * n))
    for r, (gercek, istenen, etiket) in enumerate(SECIMLER):
        ham, cam = ornek_bul(gercek, istenen, siniflar, grad_govde, onceki, sonraki)
        ax[r, 0].imshow(ham.astype("uint8")); ax[r, 0].axis("off")
        ax[r, 1].imshow(bindir(ham, cam));    ax[r, 1].axis("off")
        ax[r, 0].set_ylabel(etiket, fontsize=9)
        # sol panele satir etiketi (title yerine sol ust)
        ax[r, 0].set_title(etiket, fontsize=9, loc="left")
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
    print("kaydedildi: figure7.png / figure7.pdf")


if __name__ == "__main__":
    main()
