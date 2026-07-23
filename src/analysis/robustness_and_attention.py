# -*- coding: utf-8 -*-
"""
Niceliksel shortcut / dayaniklilik analizi  (YENIDEN EGITIM YOK, sadece inference)
---------------------------------------------------------------------------------
Editorun istedigi iki seyi uretir:

  (A) Test-zamani bozulmalar altinda dogruluk degisimi:
        - kose/vinyet maskeleme
        - aydinlatma (parlaklik) normalizasyonu
        - kontrast azaltma
        - merkez kirpma (kenar/vinyet atma)
      Model bu bozulmalara dayanikli ise, karar yuzeyin kendisine dayaniyor demektir
      (edinim/vinyet kestirmesi degil).

  (B) Grad-CAM kutlesinin kenar/vinyet bolgesindeki orani:
      Her test goruntusu icin Grad-CAM isi haritasinin ne kadari kenar seridine
      dusuyor. Bu oran, kenarin alan orandan belirgin kucukse, dikkat kenara
      kilitlenmiyor demektir. Dogru/yanlis ve LDPE/uv-PE alt kumesi ayri raporlanir.

Calistirma (proje kok klasorunden):
    python robustness_and_attention_quant.py

Cikti: konsola ozet tablolar + robustness_summary.csv, attention_border_summary.csv
"""

import os
import numpy as np
import tensorflow as tf

# ------------------------- AYARLAR -------------------------
DATA_DIR   = "data"   # sunucuda: ~/Desktop/Adem altinda   # icinde test/<sinif>/*.jpeg olan kok
SPLIT      = "test"
IMG_SIZE   = 224
MODEL_PATH = "weights/seed42/ResNet50_FINETUNED_best_model.h5"   # yeni A6000 seed-42 FT modeli

BATCH      = 32
CAM_PER_SINIF = 80           # Grad-CAM analizinde sinif basina kac goruntu (hiz icin)
BORDER_FRAC   = 0.18         # kenar seridi genisligi (goruntu boyutunun orani)
RASTGELE_TOHUM = 42
# -----------------------------------------------------------

np.random.seed(RASTGELE_TOHUM)


def sinif_listesi():
    p = os.path.join(DATA_DIR, SPLIT)
    return sorted(d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d)))


def goruntu_yukle(path):
    img = tf.keras.utils.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
    return tf.keras.utils.img_to_array(img)          # 0..255 float32


def tum_yollar(siniflar):
    yollar, etiketler = [], []
    for i, s in enumerate(siniflar):
        kd = os.path.join(DATA_DIR, SPLIT, s)
        for f in sorted(os.listdir(kd)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                yollar.append(os.path.join(kd, f)); etiketler.append(i)
    return yollar, np.array(etiketler)


# ------------------------- BOZULMALAR (RAW 0..255 uzerinde) -------------------------
def _vignette_mask():
    """Merkeze gore dairesel maske: disi (koseler) True."""
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    cx = cy = (IMG_SIZE - 1) / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return r > (IMG_SIZE / 2.0)      # kose bolgeleri


_VMASK = _vignette_mask()


def bozulma_yok(x):
    return x


def kose_maskele(x):
    """Kose/vinyet bolgesini goruntunun ortalama degeriyle doldur (bilgi sil)."""
    out = x.copy()
    m = float(out.mean())
    out[_VMASK] = m
    return out


def parlaklik_normalize(x, hedef=128.0):
    """Global parlaligi sabitle: ortalama parlakligi hedefe tasi."""
    out = x.astype(np.float32)
    out = out - out.mean() + hedef
    return np.clip(out, 0, 255)


def kontrast_azalt(x, faktor=0.5):
    m = float(x.mean())
    return np.clip(m + faktor * (x - m), 0, 255)


def merkez_kirp(x, kirp=0.80):
    """Merkezden %kirp'lik kare al, 224'e geri buyut (kenar/vinyet atilir)."""
    k = int(IMG_SIZE * kirp)
    b = (IMG_SIZE - k) // 2
    crop = x[b:b + k, b:b + k, :]
    crop = tf.image.resize(crop, (IMG_SIZE, IMG_SIZE)).numpy()
    return np.clip(crop, 0, 255)


BOZULMALAR = [
    ("Baseline (no perturbation)", bozulma_yok),
    ("Corner/vignette masking",    kose_maskele),
    ("Illumination normalization", parlaklik_normalize),
    ("Contrast reduction (0.5x)",  kontrast_azalt),
    ("Center crop (80%)",          merkez_kirp),
]


# ------------------------- (A) DAYANIKLILIK -------------------------
def dogruluk_hesapla(model, yollar, etiketler, boz):
    dogru = 0
    buf_x, buf_y = [], []

    def bosalt():
        nonlocal dogru
        if not buf_x:
            return
        arr = np.stack(buf_x, 0)
        pr = model.predict(arr, verbose=0)
        tah = np.argmax(pr, axis=1)
        dogru += int(np.sum(tah == np.array(buf_y)))
        buf_x.clear(); buf_y.clear()

    for p, y in zip(yollar, etiketler):
        x = goruntu_yukle(p)
        buf_x.append(boz(x)); buf_y.append(y)
        if len(buf_x) >= BATCH:
            bosalt()
    bosalt()
    return dogru / len(yollar)


# ------------------------- (B) GRAD-CAM (nested base model destegi) -------------------------
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


def _border_mask():
    b = int(IMG_SIZE * BORDER_FRAC)
    m = np.zeros((IMG_SIZE, IMG_SIZE), bool)
    m[:b, :] = True; m[-b:, :] = True; m[:, :b] = True; m[:, -b:] = True
    return m


def attention_kenar_orani(model, siniflar):
    grad_govde, onceki, sonraki = gradcam_kur(model)
    bmask = _border_mask()
    alan_orani = bmask.mean()

    kayitlar = []   # (sinif_adi, dogru_mu, kenar_orani)
    for i, s in enumerate(siniflar):
        kd = os.path.join(DATA_DIR, SPLIT, s)
        dosyalar = sorted(f for f in os.listdir(kd)
                          if f.lower().endswith((".jpg", ".jpeg", ".png")))[:CAM_PER_SINIF]
        for f in dosyalar:
            ham = goruntu_yukle(os.path.join(kd, f))
            cam, tah = isi_haritasi(np.expand_dims(ham.copy(), 0), grad_govde, onceki, sonraki)
            toplam = cam.sum() + 1e-8
            kenar = cam[bmask].sum() / toplam
            kayitlar.append((s, tah == i, float(kenar)))
    return kayitlar, alan_orani


# ------------------------- ANA -------------------------
def main():
    siniflar = sinif_listesi()
    print("Sinif sirasi:", siniflar)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    yollar, etiketler = tum_yollar(siniflar)
    print(f"Test goruntu sayisi: {len(yollar)}")

    # (A) dayaniklilik
    print("\n=== (A) Test-zamani bozulmalar altinda dogruluk ===")
    satirlar = []
    taban = None
    for ad, fn in BOZULMALAR:
        acc = dogruluk_hesapla(model, yollar, etiketler, fn)
        if taban is None:
            taban = acc
        delta = acc - taban
        satirlar.append((ad, acc, delta))
        print(f"  {ad:32s}  acc={acc*100:6.2f}%   delta={delta*100:+.2f} pt")
    with open("robustness_summary.csv", "w") as fh:
        fh.write("perturbation,accuracy,delta_vs_baseline\n")
        for ad, acc, d in satirlar:
            fh.write(f"{ad},{acc:.4f},{d:.4f}\n")
    print("  -> robustness_summary.csv yazildi")

    # (B) attention kenar orani
    print("\n=== (B) Grad-CAM kutlesinin kenar bolgesindeki orani ===")
    kayitlar, alan_orani = attention_kenar_orani(model, siniflar)
    arr = np.array([k[2] for k in kayitlar])
    dogru = np.array([k[1] for k in kayitlar])
    pe = np.array([k[0] in ("LDPE", "uv-PE") for k in kayitlar])
    print(f"  Kenar seridinin ALAN orani (referans)      : {alan_orani*100:5.1f}%")
    print(f"  CAM kenar orani  - tum ornekler            : {arr.mean()*100:5.1f}%  (std {arr.std()*100:.1f})")
    print(f"  CAM kenar orani  - dogru siniflananlar     : {arr[dogru].mean()*100:5.1f}%")
    print(f"  CAM kenar orani  - yanlis siniflananlar    : {arr[~dogru].mean()*100:5.1f}%")
    print(f"  CAM kenar orani  - LDPE + uv-PE alt kumesi : {arr[pe].mean()*100:5.1f}%")
    print("  (CAM kenar orani, ALAN oranindan belirgin kucukse dikkat kenara kilitlenmiyor.)")
    with open("attention_border_summary.csv", "w") as fh:
        fh.write("group,mean_cam_border_fraction,n\n")
        fh.write(f"border_area_reference,{alan_orani:.4f},0\n")
        fh.write(f"all,{arr.mean():.4f},{len(arr)}\n")
        fh.write(f"correct,{arr[dogru].mean():.4f},{int(dogru.sum())}\n")
        fh.write(f"incorrect,{arr[~dogru].mean():.4f},{int((~dogru).sum())}\n")
        fh.write(f"LDPE_uvPE,{arr[pe].mean():.4f},{int(pe.sum())}\n")
    print("  -> attention_border_summary.csv yazildi")


if __name__ == "__main__":
    main()
