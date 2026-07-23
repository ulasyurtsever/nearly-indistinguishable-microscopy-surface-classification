# -*- coding: utf-8 -*-
"""
EK ANALIZ (yeniden egitim YOK, sadece inference) -- hakem/editor D ve Grad-CAM maddeleri
========================================================================================
Bu script mevcut robustness_and_attention_quant.py ile AYNI yukleme/onisleme mantigini
kullanir, boylece uretilen sayilar makaledeki Tablo 8 ile dogrudan karsilastirilabilir.

Ekledigi iki sey:

  (1) ESIT-ALAN MERKEZ MASKESI kontrolu (hakem D):
      Kose/vinyet maskesi ile TAM AYNI sayida pikseli, ama goruntunun ORTASINDA
      maskeler. Amac: kose maskesindeki buyuk dusus 'kenara bagimlilik' mi yoksa
      sadece 'goruntunun ~%21'ini gizlemek' mi -- bunu ayirt etmek. Merkez maskesi de
      benzer dusus verirse, sorun kenar degil, kaybedilen ALAN/dagilim-disiligidir.

  (2) Grad-CAM kenar-kutle metrigine GUVEN ARALIGI + DUZGUN-DIKKAT NULL karsilastirmasi
      (hakem 3): kenar orani icin bootstrap %95 GA hesaplar ve bunu, dikkat tamamen
      duzgun (uniform) olsaydi beklenen deger olan kenar seridinin ALAN oraniyla
      karsilastirir. GA ust siniri alan oraninin altindaysa, dikkat kenardan
      istatistiksel olarak KACINIYOR demektir (kestirme/shortcut lehine degil).

Calistirma (proje kok klasorunden, tf ortaminda):
    python analiz_scriptleri/ek_analiz_centermask_gradcamCI.py

Cikti: konsol ozetleri + robustness_centermask_summary.csv, attention_border_CI.csv
"""

import os
import numpy as np
import tensorflow as tf

# ============================ AYARLAR (kendi yollarina gore duzenle) ============================
DATA_DIR   = "data"                                    # icinde test/<sinif>/*.jpeg
SPLIT      = "test"
IMG_SIZE   = 224
MODEL_PATH = "weights/seed42/ResNet50_FINETUNED_best_model.h5"
OUT_DIR    = "results"                                                 # ciktilarin yazilacagi klasor
BATCH      = 32
CAM_PER_SINIF = 80        # Grad-CAM'de sinif basina goruntu (mevcut analizle ayni)
BORDER_FRAC   = 0.18      # kenar seridi genisligi (mevcut analizle ayni -> alan orani ~%58.7)
N_BOOT        = 2000      # bootstrap tekrar sayisi
RASTGELE_TOHUM = 42
# ================================================================================================

rng = np.random.default_rng(RASTGELE_TOHUM)


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


# ------------------- maskeler -------------------
def _vignette_mask():
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    c = (IMG_SIZE - 1) / 2.0
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    return r > (IMG_SIZE / 2.0)      # koseler

_VMASK = _vignette_mask()

def _center_mask_equal_area(n_pixels):
    """Kose maskesiyle AYNI sayida pikseli, merkezde bir kare olarak maskele."""
    side = int(round(np.sqrt(n_pixels)))
    b = (IMG_SIZE - side) // 2
    m = np.zeros((IMG_SIZE, IMG_SIZE), bool)
    m[b:b + side, b:b + side] = True
    return m

_CMASK = _center_mask_equal_area(int(_VMASK.sum()))

def bozulma_yok(x):
    return x

def kose_maskele(x):
    out = x.copy(); out[_VMASK] = float(out.mean()); return out

def merkez_maskele(x):
    out = x.copy(); out[_CMASK] = float(out.mean()); return out


BOZULMALAR = [
    ("Baseline (none)",                bozulma_yok),
    ("Corner/vignette mask",           kose_maskele),
    ("Center mask (equal area)",       merkez_maskele),
]

def dogruluk_hesapla(model, yollar, etiketler, boz):
    dogru = 0; buf_x, buf_y = [], []
    def bosalt():
        nonlocal dogru
        if not buf_x: return
        pr = model.predict(np.stack(buf_x, 0), verbose=0)
        dogru += int(np.sum(np.argmax(pr, axis=1) == np.array(buf_y)))
        buf_x.clear(); buf_y.clear()
    for p, y in zip(yollar, etiketler):
        buf_x.append(boz(goruntu_yukle(p))); buf_y.append(y)
        if len(buf_x) >= BATCH: bosalt()
    bosalt()
    return dogru / len(yollar)


# ------------------- Grad-CAM (nested base destegi) -------------------
def govde_bul(model):
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.Model) and any(isinstance(l, tf.keras.layers.Conv2D) for l in layer.layers):
            return i, layer
    return None, None

def son_4b(govde):
    for layer in reversed(govde.layers):
        try: shp = layer.output_shape
        except Exception: continue
        if isinstance(shp, tuple) and len(shp) == 4: return layer
    return None

def gradcam_kur(model):
    idx, govde = govde_bul(model)
    son_conv = son_4b(govde)
    grad_govde = tf.keras.Model(govde.input, [son_conv.output, govde.output])
    return grad_govde, model.layers[:idx], model.layers[idx + 1:]

def isi_haritasi(raw, grad_govde, onceki, sonraki):
    x = raw
    for L in onceki: x = L(x, training=False)
    with tf.GradientTape() as tape:
        conv_out, govde_out = grad_govde(x); tape.watch(conv_out)
        y = govde_out
        for L in sonraki: y = L(y, training=False)
        sinif = tf.argmax(y[0]); skor = y[:, sinif]
    grads = tape.gradient(skor, conv_out)
    agirlik = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(conv_out[0] * agirlik, axis=-1)
    cam = tf.maximum(cam, 0); cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], (IMG_SIZE, IMG_SIZE))[..., 0]
    return cam.numpy(), int(sinif.numpy())

def _border_mask():
    b = int(IMG_SIZE * BORDER_FRAC)
    m = np.zeros((IMG_SIZE, IMG_SIZE), bool)
    m[:b, :] = True; m[-b:, :] = True; m[:, :b] = True; m[:, -b:] = True
    return m


def main():
    siniflar = sinif_listesi()
    print("Sinif sirasi:", siniflar)
    print(f"Kose maskesi alan orani : {_VMASK.mean()*100:.1f}%  ({int(_VMASK.sum())} piksel)")
    print(f"Merkez maskesi alan orani: {_CMASK.mean()*100:.1f}%  ({int(_CMASK.sum())} piksel)  <- esit alan")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    yollar, etiketler = tum_yollar(siniflar)
    print(f"Test goruntu sayisi: {len(yollar)}")
    os.makedirs(OUT_DIR, exist_ok=True)

    # (1) merkez vs kose maske
    print("\n=== (1) Kose maskesi vs esit-alan MERKEZ maskesi ===")
    satir = []; taban = None
    for ad, fn in BOZULMALAR:
        acc = dogruluk_hesapla(model, yollar, etiketler, fn)
        if taban is None: taban = acc
        d = acc - taban; satir.append((ad, acc, d))
        print(f"  {ad:28s} acc={acc*100:6.2f}%  delta={d*100:+.2f} pt")
    with open(os.path.join(OUT_DIR, "robustness_centermask_summary.csv"), "w") as fh:
        fh.write("perturbation,accuracy,delta_vs_baseline\n")
        for ad, acc, d in satir: fh.write(f"{ad},{acc:.4f},{d:.4f}\n")
    print("  -> robustness_centermask_summary.csv")
    print("  YORUM: merkez maskesinin dususu koseninkine yakinsa, kayip 'kenar' degil,")
    print("         gizlenen ALAN / dagitim-disiligi kaynaklidir (kestirme lehine kanit degil).")

    # (2) Grad-CAM kenar orani + bootstrap GA + uniform null
    print("\n=== (2) Grad-CAM kenar-kutle orani: bootstrap %95 GA + uniform-null ===")
    grad_govde, onceki, sonraki = gradcam_kur(model)
    bmask = _border_mask(); alan_ref = bmask.mean()
    fr_all, fr_pe = [], []
    for i, s in enumerate(siniflar):
        kd = os.path.join(DATA_DIR, SPLIT, s)
        dosyalar = sorted(f for f in os.listdir(kd) if f.lower().endswith((".jpg",".jpeg",".png")))[:CAM_PER_SINIF]
        for f in dosyalar:
            ham = goruntu_yukle(os.path.join(kd, f))
            cam, _ = isi_haritasi(np.expand_dims(ham.copy(), 0), grad_govde, onceki, sonraki)
            fr = float(cam[bmask].sum() / (cam.sum() + 1e-8))
            fr_all.append(fr)
            if s in ("LDPE", "uv-PE"): fr_pe.append(fr)

    def boot_ci(vals):
        vals = np.asarray(vals); n = len(vals)
        means = np.array([vals[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
        return vals.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)

    m, lo, hi = boot_ci(fr_all)
    mpe, lope, hipe = boot_ci(fr_pe)
    print(f"  Kenar seridi ALAN orani (uniform-null)     : {alan_ref*100:5.2f}%")
    print(f"  CAM kenar orani  - tum     : {m*100:5.2f}%  (%95 GA {lo*100:.2f}-{hi*100:.2f})  n={len(fr_all)}")
    print(f"  CAM kenar orani  - LDPE+uvPE: {mpe*100:5.2f}%  (%95 GA {lope*100:.2f}-{hipe*100:.2f})  n={len(fr_pe)}")
    karar = "KENARDAN KACINIYOR (GA ust sinir < alan orani)" if hi < alan_ref else "belirsiz"
    print(f"  Uniform-null testi: {karar}")
    with open(os.path.join(OUT_DIR, "attention_border_CI.csv"), "w") as fh:
        fh.write("group,mean,ci_low,ci_high,uniform_null_area,n\n")
        fh.write(f"all,{m:.4f},{lo:.4f},{hi:.4f},{alan_ref:.4f},{len(fr_all)}\n")
        fh.write(f"LDPE_uvPE,{mpe:.4f},{lope:.4f},{hipe:.4f},{alan_ref:.4f},{len(fr_pe)}\n")
    print("  -> attention_border_CI.csv")


if __name__ == "__main__":
    main()
