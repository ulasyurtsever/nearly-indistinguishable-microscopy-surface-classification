# -*- coding: utf-8 -*-
"""
NUMUNE-DUZEYI (specimen-level) BOOTSTRAP  --  hakem/editor A maddesi (EN ONEMLI)
================================================================================
Sorun: test kumesinde SINIF BASINA yalnizca 5 fiziksel numune var (D#_N# on ekleri).
Bir numuneye ait yuzlerce goruntu birbiriyle iliskilidir; dolayisiyla asil belirsizlik
kaynagi 'hangi seed' degil, 'hangi numunenin teste dustugu'dur. 3 seed'lik std bunu
YAKALAMAZ. Bu script, dogruluk icin numune-KUMELI (cluster) bootstrap %95 GA uretir.

Ne yapar:
  1) Egitimdeki ile AYNI test hatti (image_dataset_from_directory, shuffle=False) ile
     modeli calistirir; boylece raporlanan dogrulugu birebir yeniden uretir.
  2) Her goruntunun dosya adindan numune kimligini cikarir:  'D1_N1_img5.jpeg' -> 'D1_N1'
  3) predictions_with_specimen_seed42.csv kaydeder (filename, specimen, y_true, y_pred).
  4) Numune-kumeli bootstrap: her sinif icin 5 numuneyi yerine koyarak (with replacement)
     yeniden ornekler, tum goruntuleri havuzlar, GENEL dogrulugu ve SINIF-BAZLI recall'i
     hesaplar; N_BOOT kez tekrarlar; ortalama + %95 GA verir.

Not: Ayni scripti her seed modeli icin ayri ayri kosturup GA'lari karsilastirabilirsin
     (MODEL_PATH ve OUT adlarini degistir). Ideal olan ucunu de raporlamaktir.

Calistirma (proje kok klasorunden, tf ortaminda):
    python analiz_scriptleri/numune_bootstrap.py

Cikti: konsol + predictions_with_specimen_seed42.csv + specimen_bootstrap_summary.csv
"""

import os, re
import numpy as np
import tensorflow as tf

# ============================ AYARLAR ============================
DATA_DIR   = "data"
SPLIT      = "test"
IMG_SIZE   = (224, 224)
BATCH      = 32
MODEL_PATH = "weights/seed42/ResNet50_FINETUNED_best_model.h5"
OUT_DIR    = "results"
SEED_ETIKET = "seed42"          # cikti dosya adlarinda kullanilir
N_BOOT      = 2000
RASTGELE_TOHUM = 42
# ================================================================

rng = np.random.default_rng(RASTGELE_TOHUM)

def to_rgb(x, y):
    x = tf.image.grayscale_to_rgb(x) if x.shape[-1] == 1 else x
    return x, y

def specimen_of(path):
    """'.../LDPE/D1_N1_img5.jpeg' -> 'LDPE/D1_N1' (sinif + on ek)."""
    fn = os.path.basename(path)
    m = re.match(r"(.+?)_img\d+", fn)
    onek = m.group(1) if m else fn
    sinif = os.path.basename(os.path.dirname(path))
    return f"{sinif}/{onek}"

def main():
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, SPLIT), image_size=IMG_SIZE, batch_size=BATCH,
        label_mode="int", shuffle=False)
    class_names = test_ds.class_names
    file_paths = list(test_ds.file_paths)          # sira predict ile ayni (shuffle=False)
    print("Sinif sirasi:", class_names)
    print(f"Test goruntu sayisi: {len(file_paths)}")
    test_ds = test_ds.map(to_rgb).prefetch(tf.data.AUTOTUNE)

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    y_true, y_pred = [], []
    for images, labels in test_ds:
        p = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(p, axis=1)); y_true.extend(labels.numpy())
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    assert len(y_true) == len(file_paths), "sira/uzunluk uyusmuyor!"

    acc = float((y_true == y_pred).mean())
    print(f"\nGENEL dogruluk (yeniden uretim): {acc*100:.2f}%   "
          f"(raporlanan seed-42 ~%91.8 ile ayni olmali)")

    specimens = np.array([specimen_of(p) for p in file_paths])
    # ornek dogrulama:
    print("Ornek numune kimlikleri:", sorted(set(specimens))[:6], "...")

    # predictions_with_specimen kaydet
    pcsv = os.path.join(OUT_DIR, f"predictions_with_specimen_{SEED_ETIKET}.csv")
    with open(pcsv, "w") as fh:
        fh.write("filename,specimen,y_true,y_pred\n")
        for fp, sp, yt, yp in zip(file_paths, specimens, y_true, y_pred):
            fh.write(f"{os.path.basename(fp)},{sp},{int(yt)},{int(yp)}\n")
    print(f"-> {pcsv}")

    # sinif -> o sinifin numuneleri
    cls_of_spec = {sp: sp.split("/")[0] for sp in set(specimens)}
    by_class = {}
    for sp in set(specimens):
        by_class.setdefault(cls_of_spec[sp], []).append(sp)
    # numune -> indeksler
    idx_of = {sp: np.where(specimens == sp)[0] for sp in set(specimens)}

    def one_bootstrap():
        sel = []
        for c, sps in by_class.items():
            sps = np.array(sps)
            pick = sps[rng.integers(0, len(sps), len(sps))]   # 5'ten 5, yerine koyarak
            for sp in pick: sel.append(idx_of[sp])
        sel = np.concatenate(sel)
        yt, yp = y_true[sel], y_pred[sel]
        overall = (yt == yp).mean()
        rec = {}
        for ci, c in enumerate(class_names):
            mask = yt == ci
            rec[c] = float((yp[mask] == ci).mean()) if mask.sum() else np.nan
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

    print(f"\n=== Numune-kumeli bootstrap ({N_BOOT} tekrar) ===")
    m, lo, hi = ci(boots_overall)
    print(f"GENEL dogruluk : {m*100:5.2f}%  (%95 GA {lo*100:.2f} - {hi*100:.2f})")
    print("Sinif-bazli recall (numune-duzeyi GA):")
    rows = [("__overall__", m, lo, hi)]
    for c in class_names:
        mc, loc, hic = ci(boots_rec[c])
        print(f"  {c:8s}: {mc*100:5.2f}%  (%95 GA {loc*100:.2f} - {hic*100:.2f})")
        rows.append((c, mc, loc, hic))
    scsv = os.path.join(OUT_DIR, f"specimen_bootstrap_summary_{SEED_ETIKET}.csv")
    with open(scsv, "w") as fh:
        fh.write("group,mean,ci_low,ci_high\n")
        for g, mm, l, h in rows: fh.write(f"{g},{mm:.4f},{l:.4f},{h:.4f}\n")
    print(f"-> {scsv}")
    print("\nYORUM: Bu GA'lar, sinif basina yalnizca 5 numune oldugu icin seed-std'den")
    print("       belirgin GENIS cikacaktir; asil (numune-duzeyi) belirsizligi budur.")


if __name__ == "__main__":
    main()
