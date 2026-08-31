# -*- coding: utf-8 -*-
"""
finetune_hawkdove.py
=====================
mrince/CBRT-RoBERTa-HawkishDovish-Classifier'ı, "🗺️ Ton Haritası & Konular" /
annotate_app.py aracılığıyla toplanan İNSAN etiketleriyle fine-tune eder.

Bu script STREAMLIT DIŞINDA çalışır (eğitim ağır bir işlemdir; ayrı bir
makinede/GPU'da, terminalden çalıştırılması amaçlanmıştır). Yine de veri
katmanı için utils.py'yi import eder — bu yüzden ya (a) bu script'i,
.streamlit/secrets.toml dosyasının bulunduğu proje kök dizininde çalıştırın
(st.secrets, `streamlit run` olmadan da bu dosyayı okuyabilir), ya da
(b) --csv ile önceden dışa aktarılmış bir CSV kullanın (Supabase'e hiç
bağlanmadan, örn. ayrı bir GPU kutusunda çalıştırmak için).

KULLANIM
--------
# Supabase'den canlı veriyle (varsayılan temel model = mrince checkpoint):
python finetune_hawkdove.py --output_dir ./hawkdove_v3

# Önceden dışa aktarılmış bir CSV ile (Supabase bağlantısı gerekmez):
python finetune_hawkdove.py --csv annotations_export.csv --output_dir ./hawkdove_v3

# Bu ortamda ağ erişimi olmadığı için mekanik doğrulama YEREL bir modelle
# yapılmıştır — gerçek kullanımda --base_model'i varsayılanında bırakın:
python finetune_hawkdove.py --csv sample.csv --base_model ./tiny_test_model --epochs 1

NE YAPAR
--------
1. İnsan etiketlerini çeker (Supabase ya da CSV).
2. Kappa-set'teki (tüm annotator'ların ortak etiketlediği) cümleler üzerinden
   Fleiss' kappa hesaplar ve YAZDIRIR — eğitime geçmeden önce "uzmanlar gerçekten
   ne kadar hemfikir" sorusunun cevabını görürsünüz.
3. Çoklu-annotator cümlelerde çoğunluk oyuyla ALTIN etiket üretir; 3-kişilik
   berabere kalan (tam ayrışma) cümleleri eğitim dışı bırakıp ayrı raporlar
   (adjudication gerekiyor).
4. log_id'ye göre GRUPLU train/val/test ayrımı yapar (aynı PPK metninin
   cümleleri farklı setlere sızmaz — aksi halde değerlendirme iyimser çıkar).
5. Sınıf dengesizliğine karşı ağırlıklı çapraz-entropi kaybıyla, VAROLAN
   checkpoint'ten DEVAM ederek (sıfırdan değil) fine-tune eder.
6. Test setinde: accuracy, macro-F1, sınıf bazlı precision/recall, confusion
   matrix; ayrıca ESKİ (fine-tune öncesi) modelin aynı test setindeki performansı
   ile karşılaştırma (mevcut checkpoint indirilebiliyorsa).
7. Modeli ve tokenizer'ı --output_dir'e kaydeder + sonraki adımlar için
   talimat yazdırır (Hub'a push ya da yerel yoldan utils.py'ye bağlama).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd


LABELS = ["HAWK", "NEUT", "DOVE"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}


# =============================================================================
# 1. VERİ: insan etiketlerini çek (Supabase ya da CSV)
# =============================================================================

def load_annotations(csv_path: Optional[str]) -> pd.DataFrame:
    if csv_path:
        df = pd.read_csv(csv_path)
        print(f"[veri] CSV'den {len(df)} etiket okundu: {csv_path}")
    else:
        import utils
        df = utils.fetch_annotations()
        print(f"[veri] Supabase'ten {len(df)} etiket okundu.")
    need = {"log_id", "sent_idx", "sentence", "sentence_hash", "annotator", "label"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Etiket tablosunda eksik kolon(lar): {missing}")
    df["label"] = df["label"].astype(str).str.upper().str.strip()
    bad = ~df["label"].isin(LABELS)
    if bad.any():
        print(f"[uyarı] {bad.sum()} satırda geçersiz etiket bulundu, atılıyor: "
              f"{df.loc[bad, 'label'].unique().tolist()}")
        df = df[~bad].copy()
    return df


# =============================================================================
# 2. FLEISS' KAPPA — uzmanlar gerçekten ne kadar hemfikir?
# =============================================================================

def fleiss_kappa(df_ann: pd.DataFrame, min_raters: int = 3) -> dict:
    """
    Klasik Fleiss' kappa, YALNIZCA aynı sayıda değerlendiriciye sahip cümleler
    üzerinden hesaplanır (formülün geçerliliği bunu gerektirir). En sık görülen
    "rater sayısı"nı (n) otomatik seçer; en az `min_raters` olmalıdır.

    Dönüş: {"kappa": float, "n_items": int, "n_raters": int, "interpretation": str}
    ya da yeterli örtüşme yoksa {"kappa": None, ...}.
    """
    overlap = df_ann.groupby("sentence_hash")["annotator"].nunique()
    counts_of_n = overlap.value_counts()
    counts_of_n = counts_of_n[counts_of_n.index >= min_raters]
    if counts_of_n.empty:
        return {"kappa": None, "n_items": 0, "n_raters": 0,
                "note": "Hiçbir cümle en az {} annotator tarafından ortak etiketlenmemiş.".format(min_raters)}

    n = int(counts_of_n.index[counts_of_n.argmax()])  # en yaygın örtüşme sayısı
    target_hashes = overlap[overlap == n].index
    sub = df_ann[df_ann["sentence_hash"].isin(target_hashes)]

    # N x k matris: her satır bir cümle, her sütun bir kategori sayacı
    mat = (sub.groupby(["sentence_hash", "label"]).size()
           .unstack(fill_value=0).reindex(columns=LABELS, fill_value=0))
    # Bazı cümlelerde aynı annotator birden fazla kez oy vermiş olabilir (upsert
    # sayesinde normalde olmaz, ama emniyet için satır toplamını n'e zorla filtrele)
    mat = mat[mat.sum(axis=1) == n]
    N = len(mat)
    if N == 0:
        return {"kappa": None, "n_items": 0, "n_raters": n,
                "note": "Filtre sonrası örtüşen cümle kalmadı."}

    M = mat.values.astype(float)
    p_j = M.sum(axis=0) / (N * n)  # her kategorinin genel oranı
    P_i = ((M * (M - 1)).sum(axis=1)) / (n * (n - 1))
    P_bar = P_i.mean()
    P_e = (p_j ** 2).sum()
    kappa = (P_bar - P_e) / (1 - P_e) if (1 - P_e) != 0 else float("nan")

    if kappa < 0:
        interp = "hiçbir uyum yok / rastgeleden kötü"
    elif kappa < 0.20:
        interp = "hafif uyum"
    elif kappa < 0.40:
        interp = "zayıf-orta uyum"
    elif kappa < 0.60:
        interp = "orta uyum"
    elif kappa < 0.80:
        interp = "belirgin uyum"
    else:
        interp = "neredeyse tam uyum"

    return {"kappa": float(kappa), "n_items": int(N), "n_raters": int(n), "interpretation": interp}


# =============================================================================
# 3. ALTIN ETİKET ÜRETİMİ (çoğunluk oyu)
# =============================================================================

def build_gold_labels(df_ann: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Her (sentence_hash) için çoğunluk etiketi üretir.
    Dönüş: (gold_df, disagreement_df)
      gold_df: sentence_hash, log_id, sent_idx, sentence, label, n_raters, agreement
      disagreement_df: TAM ayrışma (ör. 3 kişi 3 farklı etiket) — eğitime alınmaz,
                       elle hakemlik (adjudication) için ayrı listelenir.
    """
    rows, disagree = [], []
    for h, g in df_ann.groupby("sentence_hash"):
        counts = Counter(g["label"])
        top_label, top_n = counts.most_common(1)[0]
        n_raters = len(g)
        # Berabere kalma kontrolü: en yüksek sayıyla eşit başka etiket var mı?
        tied = [l for l, c in counts.items() if c == top_n]
        rep = g.iloc[0]
        rec = {
            "sentence_hash": h, "log_id": rep["log_id"], "sent_idx": rep["sent_idx"],
            "sentence": rep["sentence"], "n_raters": n_raters,
        }
        if len(tied) > 1 and n_raters > 1:
            rec["tied_labels"] = tied
            disagree.append(rec)
        else:
            rec["label"] = top_label
            rec["agreement"] = top_n / n_raters
            rows.append(rec)
    gold = pd.DataFrame(rows)
    disagreement = pd.DataFrame(disagree)
    return gold, disagreement


# =============================================================================
# 4. GRUPLU (log_id) TRAIN/VAL/TEST AYRIMI — sızıntısız değerlendirme
# =============================================================================

def grouped_split(gold: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15,
                   seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Aynı PPK metnine ait cümleler tek bir sete (train YA DA val YA DA test)
    düşer. Bunu yapmazsak, aynı metnin bir cümlesi eğitimde bir cümlesi testte
    olabilir — model metnin genel üslubunu "ezberleyip" test skorunu yapay
    şekilde şişirebilir.
    """
    from sklearn.model_selection import GroupShuffleSplit

    groups = gold["log_id"].values
    n_groups = len(set(groups))
    # En az 3 grup gerekir (train/val/test her biri en az 1 grup alsın diye);
    # pratikte anlamlı bir bölme için çok daha fazlası gerekir. sklearn'ün ham
    # "n_samples=1, train set boş kalacak" hatası yerine buradan NET, eyleme
    # dönüştürülebilir bir mesaj vermek daha iyi — özellikle etiketlemenin
    # henüz birkaç belgeyle sınırlı olduğu erken aşamada karşılaşılır.
    if n_groups < 3:
        raise ValueError(
            f"Gruplu (log_id bazlı) train/val/test ayrımı için en az 3 farklı PPK "
            f"metninden ('belge'den) altın-etiketli cümle gerekiyor; şu an yalnızca "
            f"{n_groups} farklı belgeden geliyor. Daha fazla belgeyi etikettirin ya da "
            f"--skip_training ile önce yalnızca kappa/veri raporunu inceleyin."
        )
    if n_groups < 8:
        print(f"[UYARI] Yalnızca {n_groups} farklı belgeden altın-etiket var. Gruplu "
              f"bölme yine de çalışacak, ama val/test setleri çok küçük ve tek bir "
              f"belgenin cümlelerinden oluşabilir — sonuçları temkinli okuyun.")
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(gold, groups=groups))
    trainval, test = gold.iloc[trainval_idx], gold.iloc[test_idx]

    rel_val = val_size / (1 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed)
    tr_idx, val_idx = next(gss2.split(trainval, groups=trainval["log_id"].values))
    train, val = trainval.iloc[tr_idx], trainval.iloc[val_idx]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# =============================================================================
# 5. EĞİTİM
# =============================================================================

def run_training(train, val, test, base_model: str, output_dir: str,
                  epochs: float, lr: float, batch_size: int, seed: int = 42):
    import torch
    from torch import nn
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                               Trainer, TrainingArguments, set_seed)
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

    set_seed(seed)

    print(f"[model] Temel checkpoint yükleniyor: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    # ÖNEMLİ: id2label/label2id'i burada KENDİMİZ tanımlıyoruz (HAWK=0, NEUT=1,
    # DOVE=2). mrince checkpoint'inin ORİJİNAL etiket sırası farklı/belirsiz
    # olabilir (utils.py'de bu yüzden _mrince_label_map ile çalışma-anında
    # otomatik tespit ediliyor) — ama biz sınıflandırma katmanını bu etiketlerle
    # YENİDEN eğittiğimiz için katman, eğitim sonunda bizim sıramıza göre
    # kalibre olur. Fine-tune edilmiş modeli utils.py'ye bağlarken artık o
    # otomatik-tespit adımına gerek YOKTUR — id2label doğrudan güvenilir olur.

    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, padding="max_length", max_length=128)

    def to_ds(df):
        d = df.copy()
        d["labels"] = d["label"].map(LABEL2ID)
        ds = Dataset.from_pandas(d[["sentence", "labels"]], preserve_index=False)
        ds = ds.map(tokenize, batched=True)
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return ds

    train_ds, val_ds, test_ds = to_ds(train), to_ds(val), to_ds(test)

    # --- sınıf ağırlıkları (dengesizliğe karşı) -----------------------------
    counts = train["label"].value_counts().reindex(LABELS).fillna(0)
    print(f"[sınıf dağılımı - eğitim] {counts.to_dict()}")
    freq = counts.values.astype(float)
    freq[freq == 0] = 1.0  # sıfıra bölme koruması
    weights = (freq.sum() / (len(LABELS) * freq))
    class_weights = torch.tensor(weights, dtype=torch.float32)
    print(f"[sınıf ağırlıkları] {dict(zip(LABELS, weights.round(3)))}")

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, labels=[0, 1, 2], zero_division=0)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
            "f1_hawk": f1[0], "f1_neut": f1[1], "f1_dove": f1[2],
        }

    # warmup_ratio bazı transformers sürümlerinde yok/değişken; tüm sürümlerde
    # var olan warmup_steps'e elle çeviriyoruz (toplam adımın ~%6'sı — standart
    # bir varsayılan) ki script farklı transformers sürümlerinde de çalışsın.
    steps_per_epoch = max(1, len(train_ds) // max(1, batch_size))
    total_steps = int(steps_per_epoch * epochs)
    warmup_steps = max(0, int(total_steps * 0.06))

    args = TrainingArguments(
        output_dir=os.path.join(output_dir, "_checkpoints"),
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=10,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        report_to=[],
        seed=seed,
    )

    trainer = WeightedTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print("\n[eğitim] başlıyor...")
    trainer.train()

    print("\n[test] değerlendirme...")
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    preds = np.argmax(trainer.predict(test_ds).predictions, axis=-1)
    labels_true = test["label"].map(LABEL2ID).values
    cm = confusion_matrix(labels_true, preds, labels=[0, 1, 2])

    print(f"\n[test sonuçları] {test_metrics}")
    print(f"[confusion matrix] satır=gerçek, sütun=tahmin, sıra={LABELS}\n{cm}")

    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    report = {
        "base_model": base_model, "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "confusion_matrix": cm.tolist(), "labels_order": LABELS,
    }
    with open(os.path.join(output_dir, "training_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


# =============================================================================
# 6. ANA AKIŞ
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None, help="Supabase yerine bu CSV'den etiketleri oku (kolonlar: log_id, sent_idx, sentence, sentence_hash, annotator, label[, confidence])")
    ap.add_argument("--base_model", default="mrince/CBRT-RoBERTa-HawkishDovish-Classifier",
                     help="Devam edilecek checkpoint (Hub ID ya da yerel klasör).")
    ap.add_argument("--output_dir", default="./hawkdove_finetuned")
    ap.add_argument("--epochs", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--test_size", type=float, default=0.15)
    ap.add_argument("--val_size", type=float, default=0.15)
    ap.add_argument("--min_confidence", type=int, default=None,
                     help="Belirtilirse, bu değerin altındaki güvenle etiketlenen satırlar eğitim dışı bırakılır (1-3).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_training", action="store_true",
                     help="Yalnızca kappa + veri özeti raporla, eğitimi ÇALIŞTIRMA (etiketleme aşamasını izlemek için).")
    args = ap.parse_args()

    df_ann = load_annotations(args.csv)
    if df_ann.empty:
        print("[hata] Hiç etiket bulunamadı. Önce annotate_app.py ile etiketleme yapın.")
        sys.exit(1)

    if args.min_confidence is not None and "confidence" in df_ann.columns:
        before = len(df_ann)
        df_ann = df_ann[df_ann["confidence"].fillna(3) >= args.min_confidence]
        print(f"[filtre] min_confidence={args.min_confidence}: {before} -> {len(df_ann)} satır")

    print(f"\n[özet] Toplam etiket: {len(df_ann)}  ·  Annotator sayısı: {df_ann['annotator'].nunique()}  "
          f"·  Benzersiz cümle: {df_ann['sentence_hash'].nunique()}")
    print(f"[özet] Annotator başına etiket sayısı:\n{df_ann.groupby('annotator').size()}")

    kappa = fleiss_kappa(df_ann, min_raters=3)
    print(f"\n[Fleiss' kappa] {kappa}")
    if kappa.get("kappa") is not None and kappa["kappa"] < 0.40:
        print("[UYARI] Kappa 0.40'ın altında (zayıf-orta uyum). Etiketleme rehberini "
              "gözden geçirmeden modele güvenmek riskli olabilir — önce anlaşmazlık "
              "örneklerini annotator'larla birlikte inceleyin.")

    gold, disagreement = build_gold_labels(df_ann)
    print(f"\n[altın etiket] {len(gold)} cümle için çoğunluk etiketi üretildi; "
          f"{len(disagreement)} cümle TAM ayrışma nedeniyle dışarıda bırakıldı "
          f"(bkz. training_report.json yanındaki disagreements.csv).")
    print(f"[etiket dağılımı]\n{gold['label'].value_counts()}")

    os.makedirs(args.output_dir, exist_ok=True)
    if not disagreement.empty:
        disagreement.to_csv(os.path.join(args.output_dir, "disagreements.csv"), index=False)

    if args.skip_training:
        print("\n[--skip_training] Eğitim atlandı; yalnızca veri raporu üretildi.")
        return

    train, val, test = grouped_split(gold, test_size=args.test_size, val_size=args.val_size, seed=args.seed)
    print(f"\n[bölme] train={len(train)}  val={len(val)}  test={len(test)}  "
          f"(log_id bazında gruplu — aynı metnin cümleleri tek sette)")
    if len(train) < 20 or len(test) < 5:
        print("[UYARI] Eğitim/test kümesi çok küçük; sonuçlar güvenilir olmayabilir. "
              "En az birkaç yüz etiketli cümle önerilir (bkz. sohbetteki plan notu).")

    report = run_training(train, val, test, args.base_model, args.output_dir,
                          args.epochs, args.lr, args.batch_size, args.seed)

    print(f"\n[bitti] Model + tokenizer kaydedildi: {args.output_dir}")
    print(
        "\nSONRAKİ ADIMLAR\n"
        "----------------\n"
        f"1) Bu modeli önce ESKİ modelle (mrince/CBRT-...) aynı test setinde karşılaştırın "
        f"(training_report.json içindeki test_metrics'i, aynı test cümleleriyle eski model "
        f"üzerinde de çalıştırıp kıyaslayın).\n"
        f"2) Memnunsanız: ya `huggingface_hub.upload_folder('{args.output_dir}', repo_id=...)` "
        f"ile özel bir Hub deposuna yükleyin, ya da bu klasörü olduğu gibi sunucuya taşıyın.\n"
        f"3) utils.py içinde MODEL_HD = '<hub-repo-id-veya-yerel-yol>' olarak güncelleyin ve "
        f"PIPELINE_VERSION = 'v3' yapın (COMPATIBLE_VERSIONS'a da 'v3' ekleyin) — bu, "
        f"diagnose()'un eski önbelleği otomatik 'surum_eskidi' işaretlemesini sağlar.\n"
        f"4) «🗺️ Ton Haritası & Konular» sekmesinden 'Eksik/bayat kaydı hesapla'yı çalıştırıp "
        f"tüm geçmişi yeni modelle yeniden tarayın.\n"
    )


if __name__ == "__main__":
    main()
