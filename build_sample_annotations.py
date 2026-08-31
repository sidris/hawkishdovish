# -*- coding: utf-8 -*-
"""Fine-tuning script'ini (finetune_hawkdove.py) gerçek Supabase/HF Hub erişimi
olmadan uçtan uca test etmek için sentetik ama şema-doğru bir etiket CSV'si
üretir. --base_model olarak tiny_test_model/ (yerel, küçük, offline) kullanılır;
gerçek kullanımda varsayılan mrince checkpoint'i kullanılacaktır.

DİKKAT (v2): İlk sürüm yalnızca 10 cümleyi belgeler arasında permüte ediyordu;
bu, sentence_hash'in belgeler arasında ÇAKIŞMASINA ve grouped_split'in tek bir
gruba düşmesine yol açtı (gerçek bir sklearn hatasını ortaya çıkardı — bkz.
finetune_hawkdove.py'deki n_groups<3 kontrolü). Bu sürüm, build_sample_report.py
ile AYNI, GERÇEKTEN FARKLI 9 PPK metnini kullanır — gerçek üretim verisine daha
yakın (belgeler arasında neredeyse hiç birebir aynı cümle yok)."""
import hashlib

import numpy as np
import pandas as pd

rng = np.random.default_rng(11)
ANNOTATORS = ["ayse", "berk", "cem", "deniz", "elif"]


def sent_hash(s):
    return hashlib.sha256(" ".join(str(s).lower().split()).encode()).hexdigest()[:32]


# 9 GERÇEKTEN FARKLI belge (build_sample_report.py'deki PERIODS + hedef dönemle aynı) ---
DOCS = {
    1: [
        ("The Committee has decided to keep the policy rate at 50 percent.", "HAWK"),
        ("The Committee assessed that the current tight monetary stance needs to be maintained decisively.", "HAWK"),
        ("The Committee will continue to make its decisions based on the inflation outlook.", "NEUT"),
        ("Inflation remains elevated and the Committee will tighten the stance further if needed.", "HAWK"),
    ],
    2: [
        ("The Monetary Policy Committee has decided to reduce the policy rate from 50 percent to 47.5 percent.", "DOVE"),
        ("Recent data indicate that the underlying trend of inflation has started to slow.", "DOVE"),
        ("The Committee will determine the pace of the disinflation process meeting by meeting.", "NEUT"),
        ("The Committee reiterated that it remains attentive to upside risks on inflation.", "HAWK"),
    ],
    3: [
        ("The Committee has decided to lower the policy rate from 47.5 percent to 46 percent.", "DOVE"),
        ("Domestic demand continues to weaken gradually.", "DOVE"),
        ("The underlying trend of inflation declined further in February.", "DOVE"),
        ("Should upside risks to the inflation outlook materialize, the Committee stands ready to tighten again.", "HAWK"),
    ],
    4: [
        ("The Committee has decided to reduce the policy rate from 46 percent to 43 percent.", "DOVE"),
        ("Recent indicators confirm the ongoing weakening in domestic demand.", "DOVE"),
        ("The underlying trend of inflation continued to decline in May.", "DOVE"),
        ("Liquidity conditions will continue to be closely monitored.", "NEUT"),
    ],
    5: [
        ("The Committee has decided to lower the policy rate from 43 percent to 40.5 percent.", "DOVE"),
        ("The disinflation process continues in an orderly manner.", "DOVE"),
        ("Geopolitical developments and energy prices remain a source of uncertainty for the inflation outlook.", "HAWK"),
        ("The Committee will continue to assess the risks meeting by meeting.", "NEUT"),
    ],
    6: [
        ("The Committee has decided to reduce the policy rate from 40.5 percent to 39 percent.", "DOVE"),
        ("Recent data confirm the ongoing weakening in domestic demand.", "DOVE"),
        ("A further slowdown in the underlying trend of inflation was observed.", "DOVE"),
        ("The Committee will create the monetary and financial conditions necessary to reach the inflation target.", "NEUT"),
    ],
    7: [
        ("The Committee has decided to lower the policy rate from 39 percent to 38 percent.", "DOVE"),
        ("Leading indicators suggest a temporary rise in the underlying trend, mainly due to base effects.", "NEUT"),
        ("The Committee reiterated that it remains attentive to upside risks stemming from geopolitical developments.", "HAWK"),
    ],
    8: [
        ("The Committee has decided to reduce the policy rate from 38 percent to 37 percent.", "DOVE"),
        ("The underlying trend of inflation decreased slightly in April.", "DOVE"),
        ("The Committee will ensure the tightness required by the projected disinflation path.", "HAWK"),
    ],
    9: [  # hedef dönem — build_sample_report.py'deki gerçek 2026-07 cümleleri
        ("The Monetary Policy Committee has decided to keep the policy rate at 37 percent.", "HAWK"),
        ("The Committee has also maintained the Central Bank overnight lending rate and the overnight borrowing rate.", "HAWK"),
        ("Leading indicators suggest that the underlying trend will rise temporarily in July.", "NEUT"),
        ("As a result of growing uncertainty amid geopolitical developments, energy prices started trending up again.", "HAWK"),
        ("Recent data confirm the ongoing weakening in domestic demand.", "DOVE"),
        ("The underlying trend of inflation decreased slightly in June.", "DOVE"),
        ("The impact of geopolitical developments on the inflation outlook is closely monitored.", "NEUT"),
        ("The Committee reiterated that it remains highly attentive to upside risks on inflation.", "HAWK"),
        ("The tight monetary policy stance will be maintained until price stability is achieved.", "HAWK"),
        ("Monetary policy decisions are made prudently on a meeting-by-meeting basis.", "NEUT"),
        ("In case of a significant and persistent deterioration in the inflation outlook, the stance will be tightened.", "HAWK"),
        ("In case of unanticipated developments in credit and deposit markets, macroprudential measures will be used.", "DOVE"),
        ("Liquidity conditions will continue to be closely monitored and used effectively.", "NEUT"),
        ("The Committee will make its decisions in a predictable, data-driven and transparent framework.", "NEUT"),
        ("The Committee will create the monetary and financial conditions necessary to reach the 5 percent target.", "HAWK"),
    ],
}


def noisy_label(true_label, noise=0.15):
    if rng.random() < noise:
        return rng.choice([l for l in ["HAWK", "NEUT", "DOVE"] if l != true_label])
    return true_label


rows = []

# --- Kappa seti: belge 8 ve 9'un TÜM cümleleri, 5 annotator tarafından da etiketlenir ---
kappa_docs = [8, 9]
for log_id in kappa_docs:
    for sent_idx, (sentence, true_label) in enumerate(DOCS[log_id]):
        for annotator in ANNOTATORS:
            rows.append({
                "log_id": log_id, "sent_idx": sent_idx, "sentence": sentence,
                "sentence_hash": sent_hash(sentence), "annotator": annotator,
                "label": noisy_label(true_label, noise=0.12),
                "confidence": int(rng.integers(2, 4)),
            })

# --- Geri kalan belgeler: her annotator kendine düşen belgeleri tek başına etiketler ---
remaining_docs = [d for d in DOCS if d not in kappa_docs]
for i, log_id in enumerate(remaining_docs):
    annotator = ANNOTATORS[i % len(ANNOTATORS)]
    for sent_idx, (sentence, true_label) in enumerate(DOCS[log_id]):
        rows.append({
            "log_id": log_id, "sent_idx": sent_idx, "sentence": sentence,
            "sentence_hash": sent_hash(sentence), "annotator": annotator,
            "label": noisy_label(true_label, noise=0.08),
            "confidence": int(rng.integers(2, 4)),
        })

df = pd.DataFrame(rows)
# Gerçek tablodaki `unique(sentence_hash, annotator)` + upsert davranışını taklit et:
# aynı kişi aynı cümleyi (nadiren, örn. iki belgede birebir aynı metin geçerse)
# birden çok kez etiketlemiş gibi görünmesin diye son etiketi tut.
df = df.drop_duplicates(subset=["sentence_hash", "annotator"], keep="last")

df.to_csv("sample_annotations.csv", index=False)
print(f"{len(df)} satır yazıldı -> sample_annotations.csv")
print(df.groupby("annotator").size())
print(f"\nbenzersiz cümle sayısı: {df['sentence_hash'].nunique()}  ·  belge sayısı: {df['log_id'].nunique()}")
print("kappa-set belge id'leri:", kappa_docs, " diğer belgeler:", remaining_docs)
