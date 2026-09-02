# -*- coding: utf-8 -*-
"""
build_sample_report.py
=======================
report_builder.py'yi CANLI Supabase/EVDS verisi OLMADAN test etmek için
sentetik ama şema-doğru veri üretir ve örnek bir .docx rapor çıkarır.

Neden gerekli: bu ortamda gerçek Supabase/EVDS kimlik bilgileri yok, bu yüzden
utils.fetch_all_data() vb. boş döner. Burada aynı DataFrame şemaları elle
kurulup report_builder.build_report() doğrudan çağrılır — yani test edilen,
gerçek uygulamanın kullanacağı AYNI kod yoludur (yalnızca veri kaynağı farklı).

Hedef dönem (2026-07) için kullanılan PPK metni, kullanıcının paylaştığı örnek
rapordaki (23 Temmuz 2026 PPK metni) gerçek İngilizce cümlelerdir.
"""
import datetime
import numpy as np
import pandas as pd

import utils
import report_builder as rb

rng = np.random.default_rng(7)

# =============================================================================
# 1) df_logs — market_logs tablosu şeması
# =============================================================================

PERIODS = [
    # NOT: bu ilk 4 dönem, §7 backtest modelinin iç eşiğini (train_textasdata_hybrid_cpi_ridge
    # >=10 etiketli gözlem ister) test ortamında da aşabilmek için eklendi — utils.py'deki
    # GERÇEK eşik değiştirilmedi, sadece test verisi (sentetik geçmiş) uzatıldı. Ayrıca bu,
    # önceki bir testte gözlemlenen "tarihler 2025'ten başlıyor" durumunu da düzeltir.
    ("2024-03-21", 50.0, 0,
     "The Committee has decided to keep the policy rate (the one-week repo auction rate) at 50 percent. "
     "The tight monetary stance will be maintained decisively until a significant and sustained decline "
     "in the underlying trend of inflation is achieved. The Committee will continue to make its decisions "
     "based on the inflation outlook."),
    ("2024-05-23", 50.0, 0,
     "The Committee has decided to keep the policy rate (the one-week repo auction rate) at 50 percent. "
     "Recent indicators point to a continued deceleration in the underlying trend of inflation. "
     "The Committee reiterated that it remains highly attentive to inflation risks and will maintain "
     "a tight stance until price stability is achieved."),
    ("2024-07-18", 50.0, 0,
     "The Committee has decided to keep the policy rate (the one-week repo auction rate) at 50 percent. "
     "The Committee assessed that the current tight monetary stance needs to be maintained decisively. "
     "Domestic demand continues to moderate, supporting the disinflation process."),
    ("2024-09-12", 50.0, 0,
     "The Committee has decided to keep the policy rate (the one-week repo auction rate) at 50 percent. "
     "The underlying trend of inflation has started to slow, in line with the projections in the "
     "July Inflation Report. The Committee will continue to make its decisions based on the inflation outlook."),
    ("2024-11-07", 50.0, 0,
     "The Committee has decided to keep the policy rate (the one-week repo auction rate) at 50 percent. "
     "The Committee assessed that the current tight monetary stance needs to be maintained decisively "
     "until a significant and sustained decline in the underlying trend of monthly inflation is achieved. "
     "The Committee will continue to make its decisions based on the inflation outlook. "
     "Inflation remains elevated and the Committee will tighten the stance further if a significant and "
     "persistent deterioration in the inflation outlook is foreseen."),
    ("2025-01-23", 47.5, -250,
     "The Monetary Policy Committee has decided to reduce the policy rate from 50 percent to 47.5 percent. "
     "Recent data indicate that the underlying trend of inflation has started to slow. "
     "The Committee will determine the pace of the disinflation process meeting by meeting, taking a "
     "cautious and data-driven approach. The Committee reiterated that it remains attentive to upside "
     "risks on inflation and will maintain a tight stance until price stability is achieved."),
    ("2025-03-06", 46.0, -150,
     "The Committee has decided to lower the policy rate from 47.5 percent to 46 percent. "
     "Domestic demand continues to weaken gradually while the underlying trend of inflation declined "
     "further in February. The Committee will continue to make its decisions in a predictable, "
     "data-driven and transparent framework. Should upside risks to the inflation outlook materialize, "
     "the Committee stands ready to tighten the monetary stance again."),
    ("2025-06-19", 43.0, -300,
     "The Committee has decided to reduce the policy rate from 46 percent to 43 percent. "
     "Recent indicators confirm the ongoing weakening in domestic demand. The underlying trend of "
     "inflation continued to decline in May, supporting the disinflation process. Liquidity conditions "
     "will continue to be closely monitored. The Committee will make its decisions cautiously, taking "
     "into account the inflation outlook and its underlying trend."),
    ("2025-09-11", 40.5, -250,
     "The Committee has decided to lower the policy rate from 43 percent to 40.5 percent. "
     "The disinflation process continues in an orderly manner and the underlying trend of inflation "
     "decreased slightly in August. However, geopolitical developments and energy prices remain a source "
     "of uncertainty for the inflation outlook. The Committee will continue to assess the risks meeting "
     "by meeting."),
    ("2025-12-11", 39.0, -150,
     "The Committee has decided to reduce the policy rate from 40.5 percent to 39 percent. "
     "Recent data confirm the ongoing weakening in domestic demand and a further slowdown in the "
     "underlying trend of inflation. The Committee will make its policy decisions so as to create the "
     "monetary and financial conditions necessary to reach the inflation target in the medium term."),
    ("2026-03-19", 38.0, -100,
     "The Committee has decided to lower the policy rate from 39 percent to 38 percent. "
     "Leading indicators suggest a temporary rise in the underlying trend in the near term, mainly due "
     "to base effects. The Committee reiterated that it remains attentive to upside risks on inflation "
     "stemming from geopolitical developments and energy prices."),
    ("2026-05-21", 37.0, -100,
     "The Committee has decided to reduce the policy rate from 38 percent to 37 percent. "
     "The underlying trend of inflation decreased slightly in April. The Committee will determine the "
     "policy rate by taking into account realized and expected inflation and its underlying trend, in a "
     "way to ensure the tightness required by the projected disinflation path."),
]

TARGET_DONEM = "2026-07"
TARGET_DATE = "2026-07-23"
TARGET_RATE = 37.0
TARGET_DELTA_BP = 0

# Hedef dönemin cümleleri — kullanıcının paylaştığı örnek PDF'teki GERÇEK
# cümleler, orijinal metin sırasına yakın bir sırayla (rapor sayfa 3-4).
TARGET_SENTENCES = [
    ("The Monetary Policy Committee (the Committee) has decided to keep the policy rate (the one-week repo auction rate) at 37 percent.", "hawk"),
    ("The Committee has also maintained the Central Bank overnight lending rate and the overnight borrowing rate at 40 percent and 35.5 percent, respectively.", "hawk"),
    ("Leading indicators suggest that the underlying trend will rise temporarily in July.", "neut"),
    ("As a result of the growing uncertainty amid geopolitical developments, energy prices started trending up again.", "hawk"),
    ("Recent data confirm the ongoing weakening in domestic demand.", "dove"),
    ("The underlying trend of inflation decreased slightly in June.", "dove"),
    ("The impact of geopolitical developments on the inflation outlook through the cost channel, economic activity and expectations is closely monitored.", "neut"),
    ("The Committee reiterated that it remains highly attentive to upside risks on inflation.", "hawk"),
    ("The tight monetary policy stance, which will be maintained until price stability is achieved, will strengthen the disinflation process through demand, exchange rate, and expectation channels.", "hawk"),
    ("The Committee will determine the policy rate by taking into account realized and expected inflation and its underlying trend in a way to ensure the tightness required by the projected disinflation path in line with the interim targets.", "hawk"),
    ("Monetary policy decisions are made prudently on a meeting-by-meeting basis with a focus on the inflation outlook.", "neut"),
    ("In case of a significant and persistent deterioration in the inflation outlook, monetary policy stance will be tightened.", "hawk"),
    ("In case of unanticipated developments in credit and deposit markets, monetary transmission mechanism will be supported via additional macroprudential measures.", "dove"),
    ("Liquidity conditions will continue to be closely monitored and liquidity management tools will continue to be used effectively.", "neut"),
    ("The Committee will make its decisions in a predictable, data-driven and transparent framework.", "neut"),
    ("The Committee will make its policy decisions so as to create the monetary and financial conditions necessary to reach the 5 percent inflation target in the medium term.", "hawk"),
]
TARGET_TEXT = " ".join(s for s, _ in TARGET_SENTENCES)

logs_rows = []
for i, (date, rate, dbp, text) in enumerate(PERIODS):
    logs_rows.append({
        "id": i + 1, "period_date": date, "text_content": text, "source": "TCMB PPK Kararı",
        "score_dict": None, "score_abg": None, "policy_rate": rate, "delta_bp": dbp,
    })
logs_rows.append({
    "id": len(PERIODS) + 1, "period_date": TARGET_DATE, "text_content": TARGET_TEXT,
    "source": "TCMB PPK Kararı", "score_dict": None, "score_abg": None,
    "policy_rate": TARGET_RATE, "delta_bp": TARGET_DELTA_BP,
})
df_logs = pd.DataFrame(logs_rows)
df_logs["period_date"] = pd.to_datetime(df_logs["period_date"])

# =============================================================================
# 2) df_events — event_logs şeması
# =============================================================================
df_events = pd.DataFrame([
    {"id": 1, "event_date": "2026-04-02", "links": "https://example.com/haber-jeopolitik-risk"},
    {"id": 2, "event_date": "2026-06-15", "links": "https://example.com/enerji-fiyatlari"},
])

# =============================================================================
# 3) df_market — fetch_market_data_adapter çıktı şeması
# =============================================================================
all_donems = [pd.to_datetime(d).strftime("%Y-%m") for d, *_ in PERIODS] + [TARGET_DONEM]
cpi_yoy_start, cpi_yoy_end = 48.0, 29.5
market_rows = []
for i, donem in enumerate(all_donems):
    t = i / (len(all_donems) - 1)
    cpi_yoy = cpi_yoy_start + (cpi_yoy_end - cpi_yoy_start) * t + rng.normal(0, 0.4)
    cpi_mom = max(0.5, 3.2 - 2.0 * t + rng.normal(0, 0.3))
    rate = PERIODS[i][1] if i < len(PERIODS) else TARGET_RATE
    market_rows.append({
        "Donem": donem,
        "Aylık TÜFE": round(cpi_mom, 2),
        "Yıllık TÜFE": round(cpi_yoy, 2),
        "PPK Faizi": rate,
        "PKA 12 Ay Enflasyon Beklentisi": round(cpi_yoy * 0.62 + rng.normal(0, 0.5), 2),
        "İYA 12 Ay Enflasyon Beklentisi": round(cpi_yoy * 0.66 + rng.normal(0, 0.5), 2),
        "HBA 12 Ay Enflasyon Beklentisi": round(cpi_yoy * 0.95 + rng.normal(0, 0.8), 2),
        "AOFM": round(rate + rng.normal(0.3, 0.4), 2),
        "SortDate": donem + "-01",
    })
df_market = pd.DataFrame(market_rows)
df_market["AOFM-Faiz Farkı"] = df_market["AOFM"] - df_market["PPK Faizi"]

# =============================================================================
# 4) abg_df — gerçek utils.calculate_abg_scores ile (sentetik değil, GERÇEK fonksiyon)
# =============================================================================
abg_df = utils.calculate_abg_scores(df_logs)
print("[abg_df]\n", abg_df[["Donem", "abg_index", "abg_index_raw", "n_match", "hawk_count", "dove_count"]])

# =============================================================================
# 5) df_sent — fetch_sentences() çıktı şeması (gerçek assign_theme() kullanılarak)
# =============================================================================
def _mk_probs(direction):
    if direction == "hawk":
        hawk = rng.uniform(0.80, 0.995)
        dove = rng.uniform(0.001, 0.05)
    elif direction == "dove":
        dove = rng.uniform(0.65, 0.92)
        hawk = rng.uniform(0.02, 0.12)
    else:
        hawk = rng.uniform(0.15, 0.35)
        dove = rng.uniform(0.15, 0.35)
    neut = max(0.0, 1.0 - hawk - dove)
    return hawk, dove, neut

sent_rows = []
log_id_map = {row["period_date"].strftime("%Y-%m"): row["id"] for _, row in df_logs.iterrows()}

# Hedef dönem: gerçek cümleler + tasarlanmış tonlar
sent_total = len(TARGET_SENTENCES)
for idx, (sentence, direction) in enumerate(TARGET_SENTENCES):
    hawk, dove, neut = _mk_probs(direction)
    sent_rows.append({
        "log_id": log_id_map[TARGET_DONEM], "period_date": TARGET_DATE,
        "sent_idx": idx, "sent_total": sent_total, "sentence": sentence,
        "hawk": hawk, "dove": dove, "neut": neut, "diff": hawk - dove,
        "agent_label": rng.choice(["Central Bank", "Firms", "Households", "Financial Sector"]),
        "agent_conf": round(rng.uniform(0.55, 0.95), 2),
    })

# Diğer dönemler: metni gerçek cümlelere böl, yöne kabaca uygun rastgele ton ata
# (erken dönemler daha şahin, geç dönemler daha dengeli/güvercin ağırlıklı — hikaye ile tutarlı)
for i, (date, rate, dbp, text) in enumerate(PERIODS):
    donem = pd.to_datetime(date).strftime("%Y-%m")
    sents = utils.split_sentences_nlp(utils.normalize_text(text))
    hawk_bias = 0.75 - 0.55 * (i / (len(PERIODS) - 1))  # 0.75 -> 0.20 arası azalan şahinlik eğilimi
    n = len(sents)
    for idx, s in enumerate(sents):
        is_hawk = rng.random() < hawk_bias
        direction = "hawk" if is_hawk else ("dove" if rng.random() < 0.5 else "neut")
        hawk, dove, neut = _mk_probs(direction)
        sent_rows.append({
            "log_id": log_id_map[donem], "period_date": date,
            "sent_idx": idx, "sent_total": n, "sentence": s,
            "hawk": hawk, "dove": dove, "neut": neut, "diff": hawk - dove,
            "agent_label": rng.choice(["Central Bank", "Firms", "Households", "Financial Sector"]),
            "agent_conf": round(rng.uniform(0.55, 0.95), 2),
        })

df_sent = pd.DataFrame(sent_rows)
df_sent["period_date"] = pd.to_datetime(df_sent["period_date"])
df_sent["Donem"] = df_sent["period_date"].dt.strftime("%Y-%m")
# GERÇEK tema sınıflandırıcı (sözlük/regex tabanlı, model gerektirmez)
df_sent["theme_labels"] = df_sent["sentence"].map(utils.assign_themes)
df_sent["theme_label"] = df_sent["sentence"].map(utils.assign_theme)
df_sent = df_sent.sort_values(["period_date", "sent_idx"]).reset_index(drop=True)

print("\n[df_sent] dönem başına cümle sayısı:\n", df_sent.groupby("Donem").size())
print("\n[df_sent] tema dağılımı:\n", df_sent["theme_label"].value_counts())

# =============================================================================
# 6) ai_df — trend_series_from_cache() çıktısıyla AYNI şema, GERÇEK
#    postprocess_ai_series_steps() ile kalibre edilmiş (EMA/histerezis sentetik
#    değil, gerçek fonksiyon üzerinden hesaplanmış)
# =============================================================================
ai_rows = []
for i, (date, rate, dbp, text) in enumerate(PERIODS + [(TARGET_DATE, TARGET_RATE, TARGET_DELTA_BP, TARGET_TEXT)]):
    donem = pd.to_datetime(date).strftime("%Y-%m")
    d = df_sent[df_sent["Donem"] == donem]
    diff_mean = float(d["diff"].mean()) if not d.empty else 0.0
    ai_rows.append({
        "Dönem": donem, "period_date": pd.to_datetime(date),
        "Şahin Olasılık": float(d["hawk"].mean()) if not d.empty else np.nan,
        "Güvercin Olasılık": float(d["dove"].mean()) if not d.empty else np.nan,
        "Nötr Olasılık": float(d["neut"].mean()) if not d.empty else np.nan,
        "Diff (H-D)": diff_mean,
        "Diff (Full-text)": diff_mean,
        "Duruş": utils.stance_3class_from_diff(diff_mean, deadband=0.15),
        "Delta BP": dbp,
        "Aksiyon": ("Faiz İndirimi" if dbp < 0 else ("Faiz Artışı" if dbp > 0 else "Sabit Tutma")),
        "Rejim": utils.stance_3class_from_diff(diff_mean, deadband=0.15),
        "Güven": round(rng.uniform(0.7, 0.95), 2),
    })
ai_out = pd.DataFrame(ai_rows).sort_values("period_date").reset_index(drop=True)
ai_out["Aksiyon Yön"] = ai_out["Delta BP"].map(lambda x: np.nan if pd.isna(x) else (1.0 if x > 0 else (-1.0 if x < 0 else 0.0)))
ai_df = utils.postprocess_ai_series_steps(ai_out, diff_col="Diff (H-D)", span=3, z_scale=2.0, hyst=25.0)
print("\n[ai_df]\n", ai_df[["Dönem", "Diff (H-D)", "AI Score (EMA)", "AI Rejim"]])

# =============================================================================
# 7) model_pack — GERÇEK backtest fonksiyonları (utils.py), sentetik df_logs/df_market üzerinde
# =============================================================================
df_td = utils.textasdata_prepare_df_hybrid_cpi(df_logs, df_market)
print(f"\n[textasdata] hazırlanan satır sayısı: {len(df_td)} (etiketli: {df_td['delta_bp'].notna().sum()})")
model_pack = None
if not df_td.empty and df_td["delta_bp"].notna().sum() >= 5:
    # NOT: gerçek uygulamada eşik >=10'dur (bkz. app.py); bu demo veri setinde
    # yalnızca 9 dönem olduğu için burada test amaçlı 5'e düşürüldü.
    model_pack = utils.train_textasdata_hybrid_cpi_ridge(df_td, n_splits=3)
    if model_pack:
        model_pack["df_hist"] = df_td
    print("[model_pack metrics]", model_pack.get("metrics"))
else:
    print("[model_pack] yeterli etiketli gözlem yok, backtest bölümü atlanacak")

# =============================================================================
# 8) RAPORU ÜRET
# =============================================================================
out_path = rb.build_report(
    df_logs=df_logs, df_events=df_events, df_market=df_market,
    abg_df=abg_df, ai_df=ai_df, df_sent=df_sent,
    donem=TARGET_DONEM, model_pack=model_pack,
    analyst_note=(
        "Örnek analist notu: Bu alana, örneğin Claude/ChatGPT/Gemini gibi bir asistandan "
        "aldığınız bağımsız bir okuma önerisini yapıştırabilirsiniz. Yukarıdaki kullanım "
        "notunu unutmayın: bu tür bir girdi tek seferliktir ve prompt'a duyarlıdır."
    ),
    out_path="ppk_rapor_ORNEK_2026-07.docx",
)
print("\nRapor üretildi:", out_path)
