# -*- coding: utf-8 -*-
"""
annotate_app.py
================
5 profesyonel için AYRI, hafif bir cümle etiketleme arayüzü. Ana analiz
uygulamasından (app.py) BİLİNÇLİ olarak ayrı tutulmuştur: annotator'ların silme/
güncelleme gibi yönetici araçlarını görmesine gerek yok, sadece etiketleme
ekranına ihtiyaçları var. Aynı utils.py'yi (dolayısıyla aynı Supabase bağlantısını)
kullanır — ayrı bir veritabanı/kurulum gerekmez.

Çalıştırma: streamlit run annotate_app.py

Gereken tablo (bir kez, Supabase SQL editöründe):
    utils.py'nin sonundaki "20. İNSAN ETİKETLEME" bölümündeki `create table`
    ifadesine bakın (bu dosyada tekrarlanmıyor, tek bir kaynaktan yönetilsin diye).
"""
import streamlit as st
import pandas as pd
import hashlib

import utils

st.set_page_config(page_title="PPK Cümle Etiketleme", layout="centered")

# --- Annotator başına ayrı PIN — gerçek isimler ve kendi seçtiğiniz PIN'lerle
# değiştirin. Kimlik artık açılır menüden DEĞİL, hangi PIN girildiğinden
# belirleniyor — böylece PIN'i bilmeyen biri başkası adına giriş yapamaz ve
# yanlışlıkla (ya da kasıtlı) başka bir annotator'ın etiketinin üzerine yazamaz.
ANNOTATOR_PINS = {
    "4471": "Ayşe",
    "8823": "Berk",
    "1195": "Cem",
    "6640": "Deniz",
    "3308": "Elif",
}

LABEL_OPTIONS = [("🦅 Şahin", "HAWK"), ("⚖️ Nötr", "NEUT"), ("🕊️ Güvercin", "DOVE")]
CONF_OPTIONS = [("Emin değilim", 1), ("Orta", 2), ("Eminim", 3)]


# =============================================================================
# GİRİŞ
# =============================================================================
if "annot_ok" not in st.session_state:
    st.session_state["annot_ok"] = False
    st.session_state["annotator"] = None

if not st.session_state["annot_ok"]:
    st.title("🏷️ PPK Cümle Etiketleme")
    pin = st.text_input("PIN", type="password")
    if st.button("Giriş", type="primary"):
        if pin in ANNOTATOR_PINS:
            st.session_state["annot_ok"] = True
            st.session_state["annotator"] = ANNOTATOR_PINS[pin]
            st.rerun()
        else:
            st.error("Hatalı PIN.")
    st.stop()

st.title("🏷️ PPK Cümle Etiketleme")

annotator = st.session_state["annotator"]
top_l, top_r = st.columns([4, 1])
top_l.caption(f"Giriş yapan: **{annotator}**")
if top_r.button("🚪 Çıkış", use_container_width=True):
    st.session_state["annot_ok"] = False
    st.session_state["annotator"] = None
    st.rerun()

with st.expander("ℹ️ Bu ne işe yarıyor / nasıl etiketlemeliyim?"):
    st.markdown(
        "Her kart bir PPK karar metninden TEK bir cümle gösterir. Cümleyi, o cümlenin "
        "KENDİ İÇİNDE taşıdığı sıkılaştırma/gevşeme sinyaline göre işaretleyin — "
        "**🦅 Şahin**: sıkı duruşu, enflasyon endişesini, ek sıkılaştırma olasılığını "
        "vurgular. **🕊️ Güvercin**: gevşemeyi, talep/enflasyondaki zayıflamayı, "
        "destekleyici dili vurgular. **⚖️ Nötr**: sürece/çerçeveye dair, yön belirtmeyen "
        "ifadeler (\"kararlar toplantı bazında alınır\" gibi). Emin değilseniz güven "
        "seviyesini düşük işaretleyin — bu, ilerideki analizde o etiketin ağırlığını azaltır. "
        "İlk göreceğiniz cümleler (🔁 ortak küme) TÜM annotator'lara aynı sırayla gösterilir — "
        "bunlar, aramızda ne kadar hemfikir olduğumuzu ölçmek için kasıtlı olarak paylaşılan bir "
        "alt kümedir; lütfen bu kısmı atlamayın."
    )

df_logs = utils.fetch_all_data()
if df_logs is None or df_logs.empty:
    st.info("Henüz PPK kaydı yok.")
    st.stop()

pool = utils.build_sentence_pool(df_logs)
if pool.empty:
    st.info("Cümle havuzu boş.")
    st.stop()

df_ann = utils.fetch_annotations()
kappa_hashes = utils.kappa_set_hashes(pool, target_size=150)

my_done = set()
if not df_ann.empty:
    my_done = set(df_ann.loc[df_ann["annotator"] == annotator, "sentence_hash"])

pool = pool.drop_duplicates(subset=["sentence_hash"]).reset_index(drop=True)
pool["in_kappa_set"] = pool["sentence_hash"].isin(kappa_hashes)
pool["done_by_me"] = pool["sentence_hash"].isin(my_done)

# Kişiye özel deterministik sıralama anahtarı: aynı annotator her oturumda aynı
# sırayı görür, ama annotator'lar birbirinden farklı sıralarla ilerler — böylece
# koordinasyon olmadan kapsam kendiliğinden yayılır (herkes aynı cümleden başlayıp
# aynı yere kadar ilerlemez).
def _order_key(h):
    return hashlib.sha256((h + "::" + annotator).encode()).hexdigest()

pool["_order"] = pool["sentence_hash"].map(_order_key)

kappa_pending = (pool[pool["in_kappa_set"] & ~pool["done_by_me"]]
                 .sort_values("sentence_hash"))  # ortak küme: HERKESTE AYNI sıra
rest_pending = (pool[~pool["in_kappa_set"] & ~pool["done_by_me"]]
                .sort_values("_order"))  # geri kalan: kişiye özel sıra
queue = pd.concat([kappa_pending, rest_pending], ignore_index=True)

total_pool = len(pool)
done_pool = int(pool["done_by_me"].sum())
kappa_total = int(pool["in_kappa_set"].sum())
kappa_done = int((pool["in_kappa_set"] & pool["done_by_me"]).sum())

c1, c2, c3 = st.columns(3)
c1.metric("Genel ilerleme", f"{done_pool}/{total_pool}")
c2.metric("🔁 Ortak küme (kappa)", f"{kappa_done}/{kappa_total}")
c3.metric("Kalan (bu oturumda)", len(queue))

if not df_ann.empty:
    with st.expander("👥 Ekip ilerlemesi"):
        st.dataframe(utils.annotation_progress(df_ann), hide_index=True, use_container_width=True)

st.divider()

if queue.empty:
    st.success("🎉 Bu havuzdaki tüm cümleleri etiketlediniz. Yeni bir PPK kaydı eklendiğinde havuz otomatik büyür.")
    st.stop()

row = queue.iloc[0]

if row["in_kappa_set"] and not row["done_by_me"] and kappa_done < kappa_total:
    st.caption("🔁 Bu cümle ORTAK KÜMEden — tüm annotator'lar aynı cümleyi görüyor (uyum ölçümü için).")

st.caption(f"Dönem: **{row.get('Donem', '—')}**  ·  Cümle {int(row['sent_idx']) + 1} / {int(row['sent_total'])}")

# Bağlam için önceki/sonraki cümleyi soluk göster (varsa)
doc_sents = pool[pool["log_id"] == row["log_id"]].sort_values("sent_idx")
idx_in_doc = int(row["sent_idx"])
prev_s = doc_sents[doc_sents["sent_idx"] == idx_in_doc - 1]["sentence"]
next_s = doc_sents[doc_sents["sent_idx"] == idx_in_doc + 1]["sentence"]
if not prev_s.empty:
    st.markdown(f"<div style='color:#999;font-size:0.85rem;'>… {prev_s.iloc[0]}</div>", unsafe_allow_html=True)

st.markdown(
    f"<div style='font-size:1.25rem;padding:16px;border:2px solid #2F5496;"
    f"border-radius:10px;margin:8px 0;'>{row['sentence']}</div>",
    unsafe_allow_html=True,
)

if not next_s.empty:
    st.markdown(f"<div style='color:#999;font-size:0.85rem;'>{next_s.iloc[0]} …</div>", unsafe_allow_html=True)

st.write("")
label_choice = st.radio("Yön", [l[0] for l in LABEL_OPTIONS], horizontal=True, key=f"lbl_{row['sentence_hash']}")
conf_choice = st.radio("Güven", [c[0] for c in CONF_OPTIONS], horizontal=True, index=2, key=f"conf_{row['sentence_hash']}")

if st.button("💾 Kaydet ve Sonraki", type="primary", use_container_width=True):
    label_code = dict(LABEL_OPTIONS)[label_choice]
    conf_code = dict(CONF_OPTIONS)[conf_choice]
    ok = utils.insert_annotation(
        log_id=row["log_id"], sent_idx=row["sent_idx"], sentence=row["sentence"],
        annotator=annotator, label=label_code, confidence=conf_code,
    )
    if ok:
        st.toast("Kaydedildi.")
        st.rerun()
    else:
        st.error(
            "Kaydedilemedi — Supabase bağlantısı yok görünüyor. "
            "`human_annotations` tablosunun oluşturulduğundan ve secrets'ın "
            "doğru olduğundan emin olun (bkz. utils.py, bölüm 20)."
        )
