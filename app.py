import streamlit as st
import pandas as pd
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import pipeline
from collections import Counter
import re
import utils 

st.set_page_config(page_title="Piyasa Analiz", layout="wide")

# --- MODELLER ---
@st.cache_resource
def load_models():
    try: return pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except: return None
classifier = load_models()

# --- DETAYLI ANALİZ FONKSİYONU (KELİMELERİ DÖNDÜRÜR) ---
def analyze_apel_blix_detailed(text):
    text = text.lower()
    tokens = re.findall(r"[a-z']+", text)
    bigrams = [" ".join(pair) for pair in zip(tokens, tokens[1:])]
    
    token_counts = Counter(tokens)
    bigram_counts = Counter(bigrams)

    # Sözlükler
    nouns = ["inflation","growth","demand","prices","rates","policy","outlook","employment","unemployment","wages"]
    hawkish_adj = ["high","rising","strong","elevated","accelerating","robust","tightening","upside"]
    dovish_adj = ["low","falling","weak","slow","declining","subdued","easing","downside"]
    
    hawkish_single = {"tightening","restrictive","hike","risk","risks"}
    dovish_single = {"cut","easing","stimulus","recession","recovery"}

    # Set Oluşturma
    hawkish_phrases = {f"{adj} {noun}" for adj in hawkish_adj for noun in nouns}
    dovish_phrases = {f"{adj} {noun}" for adj in dovish_adj for noun in nouns}

    # Kelime Yakalama (Ekranda göstermek için)
    found_hawkish = []
    found_dovish = []

    # 1. Bigram Kontrolü
    h_bigram_score = 0
    d_bigram_score = 0
    for p in hawkish_phrases:
        if bigram_counts[p] > 0:
            h_bigram_score += bigram_counts[p]
            found_hawkish.append(f"{p} ({bigram_counts[p]})")
            
    for p in dovish_phrases:
        if bigram_counts[p] > 0:
            d_bigram_score += bigram_counts[p]
            found_dovish.append(f"{p} ({bigram_counts[p]})")

    # 2. Unigram Kontrolü
    h_single_score = 0
    d_single_score = 0
    for w in hawkish_single:
        if token_counts[w] > 0:
            h_single_score += token_counts[w]
            found_hawkish.append(f"{w} ({token_counts[w]})")
            
    for w in dovish_single:
        if token_counts[w] > 0:
            d_single_score += token_counts[w]
            found_dovish.append(f"{w} ({token_counts[w]})")

    # Skorlama
    h_total = h_bigram_score + h_single_score
    d_total = d_bigram_score + d_single_score
    total_sig = h_total + d_total
    
    final_score = (h_total - d_total) / total_sig if total_sig > 0 else 0
    
    return final_score, found_hawkish, found_dovish

def analyze_finbert(text):
    if not classifier: return 0, "neutral"
    res = classifier(text[:512])[0]
    score = res['score'] if res['label'] == "positive" else -res['score'] if res['label'] == "negative" else 0
    return score, res['label']

# --- ARAYÜZ (DASHBOARD İLK SIRADA) ---
st.title("🦅 Şahin/Güvercin Analiz Paneli")

# TAB SIRALAMASI DEĞİŞTİRİLDİ
tab1, tab2, tab3 = st.tabs(["📈 Dashboard", "📝 Veri Girişi & Analiz", "📊 Piyasa Verileri"])

# --- TAB 1: DASHBOARD ---
with tab1:
    if st.button("Grafikleri Yenile", key="dash_refresh"):
        df_logs = utils.fetch_all_data()
        
        if not df_logs.empty:
            df_logs['period_date'] = pd.to_datetime(df_logs['period_date'])
            df_logs['Donem'] = df_logs['period_date'].dt.strftime('%Y-%m')
            
            # Piyasa Verisi
            min_d = df_logs['period_date'].min().date()
            max_d = datetime.date.today()
            df_market, err = utils.fetch_market_data_adapter(min_d, max_d)
            
            # Birleştir
            merged = pd.merge(df_logs, df_market, on="Donem", how="left")
            merged = merged.sort_values("period_date")
            
            # Grafik
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # X Ekseni: Tarih
            fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['score_finbert'], name="FinBERT Skoru", line=dict(color='blue')), secondary_y=False)
            fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['score_abg'], name="Apel-Blix Skoru", line=dict(color='green', dash='dot')), secondary_y=False)
            
            if 'Yıllık TÜFE' in merged.columns:
                fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['Yıllık TÜFE'], name="Yıllık TÜFE (%)", line=dict(color='red')), secondary_y=True)
            
            if 'PPK Faizi' in merged.columns:
                 fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['PPK Faizi'], name="Faiz (%)", line=dict(color='orange')), secondary_y=True)

            fig.update_layout(title="Metin Analizi vs. Ekonomik Veriler", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Veri yok.")

# --- TAB 2: VERİ GİRİŞİ (KELİME GÖSTERİMLİ) ---
with tab2:
    c1, c2 = st.columns([1, 2])
    with c1:
        selected_date = st.date_input("Metin Tarihi", datetime.date.today())
        st.caption(f"Dönem: **{selected_date.strftime('%Y-%m')}**")
        source = st.text_input("Kaynak", "TCMB")
    with c2:
        txt = st.text_area("Metin", height=200, placeholder="İngilizce metni buraya yapıştırın...")
        
        if st.button("Analiz Et ve Kaydet", type="primary"):
            if txt:
                # 1. Analiz
                s_abg, hawks_found, doves_found = analyze_apel_blix_detailed(txt)
                s_fb, l_fb = analyze_finbert(txt)
                
                # 2. Kaydet
                utils.insert_entry(selected_date, txt, source, s_abg, s_abg, s_fb, l_fb)
                st.success(f"Kaydedildi! Skor: {s_abg:.2f}")
                
                # 3. KELİMELERİ GÖSTER (YENİ ÖZELLİK)
                st.markdown("### 🔍 Tespit Edilen İfadeler")
                k1, k2 = st.columns(2)
                with k1:
                    st.markdown("#### 🦅 Şahin (Hawkish)")
                    if hawks_found:
                        for w in hawks_found: st.write(f"- {w}")
                    else:
                        st.caption("Bulunamadı.")
                
                with k2:
                    st.markdown("#### 🕊️ Güvercin (Dovish)")
                    if doves_found:
                        for w in doves_found: st.write(f"- {w}")
                    else:
                        st.caption("Bulunamadı.")
                        
            else: st.warning("Metin girin.")

# --- TAB 3: PİYASA VERİLERİ (X EKSENİ DÖNEM) ---
with tab3:
    st.header("Piyasa Verileri")
    d1 = st.date_input("Başlangıç", datetime.date(2024, 1, 1))
    d2 = st.date_input("Bitiş", datetime.date.today())
    
    if st.button("Verileri Getir"):
        df, err = utils.fetch_market_data_adapter(d1, d2)
        if not df.empty:
            # Grafik X ekseni: Dönem
            fig_m = go.Figure()
            if 'Yıllık TÜFE' in df.columns:
                fig_m.add_trace(go.Scatter(x=df['Donem'], y=df['Yıllık TÜFE'], name="Yıllık TÜFE (%)", line=dict(color='red')))
            if 'PPK Faizi' in df.columns:
                fig_m.add_trace(go.Scatter(x=df['Donem'], y=df['PPK Faizi'], name="Faiz (%)", line=dict(color='orange')))
            
            fig_m.update_layout(title="Piyasa Görünümü (Dönem Bazlı)", xaxis_title="Dönem", yaxis_title="Değer (%)")
            st.plotly_chart(fig_m, use_container_width=True)
            
            st.dataframe(df)
        else:
            st.error(f"Veri yok: {err}")
