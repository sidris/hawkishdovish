import streamlit as st
import pandas as pd
from supabase import create_client, Client
from transformers import pipeline
from collections import Counter
import re
import plotly.express as px
import datetime

# -----------------------------------------------------------------------------
# 1. AYARLAR VE BAĞLANTILAR
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Piyasa Analiz Sistemi", layout="wide")

# Supabase Bağlantısı (Secrets'tan çeker)
@st.cache_resource
def init_supabase():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

supabase = init_supabase()

# FinBERT Modelini Yükle (Cache ile hızlandırılmış)
@st.cache_resource
def load_finbert():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

try:
    classifier = load_finbert()
except Exception as e:
    st.error(f"Model yüklenirken hata: {e}")
    classifier = None

# -----------------------------------------------------------------------------
# 2. ALGORİTMALAR (Sözlük & AI)
# -----------------------------------------------------------------------------

def analyze_with_dictionary(text):
    text = text.lower()
    tokens = re.findall(r"[a-z']+", text)
    hawkish = ["high", "rising", "elevated", "strong", "tightening", "inflation", "risk", "hike"]
    dovish = ["low", "falling", "weak", "slow", "easing", "cut", "stimulus", "decline"]
    
    c = Counter(tokens)
    h_score = sum(c[t] for t in hawkish)
    d_score = sum(c[t] for t in dovish)
    total = h_score + d_score
    
    if total == 0: return 0
    return (h_score - d_score) / total # -1 (Güvercin) ile +1 (Şahin) arası

def analyze_with_finbert(text):
    if not classifier: return 0, "neutral"
    # FinBERT max 512 token kabul eder, uzun metinleri kesiyoruz (Basitlik için)
    res = classifier(text[:512])[0]
    score = res['score']
    label = res['label']
    
    # Skoru -1 ve +1 arasına map edelim
    final_score = score if label == "positive" else -score if label == "negative" else 0
    return final_score, label

# -----------------------------------------------------------------------------
# 3. ARAYÜZ
# -----------------------------------------------------------------------------

st.title("☁️ Bulut Tabanlı Merkez Bankası Analizi")
tab1, tab2 = st.tabs(["📝 Veri Girişi & Analiz", "📈 Zaman Serisi Dashboard"])

# --- TAB 1: VERİ GİRİŞİ ---
with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Dönem Seçimi")
        # 2025-1 Formatı için Yıl ve Ay seçimi
        year = st.selectbox("Yıl", range(2020, 2030), index=5) # Default 2025
        month = st.selectbox("Ay", range(1, 13))
        source = st.text_input("Kaynak (Örn: PPK Özeti)", "TCMB")
        
    with col2:
        text_input = st.text_area("Metin", height=200, placeholder="Analiz edilecek metni buraya girin...")
        
        if st.button("Analiz Et ve Veritabanına Kaydet", type="primary"):
            if text_input:
                with st.spinner("Yapay zeka ve algoritmalar çalışıyor..."):
                    # 1. Skorları Hesapla
                    dict_score = analyze_with_dictionary(text_input)
                    fb_score, fb_label = analyze_with_finbert(text_input)
                    
                    # 2. Tarih Formatı Oluştur (Veritabanı için YYYY-MM-01)
                    period_date = f"{year}-{month:02d}-01"
                    
                    # 3. Supabase'e Yaz
                    data = {
                        "period_date": period_date,
                        "text_content": text_input,
                        "source": source,
                        "score_dict": dict_score,
                        "score_finbert": fb_score,
                        "finbert_label": fb_label
                    }
                    
                    try:
                        supabase.table("market_logs").insert(data).execute()
                        st.success(f"✅ Kayıt Başarılı! Dönem: {period_date} | FinBERT: {fb_label}")
                    except Exception as e:
                        st.error(f"Veritabanı hatası: {e}")
            else:
                st.warning("Lütfen metin giriniz.")

# --- TAB 2: DASHBOARD ---
with tab2:
    st.header("Algoritma Karşılaştırmalı Zaman Serisi")
    
    # Yenile butonu
    if st.button("Verileri Getir / Yenile"):
        # Supabase'den verileri çek
        response = supabase.table("market_logs").select("*").order("period_date").execute()
        rows = response.data
        
        if rows:
            df = pd.DataFrame(rows)
            df['period_date'] = pd.to_datetime(df['period_date'])
            
            # Grafik için veriyi düzenle (Melt)
            chart_df = df.melt(
                id_vars=['period_date', 'source'], 
                value_vars=['score_dict', 'score_finbert'],
                var_name='Algoritma', 
                value_name='Skor'
            )
            
            # İsimleri güzelleştir
            chart_df['Algoritma'] = chart_df['Algoritma'].replace({
                'score_dict': 'Geleneksel (Sözlük)', 
                'score_finbert': 'Yapay Zeka (FinBERT)'
            })
            
            # Plotly Grafiği
            fig = px.line(chart_df, x='period_date', y='Skor', color='Algoritma', 
                          title="Şahin/Güvercin Eğilimi (Zaman İçinde)",
                          markers=True, hover_data=['source'])
            
            # Referans çizgileri
            fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Nötr")
            fig.add_annotation(text="Şahin (Hawkish) 🦅", xref="paper", yref="paper", x=0, y=0.95, showarrow=False)
            fig.add_annotation(text="Güvercin (Dovish) 🕊️", xref="paper", yref="paper", x=0, y=0.05, showarrow=False)
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Ham Veriler")
            st.dataframe(df[['period_date', 'source', 'score_dict', 'score_finbert', 'finbert_label']])
            
        else:
            st.info("Veritabanında henüz kayıt yok. 'Veri Girişi' sekmesinden ekleme yapın.")
