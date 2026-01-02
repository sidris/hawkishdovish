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

# --- ANALİZ FONKSİYONLARI ---
def analyze_text(text):
    # 1. Basit Sözlük
    tokens = re.findall(r"[a-z']+", text.lower())
    c = Counter(tokens)
    h_score = sum(c[t] for t in ["high", "rising", "strong", "inflation", "risk"])
    d_score = sum(c[t] for t in ["low", "falling", "weak", "cut"])
    total = h_score + d_score
    s_dict = (h_score - d_score) / total if total > 0 else 0
    
    # 2. FinBERT
    s_fb, l_fb = 0, "neutral"
    if classifier:
        res = classifier(text[:512])[0]
        s_fb = res['score'] if res['label'] == "positive" else -res['score'] if res['label'] == "negative" else 0
        l_fb = res['label']
        
    return s_dict, s_fb, l_fb

# --- ARAYÜZ ---
st.title("🦅 Şahin/Güvercin Analiz Paneli")

tab1, tab2, tab3 = st.tabs(["📝 Veri Girişi", "📈 Dashboard", "📊 Piyasa Verileri"])

# --- TAB 1: VERİ GİRİŞİ ---
with tab1:
    c1, c2 = st.columns([1, 2])
    with c1:
        # KULLANICI TAM TARİH GİRER (Örn: 15.10.2025)
        selected_date = st.date_input("Metin Tarihi", datetime.date.today())
        # Kullanıcıya Dönemi Gösterelim
        st.caption(f"İlgili Dönem: **{selected_date.strftime('%Y-%m')}**")
        source = st.text_input("Kaynak", "TCMB")
    with c2:
        txt = st.text_area("Metin", height=150)
        if st.button("Analiz Et ve Kaydet", type="primary"):
            if txt:
                s_dict, s_fb, l_fb = analyze_text(txt)
                # Apel-Blix için şimdilik s_dict kullanıyoruz (basitlik için)
                utils.insert_entry(selected_date, txt, source, s_dict, s_dict, s_fb, l_fb)
                st.success("Kaydedildi!")
            else: st.warning("Metin girin.")

# --- TAB 2: DASHBOARD ---
with tab2:
    if st.button("Grafikleri Yenile"):
        # 1. Metin Verileri
        df_logs = utils.fetch_all_data()
        
        if not df_logs.empty:
            df_logs['period_date'] = pd.to_datetime(df_logs['period_date'])
            # Eşleşme için 'Donem' kolonu oluştur
            df_logs['Donem'] = df_logs['period_date'].dt.strftime('%Y-%m')
            
            # 2. Piyasa Verileri
            min_d = df_logs['period_date'].min().date()
            max_d = datetime.date.today()
            df_market, err = utils.fetch_market_data_adapter(min_d, max_d)
            
            # 3. Birleştirme (Donem Üzerinden)
            # Sol tarafta Metin verisi (Tam Tarihli), sağdan o ayın enflasyonu gelir
            merged = pd.merge(df_logs, df_market, on="Donem", how="left")
            merged = merged.sort_values("period_date")
            
            # 4. Grafik
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # X Ekseni: Metnin Gerçek Tarihi (period_date)
            fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['score_finbert'], name="FinBERT Skoru", line=dict(color='blue')), secondary_y=False)
            
            if 'Yıllık TÜFE' in merged.columns:
                fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['Yıllık TÜFE'], name="Yıllık TÜFE (%)", line=dict(color='red', dash='dot')), secondary_y=True)
            
            if 'PPK Faizi' in merged.columns:
                 fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['PPK Faizi'], name="Faiz (%)", line=dict(color='orange', dash='dot')), secondary_y=True)

            fig.update_layout(title="Metin Analizi vs. Ekonomik Veriler", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(merged[['period_date', 'source', 'score_finbert', 'Yıllık TÜFE', 'PPK Faizi']])
        else:
            st.warning("Veri yok.")

# --- TAB 3: PİYASA VERİLERİ ---
with tab3:
    st.header("Sadece Piyasa Verileri")
    d1 = st.date_input("Başlangıç", datetime.date(2024, 1, 1))
    d2 = st.date_input("Bitiş", datetime.date.today())
    if st.button("Getir"):
        df, err = utils.fetch_market_data_adapter(d1, d2)
        if not df.empty:
            st.dataframe(df)
            st.line_chart(df.set_index("Donem")[['Yıllık TÜFE', 'PPK Faizi']])
        else:
            st.error(f"Veri yok: {err}")
