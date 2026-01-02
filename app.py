import streamlit as st
import pandas as pd
import datetime
from transformers import pipeline
from collections import Counter
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import utils

# -----------------------------------------------------------------------------
# 1. AYARLAR
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Piyasa Analiz Sistemi", layout="wide")

# FinBERT Yükle
@st.cache_resource
def load_finbert():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

try:
    classifier = load_finbert()
except:
    classifier = None

# -----------------------------------------------------------------------------
# 2. ALGORİTMALAR
# -----------------------------------------------------------------------------
def analyze_simple_dict(text):
    text = text.lower()
    tokens = re.findall(r"[a-z']+", text)
    hawkish = ["high", "rising", "elevated", "strong", "tightening", "inflation", "risk", "hike"]
    dovish = ["low", "falling", "weak", "slow", "easing", "cut", "stimulus", "decline"]
    c = Counter(tokens)
    h_score = sum(c[t] for t in hawkish)
    d_score = sum(c[t] for t in dovish)
    total = h_score + d_score
    if total == 0: return 0
    return (h_score - d_score) / total

def analyze_apel_blix_grimaldi(text):
    text = text.lower()
    tokens = re.findall(r"[a-z']+", text)
    bigrams = [" ".join(pair) for pair in zip(tokens, tokens[1:])]
    token_counts = Counter(tokens)
    bigram_counts = Counter(bigrams)

    nouns = ["cost","costs","expenditures","consumption","growth","output","demand","activity","production","investment","productivity","labor","labour","job","jobs","participation","wage","wages","recovery","slowdown","contraction","expansion","cycle","conditions","credit","lending","borrowing","liquidity","stability","markets","volatility","uncertainty","risks","easing","rates","policy","stance","outlook","pressures","inflation","price", "prices","oil price", "oil prices","cyclical position","development","employment","unemployment","gold"]
    hawkish_adjectives = ["high", "higher","strong", "stronger","increasing", "increased","fast", "faster","elevated","rising","accelerating","robust","persistent","mounting","excessive","solid","resillent","vigorous","overheating","tightening","restrivtive","constrained","limited","upside","significant","notable"]
    dovish_adjectives = ["low", "lower","weak", "weaker","decreasing", "decreased","slow", "slower","falling","declining","subdued","soft","softer","easing","moderate","moderating","cooling","softening","downside","adverse"]
    hawkish_single = {"tight","tightening","restrictive","elevated","high","overheating","pressures","pressure","risk","risks","upside","vigilant","decisive"}
    dovish_single = {"disinflation","decline","declining","fall","falling","decrease","decreasing","lower","low","subdued","contained","anchored","cooling","slow","slower","improvement","better","easing","relief"}

    hawkish_phrases = {f"{adj} {noun}" for adj in hawkish_adjectives for noun in nouns}
    dovish_phrases = {f"{adj} {noun}" for adj in dovish_adjectives for noun in nouns}

    hawk_bigram_count = sum(bigram_counts[p] for p in hawkish_phrases)
    dove_bigram_count = sum(bigram_counts[p] for p in dovish_phrases)
    hawk_single_count = sum(token_counts[w] for w in hawkish_single)
    dove_single_count = sum(token_counts[w] for w in dovish_single)

    hawk_total = hawk_bigram_count + hawk_single_count
    dove_total = dove_bigram_count + dove_single_count
    total_signal = hawk_total + dove_total

    if total_signal == 0: return 0
    return (hawk_total - dove_total) / total_signal

def analyze_with_finbert(text):
    if not classifier: return 0, "neutral"
    res = classifier(text[:512])[0]
    score = res['score']
    label = res['label']
    final_score = score if label == "positive" else -score if label == "negative" else 0
    return final_score, label

# -----------------------------------------------------------------------------
# 3. ARAYÜZ
# -----------------------------------------------------------------------------
st.title("🦅 Şahin/Güvercin Analiz Paneli")

tab1, tab2, tab3 = st.tabs(["📝 Veri Girişi", "✏️ Düzenle/Sil", "📈 Dashboard"])

# --- TAB 1: VERİ GİRİŞİ (PICKER EKLENDİ) ---
with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Dönem Seçimi")
        # YENİ: DATE PICKER
        # Ayın 1'ini default seçelim ki verilerle uyuşsun
        default_date = datetime.date.today().replace(day=1)
        selected_date = st.date_input("Dönem Tarihi", value=default_date, format="DD/MM/YYYY")
        
        source = st.text_input("Kaynak", "TCMB PPK Özeti")
    with col2:
        text_input = st.text_area("Metin", height=200)
        
        if st.button("Kaydet ve Analiz Et", type="primary"):
            if text_input:
                with st.spinner("Analiz ediliyor..."):
                    # Skorlar
                    val_simple = analyze_simple_dict(text_input)
                    val_abg = analyze_apel_blix_grimaldi(text_input)
                    val_fb, lab_fb = analyze_with_finbert(text_input)
                    
                    # Seçilen tarihi veritabanı formatına (YYYY-MM-DD) çevir
                    # Genelde ekonomik veriler ayın 1'ine endekslidir
                    period_date = selected_date.replace(day=1)
                    
                    utils.insert_entry(period_date, text_input, source, val_simple, val_abg, val_fb, lab_fb)
                    st.success(f"✅ Kaydedildi! Dönem: {period_date}")
            else:
                st.warning("Lütfen bir metin giriniz.")

# --- TAB 2: DÜZENLEME ---
with tab2:
    df = utils.fetch_all_data()
    if not df.empty:
        opts = df.apply(lambda x: f"ID: {x['id']} | {x['period_date']} | {x['source']}", axis=1)
        sel_opt = st.selectbox("Kayıt Seç:", opts)
        try:
            sel_id = int(sel_opt.split("|")[0].replace("ID:", "").strip())
            sel_row = df[df['id'] == sel_id].iloc[0]
            
            with st.form("edit_form"):
                c1, c2 = st.columns(2)
                with c1:
                    # Tarih Picker ile güncelleme
                    curr_date = pd.to_datetime(sel_row['period_date']).date()
                    n_date = st.date_input("Dönem", value=curr_date)
                    n_src = st.text_input("Kaynak", value=sel_row['source'])
                with c2:
                    n_txt = st.text_area("Metin", value=sel_row['text_content'], height=150)
                
                if st.form_submit_button("💾 Güncelle"):
                    # Tekrar analiz yap
                    v_sim = analyze_simple_dict(n_txt)
                    v_abg = analyze_apel_blix_grimaldi(n_txt)
                    v_fb, l_fb = analyze_with_finbert(n_txt)
                    
                    utils.update_entry(sel_id, n_date, n_txt, n_src, v_sim, v_abg, v_fb, l_fb)
                    st.success("Güncellendi!")
                    st.rerun()
                    
            if st.button("🗑️ Sil"):
                utils.delete_entry(sel_id)
                st.success("Silindi")
                st.rerun()
        except Exception as e:
            st.error(f"Seçim hatası: {e}")

# --- TAB 3: DASHBOARD ---
with tab3:
    if st.button("Grafikleri Getir / Yenile"):
        df_logs = utils.fetch_all_data()
        
        if not df_logs.empty:
            df_logs['period_date'] = pd.to_datetime(df_logs['period_date'])
            df_logs = df_logs.sort_values('period_date')
            
            min_date = df_logs['period_date'].min().date()
            max_date = datetime.date.today()
            
            st.info(f"Piyasa verileri (TCMB & BIS) çekiliyor... ({min_date} - {max_date})")

            # Utils'den veri çek
            df_market, error_msg = utils.fetch_market_data_adapter(min_date, max_date)
            
            if error_msg:
                st.warning(f"Uyarı: {error_msg}")
            
            # Merge
            merged_df = df_logs
            if not df_market.empty:
                if 'Tarih' in df_market.columns:
                     df_market['Tarih'] = pd.to_datetime(df_market['Tarih'])
                merged_df = pd.merge(df_logs, df_market, left_on='period_date', right_on='Tarih', how='left')

            # 1. GRAFİK (Çift Eksen)
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Sol Eksen: Skorlar
            fig.add_trace(go.Scatter(x=merged_df['period_date'], y=merged_df['score_finbert'], name="FinBERT (AI)", line=dict(color='blue')), secondary_y=False)
            fig.add_trace(go.Scatter(x=merged_df['period_date'], y=merged_df['score_abg'], name="Apel-Blix (Sözlük)", line=dict(color='green', dash='dot')), secondary_y=False)

            # Sağ Eksen: Piyasa
            if 'Yıllık TÜFE' in merged_df.columns:
                fig.add_trace(go.Scatter(x=merged_df['period_date'], y=merged_df['Yıllık TÜFE'], name="Yıllık TÜFE (%)", line=dict(color='red')), secondary_y=True)
            if 'PPK Faizi' in merged_df.columns:
                fig.add_trace(go.Scatter(x=merged_df['period_date'], y=merged_df['PPK Faizi'], name="Faiz (%)", line=dict(color='orange')), secondary_y=True)

            fig.update_layout(title_text="Analiz vs Piyasa", hovermode="x unified", height=500)
            fig.update_yaxes(title_text="<b>Şahin/Güvercin Skoru</b>", secondary_y=False, range=[-1.1, 1.1])
            fig.update_yaxes(title_text="<b>% Oran</b>", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # 2. TABLO (İstediğiniz Format)
            st.markdown("### 📋 Veri Detayları")
            
            # Sadece dolu olanları ve birleşenleri gösterelim
            cols = ['period_date', 'source', 'score_finbert', 'score_abg']
            if 'Aylık TÜFE' in merged_df.columns: cols.append('Aylık TÜFE')
            if 'Yıllık TÜFE' in merged_df.columns: cols.append('Yıllık TÜFE')
            if 'PPK Faizi' in merged_df.columns: cols.append('PPK Faizi')
            
            display_df = merged_df[cols].copy()
            # Tarihi string yapalım ki tabloda düzgün görünsün
            display_df['period_date'] = display_df['period_date'].dt.strftime('%d-%m-%Y')
            
            st.dataframe(
                display_df.style.format({
                    "score_finbert": "{:.2f}",
                    "score_abg": "{:.2f}",
                    "Aylık TÜFE": "{:.2f}%",
                    "Yıllık TÜFE": "{:.2f}%",
                    "PPK Faizi": "{:.2f}%"
                }, na_rep="-"),
                use_container_width=True,
                height=400
            )

        else:
            st.warning("Henüz metin analizi verisi yok.")
