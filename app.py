import streamlit as st
import pandas as pd
import datetime
from transformers import pipeline
from collections import Counter
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import utils  # utils.py dosyanızın aynı klasörde olduğundan emin olun

# -----------------------------------------------------------------------------
# 1. AYARLAR VE MODEL YÜKLEME
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Piyasa Analiz Sistemi", layout="wide")

@st.cache_resource
def load_finbert():
    """FinBERT modelini önbelleğe alır."""
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

try:
    classifier = load_finbert()
except Exception as e:
    st.warning(f"Model yüklenirken gecikme olabilir: {e}")
    classifier = None

# -----------------------------------------------------------------------------
# 2. ALGORİTMALAR (METİN ANALİZİ)
# -----------------------------------------------------------------------------

def analyze_simple_dict(text):
    """Basit Kelime Sayma Yöntemi"""
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
    """Apel & Blix Grimaldi Metodolojisi (Bigram + Unigram)"""
    text = text.lower()
    tokens = re.findall(r"[a-z']+", text)
    bigrams = [" ".join(pair) for pair in zip(tokens, tokens[1:])]
    token_counts = Counter(tokens)
    bigram_counts = Counter(bigrams)

    # Sözlük Tanımları
    nouns = ["cost","costs","expenditures","consumption","growth","output","demand","activity","production","investment","productivity","labor","labour","job","jobs","participation","wage","wages","recovery","slowdown","contraction","expansion","cycle","conditions","credit","lending","borrowing","liquidity","stability","markets","volatility","uncertainty","risks","easing","rates","policy","stance","outlook","pressures","inflation","price", "prices","oil price", "oil prices","cyclical position","development","employment","unemployment","gold"]
    hawkish_adjectives = ["high", "higher","strong", "stronger","increasing", "increased","fast", "faster","elevated","rising","accelerating","robust","persistent","mounting","excessive","solid","resillent","vigorous","overheating","tightening","restrivtive","constrained","limited","upside","significant","notable"]
    dovish_adjectives = ["low", "lower","weak", "weaker","decreasing", "decreased","slow", "slower","falling","declining","subdued","soft","softer","easing","moderate","moderating","cooling","softening","downside","adverse"]
    hawkish_single = {"tight","tightening","restrictive","elevated","high","overheating","pressures","pressure","risk","risks","upside","vigilant","decisive"}
    dovish_single = {"disinflation","decline","declining","fall","falling","decrease","decreasing","lower","low","subdued","contained","anchored","cooling","slow","slower","improvement","better","easing","relief"}

    hawkish_phrases = {f"{adj} {noun}" for adj in hawkish_adjectives for noun in nouns}
    dovish_phrases = {f"{adj} {noun}" for adj in dovish_adjectives for noun in nouns}

    # Hesaplama
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
    """Yapay Zeka (FinBERT) Analizi"""
    if not classifier: return 0, "neutral"
    # Max 512 karakter (Demo için kırpıyoruz, normalde chunking yapılır)
    res = classifier(text[:512])[0]
    score = res['score']
    label = res['label']
    # Skoru -1 ile +1 arasına normalize et
    final_score = score if label == "positive" else -score if label == "negative" else 0
    return final_score, label

# -----------------------------------------------------------------------------
# 3. ARAYÜZ YAPISI
# -----------------------------------------------------------------------------

st.title("🦅 Şahin/Güvercin Analiz Paneli")
st.markdown("*Merkez Bankası Metin Analizi ve Piyasa Verileri Entegrasyonu*")

tab1, tab2, tab3 = st.tabs(["📝 Veri Girişi", "✏️ Düzenle/Sil", "📈 Dashboard"])

# --- TAB 1: VERİ GİRİŞİ (DATE PICKER EKLENDİ) ---
with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Dönem Seçimi")
        
        # YENİ: Date Picker (Varsayılan olarak bugünün ayının 1'i)
        default_date = datetime.date.today().replace(day=1)
        selected_date = st.date_input("Dönem Tarihi", value=default_date, format="DD/MM/YYYY")
        
        source = st.text_input("Kaynak", "TCMB PPK Özeti")
        
    with col2:
        text_input = st.text_area("Analiz Edilecek Metin", height=200, placeholder="Metni buraya yapıştırın...")
        
        if st.button("Kaydet ve Analiz Et", type="primary"):
            if text_input:
                with st.spinner("Algoritmalar çalışıyor..."):
                    # 1. Analizleri Yap
                    val_simple = analyze_simple_dict(text_input)
                    val_abg = analyze_apel_blix_grimaldi(text_input)
                    val_fb, lab_fb = analyze_with_finbert(text_input)
                    
                    # 2. Tarihi veritabanı formatına çevir (Ayın 1'ine sabitlemek mantıklıdır)
                    period_date = selected_date.replace(day=1)
                    
                    # 3. utils üzerinden kaydet
                    utils.insert_entry(period_date, text_input, source, val_simple, val_abg, val_fb, lab_fb)
                    st.success(f"✅ Başarıyla Kaydedildi! Dönem: {period_date}")
            else:
                st.warning("Lütfen bir metin giriniz.")

# --- TAB 2: DÜZENLEME VE SİLME ---
with tab2:
    st.header("Kayıt Yönetimi")
    df = utils.fetch_all_data()
    
    if not df.empty:
        # Seçim Kutusu için format
        opts = df.apply(lambda x: f"ID: {x['id']} | {x['period_date']} | {x['source']}", axis=1)
        sel_opt = st.selectbox("Düzenlenecek Kaydı Seçin:", opts)
        
        try:
            # ID'yi ayıkla
            sel_id = int(sel_opt.split("|")[0].replace("ID:", "").strip())
            sel_row = df[df['id'] == sel_id].iloc[0]
            
            with st.form("edit_form"):
                c1, c2 = st.columns(2)
                with c1:
                    # Mevcut tarihi date objesine çevir
                    curr_date_val = pd.to_datetime(sel_row['period_date']).date()
                    n_date = st.date_input("Dönem", value=curr_date_val)
                    n_src = st.text_input("Kaynak", value=sel_row['source'])
                with c2:
                    n_txt = st.text_area("Metin", value=sel_row['text_content'], height=150)
                
                if st.form_submit_button("💾 Değişiklikleri Kaydet"):
                    with st.spinner("Yeniden hesaplanıyor..."):
                        # Metin değiştiği için tekrar analiz etmeliyiz
                        v_sim = analyze_simple_dict(n_txt)
                        v_abg = analyze_apel_blix_grimaldi(n_txt)
                        v_fb, l_fb = analyze_with_finbert(n_txt)
                        
                        utils.update_entry(sel_id, n_date, n_txt, n_src, v_sim, v_abg, v_fb, l_fb)
                        st.success("Kayıt güncellendi!")
                        st.rerun()
                        
            # Silme Butonu (Form dışında)
            col_del, _ = st.columns([1, 4])
            with col_del:
                if st.button("🗑️ Bu Kaydı Sil"):
                    utils.delete_entry(sel_id)
                    st.success("Kayıt silindi.")
                    st.rerun()
                    
        except Exception as e:
            st.error(f"Seçim işlemi sırasında hata: {e}")
    else:
        st.info("Düzenlenecek kayıt bulunamadı.")

# --- TAB 3: DASHBOARD (GRAFİK VE TABLO) ---
with tab3:
    st.header("Analiz Sonuçları ve Piyasa Verileri")
    
    if st.button("Grafikleri Getir / Yenile"):
        # 1. Metin Verilerini Çek
        df_logs = utils.fetch_all_data()
        
        if not df_logs.empty:
            df_logs['period_date'] = pd.to_datetime(df_logs['period_date'])
            df_logs = df_logs.sort_values('period_date')
            
            # 2. Tarih Aralığını Belirle (En eski kayıttan bugüne)
            min_date = df_logs['period_date'].min().date()
            max_date = datetime.date.today()
            
            st.info(f"Piyasa verileri (EVDS & BIS) çekiliyor... Tarih Aralığı: {min_date} - {max_date}")

            # 3. utils.py üzerinden piyasa verilerini çek
            df_market, error_msg = utils.fetch_market_data_adapter(min_date, max_date)
            
            if error_msg:
                st.warning(f"⚠️ Piyasa Verisi Uyarısı: {error_msg}")
            
            # 4. Verileri Birleştir (Merge)
            merged_df = df_logs.copy()
            if not df_market.empty:
                # Merge işlemi için tarih formatlarını eşitle
                if 'Tarih' in df_market.columns:
                     df_market['Tarih'] = pd.to_datetime(df_market['Tarih'])
                
                merged_df = pd.merge(df_logs, df_market, left_on='period_date', right_on='Tarih', how='left')

            # 5. Çift Eksenli Grafik Oluştur
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Sol Eksen: Skorlar (-1 ile +1 arası)
            fig.add_trace(go.Scatter(x=merged_df['period_date'], y=merged_df['score_finbert'], name="FinBERT (AI)", line=dict(color='blue', width=2)), secondary_y=False)
            fig.add_trace(go.Scatter(x=merged_df['period_date'], y=merged_df['score_abg'], name="Apel-Blix (Sözlük)", line=dict(color='green', dash='dot')), secondary_y=False)

            # Sağ Eksen: Piyasa Verileri (% Değerler)
            # Sadece veri varsa çizdir, yoksa hata vermesin
            if 'Yıllık TÜFE' in merged_df.columns:
                fig.add_trace(go.Scatter(x=merged_df['period_date'], y=merged_df['Yıllık TÜFE'], name="Yıllık TÜFE (%)", line=dict(color='red')), secondary_y=True)
            if 'PPK Faizi' in merged_df.columns:
                fig.add_trace(go.Scatter(x=merged_df['period_date'], y=merged_df['PPK Faizi'], name="Faiz (%)", line=dict(color='orange')), secondary_y=True)

            # Grafik Ayarları
            fig.update_layout(title_text="Metin Analizi vs. Enflasyon & Faiz", hovermode="x unified", height=500)
            fig.update_yaxes(title_text="<b>Şahin/Güvercin Skoru</b>", secondary_y=False, range=[-1.1, 1.1])
            fig.update_yaxes(title_text="<b>Ekonomik Göstergeler (%)</b>", secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 6. Detaylı Veri Tablosu
            st.markdown("### 📋 Veri Detayları")
            
            # Tabloda gösterilecek kolonları seç
            cols_to_show = ['period_date', 'source', 'score_finbert', 'score_abg']
            if 'Yıllık TÜFE' in merged_df.columns: cols_to_show.append('Yıllık TÜFE')
            if 'PPK Faizi' in merged_df.columns: cols_to_show.append('PPK Faizi')
            
            # Tabloyu formatla (Okunabilirlik için)
            display_df = merged_df[cols_to_show].copy()
            display_df['period_date'] = display_df['period_date'].dt.strftime('%d-%m-%Y') # Tarihi düzgün göster
            
            st.dataframe(
                display_df.style.format({
                    "score_finbert": "{:.2f}",
                    "score_abg": "{:.2f}",
                    "Yıllık TÜFE": "{:.2f}%",
                    "PPK Faizi": "{:.2f}%"
                }, na_rep="-"), # Veri yoksa tire koy
                use_container_width=True
            )

        else:
            st.warning("Henüz hiç metin analizi kaydı yok. Lütfen 'Veri Girişi' sekmesinden veri ekleyin.")
