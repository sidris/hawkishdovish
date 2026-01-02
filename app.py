import streamlit as st
import pandas as pd
import datetime  # <--- EKSİK OLAN BU SATIR EKLENDİ
from supabase import create_client, Client
from transformers import pipeline
from collections import Counter
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import utils  # utils.py dosyanızın yanına olduğundan emin olun

# -----------------------------------------------------------------------------
# 1. AYARLAR VE BAĞLANTILAR
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Piyasa Analiz Sistemi", layout="wide")

# Supabase Bağlantısı
@st.cache_resource
def init_supabase():
    try:
        # Hem local (secrets.toml) hem cloud (st.secrets) uyumlu
        if "supabase" in st.secrets:
            url = st.secrets["supabase"]["url"]
            key = st.secrets["supabase"]["key"]
        else:
            url = st.secrets["SUPABASE_URL"]
            key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception as e:
        st.error(f"Supabase bağlantı hatası: {e}")
        return None

supabase = init_supabase()

# FinBERT Modelini Yükle
@st.cache_resource
def load_finbert():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

try:
    classifier = load_finbert()
except:
    classifier = None

# -----------------------------------------------------------------------------
# 2. ALGORİTMALAR (METİN ANALİZİ)
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
# 3. VERİTABANI İŞLEMLERİ
# -----------------------------------------------------------------------------

def fetch_all_data():
    response = supabase.table("market_logs").select("*").order("period_date", desc=True).execute()
    return pd.DataFrame(response.data)

def delete_entry(record_id):
    supabase.table("market_logs").delete().eq("id", record_id).execute()

def update_entry(record_id, date, text, source):
    s_simple = analyze_simple_dict(text)
    s_abg = analyze_apel_blix_grimaldi(text)
    s_fb, l_fb = analyze_with_finbert(text)
    
    update_data = {
        "period_date": str(date), "text_content": text, "source": source,
        "score_dict": s_simple, "score_abg": s_abg, 
        "score_finbert": s_fb, "finbert_label": l_fb
    }
    supabase.table("market_logs").update(update_data).eq("id", record_id).execute()

# -----------------------------------------------------------------------------
# 4. ARAYÜZ
# -----------------------------------------------------------------------------

st.title("🦅 Şahin/Güvercin Analiz Paneli")
st.markdown("*Metin Analizi ve Gerçek Piyasa Verileri (TCMB & BIS)*")

tab1, tab2, tab3 = st.tabs(["📝 Veri Girişi", "✏️ Düzenle/Sil", "📈 Dashboard"])

# --- TAB 1: VERİ GİRİŞİ ---
with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Dönem")
        year = st.selectbox("Yıl", range(2020, 2030), index=5)
        month = st.selectbox("Ay", range(1, 13))
        source = st.text_input("Kaynak", "TCMB")
    with col2:
        text_input = st.text_area("Metin", height=200)
        if st.button("Kaydet ve Analiz Et", type="primary"):
            if text_input:
                with st.spinner("Analiz ediliyor..."):
                    val_simple = analyze_simple_dict(text_input)
                    val_abg = analyze_apel_blix_grimaldi(text_input)
                    val_fb, lab_fb = analyze_with_finbert(text_input)
                    
                    period_date = f"{year}-{month:02d}-01"
                    
                    data = {
                        "period_date": period_date, "text_content": text_input, "source": source,
                        "score_dict": val_simple, "score_abg": val_abg,
                        "score_finbert": val_fb, "finbert_label": lab_fb
                    }
                    try:
                        supabase.table("market_logs").insert(data).execute()
                        st.success(f"✅ Kaydedildi!")
                    except Exception as e:
                        st.error(f"Kayıt Hatası: {e}")

# --- TAB 2: DÜZENLEME ---
with tab2:
    df = fetch_all_data()
    if not df.empty:
        opts = df.apply(lambda x: f"ID: {x['id']} | {x['period_date']} | {x['source']}", axis=1)
        sel_opt = st.selectbox("Kayıt Seç:", opts)
        try:
            sel_id = int(sel_opt.split("|")[0].replace("ID:", "").strip())
            sel_row = df[df['id'] == sel_id].iloc[0]
            
            with st.form("edit_form"):
                c1, c2 = st.columns(2)
                with c1:
                    n_date = st.date_input("Dönem", value=pd.to_datetime(sel_row['period_date']).date())
                    n_src = st.text_input("Kaynak", value=sel_row['source'])
                with c2:
                    n_txt = st.text_area("Metin", value=sel_row['text_content'], height=150)
                
                if st.form_submit_button("💾 Güncelle"):
                    update_entry(sel_id, n_date, n_txt, n_src)
                    st.success("Güncellendi!")
                    st.rerun()
                    
            if st.button("🗑️ Sil"):
                delete_entry(sel_id)
                st.success("Silindi")
                st.rerun()
        except Exception as e:
            st.error(f"Seçim hatası: {e}")

# --- TAB 3: DASHBOARD (CANLI VERİ ENTEGRASYONU) ---
with tab3:
    if st.button("Grafikleri Getir / Yenile"):
        # 1. Metin Analiz Verilerini Çek (Supabase)
        df_logs = fetch_all_data()
        
        if not df_logs.empty:
            df_logs['period_date'] = pd.to_datetime(df_logs['period_date'])
            df_logs = df_logs.sort_values('period_date')
            
            # 2. Tarih Aralığını Belirle
            min_date = df_logs['period_date'].min().date()
            max_date = datetime.date.today() # ARTIK HATA VERMEYECEK
            
            st.info(f"Piyasa verileri çekiliyor: {min_date} - {max_date} arası...")

            # 3. UTILS.PY İLE GERÇEK VERİLERİ ÇEK
            df_market, error_msg = utils.fetch_market_data_adapter(min_date, max_date)
            
            if error_msg:
                st.warning(f"Piyasa verileri tam çekilemedi: {error_msg}")
            
            if df_market.empty:
                st.error("Piyasa verisi bulunamadı veya EVDS Key eksik.")
            else:
                # 4. Verileri Birleştir (Merge)
                if 'Tarih' in df_market.columns:
                     df_market['Tarih'] = pd.to_datetime(df_market['Tarih'])
                
                # İki tabloyu birleştir
                merged_df = pd.merge(df_logs, df_market, left_on='period_date', right_on='Tarih', how='left')

                # 5. GRAFİK OLUŞTUR (ÇİFT EKSEN)
                fig = make_subplots(specs=[[{"secondary_y": True}]])

                # --- SOL EKSEN (METİN SKORLARI) ---
                fig.add_trace(go.Scatter(x=merged_df['period_date'], y=merged_df['score_finbert'], name="FinBERT (AI)", line=dict(color='blue')), secondary_y=False)
                fig.add_trace(go.Scatter(x=merged_df['period_date'], y=merged_df['score_abg'], name="Apel-Blix (Sözlük)", line=dict(color='green', dash='dot')), secondary_y=False)

                # --- SAĞ EKSEN (TCMB/BIS VERİLERİ) ---
                if 'Yıllık TÜFE' in merged_df.columns:
                    fig.add_trace(go.Scatter(x=merged_df['period_date'], y=merged_df['Yıllık TÜFE'], name="Yıllık TÜFE (%)", line=dict(color='red')), secondary_y=True)
                
                if 'PPK Faizi' in merged_df.columns:
                    fig.add_trace(go.Scatter(x=merged_df['period_date'], y=merged_df['PPK Faizi'], name="Faiz (%)", line=dict(color='orange')), secondary_y=True)

                # --- AYARLAR ---
                fig.update_layout(title_text="Metin Analizi vs Piyasa Göstergeleri (Canlı Veri)", hovermode="x unified")
                fig.update_yaxes(title_text="<b>Şahin/Güvercin Skoru</b>", secondary_y=False, range=[-1.1, 1.1])
                fig.update_yaxes(title_text="<b>Enflasyon / Faiz (%)</b>", secondary_y=True)

                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### Birleştirilmiş Veri Seti")
                cols_to_show = ['period_date', 'source', 'score_finbert', 'score_abg']
                if 'Yıllık TÜFE' in merged_df.columns: cols_to_show.append('Yıllık TÜFE')
                if 'PPK Faizi' in merged_df.columns: cols_to_show.append('PPK Faizi')
                
                st.dataframe(merged_df[cols_to_show])
        else:
            st.warning("Henüz metin analizi verisi yok. Lütfen 'Veri Girişi' sekmesinden veri ekleyin.")
