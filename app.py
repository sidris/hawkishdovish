import streamlit as st
import pandas as pd
from supabase import create_client, Client
from transformers import pipeline
from collections import Counter
import re
import plotly.express as px

# -----------------------------------------------------------------------------
# 1. AYARLAR VE BAĞLANTILAR
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Piyasa Analiz Sistemi", layout="wide")

@st.cache_resource
def init_supabase():
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        return create_client(url, key)
    except KeyError:
        st.error("Supabase sırları bulunamadı.")
        return None

supabase = init_supabase()

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

# --- A. Basit Sözlük (Sizin ilk yönteminiz) ---
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

# --- B. Apel & Blix Grimaldi (Düzeltilmiş Kodunuz) ---
def analyze_apel_blix_grimaldi(text):
    text = text.lower()
    tokens = re.findall(r"[a-z']+", text)
    bigrams = [" ".join(pair) for pair in zip(tokens, tokens[1:])]
    
    token_counts = Counter(tokens)
    bigram_counts = Counter(bigrams)

    # Kelime Listeleri (Orijinal kodunuzdan)
    nouns = ["cost","costs","expenditures","consumption","growth","output","demand","activity","production","investment","productivity","labor","labour","job","jobs","participation","wage","wages","recovery","slowdown","contraction","expansion","cycle","conditions","credit","lending","borrowing","liquidity","stability","markets","volatility","uncertainty","risks","easing","rates","policy","stance","outlook","pressures","inflation","price", "prices","oil price", "oil prices","cyclical position","development","employment","unemployment","gold"]
    hawkish_adjectives = ["high", "higher","strong", "stronger","increasing", "increased","fast", "faster","elevated","rising","accelerating","robust","persistent","mounting","excessive","solid","resillent","vigorous","overheating","tightening","restrivtive","constrained","limited","upside","significant","notable"]
    dovish_adjectives = ["low", "lower","weak", "weaker","decreasing", "decreased","slow", "slower","falling","declining","subdued","soft","softer","easing","moderate","moderating","cooling","softening","downside","adverse"]
    
    # Tekil Kelimeler
    hawkish_single = {"tight","tightening","restrictive","elevated","high","overheating","pressures","pressure","risk","risks","upside","vigilant","decisive"}
    dovish_single = {"disinflation","decline","declining","fall","falling","decrease","decreasing","lower","low","subdued","contained","anchored","cooling","slow","slower","improvement","better","easing","relief"}

    # Setleri oluştur
    hawkish_phrases = {f"{adj} {noun}" for adj in hawkish_adjectives for noun in nouns}
    dovish_phrases = {f"{adj} {noun}" for adj in dovish_adjectives for noun in nouns}

    # Skorlama (Hata düzeltmesi: dove__pct yerine doğru hesaplama)
    hawk_bigram_count = sum(bigram_counts[p] for p in hawkish_phrases)
    dove_bigram_count = sum(bigram_counts[p] for p in dovish_phrases)
    
    hawk_single_count = sum(token_counts[w] for w in hawkish_single)
    dove_single_count = sum(token_counts[w] for w in dovish_single)

    hawk_total = hawk_bigram_count + hawk_single_count
    dove_total = dove_bigram_count + dove_single_count
    total_signal = hawk_total + dove_total

    if total_signal == 0:
        return 0
    
    # Skoru -1 (Güvercin) ile +1 (Şahin) arasına normalize edelim
    # Orijinal kodunuz yüzde kullanıyordu, grafikte çizmek için -1/+1 skalası daha iyidir.
    return (hawk_total - dove_total) / total_signal

# --- C. FinBERT (Yapay Zeka) ---
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
    # Tüm algoritmaları çalıştır
    s_simple = analyze_simple_dict(text)
    s_abg = analyze_apel_blix_grimaldi(text) # Yeni
    s_fb, l_fb = analyze_with_finbert(text)
    
    update_data = {
        "period_date": str(date), "text_content": text, "source": source,
        "score_dict": s_simple, "score_abg": s_abg, # Yeni
        "score_finbert": s_fb, "finbert_label": l_fb
    }
    supabase.table("market_logs").update(update_data).eq("id", record_id).execute()

# -----------------------------------------------------------------------------
# 4. ARAYÜZ
# -----------------------------------------------------------------------------

st.title("🦅 Şahin/Güvercin Analiz Paneli")
st.markdown("*Algoritma Karşılaştırmalı: Basit Sözlük vs. Apel-Blix-Grimaldi vs. FinBERT*")

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
                with st.spinner("3 farklı algoritma çalışıyor..."):
                    # Skorlar
                    val_simple = analyze_simple_dict(text_input)
                    val_abg = analyze_apel_blix_grimaldi(text_input) # Apel-Blix
                    val_fb, lab_fb = analyze_with_finbert(text_input)
                    
                    period_date = f"{year}-{month:02d}-01"
                    
                    data = {
                        "period_date": period_date, "text_content": text_input, "source": source,
                        "score_dict": val_simple, 
                        "score_abg": val_abg,  # Veritabanına yeni alan
                        "score_finbert": val_fb, "finbert_label": lab_fb
                    }
                    try:
                        supabase.table("market_logs").insert(data).execute()
                        st.success(f"✅ Kaydedildi! ABG Skoru: {val_abg:.2f}")
                    except Exception as e:
                        st.error(f"Kayıt Hatası (SQL kolonunu eklediniz mi?): {e}")

# --- TAB 2: DÜZENLEME ---
with tab2:
    df = fetch_all_data()
    if not df.empty:
        opts = df.apply(lambda x: f"ID: {x['id']} | {x['period_date']} | {x['source']}", axis=1)
        sel_opt = st.selectbox("Kayıt Seç:", opts)
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

# --- TAB 3: DASHBOARD ---
with tab3:
    if st.button("Grafikleri Yenile"):
        df = fetch_all_data()
        if not df.empty:
            df['period_date'] = pd.to_datetime(df['period_date'])
            
            # Grafik verisi hazırlığı (3 Algoritma)
            # Eğer eski kayıtlarda score_abg yoksa (NaN ise) 0 yapalım
            if 'score_abg' in df.columns:
                df['score_abg'] = df['score_abg'].fillna(0)
            else:
                df['score_abg'] = 0

            chart_df = df.melt(
                id_vars=['period_date', 'source'], 
                value_vars=['score_dict', 'score_abg', 'score_finbert'], 
                var_name='Algoritma', value_name='Skor'
            )
            
            # İsimlendirme
            names = {
                'score_dict': 'Basit Sözlük',
                'score_abg': 'Apel Blix Grimaldi (Klasik)',
                'score_finbert': 'FinBERT (Yapay Zeka)'
            }
            chart_df['Algoritma'] = chart_df['Algoritma'].map(names)
            
            fig = px.line(chart_df, x='period_date', y='Skor', color='Algoritma', 
                          title="Şahin/Güvercin Trend Analizi", markers=True)
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Nötr")
            fig.add_annotation(text="Şahin 🦅", xref="paper", yref="paper", x=0, y=0.95, showarrow=False)
            fig.add_annotation(text="Güvercin 🕊️", xref="paper", yref="paper", x=0, y=0.05, showarrow=False)
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Ham Veriler")
            st.dataframe(df)
        else:
            st.warning("Veri yok.")
