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

# --- SESSION STATE (DÜZENLEME MODU İÇİN) ---
if 'form_data' not in st.session_state:
    st.session_state['form_data'] = {
        'id': None,
        'date': datetime.date.today().replace(day=1),
        'source': "TCMB",
        'text': ""
    }

# --- MODELLER ---
@st.cache_resource
def load_models():
    try: return pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except: return None
classifier = load_models()

# --- ÖZELLEŞTİRİLMİŞ SÖZLÜKLER (SİZİN LİSTENİZ) ---
NOUNS = ["cost","costs","expenditures","consumption","growth","output","demand","activity","production","investment","productivity","labor","labour","job","jobs","participation","wage","wages","recovery","slowdown","contraction","expansion","cycle","conditions","credit","lending","borrowing","liquidity","stability","markets","volatility","uncertainty","risks","easing","rates","policy","stance","outlook","pressures","inflation","price", "prices","wage", "wages","oil price", "oil prices","cyclical position","growth","development","employment","unemployment","recovery","cost", "costs","gold"]

HAWKISH_ADJ = ["high", "higher","strong", "stronger","increasing", "increased","fast", "faster","elevated","rising","accelerating","robust","persistent","mounting","excessive","solid","resillent","vigorous","overheating","tightening","restrivtive","constrained","limited","upside","significant","notable"]

DOVISH_ADJ = ["low", "lower","weak", "weaker","decreasing", "decreased","slow", "slower","falling","declining","subdued","weak","weaker","soft","softer","easing","slow","slower","moderate","moderating","cooling","softening","downside","adverse"]

HAWKISH_SINGLE = {"tight","tightening","restrictive","elevated","high","overheating","pressures","pressure","risk","risks","upside","vigilant","decisive"}

DOVISH_SINGLE = {"disinflation","decline","declining","fall","falling","decrease","decreasing","lower","low","subdued","contained","anchored","cooling","slow","slower","improvement","improvement","better","easing","relief"}

# Setleri Hazırla (Performans için)
HAWKISH_PHRASES = {f"{adj} {noun}" for adj in HAWKISH_ADJ for noun in NOUNS}
DOVISH_PHRASES = {f"{adj} {noun}" for adj in DOVISH_ADJ for noun in NOUNS}

# --- ANALİZ FONKSİYONLARI ---
def analyze_apel_blix_detailed(text):
    text = text.lower()
    # Noktalama işaretlerini temizle
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    
    bigrams = [" ".join(pair) for pair in zip(tokens, tokens[1:])]
    token_counts = Counter(tokens)
    bigram_counts = Counter(bigrams)

    found_hawkish = []
    found_dovish = []

    h_score = 0; d_score = 0

    # Bigram Kontrolü
    for p in HAWKISH_PHRASES:
        if bigram_counts[p] > 0:
            count = bigram_counts[p]
            h_score += count
            found_hawkish.append(f"{p} ({count})")
            
    for p in DOVISH_PHRASES:
        if bigram_counts[p] > 0:
            count = bigram_counts[p]
            d_score += count
            found_dovish.append(f"{p} ({count})")

    # Unigram Kontrolü
    for w in HAWKISH_SINGLE:
        if token_counts[w] > 0:
            count = token_counts[w]
            h_score += count
            found_hawkish.append(f"{w} ({count})")
            
    for w in DOVISH_SINGLE:
        if token_counts[w] > 0:
            count = token_counts[w]
            d_score += count
            found_dovish.append(f"{w} ({count})")

    total = h_score + d_score
    final_score = (h_score - d_score) / total if total > 0 else 0
    
    return final_score, found_hawkish, found_dovish

def analyze_finbert(text):
    if not classifier: return 0, "neutral"
    # FinBERT max 512 token kabul eder
    res = classifier(text[:512])[0]
    score = res['score'] if res['label'] == "positive" else -res['score'] if res['label'] == "negative" else 0
    return score, res['label']

# --- ARAYÜZ BAŞLIYOR ---
st.title("🦅 Şahin/Güvercin Analiz Paneli")

# TABLARI OLUŞTUR
tab1, tab2, tab3 = st.tabs(["📈 Dashboard", "📝 Veri Girişi & Yönetimi", "📊 Piyasa Verileri"])

# ==============================================================================
# TAB 1: DASHBOARD (OTOMATİK YÜKLENİR)
# ==============================================================================
with tab1:
    # Verileri direkt çek (Butonsuz)
    with st.spinner("Dashboard güncelleniyor..."):
        df_logs = utils.fetch_all_data()
        
    if not df_logs.empty:
        # Tarih İşlemleri
        df_logs['period_date'] = pd.to_datetime(df_logs['period_date'])
        df_logs['Donem'] = df_logs['period_date'].dt.strftime('%Y-%m')
        
        # Piyasa Verisi (Otomatik tarih aralığı)
        min_d = df_logs['period_date'].min().date()
        max_d = datetime.date.today()
        
        # utils.py cache kullandığı için hızlıdır
        df_market, err = utils.fetch_market_data_adapter(min_d, max_d)
        
        # Birleştirme
        merged = pd.merge(df_logs, df_market, on="Donem", how="left")
        merged = merged.sort_values("period_date")
        
        # --- GRAFİK ---
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Sol Eksen: Skorlar
        fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['score_finbert'], name="FinBERT (AI)", line=dict(color='blue')), secondary_y=False)
        fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['score_abg'], name="Apel-Blix (Sözlük)", line=dict(color='green', dash='dot')), secondary_y=False)
        
        # Sağ Eksen: Piyasa
        if 'Yıllık TÜFE' in merged.columns:
            fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['Yıllık TÜFE'], name="Yıllık TÜFE (%)", line=dict(color='red')), secondary_y=True)
        if 'PPK Faizi' in merged.columns:
                fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['PPK Faizi'], name="Faiz (%)", line=dict(color='orange')), secondary_y=True)

        fig.update_layout(title="Metin Analizi ve Piyasa Göstergeleri", hovermode="x unified", height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Yenileme butonu (Opsiyonel, manuel tetiklemek için)
        if st.button("🔄 Verileri Tazele"):
            st.cache_data.clear()
            st.rerun()
    else:
        st.info("Henüz analiz verisi yok. 'Veri Girişi' sekmesinden ekleme yapın.")

# ==============================================================================
# TAB 2: VERİ GİRİŞİ & YÖNETİMİ
# ==============================================================================
with tab2:
    st.subheader("Veri İşlemleri")
    
    # Tüm verileri çek (Çakışma kontrolü ve liste için)
    df_all = utils.fetch_all_data()
    if not df_all.empty:
        df_all['period_date'] = pd.to_datetime(df_all['period_date'])
        
    # Session State'den verileri al
    current_id = st.session_state['form_data']['id']
    
    with st.container(border=True):
        c1, c2 = st.columns([1, 2])
        with c1:
            val_date = st.session_state['form_data']['date']
            selected_date = st.date_input("Tarih", value=val_date)
            
            val_source = st.session_state['form_data']['source']
            source = st.text_input("Kaynak", value=val_source)
            
            # Dönem Bilgisi
            current_period_str = selected_date.strftime('%Y-%m-%d')
            st.caption(f"Kayıt Dönemi: **{current_period_str}**")
            
        with c2:
            val_text = st.session_state['form_data']['text']
            txt = st.text_area("Metin", value=val_text, height=200, placeholder="Metni buraya yapıştırın...")
        
        # --- BUTON MANTIĞI ---
        col_btn1, col_btn2 = st.columns([1, 5])
        
        with col_btn1:
            if st.button("💾 Kaydet / Analiz Et", type="primary"):
                if txt:
                    # 1. Analiz Yap
                    s_abg, hawks, doves = analyze_apel_blix_detailed(txt)
                    s_fb, l_fb = analyze_finbert(txt)
                    
                    # 2. Çakışma Kontrolü (Overwrite Logic)
                    existing_record = None
                    if not df_all.empty:
                        # Seçilen tarih veritabanında var mı?
                        mask = df_all['period_date'] == pd.to_datetime(selected_date)
                        if mask.any():
                            existing_record = df_all[mask].iloc[0]
                    
                    if existing_record:
                        # VARSA GÜNCELLE
                        target_id = int(existing_record['id'])
                        st.warning(f"⚠️ {selected_date} tarihine ait kayıt bulundu. Mevcut kayıt güncellendi.")
                        utils.update_entry(target_id, selected_date, txt, source, s_abg, s_abg, s_fb, l_fb)
                    elif current_id:
                        # LİSTEDEN SEÇİLDİYSE GÜNCELLE
                        utils.update_entry(current_id, selected_date, txt, source, s_abg, s_abg, s_fb, l_fb)
                        st.success("Kayıt güncellendi!")
                    else:
                        # YOKSA YENİ EKLE
                        utils.insert_entry(selected_date, txt, source, s_abg, s_abg, s_fb, l_fb)
                        st.success("Yeni kayıt eklendi!")
                    
                    # Formu sıfırla
                    st.session_state['form_data'] = {'id': None, 'date': datetime.date.today(), 'source': "TCMB", 'text': ""}
                    st.rerun()
                else:
                    st.error("Metin alanı boş olamaz.")

        with col_btn2:
            if st.button("Temizle"):
                st.session_state['form_data'] = {'id': None, 'date': datetime.date.today(), 'source': "TCMB", 'text': ""}
                st.rerun()

        # --- OTOMATİK KELİME GÖSTERİMİ ---
        # Metin kutusu dolu olduğu sürece burası çalışır (Seçim yapılınca otomatik çalışır)
        if txt:
            s_abg_live, hawks_live, doves_live = analyze_apel_blix_detailed(txt)
            
            st.markdown("---")
            st.info(f"**Analiz Önizlemesi:** Apel-Blix Skoru: `{s_abg_live:.2f}`")
            
            exp_res = st.expander("🔍 Tespit Edilen Kelimeler", expanded=True)
            with exp_res:
                k1, k2 = st.columns(2)
                with k1:
                    st.markdown(f"**🦅 Şahin ({len(hawks_live)})**")
                    if hawks_live:
                        for w in hawks_live: st.write(f"- {w}")
                    else: st.caption("-")
                with k2:
                    st.markdown(f"**🕊️ Güvercin ({len(doves_live)})**")
                    if doves_live:
                        for w in doves_live: st.write(f"- {w}")
                    else: st.caption("-")

    # ---------------------------------------------------------
    # GEÇMİŞ KAYITLAR LİSTESİ
    # ---------------------------------------------------------
    st.markdown("### 📋 Geçmiş Kayıtlar")
    
    if not df_all.empty:
        # Görüntüleme DF
        df_display = df_all.copy()
        df_display['Dönem'] = df_display['period_date'].dt.strftime('%Y-%m')
        
        grid_df = df_display[['id', 'Dönem', 'period_date', 'source', 'score_abg']].sort_values('period_date', ascending=False)
        
        # Seçilebilir Tablo
        event = st.dataframe(
            grid_df,
            on_select="rerun",
            selection_mode="single-row",
            use_container_width=True,
            hide_index=True,
            column_config={
                "period_date": st.column_config.DateColumn("Tam Tarih", format="DD.MM.YYYY"),
                "score_abg": st.column_config.ProgressColumn("Skor", min_value=-1, max_value=1, format="%.2f"),
                "id": st.column_config.NumberColumn("ID", width="small")
            }
        )

        # SEÇİM YAPILINCA FORMU DOLDUR
        if len(event.selection.rows) > 0:
            sel_idx = event.selection.rows[0]
            sel_id = grid_df.iloc[sel_idx]['id']
            
            # State güncelle (Sadece ID değiştiyse)
            if st.session_state['form_data']['id'] != sel_id:
                orig = df_all[df_all['id'] == sel_id].iloc[0]
                st.session_state['form_data'] = {
                    'id': int(orig['id']),
                    'date': pd.to_datetime(orig['period_date']).date(),
                    'source': orig['source'],
                    'text': orig['text_content']
                }
                st.rerun()
    else:
        st.write("Kayıt yok.")

# ==============================================================================
# TAB 3: PİYASA VERİLERİ
# ==============================================================================
with tab3:
    st.header("Piyasa Verileri")
    col_d1, col_d2 = st.columns(2)
    d1 = col_d1.date_input("Başlangıç", datetime.date(2023, 1, 1))
    d2 = col_d2.date_input("Bitiş", datetime.date.today())
    
    if st.button("Verileri Getir", type="primary"):
        df, err = utils.fetch_market_data_adapter(d1, d2)
        if not df.empty:
            # Grafik
            fig_m = go.Figure()
            if 'Yıllık TÜFE' in df.columns:
                fig_m.add_trace(go.Scatter(x=df['Donem'], y=df['Yıllık TÜFE'], name="Yıllık TÜFE (%)", line=dict(color='red')))
            if 'PPK Faizi' in df.columns:
                fig_m.add_trace(go.Scatter(x=df['Donem'], y=df['PPK Faizi'], name="Faiz (%)", line=dict(color='orange')))
            
            fig_m.update_layout(title="Piyasa Görünümü", xaxis_title="Dönem", yaxis_title="Değer (%)")
            st.plotly_chart(fig_m, use_container_width=True)
            st.dataframe(df)
        else:
            st.error(f"Veri yok: {err}")
