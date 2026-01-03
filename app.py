import streamlit as st
import pandas as pd
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import pipeline
import utils 

st.set_page_config(page_title="Piyasa Analiz", layout="wide")

# --- SESSION STATE ---
if 'form_data' not in st.session_state:
    st.session_state['form_data'] = {
        'id': None,
        'date': datetime.date.today().replace(day=1),
        'source': "TCMB",
        'text': ""
    }

# --- AI MODELİ (FINBERT) ---
@st.cache_resource
def load_models():
    try: return pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except: return None
classifier = load_models()

def analyze_finbert(text):
    if not classifier: return 0, "neutral"
    res = classifier(text[:512])[0]
    score = res['score'] if res['label'] == "positive" else -res['score'] if res['label'] == "negative" else 0
    return score, res['label']

# --- ARAYÜZ ---
st.title("🦅 Şahin/Güvercin Analiz Paneli")

tab1, tab2, tab3 = st.tabs(["📈 Dashboard", "📝 Veri Girişi & Yönetimi", "📊 Piyasa Verileri"])

# ==============================================================================
# TAB 1: DASHBOARD
# ==============================================================================
with tab1:
    with st.spinner("Veriler yükleniyor..."):
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
        
        # FinBERT
        fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['score_finbert'], name="FinBERT (AI)", line=dict(color='blue')), secondary_y=False)
        # Gelişmiş ABG
        fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['score_abg'], name="ABG (Gelişmiş Algoritma)", line=dict(color='green', dash='dot')), secondary_y=False)
        
        if 'Yıllık TÜFE' in merged.columns:
            fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['Yıllık TÜFE'], name="Yıllık TÜFE (%)", line=dict(color='red')), secondary_y=True)
        if 'PPK Faizi' in merged.columns:
                fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['PPK Faizi'], name="Faiz (%)", line=dict(color='orange')), secondary_y=True)

        fig.update_layout(title="Metin Analizi ve Ekonomi", hovermode="x unified", height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        if st.button("🔄 Yenile"):
            st.cache_data.clear()
            st.rerun()
    else:
        st.info("Kayıt yok.")

# ==============================================================================
# TAB 2: VERİ GİRİŞİ & YÖNETİMİ
# ==============================================================================
with tab2:
    st.subheader("Veri İşlemleri")
    df_all = utils.fetch_all_data()
    if not df_all.empty: df_all['period_date'] = pd.to_datetime(df_all['period_date'])
    
    current_id = st.session_state['form_data']['id']
    
    with st.container(border=True):
        c1, c2 = st.columns([1, 2])
        with c1:
            val_date = st.session_state['form_data']['date']
            selected_date = st.date_input("Tarih", value=val_date)
            val_source = st.session_state['form_data']['source']
            source = st.text_input("Kaynak", value=val_source)
            st.caption(f"Dönem: **{selected_date.strftime('%Y-%m')}**")
            
        with c2:
            val_text = st.session_state['form_data']['text']
            txt = st.text_area("Metin", value=val_text, height=200)
        
        # BUTONLAR
        col_b1, col_b2, col_b3 = st.columns([2, 1, 1])
        
        with col_b1:
            if st.button("💾 Kaydet / Analiz Et", type="primary"):
                if txt:
                    # 1. YENİ GELİŞMİŞ ALGORİTMA İLE ANALİZ
                    s_abg, hawks, doves = utils.run_full_analysis(txt)
                    s_fb, l_fb = analyze_finbert(txt)
                    
                    # 2. Kayıt Mantığı
                    if current_id:
                        utils.update_entry(current_id, selected_date, txt, source, s_abg, s_abg, s_fb, l_fb)
                        st.success("Güncellendi!")
                    else:
                        # Tarih çakışması kontrolü
                        existing = None
                        if not df_all.empty:
                            mask = df_all['period_date'] == pd.to_datetime(selected_date)
                            if mask.any(): existing = df_all[mask].iloc[0]
                        
                        if existing:
                            utils.update_entry(int(existing['id']), selected_date, txt, source, s_abg, s_abg, s_fb, l_fb)
                            st.warning("Aynı tarihte kayıt vardı, üzerine yazıldı.")
                        else:
                            utils.insert_entry(selected_date, txt, source, s_abg, s_abg, s_fb, l_fb)
                            st.success("Yeni kayıt eklendi!")
                    
                    st.session_state['form_data'] = {'id': None, 'date': datetime.date.today(), 'source': "TCMB", 'text': ""}
                    st.rerun()

        with col_b2:
            if st.button("Temizle"):
                st.session_state['form_data'] = {'id': None, 'date': datetime.date.today(), 'source': "TCMB", 'text': ""}
                st.rerun()

        with col_b3:
            # SİLME BUTONU
            if current_id:
                if st.button("🗑️ Sil", type="primary"):
                    utils.delete_entry(current_id)
                    st.success("Silindi!")
                    st.session_state['form_data'] = {'id': None, 'date': datetime.date.today(), 'source': "TCMB", 'text': ""}
                    st.rerun()

        # CANLI ANALİZ GÖSTERİMİ
        if txt:
            s_live, h_live, d_live = utils.run_full_analysis(txt)
            st.markdown("---")
            st.info(f"**Gelişmiş Skor:** `{s_live:.2f}`")
            
            exp = st.expander("🔍 Kelime Detayları (Algoritma Çıktısı)", expanded=True)
            with exp:
                k1, k2 = st.columns(2)
                with k1:
                    st.markdown(f"**🦅 Şahin ({len(h_live)})**")
                    for w in h_live: st.write(f"- {w}")
                with k2:
                    st.markdown(f"**🕊️ Güvercin ({len(d_live)})**")
                    for w in d_live: st.write(f"- {w}")

    # LİSTE
    st.markdown("### 📋 Kayıtlar")
    if not df_all.empty:
        df_show = df_all.copy()
        df_show['Dönem'] = df_show['period_date'].dt.strftime('%Y-%m')
        
        event = st.dataframe(
            df_show[['id', 'Dönem', 'period_date', 'source', 'score_abg']].sort_values('period_date', ascending=False),
            on_select="rerun", selection_mode="single-row", use_container_width=True, hide_index=True
        )
        
        if len(event.selection.rows) > 0:
            sel_idx = event.selection.rows[0]
            sel_id = df_show.iloc[sel_idx]['id']
            if st.session_state['form_data']['id'] != sel_id:
                orig = df_all[df_all['id'] == sel_id].iloc[0]
                st.session_state['form_data'] = {
                    'id': int(orig['id']),
                    'date': pd.to_datetime(orig['period_date']).date(),
                    'source': orig['source'],
                    'text': orig['text_content']
                }
                st.rerun()

# ==============================================================================
# TAB 3: PİYASA
# ==============================================================================
with tab3:
    st.header("Piyasa Verileri")
    c1, c2 = st.columns(2)
    d1 = c1.date_input("Başlangıç", datetime.date(2023, 1, 1))
    d2 = c2.date_input("Bitiş", datetime.date.today())
    
    if st.button("Getir"):
        df, err = utils.fetch_market_data_adapter(d1, d2)
        if not df.empty:
            fig_m = go.Figure()
            if 'Yıllık TÜFE' in df.columns:
                fig_m.add_trace(go.Scatter(x=df['Donem'], y=df['Yıllık TÜFE'], name="Yıllık TÜFE", line=dict(color='red')))
            if 'PPK Faizi' in df.columns:
                fig_m.add_trace(go.Scatter(x=df['Donem'], y=df['PPK Faizi'], name="Faiz", line=dict(color='orange')))
            st.plotly_chart(fig_m, use_container_width=True)
            st.dataframe(df)
        else:
            st.error(f"Hata: {err}")
