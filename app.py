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
        'date': datetime.date.today(),
        'source': "TCMB",
        'text': ""
    }

# --- MODELLER ---
@st.cache_resource
def load_models():
    try: return pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except: return None
classifier = load_models()

# --- ANALİZ FONKSİYONLARI ---
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

    hawkish_phrases = {f"{adj} {noun}" for adj in hawkish_adj for noun in nouns}
    dovish_phrases = {f"{adj} {noun}" for adj in dovish_adj for noun in nouns}

    found_hawkish = []
    found_dovish = []

    h_score = 0; d_score = 0

    # Bigram Sayımı
    for p in hawkish_phrases:
        if bigram_counts[p] > 0:
            h_score += bigram_counts[p]
            found_hawkish.append(f"{p} ({bigram_counts[p]})")
    for p in dovish_phrases:
        if bigram_counts[p] > 0:
            d_score += bigram_counts[p]
            found_dovish.append(f"{p} ({bigram_counts[p]})")

    # Unigram Sayımı
    for w in hawkish_single:
        if token_counts[w] > 0:
            h_score += token_counts[w]
            found_hawkish.append(f"{w} ({token_counts[w]})")
    for w in dovish_single:
        if token_counts[w] > 0:
            d_score += token_counts[w]
            found_dovish.append(f"{w} ({token_counts[w]})")

    total = h_score + d_score
    final_score = (h_score - d_score) / total if total > 0 else 0
    return final_score, found_hawkish, found_dovish

def analyze_finbert(text):
    if not classifier: return 0, "neutral"
    res = classifier(text[:512])[0]
    score = res['score'] if res['label'] == "positive" else -res['score'] if res['label'] == "negative" else 0
    return score, res['label']

# --- ARAYÜZ ---
st.title("🦅 Şahin/Güvercin Analiz Paneli")

# TABLARI OLUŞTUR
tab1, tab2, tab3 = st.tabs(["📈 Dashboard", "📝 Veri Girişi & Yönetimi", "📊 Piyasa Verileri"])

# ==============================================================================
# TAB 1: DASHBOARD
# ==============================================================================
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

# ==============================================================================
# TAB 2: VERİ GİRİŞİ & YÖNETİMİ (HEPSİ BİR ARADA)
# ==============================================================================
with tab2:
    # ---------------------------------------------------------
    # BÖLÜM 1: FORM ALANI
    # ---------------------------------------------------------
    st.subheader("Veri Giriş / Düzenleme")
    
    # Form verilerini session_state'den al (Eğer listeden seçildiyse dolu gelir)
    current_id = st.session_state['form_data']['id']
    
    with st.container(border=True):
        c1, c2 = st.columns([1, 2])
        with c1:
            # Tarih Girişi
            val_date = st.session_state['form_data']['date']
            selected_date = st.date_input("Tarih", value=val_date)
            
            # Kaynak Girişi
            val_source = st.session_state['form_data']['source']
            source = st.text_input("Kaynak", value=val_source)
            
            st.info(f"Dönem: **{selected_date.strftime('%Y-%m')}** olarak kaydedilecek.")
            
        with c2:
            # Metin Girişi
            val_text = st.session_state['form_data']['text']
            txt = st.text_area("Metin", value=val_text, height=180, placeholder="Metni buraya yapıştırın...")
        
        # Butonlar
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
        
        with col_btn1:
            # Buton yazısı duruma göre değişir
            btn_label = "Güncelle" if current_id else "Kaydet"
            btn_type = "primary" if current_id else "secondary"
            
            if st.button(f"💾 {btn_label}", type="primary"):
                if txt:
                    # Analiz Yap
                    s_abg, hawks, doves = analyze_apel_blix_detailed(txt)
                    s_fb, l_fb = analyze_finbert(txt)
                    
                    if current_id:
                        # GÜNCELLEME
                        utils.update_entry(current_id, selected_date, txt, source, s_abg, s_abg, s_fb, l_fb)
                        st.success("Kayıt güncellendi!")
                    else:
                        # YENİ KAYIT
                        utils.insert_entry(selected_date, txt, source, s_abg, s_abg, s_fb, l_fb)
                        st.success("Yeni kayıt eklendi!")
                    
                    # Formu temizle
                    st.session_state['form_data'] = {'id': None, 'date': datetime.date.today(), 'source': "TCMB", 'text': ""}
                    st.rerun()
                else:
                    st.warning("Metin boş olamaz.")

        with col_btn2:
            if st.button("❌ Temizle"):
                st.session_state['form_data'] = {'id': None, 'date': datetime.date.today(), 'source': "TCMB", 'text': ""}
                st.rerun()

        # ANALİZ SONUÇLARI (HEMEN ALTINDA)
        if txt:
            s_abg_live, hawks_live, doves_live = analyze_apel_blix_detailed(txt)
            exp_res = st.expander("🔍 Kelime Analiz Detayları (Önizleme)", expanded=True)
            with exp_res:
                k1, k2 = st.columns(2)
                with k1:
                    st.markdown(f"**🦅 Şahin İfadeler**")
                    if hawks_live:
                        for w in hawks_live: st.write(f"- {w}")
                    else: st.caption("Yok")
                with k2:
                    st.markdown(f"**🕊️ Güvercin İfadeler**")
                    if doves_live:
                        for w in doves_live: st.write(f"- {w}")
                    else: st.caption("Yok")

    # ---------------------------------------------------------
    # BÖLÜM 2: KAYIT LİSTESİ (SEÇİLEBİLİR TABLO)
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("📋 Geçmiş Kayıtlar")
    st.caption("Düzenlemek için listeden bir satır seçin.")

    df_all = utils.fetch_all_data()

    if not df_all.empty:
        # Görüntüleme için düzenleme
        df_display = df_all.copy()
        df_display['period_date'] = pd.to_datetime(df_display['period_date'])
        # İlk sütun olarak Dönem (YYYY-MM) gösterelim
        df_display['Dönem'] = df_display['period_date'].dt.strftime('%Y-%m')
        
        # Tabloda gösterilecek sütunlar
        grid_df = df_display[['id', 'Dönem', 'period_date', 'source', 'score_abg']].sort_values('period_date', ascending=False)
        
        # Seçilebilir Tablo (Streamlit native selection)
        event = st.dataframe(
            grid_df,
            on_select="rerun", # Seçince sayfayı yenile
            selection_mode="single-row",
            use_container_width=True,
            hide_index=True,
            column_config={
                "period_date": st.column_config.DateColumn("Tam Tarih", format="DD.MM.YYYY"),
                "score_abg": st.column_config.ProgressColumn("Skor", min_value=-1, max_value=1, format="%.2f"),
                "id": st.column_config.NumberColumn("ID", width="small")
            }
        )

        # SEÇİM YAPILDIĞINDA FORMU DOLDUR
        if len(event.selection.rows) > 0:
            selected_row_index = event.selection.rows[0]
            selected_db_id = grid_df.iloc[selected_row_index]['id']
            
            # Eğer şu anki formdaki ID farklıysa (yeni seçim yapıldıysa) state'i güncelle
            if st.session_state['form_data']['id'] != selected_db_id:
                # Orijinal veriyi bul
                original_row = df_all[df_all['id'] == selected_db_id].iloc[0]
                
                st.session_state['form_data'] = {
                    'id': int(original_row['id']),
                    'date': pd.to_datetime(original_row['period_date']).date(),
                    'source': original_row['source'],
                    'text': original_row['text_content']
                }
                st.rerun()

    else:
        st.info("Henüz kayıt yok.")

# ==============================================================================
# TAB 3: PİYASA VERİLERİ
# ==============================================================================
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
