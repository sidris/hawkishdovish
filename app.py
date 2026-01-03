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

# --- AI ---
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
    with st.spinner("Yükleniyor..."):
        df_logs = utils.fetch_all_data()
    
    if not df_logs.empty:
        df_logs['period_date'] = pd.to_datetime(df_logs['period_date'])
        df_logs['Donem'] = df_logs['period_date'].dt.strftime('%Y-%m')
        
        min_d = df_logs['period_date'].min().date()
        max_d = datetime.date.today()
        df_market, err = utils.fetch_market_data_adapter(min_d, max_d)
        
        merged = pd.merge(df_logs, df_market, on="Donem", how="left")
        merged = merged.sort_values("period_date")
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['score_finbert'], name="FinBERT (AI)", line=dict(color='blue')), secondary_y=False)
        fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['score_abg'], name="ABG (N-Gram)", line=dict(color='green', dash='dot')), secondary_y=False)
        
        if 'Yıllık TÜFE' in merged.columns:
            fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['Yıllık TÜFE'], name="Yıllık TÜFE (%)", line=dict(color='red')), secondary_y=True)
        if 'PPK Faizi' in merged.columns:
            fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['PPK Faizi'], name="Faiz (%)", line=dict(color='orange')), secondary_y=True)

        fig.update_layout(title="Metin Analizi ve Ekonomi", hovermode="x unified", height=500)
        st.plotly_chart(fig, use_container_width=True)
        if st.button("🔄 Yenile"): st.cache_data.clear(); st.rerun()
    else: st.info("Kayıt yok.")

# ==============================================================================
# TAB 2: VERİ GİRİŞİ (ÇAKIŞMA KONTROLLÜ)
# ==============================================================================
with tab2:
    st.subheader("Veri İşlemleri")
    
    # Tüm verileri çek (Kontrol için)
    df_all = utils.fetch_all_data()
    if not df_all.empty: 
        df_all['period_date'] = pd.to_datetime(df_all['period_date'])
        # Karşılaştırma için sadece tarih kısmını al
        df_all['date_only'] = df_all['period_date'].dt.date
    
    current_id = st.session_state['form_data']['id']
    
    with st.container(border=True):
        c1, c2 = st.columns([1, 2])
        with c1:
            val_date = st.session_state['form_data']['date']
            # Kullanıcı tarihi değiştirdiği an bu değişken güncellenir
            selected_date = st.date_input("Tarih", value=val_date)
            
            val_source = st.session_state['form_data']['source']
            source = st.text_input("Kaynak", value=val_source)
            st.caption(f"Dönem: **{selected_date.strftime('%Y-%m')}**")
            
            # --- ÇAKIŞMA KONTROLÜ ---
            collision_record = None
            if not df_all.empty:
                # Seçilen tarih veritabanında var mı?
                mask = df_all['date_only'] == selected_date
                if mask.any():
                    collision_record = df_all[mask].iloc[0]
            
            # Eğer çakışma varsa ve şu an o kaydı düzenlemiyorsak (Yeni bir tarih seçtiysek)
            if collision_record:
                # Eğer listeden seçip düzenlemiyorsak (ID'ler farklıysa), uyarı ver
                if current_id != collision_record['id']:
                    st.warning(f"⚠️ **DİKKAT:** {selected_date} tarihinde zaten bir kayıt var!")
                    st.markdown(f"*Kaydet tuşuna basarsanız mevcut verinin **üzerine yazılacaktır**.*")

        with c2:
            val_text = st.session_state['form_data']['text']
            txt = st.text_area("Metin", value=val_text, height=200)
        
        col_b1, col_b2, col_b3 = st.columns([2, 1, 1])
        with col_b1:
            # Buton Metni Duruma Göre Değişsin
            btn_text = "💾 Kaydet / Analiz Et"
            if collision_record and (current_id != collision_record['id']):
                btn_text = "⚠️ Üzerine Yaz ve Kaydet"
            elif current_id:
                btn_text = "💾 Güncelle"

            if st.button(btn_text, type="primary"):
                if txt:
                    # Analiz
                    s_abg, h_cnt, d_cnt, hawks, doves = utils.run_full_analysis(txt)
                    s_fb, l_fb = analyze_finbert(txt)
                    
                    # KAYIT SENARYOLARI
                    
                    # 1. Senaryo: Listeden seçilen kaydı güncelliyoruz
                    if current_id:
                        utils.update_entry(current_id, selected_date, txt, source, s_abg, s_abg, s_fb, l_fb)
                        st.success("Kayıt başarıyla güncellendi!")
                        
                    # 2. Senaryo: Yeni tarih seçtik ama o tarihte kayıt varmış (Collision) -> ÜZERİNE YAZ
                    elif collision_record:
                        target_id = int(collision_record['id'])
                        utils.update_entry(target_id, selected_date, txt, source, s_abg, s_abg, s_fb, l_fb)
                        st.warning(f"{selected_date} tarihli eski kayıt silindi, yeni veri üzerine yazıldı.")
                        
                    # 3. Senaryo: Tarih boş, yeni kayıt
                    else:
                        utils.insert_entry(selected_date, txt, source, s_abg, s_abg, s_fb, l_fb)
                        st.success("Yeni kayıt eklendi!")
                    
                    # Formu Temizle
                    st.session_state['form_data'] = {'id': None, 'date': datetime.date.today(), 'source': "TCMB", 'text': ""}
                    st.rerun()
                else:
                    st.error("Lütfen metin giriniz.")

        with col_b2:
            if st.button("Temizle"):
                st.session_state['form_data'] = {'id': None, 'date': datetime.date.today(), 'source': "TCMB", 'text': ""}
                st.rerun()

        with col_b3:
            if current_id:
                if st.button("🗑️ Sil", type="primary"):
                    utils.delete_entry(current_id)
                    st.success("Silindi!")
                    st.session_state['form_data'] = {'id': None, 'date': datetime.date.today(), 'source': "TCMB", 'text': ""}
                    st.rerun()

        # CANLI ANALİZ VE YÜZDELİK GÖSTERİM
        if txt:
            s_live, h_live_cnt, d_live_cnt, h_list, d_list = utils.run_full_analysis(txt)
            total_sigs = h_live_cnt + d_live_cnt
            if total_sigs > 0:
                h_pct = (h_live_cnt / total_sigs) * 100
                d_pct = (d_live_cnt / total_sigs) * 100
                tone_label = "ŞAHİN" if h_pct > d_pct else "GÜVERCİN" if d_pct > h_pct else "DENGELİ"
            else:
                h_pct = 0; d_pct = 0
                tone_label = "NÖTR"
            
            st.markdown("---")
            c_score1, c_score2 = st.columns(2)
            with c_score1: st.metric(label="Şahin (Hawkish)", value=f"%{h_pct:.1f}", delta=f"{h_live_cnt} Sinyal")
            with c_score2: st.metric(label="Güvercin (Dovish)", value=f"%{d_pct:.1f}", delta=f"{d_live_cnt} Sinyal")
            
            st.progress(h_pct / 100)
            st.caption(f"Genel Ton: **{tone_label}**")

            exp = st.expander("🔍 Tespit Edilen İfadeler (N-Gram)", expanded=True)
            with exp:
                k1, k2 = st.columns(2)
                with k1:
                    st.markdown(f"**🦅 Şahin İfadeler**")
                    if h_list:
                        for w in h_list: st.write(f"- {w}")
                    else: st.write("- Yok")
                with k2:
                    st.markdown(f"**🕊️ Güvercin İfadeler**")
                    if d_list:
                        for w in d_list: st.write(f"- {w}")
                    else: st.write("- Yok")

    # LİSTE
    st.markdown("### 📋 Geçmiş Kayıtlar")
    if not df_all.empty:
        df_show = df_all.copy()
        df_show['Dönem'] = df_show['period_date'].dt.strftime('%Y-%m')
        event = st.dataframe(df_show[['id', 'Dönem', 'period_date', 'source', 'score_abg']].sort_values('period_date', ascending=False),
            on_select="rerun", selection_mode="single-row", use_container_width=True, hide_index=True)
        if len(event.selection.rows) > 0:
            sel_id = df_show.iloc[event.selection.rows[0]]['id']
            if st.session_state['form_data']['id'] != sel_id:
                orig = df_all[df_all['id'] == sel_id].iloc[0]
                st.session_state['form_data'] = {'id': int(orig['id']), 'date': pd.to_datetime(orig['period_date']).date(), 'source': orig['source'], 'text': orig['text_content']}
                st.rerun()

# TAB 3: PİYASA
with tab3:
    st.header("Piyasa Verileri")
    c1, c2 = st.columns(2)
    d1 = c1.date_input("Başlangıç", datetime.date(2023, 1, 1))
    d2 = c2.date_input("Bitiş", datetime.date.today())
    if st.button("Getir"):
        df, err = utils.fetch_market_data_adapter(d1, d2)
        if not df.empty:
            fig_m = go.Figure()
            if 'Yıllık TÜFE' in df.columns: fig_m.add_trace(go.Scatter(x=df['Donem'], y=df['Yıllık TÜFE'], name="Yıllık TÜFE", line=dict(color='red')))
            if 'PPK Faizi' in df.columns: fig_m.add_trace(go.Scatter(x=df['Donem'], y=df['PPK Faizi'], name="Faiz", line=dict(color='orange')))
            st.plotly_chart(fig_m, use_container_width=True)
            st.dataframe(df)
        else: st.error(f"Hata: {err}")
