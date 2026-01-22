import streamlit as st
import pandas as pd
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import uuid
import numpy as np

# Utils dosyasını çağırıyoruz (Arka plandaki motor)
import utils 

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Piyasa Analiz Paneli", layout="wide", page_icon="🦅")

st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 5rem; }
    h1 { font-size: 1.8rem !important; }
    .stDataFrame { font-size: 0.8rem; }
    .stButton button { border-radius: 8px; font-size: 0.9rem; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
</style>
""", unsafe_allow_html=True)

# --- GÜVENLİK ---
APP_PWD = "SahinGuvercin34"      
ADMIN_PWD = "SahinGuvercin06"    

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br><h3 style='text-align: center;'>🔐 Güvenli Giriş</h3>", unsafe_allow_html=True)
        pwd_input = st.text_input("Uygulama Şifresi", type="password")
        if st.button("Giriş Yap", type="primary", use_container_width=True):
            if pwd_input == APP_PWD:
                st.session_state['logged_in'] = True
                st.rerun()
            else:
                st.error("Hatalı Şifre!")
    st.stop()

# --- SESSION STATE BAŞLATMA ---
defaults = {
    'form_data': {'id': None, 'date': datetime.date.today().replace(day=1), 'source': "TCMB", 'text': ""},
    'table_key': str(uuid.uuid4()),
    'collision_state': {'active': False, 'target_id': None, 'pending_text': None, 'target_date': None},
    'update_state': {'active': False, 'pending_text': None},
    'stop_words_deep': [],
    'stop_words_cloud': [],
    'ai_trend_df': None,
    'textasdata_model': None,
    'watch_terms': ["inflation", "disinflation", "stability", "growth", "interest rate", "policy rate", "tightened", "risks", "global"]
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --- YARDIMCI UI FONKSİYONLARI ---
def add_cloud_stop():
    word = st.session_state.get("cloud_stop_in", "").strip()
    if word and word not in st.session_state['stop_words_cloud']:
        st.session_state['stop_words_cloud'].append(word)
    st.session_state["cloud_stop_in"] = ""

def reset_form():
    st.session_state['form_data'] = {'id': None, 'date': datetime.date.today(), 'source': "TCMB", 'text': ""}
    st.session_state['collision_state'] = {'active': False, 'target_id': None, 'pending_text': None, 'target_date': None}
    st.session_state['update_state'] = {'active': False, 'pending_text': None}
    st.session_state['table_key'] = str(uuid.uuid4())

# --- HEADER ---
c_head1, c_head2 = st.columns([8, 1])
with c_head1: 
    st.title("🦅 Şahin/Güvercin Analiz Paneli")
with c_head2: 
    if st.button("Çıkış", type="secondary"): 
        st.session_state['logged_in'] = False
        st.rerun()

# --- TAB YAPISI ---
tab1, tab2, tab3, tab4, tab_textdata, tab6, tab7, tab_roberta, tab_imp = st.tabs([
    "📈 Dashboard", "📝 Veri Girişi", "📊 Veriler", "🔍 Frekans", 
    "📚 Text as Data", "☁️ WordCloud", "📜 ABF (2019)", "🧠 CB-RoBERTa", "📅 Haberler"
])

# ==============================================================================
# TAB 1: DASHBOARD
# ==============================================================================
with tab1:
    with st.spinner("Veriler Hazırlanıyor..."):
        df_logs = utils.fetch_all_data()
        df_events = utils.fetch_events() 
    
    if not df_logs.empty:
        # Ön İşleme
        df_logs['period_date'] = pd.to_datetime(df_logs['period_date'])
        df_logs['Donem'] = df_logs['period_date'].dt.strftime('%Y-%m')
        df_logs['word_count'] = df_logs['text_content'].apply(lambda x: len(str(x).split()) if x else 0)
        df_logs['flesch_score'] = df_logs['text_content'].apply(lambda x: utils.calculate_flesch_reading_ease(str(x)))
        
        # ABG Skoru (Utils'den)
        abg_df = utils.calculate_abg_scores(df_logs)
        abg_df['abg_dashboard_val'] = (abg_df['abg_index'] - 1.0) * 100
        
        # Market Verisi
        min_d = df_logs['period_date'].min().date()
        max_d = datetime.date.today()
        df_market, err = utils.fetch_market_data_adapter(min_d, max_d)
        
        # MERGE İŞLEMLERİ
        merged = pd.merge(df_logs, df_market, on="Donem", how="left")
        merged = pd.merge(merged, abg_df[['period_date', 'abg_dashboard_val']], on='period_date', how='left')
        
        # AI Trend Verisi Merge (Session State'den)
        ai_df = st.session_state.get("ai_trend_df")
        if ai_df is not None and not ai_df.empty:
            ai_tmp = ai_df.copy()
            # Kolon isimlerini normalize et
            if "AI Score (EMA)" in ai_tmp.columns:
                ai_tmp["AI_EMA"] = pd.to_numeric(ai_tmp["AI Score (EMA)"], errors="coerce")
            if "AI Score (Calib)" in ai_tmp.columns:
                ai_tmp["AI_CALIB"] = pd.to_numeric(ai_tmp["AI Score (Calib)"], errors="coerce")
            
            # Dönem formatı
            col_date = "period_date" if "period_date" in ai_tmp.columns else "period_date" # fallback
            if "Dönem" in ai_tmp.columns:
                 ai_tmp["Donem"] = ai_tmp["Dönem"].astype(str)
            else:
                 ai_tmp["Donem"] = pd.to_datetime(ai_tmp[col_date]).dt.strftime("%Y-%m")

            # Duplicate temizle ve birleştir
            ai_cols = [c for c in ["Donem", "AI_EMA", "AI_CALIB"] if c in ai_tmp.columns]
            ai_tmp = ai_tmp[ai_cols].drop_duplicates(subset=["Donem"])
            merged = pd.merge(merged, ai_tmp, on="Donem", how="left")
        else:
            merged["AI_EMA"] = np.nan
            merged["AI_CALIB"] = np.nan

        merged = merged.sort_values("period_date")
        
        # Numeric dönüşümler
        for col in ['Yıllık TÜFE', 'PPK Faizi']:
            if col in merged.columns: merged[col] = pd.to_numeric(merged[col], errors='coerce')

        # --- GRAFİK ÇİZİMİ ---
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # 1. Metin Uzunluğu (Bar - Arka Plan)
        fig.add_trace(go.Bar(
            x=merged['period_date'], y=merged['word_count'], 
            name="Kelime Sayısı", marker=dict(color='gray'), opacity=0.1, 
            yaxis="y3", hoverinfo="x+y+name"
        ))
        
        # 2. Skorlar
        fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['abg_dashboard_val'], name="ABG Endeksi (2019)", line=dict(color='navy', width=2), yaxis="y"))
        
        if merged["AI_EMA"].notna().any():
            fig.add_trace(go.Scatter(x=merged['period_date'], y=merged["AI_EMA"], name="AI RoBERTa (EMA)", line=dict(color='green', width=3), yaxis="y"))
        
        # 3. Makro Veriler
        if 'Yıllık TÜFE' in merged.columns: 
            fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['Yıllık TÜFE'], name="TÜFE (%)", line=dict(color='red', dash='dot'), yaxis="y"))
        if 'PPK Faizi' in merged.columns: 
            fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['PPK Faizi'], name="Faiz (%)", line=dict(color='orange', dash='dot'), yaxis="y"))
        
        # Layout Ayarları
        fig.update_layout(
            title="Merkez Bankası Analiz Paneli", hovermode="x unified", height=550,
            yaxis=dict(title="Skor & Oranlar", range=[-150, 150]),
            yaxis2=dict(visible=False, overlaying="y", side="right"),
            yaxis3=dict(title="Kelime", overlaying="y", side="right", showgrid=False, visible=False, range=[0, merged['word_count'].max() * 3]),
            legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5)
        )
        
        # Bölge Renklendirme (Şahin/Güvercin)
        fig.add_hrect(y0=0, y1=150, fillcolor="red", opacity=0.05, line_width=0, layer="below")
        fig.add_hrect(y0=-150, y1=0, fillcolor="blue", opacity=0.05, line_width=0, layer="below")
        fig.add_hline(y=0, line_width=2, line_color="black")

        # Olaylar (Haberler)
        if not df_events.empty:
            for _, ev in df_events.iterrows():
                d_str = pd.to_datetime(ev['event_date']).strftime('%Y-%m-%d')
                first_link = ev['links'].split('\n')[0] if ev['links'] else ""
                fig.add_shape(type="line", x0=d_str, x1=d_str, y0=-140, y1=140, line=dict(color="purple", width=1, dash="dot"))
                fig.add_annotation(x=d_str, y=-145, text=f"<a href='{first_link}'>ℹ️</a>", showarrow=False, font=dict(size=14))

        st.plotly_chart(fig, use_container_width=True)

        # --- AI TREND ALT BÖLÜMÜ ---
        st.divider()
        c_ai1, c_ai2 = st.columns([3, 1])
        with c_ai1:
            st.subheader("🤖 Yapay Zeka Trendi (RoBERTa)")
        with c_ai2:
            if st.button("🚀 AI Hesapla", key="dash_calc_ai"):
                with st.spinner("AI Modeli çalışıyor..."):
                    if utils.HAS_TRANSFORMERS:
                        res = utils.calculate_ai_trend_series(utils.fetch_all_data())
                        if not res.empty:
                            st.session_state['ai_trend_df'] = res
                            st.success("Hesaplandı!")
                            st.rerun()
                    else:
                        st.error("Model kütüphaneleri eksik.")

        if st.session_state.get('ai_trend_df') is not None:
             # Utils'deki grafik fonksiyonunu kullan
             if hasattr(utils, 'create_ai_trend_chart'):
                 fig_ai = utils.create_ai_trend_chart(st.session_state['ai_trend_df'])
                 if fig_ai: st.plotly_chart(fig_ai, use_container_width=True)
        
        if st.button("🔄 Verileri Yenile"):
            st.cache_data.clear()
            st.rerun()

    else:
        st.info("Veritabanında henüz kayıt yok. 'Veri Girişi' sekmesinden ekleyin.")

# ==============================================================================
# TAB 2: VERİ GİRİŞİ
# ==============================================================================
with tab2:
    st.header("📝 Veri Girişi ve Düzenleme")
    
    df_all = utils.fetch_all_data()
    
    # Form Alanı
    with st.container(border=True):
        col_act, col_info = st.columns([1, 4])
        with col_act:
            if st.button("➕ Temizle / Yeni", type="secondary", use_container_width=True):
                reset_form()
                st.rerun()
        
        st.markdown("---")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            sel_date = st.date_input("Tarih", value=st.session_state['form_data']['date'])
            source = st.text_input("Kaynak", value=st.session_state['form_data']['source'])
        with c2:
            txt = st.text_area("Metin İçeriği", value=st.session_state['form_data']['text'], height=200)

        # Buton Mantığı (Kaydet / Güncelle / Çakışma Yönetimi)
        current_id = st.session_state['form_data']['id']
        
        # 1. Çakışma Durumu
        if st.session_state['collision_state']['active']:
            st.error(f"⚠️ Bu tarihte ({st.session_state['collision_state']['target_date']}) zaten kayıt var!")
            ov_pass = st.text_input("Admin Şifresi (Üzerine Yazmak İçin)", type="password")
            
            c_ov1, c_ov2 = st.columns(2)
            if c_ov1.button("🚨 Üzerine Yaz", type="primary"):
                if ov_pass == ADMIN_PWD:
                    p_txt = st.session_state['collision_state']['pending_text']
                    t_id = st.session_state['collision_state']['target_id']
                    # Analiz yap
                    res = utils.run_full_analysis(p_txt) # s_abg, h_cnt... döner
                    utils.update_entry(t_id, sel_date, p_txt, source, res[0], res[0])
                    st.success("Güncellendi!")
                    reset_form()
                    st.rerun()
                else:
                    st.error("Şifre Hatalı")
            if c_ov2.button("İptal"):
                reset_form()
                st.rerun()
        
        # 2. Güncelleme Durumu
        elif st.session_state['update_state']['active']:
            st.warning(f"ID: {current_id} güncelleniyor...")
            up_pass = st.text_input("Admin Şifresi", type="password")
            if st.button("💾 Değişiklikleri Kaydet", type="primary"):
                if up_pass == ADMIN_PWD:
                    p_txt = st.session_state['update_state']['pending_text']
                    res = utils.run_full_analysis(p_txt)
                    utils.update_entry(current_id, sel_date, p_txt, source, res[0], res[0])
                    st.success("Başarılı!")
                    reset_form()
                    st.rerun()
                else:
                    st.error("Şifre Hatalı")
        
        # 3. Normal Kayıt
        else:
            btn_txt = "💾 Güncelle" if current_id else "💾 Kaydet"
            if st.button(btn_txt, type="primary"):
                if not txt:
                    st.error("Metin boş olamaz.")
                else:
                    # Çakışma kontrolü
                    collision = None
                    if not df_all.empty:
                        df_all['d_check'] = pd.to_datetime(df_all['period_date']).dt.date
                        mask = df_all['d_check'] == sel_date
                        if mask.any(): collision = df_all[mask].iloc[0]
                    
                    is_self = current_id and collision is not None and int(collision['id']) == current_id
                    
                    if is_self:
                        st.session_state['update_state'] = {'active': True, 'pending_text': txt}
                        st.rerun()
                    elif collision is not None:
                        st.session_state['collision_state'] = {'active': True, 'target_id': int(collision['id']), 'target_date': sel_date, 'pending_text': txt}
                        st.rerun()
                    else:
                        # Yeni Kayıt
                        res = utils.run_full_analysis(txt)
                        utils.insert_entry(sel_date, txt, source, res[0], res[0])
                        st.success("Kaydedildi!")
                        reset_form()
                        st.rerun()
            
            # Silme Butonu
            if current_id:
                with st.popover("🗑️ Kaydı Sil"):
                    d_pass = st.text_input("Şifre", type="password", key="del_pass")
                    if st.button("Onayla ve Sil"):
                        if d_pass == ADMIN_PWD:
                            utils.delete_entry(current_id)
                            st.success("Silindi.")
                            reset_form()
                            st.rerun()
                        else:
                            st.error("Hatalı")

    # Liste
    st.subheader("📋 Kayıt Listesi")
    if not df_all.empty:
        df_show = df_all.copy()
        df_show['Dönem'] = pd.to_datetime(df_show['period_date']).dt.strftime('%Y-%m')
        
        event = st.dataframe(
            df_show[['id', 'Dönem', 'source']], 
            selection_mode="single-row", on_select="rerun", 
            hide_index=True, use_container_width=True, key=st.session_state['table_key']
        )
        
        if len(event.selection.rows) > 0:
            sel_idx = event.selection.rows[0]
            sel_row = df_show.iloc[sel_idx]
            # Seçileni forma yükle
            if st.session_state['form_data']['id'] != sel_row['id']:
                st.session_state['form_data'] = {
                    'id': int(sel_row['id']),
                    'date': pd.to_datetime(sel_row['period_date']).date(),
                    'source': sel_row['source'],
                    'text': sel_row['text_content']
                }
                st.rerun()

# ==============================================================================
# TAB 3: VERİLER
# ==============================================================================
with tab3:
    st.header("📊 Piyasa Verileri (TÜFE & Faiz)")
    c1, c2, c3 = st.columns([1, 1, 1])
    d1 = c1.date_input("Başlangıç", datetime.date(2023, 1, 1))
    d2 = c2.date_input("Bitiş", datetime.date.today())
    
    if c3.button("Verileri Getir"):
        df_m, err = utils.fetch_market_data_adapter(d1, d2)
        if not df_m.empty:
            fig = go.Figure()
            if 'Yıllık TÜFE' in df_m.columns: fig.add_trace(go.Scatter(x=df_m['Donem'], y=df_m['Yıllık TÜFE'], name="Yıllık TÜFE", line=dict(color='red')))
            if 'PPK Faizi' in df_m.columns: fig.add_trace(go.Scatter(x=df_m['Donem'], y=df_m['PPK Faizi'], name="Faiz", line=dict(color='orange')))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_m, use_container_width=True)
        else:
            st.error(f"Veri bulunamadı: {err}")

# ==============================================================================
# TAB 4: FREKANS
# ==============================================================================
with tab4:
    st.header("🔍 Kelime Frekans Analizi")
    
    # Kelime Ekleme/Çıkarma UI
    c1, c2 = st.columns([3, 1])
    new_term = c1.text_input("İzlenecek kelime ekle (Enter)", placeholder="örn: likidite, sıkılaşma")
    if new_term:
        if new_term not in st.session_state["watch_terms"]:
            st.session_state["watch_terms"].append(new_term)
            st.rerun()
            
    if c2.button("Sıfırla"):
        st.session_state["watch_terms"] = ["inflation", "growth", "interest rate"]
        st.rerun()

    # Etiketler
    st.write("İzlenenler: " + ", ".join([f"`{t}`" for t in st.session_state["watch_terms"]]))
    
    # Analiz
    df_all = utils.fetch_all_data()
    if not df_all.empty:
        # Utils'de fonksiyon varsa kullan, yoksa fallback
        if hasattr(utils, "build_watch_terms_timeseries"):
            freq_df = utils.build_watch_terms_timeseries(df_all, st.session_state["watch_terms"])
        else:
            # Basit fallback logic (frontend içinde)
            rows = []
            for _, r in df_all.iterrows():
                txt = str(r.get("text_content","")).lower()
                rec = {"period_date": r["period_date"]}
                for t in st.session_state["watch_terms"]:
                    rec[t] = txt.count(t.lower())
                rows.append(rec)
            freq_df = pd.DataFrame(rows).sort_values("period_date")

        if not freq_df.empty:
            fig = go.Figure()
            for t in st.session_state["watch_terms"]:
                if t in freq_df.columns and freq_df[t].sum() > 0:
                    fig.add_trace(go.Scatter(x=freq_df["period_date"], y=freq_df[t], name=t, mode='lines+markers'))
            st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    st.subheader("Metin Karşılaştırma (Diff)")
    if not df_all.empty:
        dates = df_all.sort_values("period_date", ascending=False)['period_date'].astype(str).tolist()
        c_d1, c_d2 = st.columns(2)
        d_old = c_d1.selectbox("Eski Tarih", dates, index=min(1, len(dates)-1))
        d_new = c_d2.selectbox("Yeni Tarih", dates, index=0)
        
        if st.button("Karşılaştır"):
            t1 = df_all[df_all['period_date'].astype(str)==d_old].iloc[0]['text_content']
            t2 = df_all[df_all['period_date'].astype(str)==d_new].iloc[0]['text_content']
            html = utils.generate_diff_html(t1, t2)
            st.markdown(html, unsafe_allow_html=True)

# ==============================================================================
# TAB 5: TEXT AS DATA (OPTIMIZED)
# ==============================================================================
with tab_textdata:
    st.header("📚 Text as Data: HYBRID + CPI Tahmin Modeli")
    st.info("TF-IDF (Kelime + Karakter) ve Makroekonomik verileri birleştirerek Faiz Karar Metninden `delta_bp` tahmini yapar.")

    if not utils.HAS_ML_DEPS:
        st.error("ML Kütüphaneleri eksik.")
        st.stop()

    df_logs = utils.fetch_all_data()
    
    # Veri Hazırlığı (Utils üzerinden)
    with st.spinner("Veri seti hazırlanıyor..."):
        min_d = pd.to_datetime(df_logs['period_date']).min().date()
        df_market, _ = utils.fetch_market_data_adapter(min_d, datetime.date.today())
        
        # Utils fonksiyonunu çağırıyoruz
        df_td = utils.textasdata_prepare_df_hybrid_cpi(
            df_logs, df_market, 
            text_col="text_content", date_col="period_date", 
            y_col="delta_bp", rate_col="policy_rate"
        )
    
    if df_td.empty or len(df_td) < 10:
        st.warning(f"Yetersiz veri. (Gözlem sayısı: {len(df_td)})")
        st.stop()

    # Ayarlar
    c1, c2 = st.columns(2)
    alpha = c1.number_input("Ridge Alpha", 1.0, 50.0, 10.0)
    min_df = c2.number_input("Min DF", 1, 5, 2)
    
    if st.button("🚀 Modeli Eğit", type="primary"):
        with st.spinner("Model eğitiliyor..."):
            # Utils fonksiyonunu çağırıyoruz (Frontend'de kod tekrarı yok!)
            model_pack = utils.train_textasdata_hybrid_cpi_ridge(
                df_td, alpha=alpha, min_df=min_df, n_splits=5
            )
            st.session_state['textasdata_model'] = model_pack
            st.success("Model eğitildi!")

    # Sonuçlar
    pack = st.session_state.get('textasdata_model')
    if pack:
        metrics = pack.get('metrics', {})
        cm1, cm2, cm3 = st.columns(3)
        cm1.metric("MAE", f"{metrics.get('mae',0):.1f}")
        cm2.metric("RMSE", f"{metrics.get('rmse',0):.1f}")
        cm3.metric("R2", f"{metrics.get('r2',0):.2f}")
        
        # Grafik
        pred_df = pack.get('pred_df')
        if pred_df is not None:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=pred_df['period_date'], y=pred_df['delta_bp'], name="Gerçek"))
            fig.add_trace(go.Scatter(x=pred_df['period_date'], y=pred_df['pred_delta_bp'], name="Tahmin"))
            st.plotly_chart(fig, use_container_width=True)
            
        # Canlı Tahmin
        st.divider()
        st.subheader("🔮 Metin Tahmini")
        txt_in = st.text_area("Metni buraya yapıştırın...")
        if st.button("Tahmin Et"):
            res = utils.predict_textasdata_hybrid_cpi(pack, df_td, txt_in)
            bp = res.get('pred_delta_bp', 0)
            st.metric("Tahmini Değişim (bps)", f"{bp:.0f}")

# ==============================================================================
# TAB 6: WORDCLOUD
# ==============================================================================
with tab6:
    st.header("☁️ Kelime Bulutu")
    df_wc = utils.fetch_all_data()
    if not df_wc.empty:
        # Stopword yönetimi
        col_in, col_list = st.columns([1, 2])
        col_in.text_input("Hariç tutulacak kelime", key="cloud_stop_in", on_change=add_cloud_stop)
        col_list.write(st.session_state['stop_words_cloud'])
        
        dates = df_wc['period_date'].astype(str).tolist()
        sel_date = st.selectbox("Dönem", ["Hepsi"] + dates)
        
        if st.button("Oluştur"):
            if sel_date == "Hepsi":
                text = " ".join(df_wc['text_content'].astype(str))
            else:
                text = df_wc[df_wc['period_date'].astype(str) == sel_date].iloc[0]['text_content']
            
            fig = utils.generate_wordcloud_img(text, st.session_state['stop_words_cloud'])
            if fig: st.pyplot(fig)

# ==============================================================================
# TAB 7: ABF ANALİZİ
# ==============================================================================
with tab7:
    st.header("📜 ABF (2019) Sözlük Analizi")
    df_abg = utils.fetch_all_data()
    if not df_abg.empty:
        abg_scored = utils.calculate_abg_scores(df_abg)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=abg_scored['period_date'], y=abg_scored['abg_index'], 
            mode='lines+markers', name='Hawkishness'
        ))
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
        
        # Detay Analiz
        sel_per = st.selectbox("Detaylı İncele", abg_scored['period_date'].astype(str).tolist())
        row = df_abg[df_abg['period_date'].astype(str) == sel_per].iloc[0]
        
        res = utils.analyze_hawk_dove(row['text_content'], utils.DICT)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Şahin Sayısı", res['hawk_count'])
        c2.metric("Güvercin Sayısı", res['dove_count'])
        c3.metric("Net Skor", f"{res['net_hawkishness']:.2f}")
        
        st.dataframe(pd.DataFrame(res['matches']))

# ==============================================================================
# TAB 8: ROBERTA
# ==============================================================================
with tab_roberta:
    st.header("🧠 CB-RoBERTa Analizi")
    
    # 1. Trend Grafiği
    st.subheader("Tarihsel Trend")
    if st.session_state.get('ai_trend_df') is not None:
        if hasattr(utils, 'create_ai_trend_chart'):
            fig = utils.create_ai_trend_chart(st.session_state['ai_trend_df'])
            if fig: st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Hesaplanmış trend verisi yok. Dashboard veya burada hesaplatabilirsiniz.")
        if st.button("Hesapla"):
            res = utils.calculate_ai_trend_series(utils.fetch_all_data())
            st.session_state['ai_trend_df'] = res
            st.rerun()

    st.divider()
    
    # 2. Tekil Cümle Analizi
    st.subheader("Cümle Bazlı Ayrıştırma")
    txt_rob = st.text_area("Metin (Analiz için)", height=150)
    
    if st.button("Cümleleri Analiz Et"):
        if hasattr(utils, 'analyze_sentences_with_roberta'):
            with st.spinner("Model çalışıyor..."):
                df_sent = utils.analyze_sentences_with_roberta(txt_rob)
                
                if not df_sent.empty and "Diff (H-D)" in df_sent.columns:
                    c1, c2 = st.columns(2)
                    c1.metric("Ortalama Şahinlik", f"{df_sent['Diff (H-D)'].mean():.2f}")
                    
                    st.dataframe(
                        df_sent.style.background_gradient(subset=['Diff (H-D)'], cmap="RdBu_r", vmin=-1, vmax=1),
                        use_container_width=True
                    )
                else:
                    st.warning("Sonuç bulunamadı.")
        else:
            st.error("Utils dosyasında 'analyze_sentences_with_roberta' fonksiyonu eksik.")

# ==============================================================================
# TAB 9: HABERLER
# ==============================================================================
with tab_imp:
    st.header("📅 Haber Kayıtları (Events)")
    
    c1, c2 = st.columns([1, 3])
    e_date = c1.date_input("Haber Tarihi")
    e_links = c2.text_area("Linkler (Her satıra bir tane)")
    
    if st.button("Haber Ekle"):
        utils.add_event(e_date, e_links)
        st.success("Eklendi")
        st.rerun()
        
    st.divider()
    
    events = utils.fetch_events()
    if not events.empty:
        for _, row in events.iterrows():
            with st.expander(f"{row['event_date']} - Haberler"):
                st.write(row['links'])
                if st.button("Sil", key=f"del_ev_{row['id']}"):
                    utils.delete_event(row['id'])
                    st.rerun()
