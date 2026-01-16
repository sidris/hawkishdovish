import streamlit as st
import pandas as pd
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt 
import utils 
import uuid

st.set_page_config(page_title="Piyasa Analiz", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 5rem; }
    h1 { font-size: 1.8rem !important; }
    .stDataFrame { font-size: 0.8rem; }
    .stButton button { border-radius: 20px; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# --- GÜVENLİK ---
APP_PWD = "SahinGuvercin34"      
ADMIN_PWD = "SahinGuvercin06"    

if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if not st.session_state['logged_in']:
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        st.markdown("<br><h3 style='text-align: center;'>🔐 Güvenli Giriş</h3>", unsafe_allow_html=True)
        pwd_input = st.text_input("Uygulama Şifresi", type="password")
        if st.button("Giriş Yap", type="primary"):
            if pwd_input == APP_PWD: st.session_state['logged_in'] = True; st.success("Başarılı!"); st.rerun()
            else: st.error("Hatalı!")
    st.stop()

# --- STATE ---
if 'form_data' not in st.session_state: st.session_state['form_data'] = {'id': None, 'date': datetime.date.today().replace(day=1), 'source': "TCMB", 'text': ""}
if 'table_key' not in st.session_state: st.session_state['table_key'] = str(uuid.uuid4())
if 'collision_state' not in st.session_state: st.session_state['collision_state'] = {'active': False, 'target_id': None, 'pending_text': None, 'target_date': None}
if 'update_state' not in st.session_state: st.session_state['update_state'] = {'active': False, 'pending_text': None}
if 'stop_words_deep' not in st.session_state: st.session_state['stop_words_deep'] = []
if 'stop_words_cloud' not in st.session_state: st.session_state['stop_words_cloud'] = []

def add_deep_stop():
    word = st.session_state.get("deep_stop_in", "").strip()
    if word and word not in st.session_state['stop_words_deep']: st.session_state['stop_words_deep'].append(word)
    st.session_state["deep_stop_in"] = ""
def add_cloud_stop():
    word = st.session_state.get("cloud_stop_in", "").strip()
    if word and word not in st.session_state['stop_words_cloud']: st.session_state['stop_words_cloud'].append(word)
    st.session_state["cloud_stop_in"] = ""
def reset_form():
    st.session_state['form_data'] = {'id': None, 'date': datetime.date.today(), 'source': "TCMB", 'text': ""}
    st.session_state['collision_state'] = {'active': False, 'target_id': None, 'pending_text': None, 'target_date': None}
    st.session_state['update_state'] = {'active': False, 'pending_text': None}
    st.session_state['table_key'] = str(uuid.uuid4())

c_head1, c_head2 = st.columns([6, 1])
with c_head1: st.title("🦅 Şahin/Güvercin Paneli")
with c_head2: 
    if st.button("Çıkış"): st.session_state['logged_in'] = False; st.rerun()

# SEKME YAPISI
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab_struct, tab_roberta, tab_imp = st.tabs([
    "📈 Dashboard", "📝 Veri Girişi", "📊 Veriler", "🔍 Frekans", "🤖 Faiz Tahmini", "☁️ WordCloud", "📜 ABF (2019)", 
    "🏗️ Yapısal Analiz", "🧠 CB-RoBERTa", "📅 Haberler"
])

# ==============================================================================
# TAB 1: DASHBOARD
# ==============================================================================
with tab1:
    with st.spinner("Veriler ve Yapay Zeka Analizleri Yükleniyor (İlk açılışta model indirildiği için yavaş olabilir)..."):
        df_logs = utils.fetch_all_data()
        df_events = utils.fetch_events() 
    
    if not df_logs.empty:
        # Tarih formatlama
        df_logs['period_date'] = pd.to_datetime(df_logs['period_date'])
        df_logs['Donem'] = df_logs['period_date'].dt.strftime('%Y-%m')
        
        # Metrik Hesaplamaları
        df_logs['word_count'] = df_logs['text_content'].apply(lambda x: len(str(x).split()) if x else 0)
        df_logs['flesch_score'] = df_logs['text_content'].apply(lambda x: utils.calculate_flesch_reading_ease(str(x)))
        df_logs['score_abg_scaled'] = df_logs['score_abg'].apply(lambda x: x*100 if abs(x) <= 1 else x)

        # ABG (Klasik) Verisi
        abg_df = utils.calculate_abg_scores(df_logs)
        abg_df['abg_dashboard_val'] = (abg_df['abg_index'] - 1.0) * 100
        
        # RoBERTa (AI) Verisi
        if utils.HAS_ROBERTA_LIB:
            roberta_series = utils.calculate_roberta_series(df_logs)
        else:
            roberta_series = pd.DataFrame()

        # Piyasa Verisi
        min_d = df_logs['period_date'].min().date()
        max_d = datetime.date.today()
        df_market, err = utils.fetch_market_data_adapter(min_d, max_d)
        
        # VERİLERİ BİRLEŞTİR (MERGE)
        merged = pd.merge(df_logs, df_market, on="Donem", how="left")
        merged = pd.merge(merged, abg_df[['period_date', 'abg_dashboard_val']], on='period_date', how='left')
        
        if not roberta_series.empty:
            roberta_series['period_date'] = pd.to_datetime(roberta_series['period_date'])
            merged = pd.merge(merged, roberta_series, on='period_date', how='left')

        merged = merged.sort_values("period_date")
        
        # Sayısal Dönüşümler (Hata önlemek için)
        if 'Yıllık TÜFE' in merged.columns: merged['Yıllık TÜFE'] = pd.to_numeric(merged['Yıllık TÜFE'], errors='coerce')
        if 'PPK Faizi' in merged.columns: merged['PPK Faizi'] = pd.to_numeric(merged['PPK Faizi'], errors='coerce')
        
        # --- GRAFİK OLUŞTURMA (PLOTLY) ---
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # 1. Arka Plan: Metin Uzunluğu (Gri Bar) - Sağ Eksen
        fig.add_trace(go.Bar(
            x=merged['period_date'], 
            y=merged['word_count'], 
            name="Metin Uzunluğu", 
            marker=dict(color='rgba(200, 200, 200, 0.3)'), # Şeffaf gri
            yaxis="y3", 
            hoverinfo="x+y+name"
        ))
        
        # 2. Okunabilirlik (Flesch) - Yeşil Baloncuklar - Sağ Eksen (Y3)
        fig.add_trace(go.Scatter(
            x=merged['period_date'], 
            y=merged['flesch_score'], 
            name="Okunabilirlik (Flesch)", 
            mode='markers', 
            marker=dict(color='#2ecc71', size=8, symbol='circle', opacity=0.9),
            yaxis="y3" 
        ))
        
        # 3. Klasik ABG Skoru (Lacivert Çizgi)
        fig.add_trace(go.Scatter(
            x=merged['period_date'], 
            y=merged['abg_dashboard_val'], 
            name="ABF Endeksi (Klasik)", 
            line=dict(color='#000080', width=3), # Navy Blue
            yaxis="y"
        ))
        
        # 4. RoBERTa AI Skoru (Kırmızı, Kalın, Tireli)
        # GÜNCELLEME: Tooltip'e doğrudan "Güvercin %56.4" gibi metni basıyoruz.
        if 'roberta_index' in merged.columns:
            fig.add_trace(go.Scatter(
                x=merged['period_date'], 
                y=merged['roberta_index'], 
                name="AI Sentiment (RoBERTa)", 
                line=dict(color='#FF0000', width=4, dash='dash'), 
                mode='lines+markers', # Noktaların üzerine gelmeyi kolaylaştırır
                yaxis="y",
                text=merged['roberta_desc'] if 'roberta_desc' in merged.columns else "", # <--- ÖZEL METİN BURADA
                hovertemplate='<b>%{text}</b><br>Endeks: %{y:.1f}<extra></extra>' # Tooltip formatı
            ))

        # 5. Piyasa Verileri (İnce Çizgiler)
        if 'Yıllık TÜFE' in merged.columns: 
            fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['Yıllık TÜFE'], name="Yıllık TÜFE (%)", line=dict(color='gray', dash='dot', width=1), yaxis="y"))
        if 'PPK Faizi' in merged.columns: 
            fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['PPK Faizi'], name="Faiz (%)", line=dict(color='orange', dash='dot', width=1), yaxis="y"))

        # Düzen (Layout)
        layout_shapes = [
            dict(type="line", xref="paper", yref="y", x0=0, x1=1, y0=0, y1=0, line=dict(color="black", width=2), layer="below"),
        ]
        
        layout_annotations = [
            dict(x=0.02, y=110, xref="paper", yref="y", text="🦅 ŞAHİN BÖLGESİ", showarrow=False, font=dict(size=12, color="darkred", weight="bold"), xanchor="left"),
            dict(x=0.02, y=-110, xref="paper", yref="y", text="🕊️ GÜVERCİN BÖLGESİ", showarrow=False, font=dict(size=12, color="darkblue", weight="bold"), xanchor="left")
        ]
        
        governors = [("2020-11-01", "N.Ağbal"), ("2021-04-01", "Ş.Kavcıoğlu"), ("2023-06-01", "H.G.Erkan"), ("2024-02-01", "F.Karahan")]
        for start_date, name in governors:
            layout_shapes.append(dict(type="line", xref="x", yref="paper", x0=start_date, x1=start_date, y0=0, y1=1, line=dict(color="gray", width=1, dash="longdash"), layer="below"))
            layout_annotations.append(dict(x=start_date, y=1.05, xref="x", yref="paper", text=f"<b>{name}</b>", showarrow=False, xanchor="left", font=dict(size=9, color="#555")))

        event_links_display = []
        if not df_events.empty:
            for _, ev in df_events.iterrows():
                ev_date = pd.to_datetime(ev['event_date']).strftime('%Y-%m-%d')
                layout_shapes.append(dict(type="line", xref="x", yref="paper", x0=ev_date, x1=ev_date, y0=0, y1=1, line=dict(color="purple", width=2, dash="dot")))
                first_link = ev['links'].split('\n')[0] if ev['links'] else ""
                layout_annotations.append(dict(x=ev_date, y=0.05, xref="x", yref="paper", text=f"ℹ️", showarrow=False, xanchor="left", font=dict(size=14, color="purple")))
                if ev['links']: event_links_display.append({"Tarih": ev_date, "Linkler": [l.strip() for l in ev['links'].split('\n') if l.strip()]})

        fig.update_layout(
            title="Merkez Bankası Sentiment Analizi (AI vs Klasik)", 
            hovermode="x unified", 
            height=650,
            shapes=layout_shapes, 
            annotations=layout_annotations, 
            showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5),
            yaxis=dict(title="Sentiment Endeksi (-100 / +100)", range=[-130, 130], zeroline=False),
            yaxis2=dict(visible=False, overlaying="y", side="right"),
            yaxis3=dict(title="Kelime / Okunabilirlik", overlaying="y", side="right", showgrid=False, visible=False, range=[0, merged['word_count'].max() * 2])
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if event_links_display:
            with st.expander("📅 Grafikteki Önemli Tarihler ve Haber Linkleri"):
                for item in event_links_display:
                    st.markdown(f"**{item['Tarih']}**")
                    for link in item['Linkler']: st.markdown(f"- [Haber Linki]({link})")
                        
        if st.button("🔄 Verileri Yenile (Cache Temizle)"): 
            st.cache_data.clear()
            st.rerun()
            
    else: st.info("Kayıt yok.")

# ==============================================================================
# TAB 2: VERİ GİRİŞİ
# ==============================================================================
with tab2:
    st.subheader("Veri İşlemleri")
    with st.container():
        df_all = utils.fetch_all_data()
        if not df_all.empty: 
            df_all['period_date'] = pd.to_datetime(df_all['period_date']); df_all['date_only'] = df_all['period_date'].dt.date
            current_id = st.session_state['form_data']['id']
            with st.container(border=True):
                if st.button("➕ YENİ VERİ GİRİŞİ (Temizle)", type="secondary"): reset_form(); st.rerun()
                st.markdown("---")
                c1, c2 = st.columns([1, 2])
                with c1:
                    val_date = st.session_state['form_data']['date']; selected_date = st.date_input("Tarih", value=val_date)
                    val_source = st.session_state['form_data']['source']; source = st.text_input("Kaynak", value=val_source)
                with c2:
                    val_text = st.session_state['form_data']['text']; txt = st.text_area("Metin", value=val_text, height=200)
                st.markdown("---")
                if st.session_state['collision_state']['active']:
                    st.error("⚠️ Kayıt Çakışması"); admin_pass = st.text_input("Admin Şifresi", type="password", key="overwrite_pass")
                    if st.button("🚨 Üzerine Yaz", type="primary"):
                        if admin_pass == ADMIN_PWD:
                            p_txt = st.session_state['collision_state']['pending_text']; t_id = st.session_state['collision_state']['target_id']
                            s_abg, _, _, _, _, _, _, _ = utils.run_full_analysis(p_txt)
                            utils.update_entry(t_id, selected_date, p_txt, source, s_abg, s_abg); st.success("Başarılı!"); reset_form(); st.rerun()
                        else: st.error("Hatalı!")
                    if st.button("❌ İptal"): st.session_state['collision_state']['active'] = False; st.rerun()
                elif st.session_state['update_state']['active']:
                    st.warning("Güncelleme Onayı"); update_pass = st.text_input("Admin Şifresi", type="password", key="update_pass")
                    if st.button("💾 Güncelle", type="primary"):
                        if update_pass == ADMIN_PWD:
                            p_txt = st.session_state['update_state']['pending_text']
                            s_abg, _, _, _, _, _, _, _ = utils.run_full_analysis(p_txt)
                            utils.update_entry(current_id, selected_date, p_txt, source, s_abg, s_abg); st.success("Güncellendi!"); reset_form(); st.rerun()
                        else: st.error("Hatalı!")
                    if st.button("❌ İptal"): st.session_state['update_state']['active'] = False; st.rerun()
                else:
                    if st.button("💾 Güncelle" if current_id else "💾 Kaydet", type="primary"):
                        if txt:
                            collision_record = None
                            if not df_all.empty:
                                mask = df_all['date_only'] == selected_date
                                if mask.any(): collision_record = df_all[mask].iloc[0]
                            is_self_update = current_id and ((collision_record is None) or (collision_record is not None and int(collision_record['id']) == current_id))
                            if is_self_update: st.session_state['update_state'] = {'active': True, 'pending_text': txt}; st.rerun()
                            elif collision_record is not None: st.session_state['collision_state'] = {'active': True, 'target_id': int(collision_record['id']), 'target_date': selected_date, 'pending_text': txt}; st.rerun()
                            else:
                                s_abg, _, _, _, _, _, _, _ = utils.run_full_analysis(txt)
                                utils.insert_entry(selected_date, txt, source, s_abg, s_abg); st.success("Eklendi!"); reset_form(); st.rerun()
                        else: st.error("Metin boş.")
                    if current_id:
                        with st.popover("🗑️ Sil"):
                            del_pass = st.text_input("Şifre", type="password", key="del_pass")
                            if st.button("🔥 Sil"):
                                if del_pass == ADMIN_PWD: utils.delete_entry(current_id); st.success("Silindi!"); reset_form(); st.rerun()
                                else: st.error("Hatalı!")
            st.markdown("### 📋 Kayıtlar")
            df_show = df_all.copy(); df_show['Dönem'] = df_show['period_date'].dt.strftime('%Y-%m'); df_show['Skor'] = df_show['score_abg'].apply(lambda x: x*100 if abs(x)<=1 else x)
            event = st.dataframe(df_show[['id', 'Dönem', 'Skor']], on_select="rerun", selection_mode="single-row", use_container_width=True, hide_index=True, key=st.session_state['table_key'])
            if len(event.selection.rows) > 0:
                sel_id = df_show.iloc[event.selection.rows[0]]['id']
                if st.session_state['form_data']['id'] != sel_id:
                    orig = df_all[df_all['id'] == sel_id].iloc[0]
                    st.session_state['form_data'] = {'id': int(orig['id']), 'date': pd.to_datetime(orig['period_date']).date(), 'source': orig['source'], 'text': orig['text_content']}
                    st.rerun()

with tab3:
    st.header("Piyasa Verileri"); d1 = st.date_input("Başlangıç", datetime.date(2023, 1, 1)); d2 = st.date_input("Bitiş", datetime.date.today())
    if st.button("Getir"):
        df, err = utils.fetch_market_data_adapter(d1, d2)
        if not df.empty: st.plotly_chart(go.Figure(data=[go.Scatter(x=df['Donem'], y=df[c], name=c) for c in df.columns if c not in ['Donem','SortDate']]), use_container_width=True); st.dataframe(df)
        else: st.error(err)

with tab4:
    st.header("🔍 Frekans")
    df_all = utils.fetch_all_data()
    if not df_all.empty:
        df_all['period_date'] = pd.to_datetime(df_all['period_date'])
        freq_df, terms = utils.get_top_terms_series(df_all, 7)
        st.plotly_chart(go.Figure(data=[go.Scatter(x=freq_df['period_date'], y=freq_df[t], name=t) for t in terms]), use_container_width=True)
    else: st.info("Veri yok")

with tab5:
    st.header("🤖 Faiz Tahmini")
    df_logs = utils.fetch_all_data()
    if not df_logs.empty and len(df_logs) > 10:
        min_d = pd.to_datetime(df_logs['period_date']).min().date()
        df_m, _ = utils.fetch_market_data_adapter(min_d, datetime.date.today())
        # prepare_ml_dataset artık KeyError hatası vermez
        ml_df = utils.prepare_ml_dataset(df_logs, df_m)
        pred = utils.AdvancedMLPredictor(); stat = pred.train(ml_df)
        if stat == "OK":
            st.success("Model Eğitildi")
            sel_p = st.selectbox("Test Dönemi", df_logs['period_date'].astype(str).tolist())
            if sel_p:
                txt = df_logs[df_logs['period_date'] == sel_p].iloc[0]['text_content']
                res = pred.predict(txt)
                if res: st.json(res)
        else: st.error(stat)
    else: st.info("Yetersiz veri")

with tab6:
    st.header("☁️ WordCloud")
    df_all = utils.fetch_all_data()
    if not df_all.empty:
        sel = st.selectbox("Dönem", ["Tüm"] + df_all['period_date'].astype(str).tolist())
        txt = " ".join(df_all['text_content'].tolist()) if sel == "Tüm" else df_all[df_all['period_date'] == sel].iloc[0]['text_content']
        fig = utils.generate_wordcloud_img(txt)
        if fig: st.pyplot(fig)

with tab7:
    st.header("📜 ABF (2019)"); df_all = utils.fetch_all_data()
    if not df_all.empty:
        abg = utils.calculate_abg_scores(df_all)
        st.plotly_chart(go.Figure(data=go.Scatter(x=abg['period_date'], y=abg['abg_index'])), use_container_width=True)

with tab_struct:
    st.header("🏗️ Yapısal Analiz")
    df_all = utils.fetch_all_data()
    if not df_all.empty:
        sel = st.selectbox("Dönem", df_all['period_date'].astype(str).tolist(), key="str_sel")
        if sel:
            txt = df_all[df_all['period_date'] == sel].iloc[0]['text_content']
            res = utils.analyze_hawk_dove_structural(txt)
            st.metric("Skor", f"{res['net_hawkishness']:.4f}")
            # KeyError hatası artık yok
            st.dataframe(res['matches_df'])

with tab_roberta:
    st.header("🧠 CB-RoBERTa")
    df_all = utils.fetch_all_data()
    if not df_all.empty:
        sel = st.selectbox("Dönem", df_all['period_date'].astype(str).tolist(), key="rob_sel")
        if sel and st.button("Analiz Et"):
            txt = df_all[df_all['period_date'] == sel].iloc[0]['text_content']
            with st.spinner("Analiz ediliyor..."):
                res = utils.analyze_with_roberta(txt)
                if isinstance(res, dict):
                    st.metric(res['best_label'], f"%{res['best_score']*100:.1f}")
                    # Cümle ayrıştırma hatası artık yok
                    df_sent = utils.analyze_sentences_with_roberta(txt)
                    
                    if not df_sent.empty:
                        def color_coding(val):
                            color = 'black'
                            if 'Şahin' in val: color = 'red'
                            elif 'Güvercin' in val: color = 'green'
                            return f'color: {color}; font-weight: bold;'
                        st.dataframe(df_sent.style.map(color_coding, subset=['Etiket']), use_container_width=True)
                    else: st.info("Cümle bulunamadı.")
                else: st.error("Hata")

with tab_imp:
    st.header("📅 Haberler")
    evs = utils.fetch_events()
    st.dataframe(evs)
    d = st.date_input("Tarih"); l = st.text_area("Linkler")
    if st.button("Ekle") and l: utils.add_event(d, l); st.rerun()
