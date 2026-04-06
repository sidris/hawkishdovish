import streamlit as st
import pandas as pd
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
import utils 
import uuid
import numpy as np





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
APP_PWD = "SahinGuvercin35"      
ADMIN_PWD = "SahinGuvercin06"    

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        st.markdown("<br><h3 style='text-align: center;'>🔐 Güvenli Giriş</h3>", unsafe_allow_html=True)
        pwd_input = st.text_input("Uygulama Şifresi", type="password")
        if st.button("Giriş Yap", type="primary"):
            if pwd_input == APP_PWD:
                st.session_state['logged_in'] = True; st.success("Başarılı!"); st.rerun()
            else: st.error("Hatalı!")
    st.stop()

# --- SESSION & STATE ---
if 'form_data' not in st.session_state: st.session_state['form_data'] = {'id': None, 'date': datetime.date.today().replace(day=1), 'source': "TCMB", 'text': ""}
if 'table_key' not in st.session_state: st.session_state['table_key'] = str(uuid.uuid4())
if 'collision_state' not in st.session_state: st.session_state['collision_state'] = {'active': False, 'target_id': None, 'pending_text': None, 'target_date': None}
if 'update_state' not in st.session_state: st.session_state['update_state'] = {'active': False, 'pending_text': None}

if 'stop_words_deep' not in st.session_state: st.session_state['stop_words_deep'] = []
if 'stop_words_cloud' not in st.session_state: st.session_state['stop_words_cloud'] = []

def add_deep_stop():
    word = st.session_state.get("deep_stop_in", "").strip()
    if word and word not in st.session_state['stop_words_deep']:
        st.session_state['stop_words_deep'].append(word)
    st.session_state["deep_stop_in"] = ""

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

c_head1, c_head2 = st.columns([6, 1])
with c_head1: st.title("🦅 Şahin/Güvercin Paneli")
with c_head2: 
    if st.button("Çıkış"): st.session_state['logged_in'] = False; st.rerun()

tab1, tab2, tab3, tab4, tab_textdata, tab6, tab7, tab_roberta, tab_imp = st.tabs([
    "📈 Dashboard",
    "📝 Veri Girişi",
    "📊 Veriler",
    "🔍 Frekans",
    "📚 Text as Data (TF-IDF)",
    "☁️ WordCloud",
    "📜 ABG (2019)",
    "🧠 CB-RoBERTa",
    "📅 Haberler"
])




# --- SESSION & STATE --- bloğunun içine ekleyin:
if 'ai_trend_df' not in st.session_state: st.session_state['ai_trend_df'] = None
if 'ai_step' not in st.session_state: st.session_state['ai_step'] = 3

# ==============================================================================
# TAB 1: DASHBOARD
# ==============================================================================
with tab1:
    with st.spinner("Veriler Yükleniyor..."):
        df_logs = utils.fetch_all_data()
        df_events = utils.fetch_events() 
    
    # 1. Ana Veri Kontrolü
    if not df_logs.empty:
        df_logs['period_date'] = pd.to_datetime(df_logs['period_date'])
        df_logs['Donem'] = df_logs['period_date'].dt.strftime('%Y-%m')
        df_logs['word_count'] = df_logs['text_content'].apply(lambda x: len(str(x).split()) if x else 0)
        df_logs['flesch_score'] = df_logs['text_content'].apply(lambda x: utils.calculate_flesch_reading_ease(str(x)))
        df_logs['score_abg_scaled'] = df_logs['score_abg'].apply(lambda x: x*100 if abs(x) <= 1 else x)

        abg_df = utils.calculate_abg_scores(df_logs)
        abg_df['abg_dashboard_val'] = (abg_df['abg_index'] - 1.0) * 100
        
        min_d = df_logs['period_date'].min().date()
        max_d = datetime.date.today()
        df_market, err = utils.fetch_market_data_adapter(min_d, max_d)
        
        merged = pd.merge(df_logs, df_market, on="Donem", how="left")
        merged = pd.merge(merged, abg_df[['period_date', 'abg_dashboard_val']], on='period_date', how='left')
        
        merged = merged.sort_values("period_date")

                # --- AI SCORE (mrince) merge ---
        ai_df = st.session_state.get("ai_trend_df")
        if ai_df is not None and not ai_df.empty:
            ai_tmp = ai_df.copy()
        
            ai_tmp["AI_EMA"] = pd.to_numeric(ai_tmp.get("AI Score (EMA)", np.nan), errors="coerce")
            ai_tmp["AI_CALIB"] = pd.to_numeric(ai_tmp.get("AI Score (Calib)", np.nan), errors="coerce")
        
            if "Dönem" in ai_tmp.columns:
                ai_tmp["Donem"] = ai_tmp["Dönem"].astype(str)
            else:
                ai_tmp["Donem"] = pd.to_datetime(ai_tmp["period_date"]).dt.strftime("%Y-%m")
        
            ai_tmp = ai_tmp[["Donem", "AI_EMA", "AI_CALIB"]].drop_duplicates(subset=["Donem"])
            merged = pd.merge(merged, ai_tmp, on="Donem", how="left")
        else:
            merged["AI_EMA"] = np.nan
            merged["AI_CALIB"] = np.nan




        
        ai_df = st.session_state.get("ai_trend_df")
        if ai_df is not None and not ai_df.empty:
            ai_tmp = ai_df.copy()
        
            # Güvenli kolon seçimi: AI Score (EMA) varsa onu, yoksa Net Skor'u kullan
            if "AI Score (EMA)" in ai_tmp.columns:
                ai_tmp["AI_DASH"] = pd.to_numeric(ai_tmp["AI Score (EMA)"], errors="coerce")
            else:
                ai_tmp["AI_DASH"] = pd.to_numeric(ai_tmp.get("Net Skor", np.nan), errors="coerce")
        
            # Dönem kolonu Donem ile aynı formatta olmalı (YYYY-MM)
            if "Dönem" in ai_tmp.columns:
                ai_tmp["Donem"] = ai_tmp["Dönem"].astype(str)
            elif "period_date" in ai_tmp.columns:
                ai_tmp["Donem"] = pd.to_datetime(ai_tmp["period_date"]).dt.strftime("%Y-%m")
            else:
                ai_tmp["Donem"] = None
        
            ai_tmp = ai_tmp.dropna(subset=["Donem"])
            ai_tmp = ai_tmp[["Donem", "AI_DASH"]].drop_duplicates(subset=["Donem"])
        
            merged = pd.merge(merged, ai_tmp, on="Donem", how="left")
        else:
            merged["AI_DASH"] = np.nan




        
        if 'Yıllık TÜFE' in merged.columns: merged['Yıllık TÜFE'] = pd.to_numeric(merged['Yıllık TÜFE'], errors='coerce')
        if 'Aylık TÜFE' in merged.columns: merged['Aylık TÜFE'] = pd.to_numeric(merged['Aylık TÜFE'], errors='coerce')
        if 'PPK Faizi' in merged.columns: merged['PPK Faizi'] = pd.to_numeric(merged['PPK Faizi'], errors='coerce')
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Word Count
        fig.add_trace(go.Bar(x=merged['period_date'], y=merged['word_count'], name="Metin Uzunluğu", marker=dict(color='gray'), opacity=0.10, yaxis="y3", hoverinfo="x+y+name"))
        
        # --- SKORLAR ---
        fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['score_abg_scaled'], name="Şahin/Güvercin-Hibrit", line=dict(color='black', width=2, dash='dot'), marker=dict(size=6, color='black'), yaxis="y"))
        fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['abg_dashboard_val'], name="Şahin/Güvercin ABG 2019", line=dict(color='navy', width=4), yaxis="y"))

                # --- AI Score (mrince) çizgileri ---
        if merged["AI_EMA"].notna().any():
            fig.add_trace(go.Scatter(
                x=merged["period_date"], y=merged["AI_EMA"],
                name="CB-RoBERTa (EMA)",
                line=dict(color="green", width=3),
                yaxis="y"
            ))
        
        if merged["AI_CALIB"].notna().any():
            fig.add_trace(go.Scatter(
                x=merged["period_date"], y=merged["AI_CALIB"],
                name="CB-RoBERTa (Calib)",
                line=dict(color="green", width=2, dash="dot"),
                yaxis="y"
            ))



        
        if 'Yıllık TÜFE' in merged.columns: fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['Yıllık TÜFE'], name="Yıllık TÜFE (%)", line=dict(color='red', dash='dot'), yaxis="y"))
        if 'Aylık TÜFE' in merged.columns: fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['Aylık TÜFE'], name="Aylık TÜFE (%)", line=dict(color='crimson', dash='dash', width=1), yaxis="y"))
        if 'PPK Faizi' in merged.columns: fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['PPK Faizi'], name="Faiz (%)", line=dict(color='orange', dash='dot'), yaxis="y"))
        fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['flesch_score'], name="Okunabilirlik (Flesch)", mode='markers', marker=dict(color='teal', size=8, opacity=0.8), yaxis="y"))

        layout_shapes = [
            dict(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=0, y1=150, fillcolor="rgba(255, 0, 0, 0.08)", line_width=0, layer="below"),
            dict(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=-150, y1=0, fillcolor="rgba(0, 0, 255, 0.08)", line_width=0, layer="below"),
            dict(type="line", xref="paper", yref="y", x0=0, x1=1, y0=0, y1=0, line=dict(color="black", width=3), layer="below"),
        ]
        layout_annotations = [
            dict(x=0.02, y=130, xref="paper", yref="y", text="🦅 ŞAHİN", showarrow=False, font=dict(size=14, color="darkred", weight="bold"), xanchor="left"),
            dict(x=0.02, y=-130, xref="paper", yref="y", text="🕊️ GÜVERCİN", showarrow=False, font=dict(size=14, color="darkblue", weight="bold"), xanchor="left")
        ]
        governors = [("2020-11-01", "Naci Ağbal"), ("2021-04-01", "Şahap Kavcıoğlu"), ("2023-06-01", "Hafize Gaye Erkan"), ("2024-02-01", "Fatih Karahan")]
        for start_date, name in governors:
            layout_shapes.append(dict(type="line", xref="x", yref="paper", x0=start_date, x1=start_date, y0=0, y1=1, line=dict(color="gray", width=1, dash="longdash"), layer="below"))
            layout_annotations.append(dict(x=start_date, y=1.02, xref="x", yref="paper", text=f" <b>{name.split()[0][0]}.{name.split()[-1]}</b>", showarrow=False, xanchor="left", font=dict(size=9, color="#555")))

        event_links_display = []
        if not df_events.empty:
            for _, ev in df_events.iterrows():
                ev_date = pd.to_datetime(ev['event_date']).strftime('%Y-%m-%d')
                
                # Olay çizgisi (%20 kısa versiyon)
                layout_shapes.append(dict(
                    type="line", xref="x", yref="paper",
                    x0=ev_date, x1=ev_date, 
                    y0=0, y1=0.2, 
                    line=dict(color="purple", width=2, dash="dot")
                ))

                first_link = ev['links'].split('\n')[0] if ev['links'] else ""
                layout_annotations.append(dict(
                    x=ev_date, y=0.05, xref="x", yref="paper",
                    text=f"ℹ️ <a href='{first_link}' target='_blank'>Haber</a>",
                    showarrow=False, xanchor="left",
                    font=dict(size=10, color="purple"),
                    bgcolor="rgba(255,255,255,0.7)"
                ))
                if ev['links']:
                    links_list = [l.strip() for l in ev['links'].split('\n') if l.strip()]
                    event_links_display.append({"Tarih": ev_date, "Linkler": links_list})

        fig.update_layout(
            title="Merkez Bankası Analiz Paneli", hovermode="x unified", height=600,
            shapes=layout_shapes, annotations=layout_annotations, showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
            yaxis=dict(title="Skor & Oranlar", range=[-150, 150], zeroline=False),
            yaxis2=dict(visible=False, overlaying="y", side="right"),
            yaxis3=dict(title="Kelime", overlaying="y", side="right", showgrid=False, visible=False, range=[0, merged['word_count'].max() * 2])
        )
        st.plotly_chart(fig, use_container_width=True)
        
       # --- YENİ EKLENEN AI TREND GRAFİĞİ (GÜVENLİ VERSİYON) ---
        st.markdown("---")
        st.subheader("🤖 Yapay Zeka (RoBERTa) Trendi")
        
        # Session state dolu mu diye bakıyoruz
        if st.session_state.get('ai_trend_df') is not None and not st.session_state['ai_trend_df'].empty:
            # Grafiği oluşturmaya çalışıyoruz
            fig_ai = utils.create_ai_trend_chart_step(st.session_state['ai_trend_df'], step=st.session_state.get('ai_step', 3)) if hasattr(utils, 'create_ai_trend_chart_step') else utils.create_ai_trend_chart(st.session_state['ai_trend_df'])
            
            # KONTROL: Grafik oluştu mu? (None değilse çiz)
            if fig_ai:
                st.plotly_chart(fig_ai, use_container_width=True, key="ai_chart_dashboard")
            else:
                st.warning("Grafik oluşturulamadı (Veri seti boş olabilir).")
        else:
            st.info("Yapay zeka analizi hesaplama gücü gerektirir. Görüntülemek için aşağıdaki butonu kullanın.")
            if st.button("🚀 AI Analizini Başlat (Dashboard)", key="btn_ai_dash"):
                if not utils.HAS_TRANSFORMERS:
                    st.error("Kütüphaneler eksik.")
                else:
                    with st.spinner("AI Modeli tüm geçmişi tarıyor... Lütfen bekleyin..."):
                        df_all_data = utils.fetch_all_data()
                        # Hesaplamayı yap ve kaydet
                        res_df = utils.calculate_ai_trend_series(df_all_data)
                        
                        if res_df.empty:
                            st.error("Analiz sonucunda hiç veri dönmedi! (Utils dosyasındaki Debug çıktılarına bakın)")
                        else:
                            st.session_state['ai_trend_df'] = res_df
                            st.rerun()
        # -----------------------------------------------------------------------

        if event_links_display:
            with st.expander("📅 Grafikteki Önemli Tarihler ve Haber Linkleri", expanded=False):
                for item in event_links_display:
                    st.markdown(f"**{item['Tarih']}**")
                    for link in item['Linkler']:
                        st.markdown(f"- [Haber Linki]({link})")
                        
       # --- TAB 1'in SON KISMI ---
        
        if st.button("🔄 Yenile"): st.cache_data.clear(); st.rerun()

    # Eğer kayıt yoksa çalışacak ELSE bloğu (Buradaki girintiye dikkat)
    else: 
        st.info("Kayıt yok.")

# ==============================================================================
# TAB 2: VERİ GİRİŞİ
# ==============================================================================
with tab2:
    st.subheader("Veri İşlemleri")
    st.info("ℹ️ **BİLGİ:** Aşağıdaki geçmiş kayıtlar listesinden istediğiniz dönemi seçerek, hangi cümlelerin hesaplamaya alındığını görebilirsiniz.")
    
    with st.container():
        df_all = utils.fetch_all_data()
        # ... (Tab 2 kodları devam eder)
        if not df_all.empty: 
            df_all['period_date'] = pd.to_datetime(df_all['period_date'])
            df_all['date_only'] = df_all['period_date'].dt.date
            current_id = st.session_state['form_data']['id']
            with st.container(border=True):
                if st.button("➕ YENİ VERİ GİRİŞİ (Temizle)", type="secondary"): reset_form(); st.rerun()
                st.markdown("---")
                c1, c2 = st.columns([1, 2])
                with c1:
                    val_date = st.session_state['form_data']['date']; selected_date = st.date_input("Tarih", value=val_date)
                    val_source = st.session_state['form_data']['source']; source = st.text_input("Kaynak", value=val_source)
                    st.caption(f"Dönem: **{selected_date.strftime('%Y-%m')}**")
                with c2:
                    val_text = st.session_state['form_data']['text']; txt = st.text_area("Metin", value=val_text, height=200, placeholder="Metni buraya yapıştırın...")
                st.markdown("---")
                if st.session_state['collision_state']['active']:
                    st.error("⚠️ Kayıt Çakışması"); admin_pass = st.text_input("Admin Şifresi", type="password", key="overwrite_pass")
                    if st.button("🚨 Üzerine Yaz", type="primary"):
                        if admin_pass == ADMIN_PWD:
                            p_txt = st.session_state['collision_state']['pending_text']; t_id = st.session_state['collision_state']['target_id']
                            s_abg, h_cnt, d_cnt, hawks, doves, h_ctx, d_ctx, flesch = utils.run_full_analysis(p_txt)
                            utils.update_entry(t_id, selected_date, p_txt, source, s_abg, s_abg); st.success("Başarılı!"); reset_form(); st.rerun()
                        else: st.error("Hatalı!")
                    if st.button("❌ İptal"): st.session_state['collision_state']['active'] = False; st.rerun()
                elif st.session_state['update_state']['active']:
                    st.warning("Güncelleme Onayı"); update_pass = st.text_input("Admin Şifresi", type="password", key="update_pass")
                    if st.button("💾 Güncelle", type="primary"):
                        if update_pass == ADMIN_PWD:
                            p_txt = st.session_state['update_state']['pending_text']
                            s_abg, h_cnt, d_cnt, hawks, doves, h_ctx, d_ctx, flesch = utils.run_full_analysis(p_txt)
                            utils.update_entry(current_id, selected_date, p_txt, source, s_abg, s_abg); st.success("Güncellendi!"); reset_form(); st.rerun()
                        else: st.error("Hatalı!")
                    if st.button("❌ İptal"): st.session_state['update_state']['active'] = False; st.rerun()
                else:
                    btn_label = "💾 Güncelle" if current_id else "💾 Kaydet"
                    if st.button(btn_label, type="primary"):
                        if txt:
                            collision_record = None
                            if not df_all.empty:
                                mask = df_all['date_only'] == selected_date
                                if mask.any(): collision_record = df_all[mask].iloc[0]
                            is_self_update = current_id and ((collision_record is None) or (collision_record is not None and int(collision_record['id']) == current_id))
                            if is_self_update: st.session_state['update_state'] = {'active': True, 'pending_text': txt}; st.rerun()
                            elif collision_record is not None: st.session_state['collision_state'] = {'active': True, 'target_id': int(collision_record['id']), 'target_date': selected_date, 'pending_text': txt}; st.rerun()
                            else:
                                s_abg, h_cnt, d_cnt, hawks, doves, h_ctx, d_ctx, flesch = utils.run_full_analysis(txt)
                                utils.insert_entry(selected_date, txt, source, s_abg, s_abg); st.success("Eklendi!"); reset_form(); st.rerun()
                        else: st.error("Metin boş.")
                    if current_id:
                        with st.popover("🗑️ Sil"):
                            del_pass = st.text_input("Şifre", type="password", key="del_pass")
                            if st.button("🔥 Sil"):
                                if del_pass == ADMIN_PWD: utils.delete_entry(current_id); st.success("Silindi!"); reset_form(); st.rerun()
                                else: st.error("Hatalı!")
                if txt:
                    s_live, h_cnt, d_cnt, h_list, d_list, h_ctx, d_ctx, flesch_live = utils.run_full_analysis(txt)
                    st.markdown("---"); st.subheader("🔍 Analiz Detayları")
                    c1, c2, c3 = st.columns(3)
                    with c1: st.metric("Şahin", f"{h_cnt}")
                    with c2: st.metric("Güvercin", f"{d_cnt}")
                    with c3: st.metric("Flesch", f"{flesch_live:.1f}")
                    st.caption(f"**Net Skor:** {s_live:.2f}")
                    with st.expander("📄 Tespit Edilen Cümleler", expanded=True):
                        k1, k2 = st.columns(2)
                        with k1:
                            st.markdown("#### 🦅 Şahin")
                            if h_list:
                                for item in h_list:
                                    t = item.split(' (')[0]; st.markdown(f"**{item}**")
                                    if t in h_ctx: 
                                        for s in h_ctx[t]: st.caption(f"📝 {s}")
                            else: st.write("- Yok")
                        with k2:
                            st.markdown("#### 🕊️ Güvercin")
                            if d_list:
                                for item in d_list:
                                    t = item.split(' (')[0]; st.markdown(f"**{item}**")
                                    if t in d_ctx: 
                                        for s in d_ctx[t]: st.caption(f"📝 {s}")
                            else: st.write("- Yok")
            st.markdown("### 📋 Kayıtlar")
            df_show = df_all.copy()
            df_show['Dönem'] = df_show['period_date'].dt.strftime('%Y-%m')
            df_show['Görsel Skor'] = df_show['score_abg'].apply(lambda x: x*100 if abs(x)<=1 else x)
            event = st.dataframe(df_show[['id', 'Dönem', 'Görsel Skor']], on_select="rerun", selection_mode="single-row", use_container_width=True, hide_index=True, key=st.session_state['table_key'])
            if len(event.selection.rows) > 0:
                sel_id = df_show.iloc[event.selection.rows[0]]['id']
                if st.session_state['collision_state']['active'] or st.session_state['update_state']['active']: st.session_state['collision_state']['active'] = False; st.session_state['update_state']['active'] = False
                if st.session_state['form_data']['id'] != sel_id:
                    orig = df_all[df_all['id'] == sel_id].iloc[0]
                    st.session_state['form_data'] = {'id': int(orig['id']), 'date': pd.to_datetime(orig['period_date']).date(), 'source': orig['source'], 'text': orig['text_content']}
                    st.rerun()

with tab_imp:
    st.header("📅 Haberler (Event Logs)")
    st.caption("Seçtiğin tarihe haber linkleri ekle. Dashboard grafiğinde mor çizgiler olarak görünür.")

    try:
        # --- 1) Yeni event ekleme ---
        c1, c2 = st.columns([1, 3])
        with c1:
            ev_date = st.date_input("Tarih", value=datetime.date.today(), key="ev_date_in")
        with c2:
            ev_links = st.text_area(
                "Haber linkleri (her satıra 1 link)",
                height=140,
                placeholder="https://...\nhttps://...\n...",
                key="ev_links_in"
            )

        col_add, col_refresh = st.columns([1, 1])
        with col_add:
            if st.button("➕ Kaydet", type="primary", key="btn_add_event"):
                links_clean = "\n".join([l.strip() for l in (ev_links or "").splitlines() if l.strip()])
                if not links_clean:
                    st.warning("En az 1 link gir.")
                else:
                    utils.add_event(ev_date, links_clean)
                    st.success("Eklendi!")
                    st.rerun()

        with col_refresh:
            if st.button("🔄 Yenile", key="btn_refresh_events"):
                st.rerun()

        st.divider()

        # --- 2) Event listesi ---
        df_events = utils.fetch_events()

        if df_events is None or df_events.empty:
            st.info("Henüz kayıtlı haber yok.")
        else:
            df_events = df_events.copy()

            # kolon güvenliği
            if "event_date" in df_events.columns:
                df_events["event_date"] = pd.to_datetime(df_events["event_date"], errors="coerce")
                df_events = df_events.dropna(subset=["event_date"]).sort_values("event_date", ascending=False)

            st.subheader("📌 Kayıtlı Haberler")

            # Görsel liste (kart gibi)
            for _, row in df_events.iterrows():
                rid = row.get("id", None)
                d = row.get("event_date", None)
                d_str = pd.to_datetime(d).strftime("%Y-%m-%d") if pd.notna(d) else "—"

                links_raw = row.get("links", "") or ""
                links_list = [l.strip() for l in links_raw.splitlines() if l.strip()]

                with st.container(border=True):
                    top1, top2 = st.columns([5, 1])
                    with top1:
                        st.markdown(f"### {d_str}")
                        if links_list:
                            for l in links_list:
                                st.markdown(f"- {l}")
                        else:
                            st.caption("Link yok")

                    with top2:
                        if rid is not None:
                            if st.button("🗑️ Sil", key=f"del_event_{rid}"):
                                utils.delete_event(int(rid))
                                st.success("Silindi!")
                                st.rerun()

            # Ham tablo da isteyen için
            with st.expander("📋 Ham Tablo", expanded=False):
                st.dataframe(df_events, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error("Haberler sekmesi hata aldı.")
        st.exception(e)






with tab3:
    st.header("Piyasa Verileri")
    d1 = st.date_input("Başlangıç", datetime.date(2023, 1, 1))
    d2 = st.date_input("Bitiş", datetime.date.today())
    if st.button("Getir", key="get_market"):
        utils.fetch_market_data_adapter.clear()
        df, err = utils.fetch_market_data_adapter(d1, d2)
        # Debug bilgisi
        debug_info = st.session_state.get("_tufe_debug", [])
        if debug_info:
            with st.expander("🔧 Debug (TÜFE fetch)", expanded=True):
                for line in debug_info:
                    st.text(line)
        if not df.empty:
            fig_m = go.Figure()
            if 'Yıllık TÜFE' in df.columns: fig_m.add_trace(go.Scatter(x=df['Donem'], y=df['Yıllık TÜFE'], name="Yıllık TÜFE", line=dict(color='red')))
            if 'Aylık TÜFE' in df.columns: fig_m.add_trace(go.Scatter(x=df['Donem'], y=df['Aylık TÜFE'], name="Aylık TÜFE", line=dict(color='blue', dash='dot')))
            if 'PPK Faizi' in df.columns: fig_m.add_trace(go.Scatter(x=df['Donem'], y=df['PPK Faizi'], name="Faiz", line=dict(color='orange')))
            st.plotly_chart(fig_m, use_container_width=True)
            st.dataframe(df, use_container_width=True)
        else: st.error(f"Hata: {err}")


def _render_tab4_frequency():
    st.header("🔍 Frekans (İzlenen Terimler)")

    # -----------------------------
    # 0) Varsayılan izlenecek terimler
    # -----------------------------
    DEFAULT_WATCH_TERMS = [
        "inflation", "disinflation", "stability", "growth", "gdp",
        "interest rate", "policy rate", "lowered", "macroprudential",
        "target", "monetary policy", "tightened", "risks", "exchange rate",
        "prudently", "global", "recession", "food"
    ]

    if "watch_terms" not in st.session_state:
        st.session_state["watch_terms"] = DEFAULT_WATCH_TERMS.copy()

    # -----------------------------
    # 1) Helper fonksiyonlar (TAB içinde güvenli)
    # -----------------------------
    def add_watch_term():
        t = (st.session_state.get("watch_term_in", "") or "").strip().lower()
        if not t:
            return
        if t not in st.session_state["watch_terms"]:
            st.session_state["watch_terms"].append(t)
        # widget key'ine dokunma (StreamlitAPIException önlemi)

    def reset_watch_terms():
        st.session_state["watch_terms"] = DEFAULT_WATCH_TERMS.copy()

    # ✅ YENİ: grafikteki tüm kelimeleri sıfırla (listeyi boşalt)
    def clear_watch_terms():
        st.session_state["watch_terms"] = []

    def _fallback_build_watch_terms_timeseries(df_in: pd.DataFrame, terms: list) -> pd.DataFrame:
        """utils'te fonksiyon yoksa diye basit fallback: substring count"""
        rows = []
        for _, r in df_in.iterrows():
            txt = str(r.get("text_content", "") or "").lower()
            rec = {"period_date": r["period_date"], "Donem": r["Donem"]}
            for term in terms:
                rec[term] = txt.count(term.lower())
            rows.append(rec)
        return pd.DataFrame(rows).sort_values("period_date").reset_index(drop=True)

    # -----------------------------
    # 2) Veri çek / temizle
    # -----------------------------
    df_all = utils.fetch_all_data()
    if df_all is None or df_all.empty:
        st.info("Yeterli veri yok.")
        return

    df_all = df_all.copy()
    df_all["period_date"] = pd.to_datetime(df_all.get("period_date", None), errors="coerce")
    df_all = df_all.dropna(subset=["period_date"]).sort_values("period_date", ascending=False).reset_index(drop=True)
    df_all["Donem"] = df_all["period_date"].dt.strftime("%Y-%m")

    # -----------------------------
    # 3) UI: kelime ekleme / reset / silme
    # -----------------------------
    st.caption(
        "Bu bölüm sadece izlediğin kelimeleri gösterir. "
        "Yeni kelime eklersen ve metinlerde geçiyorsa otomatik seriye girer."
    )

    # ✅ YENİ: 3 kolon (Varsayılan reset + kelimeleri sıfırla)
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        st.text_input(
            "➕ Kelime veya phrase ekle (Enter)",
            key="watch_term_in",
            on_change=add_watch_term,
            placeholder="ör: liquidity, demand, wage, credit growth"
        )
    with c2:
        if st.button("↩️ Reset (Varsayılan)", type="secondary"):
            reset_watch_terms()
            st.rerun()
    with c3:
        if st.button("🧹 Sıfırla (Tüm Kelimeler)", type="secondary"):
            clear_watch_terms()
            st.rerun()

    terms = st.session_state.get("watch_terms", [])
    if not terms:
        st.warning("İzlenen kelime yok. Yukarıdan ekleyebilirsin.")
        return

    st.write("Aktif izlenen terimler:")
    cols = st.columns(6)
    for i, term in enumerate(list(terms)):
        if cols[i % 6].button(f"{term} ✖", key=f"watch_del_{term}"):
            st.session_state["watch_terms"].remove(term)
            st.rerun()

    st.divider()

       # -----------------------------
    # 4) Grafik: dönemlere snap hover + üstte tooltip + x ekseninde dönemler
    # -----------------------------
    # ts_df üretimini GARANTİ ET
    if hasattr(utils, "build_watch_terms_timeseries"):
        ts_df = utils.build_watch_terms_timeseries(df_all, terms)
    else:
        ts_df = _fallback_build_watch_terms_timeseries(df_all, terms)

    if ts_df is None or ts_df.empty:
        st.info("Grafik için yeterli veri yok.")
        return

    # bazı durumlarda term kolonu oluşmayabilir (güvenlik)
    missing = [t for t in terms if t not in ts_df.columns]
    if missing:
        st.warning(f"Bu terimler seride bulunamadı: {', '.join(missing)}")
        terms = [t for t in terms if t in ts_df.columns]
        if not terms:
            return

    ts_df = ts_df.sort_values("period_date").reset_index(drop=True)

    import plotly.graph_objects as go

    fig = go.Figure()

    # çizgiler (hover kapalı)
    for term in terms:
        fig.add_trace(
            go.Scatter(
                x=ts_df["Donem"],
                y=ts_df[term],
                mode="lines+markers",
                name=term,
                hoverinfo="skip"
            )
        )

    # hover metni (yan yana sütun)
    def _build_hover_rows(row, terms_list, ncols=3):
        items = [f"{t}: {int(row[t])}" for t in terms_list]
        cols = [items[i::ncols] for i in range(ncols)]
        maxlen = max(len(c) for c in cols) if cols else 0
        lines = []
        for i in range(maxlen):
            parts = [c[i] for c in cols if i < len(c)]
            lines.append(" | ".join(parts))
        return "<br>".join(lines)

    hover_text = [_build_hover_rows(r, terms, ncols=3) for _, r in ts_df.iterrows()]

    # tooltip üstte dursun diye taşıyıcı trace'i yukarı koy
    y_top = float(ts_df[terms].max().max()) * 1.15 if len(terms) else 1.0

    fig.add_trace(
        go.Scatter(
            x=ts_df["Donem"],
            y=[y_top] * len(ts_df),
            mode="markers",
            marker=dict(opacity=0),
            showlegend=False,
            customdata=hover_text,
            hovertemplate="<b>%{x}</b><br>%{customdata}<extra></extra>"
        )
    )

    # x ekseni: dönemleri yaz + dönemlere snap + spike
    x_ticks = ts_df["Donem"].tolist()
    fig.update_xaxes(
        tickmode="array",
        tickvals=x_ticks,
        ticktext=x_ticks,
        tickangle=-45,
        showspikes=True,
        spikemode="across",
        spikesnap="data",
        spikethickness=1
    )

    fig.update_layout(
        hovermode="x",
        hoverlabel=dict(font_size=11, align="left"),
        margin=dict(t=40, l=10, r=10, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)




    # -----------------------------
    # 5) Diff analizi
    # -----------------------------
    st.markdown("---")
    st.subheader("🔄 Metin Farkı (Diff) Analizi")

    c_diff1, c_diff2 = st.columns(2)
    with c_diff1:
        sel_date1 = st.selectbox("Eski Metin:", df_all["Donem"].tolist(), index=min(1, len(df_all) - 1), key="diff_old")
    with c_diff2:
        sel_date2 = st.selectbox("Yeni Metin:", df_all["Donem"].tolist(), index=0, key="diff_new")

    if st.button("Farkları Göster", type="primary", key="btn_diff"):
        t1 = str(df_all[df_all["Donem"] == sel_date1].iloc[0].get("text_content", "") or "")
        t2 = str(df_all[df_all["Donem"] == sel_date2].iloc[0].get("text_content", "") or "")
        diff_html = utils.generate_diff_html(t1, t2)

        st.markdown(f"**Kırmızı:** {sel_date1}'den silinenler | **Yeşil:** {sel_date2}'ye eklenenler")
        with st.container(border=True, height=400):
            st.markdown(diff_html, unsafe_allow_html=True)




    # ==============================================================================
    # TAB: TEXT AS DATA (TF-IDF) — HYBRID + CPI delta_bp tahmini
    # ==============================================================================

with tab4:
    _render_tab4_frequency()
def _render_tab_text_as_data():
    st.header("📚 Text as Data (TF-IDF) — HYBRID + CPI PPK Kararı (delta_bp) Tahmini")



    if not utils.HAS_ML_DEPS:
        st.error("ML kütüphaneleri eksik (sklearn).")
        return

    df_logs = utils.fetch_all_data()
    if df_logs is None or df_logs.empty:
        st.info("Veri yok.")
        return

    df_logs = df_logs.copy()
    df_logs["period_date"] = pd.to_datetime(df_logs["period_date"], errors="coerce")
    df_logs = df_logs.dropna(subset=["period_date"]).sort_values("period_date")

    # numeric kolonları sayısala çevir
    for c in ["policy_rate", "delta_bp"]:
        if c in df_logs.columns:
            df_logs[c] = pd.to_numeric(df_logs[c], errors="coerce")

    # CPI / market data çek
    min_d = df_logs["period_date"].min().date()
    max_d = datetime.date.today()
    df_market, err = utils.fetch_market_data_adapter(min_d, max_d)
    if err:
        st.warning(f"Market veri uyarısı: {err}")

    # HYBRID + CPI prepare
    df_td = utils.textasdata_prepare_df_hybrid_cpi(
        df_logs,
        df_market,
        text_col="text_content",
        date_col="period_date",
        y_col="delta_bp",
        rate_col="policy_rate"
    )

    if df_td.empty or df_td["delta_bp"].notna().sum() < 10:
        st.warning("HYBRID+CPI eğitim için yeterli gözlem yok. (En az ~10 kayıt önerilir)")
        return

    # -------------------------
    # 1) Model ayarları
    # -------------------------
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.info(
            "Bu sekme **English TF-IDF (word+char)** + **faiz geçmişi** + **TÜFE (lagged)** ile "
            "**delta_bp (bps)** tahmin eder. Walk-forward backtest gösterir."
        )
    with c2:
        min_df = st.number_input("min_df", min_value=1, max_value=10, value=2, step=1)
    with c3:
        alpha = st.number_input("Ridge alpha", min_value=0.1, max_value=80.0, value=10.0, step=1.0)

    if "textasdata_model" not in st.session_state:
        st.session_state["textasdata_model"] = None

    if st.button("🚀 Modeli Eğit / Yenile (HYBRID + CPI)", type="primary"):
        with st.spinner("Eğitiliyor + walk-forward backtest..."):
            out = utils.train_textasdata_hybrid_cpi_ridge(
                df_td,
                min_df=int(min_df),
                alpha=float(alpha),
                n_splits=6,
                word_ngram=(1, 2),
                char_ngram=(3, 5),
                max_features_word=12000,
                max_features_char=20000
            )
            st.session_state["textasdata_model"] = out
        st.success("Hazır!")

    model_pack = st.session_state.get("textasdata_model")
    if not model_pack:
        st.info("Başlamak için yukarıdaki butona bas.")
        return

    # -------------------------
    # 2) Backtest Özeti
    # -------------------------
    metrics = model_pack.get("metrics", {}) or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE (bps)", f"{metrics.get('mae', np.nan):.1f}")
    c2.metric("RMSE (bps)", f"{metrics.get('rmse', np.nan):.1f}")
    c3.metric("R²", f"{metrics.get('r2', np.nan):.2f}")
    c4.metric("Gözlem", f"{metrics.get('n', 0)}")

    df_pred = model_pack.get("pred_df")
    if df_pred is not None and not df_pred.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_pred["period_date"], y=df_pred["delta_bp"],
            name="Gerçek delta_bp", opacity=0.45
        ))
        fig.add_trace(go.Scatter(
            x=df_pred["period_date"], y=df_pred["pred_delta_bp"],
            name="Walk-forward Tahmin", mode="lines+markers"
        ))
        fig.add_hline(y=0, line_color="black", opacity=0.25)
        fig.update_layout(
            title="Text-as-Data HYBRID + CPI Backtest — delta_bp (bps)",
            hovermode="x unified",
            height=420,
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # 3) Kelime Etkileri (word TF-IDF)
    # -------------------------
    coef_df = model_pack.get("coef_df")
    if coef_df is not None and not coef_df.empty:
        st.subheader("🧠 Which words push hike/cut? (word TF-IDF coefficients)")
        k = st.slider("Show top K", 10, 60, 25, step=5)

        cpos, cneg = st.columns(2)
        with cpos:
            st.markdown("### 🔺 Hike-leaning (positive)")
            st.dataframe(coef_df.sort_values("coef", ascending=False).head(int(k)),
                         use_container_width=True, hide_index=True)
        with cneg:
            st.markdown("### 🔻 Cut-leaning (negative)")
            st.dataframe(coef_df.sort_values("coef", ascending=True).head(int(k)),
                         use_container_width=True, hide_index=True)

    # -------------------------
    # 4) Tek Metin Tahmini
    # -------------------------
    st.divider()
    st.subheader("🔮 Single-text Prediction (HYBRID + CPI)")

    last_rate = float(df_td["policy_rate"].dropna().iloc[-1]) if df_td["policy_rate"].notna().any() else np.nan
    st.caption(f"Last known policy_rate: {last_rate if np.isfinite(last_rate) else '—'}")

    txt = st.text_area("Paste the statement text", height=220, placeholder="Paste PPK statement...")

    if st.button("🧾 Predict (HYBRID + CPI)", type="secondary"):
        if not txt or len(txt.strip()) < 30:
            st.warning("Text too short.")
        else:
            pred = utils.predict_textasdata_hybrid_cpi(model_pack, df_td, txt)
            pred_bp = float(pred.get("pred_delta_bp", 0.0))
            implied = (last_rate + pred_bp / 100.0) if np.isfinite(last_rate) else np.nan

            c1, c2 = st.columns(2)
            c1.metric("Predicted delta_bp", f"{pred_bp:.0f} bps")
            c2.metric("Implied policy_rate", f"{implied:.2f}" if np.isfinite(implied) else "—")

    def textasdata_prepare_df_hybrid_cpi(
        df_logs: pd.DataFrame,
        df_market: pd.DataFrame,
        text_col: str = "text_content",
        date_col: str = "period_date",
        y_col: str = "delta_bp",
        rate_col: str = "policy_rate",
    ) -> pd.DataFrame:
        """
        Amaç: delta_bp (bps) tahmini için text+numeric+TÜFE özellikli dataset hazırlamak.

        Çıktı kolonları:
          - period_date (datetime)
          - text (clean)
          - delta_bp (float)  [target]
          - policy_rate (float)
          - cpi_yoy, cpi_mom (float)  [market'ten]
          - cpi_yoy_l1, cpi_mom_l1 (lag)
          - prev_delta_bp (lag)
          - prev_policy_rate (lag)
        """
        if df_logs is None or df_logs.empty:
            return pd.DataFrame()

        df = df_logs.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col)

        # Sayısal kolonlar
        if y_col in df.columns:
            df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
        else:
            df[y_col] = np.nan

        if rate_col in df.columns:
            df[rate_col] = pd.to_numeric(df[rate_col], errors="coerce")
        else:
            df[rate_col] = np.nan

        # Donem anahtarı
        df["Donem"] = df[date_col].dt.strftime("%Y-%m")

        # Market merge (TÜFE)
        if df_market is not None and not df_market.empty and "Donem" in df_market.columns:
            mk = df_market.copy()
            # beklenen kolonlar: "Yıllık TÜFE", "Aylık TÜFE" (senin adapter bunları döndürüyor)
            if "Yıllık TÜFE" in mk.columns:
                mk["Yıllık TÜFE"] = pd.to_numeric(mk["Yıllık TÜFE"], errors="coerce")
            if "Aylık TÜFE" in mk.columns:
                mk["Aylık TÜFE"] = pd.to_numeric(mk["Aylık TÜFE"], errors="coerce")

            df = pd.merge(df, mk[["Donem"] + [c for c in ["Yıllık TÜFE", "Aylık TÜFE"] if c in mk.columns]],
                          on="Donem", how="left")

        # Text clean
        df["text"] = df[text_col].fillna("").astype(str).apply(normalize_tr_text)

        # CPI feature names
        df["cpi_yoy"] = pd.to_numeric(df.get("Yıllık TÜFE", np.nan), errors="coerce")
        df["cpi_mom"] = pd.to_numeric(df.get("Aylık TÜFE", np.nan), errors="coerce")

        # Lagged CPI
        df["cpi_yoy_l1"] = df["cpi_yoy"].shift(1)
        df["cpi_mom_l1"] = df["cpi_mom"].shift(1)

        # Lagged policy vars
        df["prev_delta_bp"] = pd.to_numeric(df[y_col].shift(1), errors="coerce")
        df["prev_policy_rate"] = pd.to_numeric(df[rate_col].shift(1), errors="coerce")

        # Son temizlik
        out = df[[date_col, "text", y_col, rate_col, "cpi_yoy", "cpi_mom", "cpi_yoy_l1", "cpi_mom_l1", "prev_delta_bp", "prev_policy_rate"]].copy()
        out = out.rename(columns={date_col: "period_date"})
        return out


    def train_textasdata_hybrid_cpi_ridge(
        df_td: pd.DataFrame,
        min_df: int = 2,
        alpha: float = 10.0,
        n_splits: int = 6,
        word_ngram=(1, 2),
        char_ngram=(3, 5),
        max_features_word: int = 12000,
        max_features_char: int = 20000,
    ):
        """
        HYBRID model:
          - word TF-IDF
          - char TF-IDF
          - numeric features (policy_rate, prev_*, cpi*)
        Target: delta_bp

        Walk-forward backtest: TimeSeriesSplit
        Returns: dict(model, metrics, pred_df, coef_df)
        """
        if not HAS_ML_DEPS:
            return {}

        # --- Input guard ---
        if df_td is None or df_td.empty:
            return {}

        df = df_td.copy().sort_values("period_date").reset_index(drop=True)
        df["delta_bp"] = pd.to_numeric(df["delta_bp"], errors="coerce")

        # target boş olanları at
        df = df.dropna(subset=["delta_bp", "text"])
        if len(df) < max(10, n_splits + 3):
            return {}

        # numeric features
        num_cols = ["policy_rate", "prev_delta_bp", "prev_policy_rate", "cpi_yoy", "cpi_mom", "cpi_yoy_l1", "cpi_mom_l1"]
        for c in num_cols:
            if c not in df.columns:
                df[c] = np.nan
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(method="ffill").fillna(0.0)

        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge

        # preprocess
        word_vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=word_ngram,
            min_df=int(min_df),
            max_features=int(max_features_word),
            sublinear_tf=True
        )
        char_vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=char_ngram,
            min_df=int(min_df),
            max_features=int(max_features_char),
            sublinear_tf=True
        )

        pre = ColumnTransformer(
            transformers=[
                ("w", word_vec, "text"),
                ("c", char_vec, "text"),
                ("n", Pipeline([("sc", StandardScaler(with_mean=False))]), num_cols),
            ],
            remainder="drop",
            sparse_threshold=0.3
        )

        model = Pipeline([
            ("prep", pre),
            ("reg", Ridge(alpha=float(alpha), random_state=42))
        ])

        X = df[["text"] + num_cols]
        y = df["delta_bp"].values.astype(float)

        # --- Walk-forward ---
        tscv = TimeSeriesSplit(n_splits=min(int(n_splits), max(2, len(df) // 4)))
        pred = np.full(len(df), np.nan)

        for tr, te in tscv.split(X):
            model.fit(X.iloc[tr], y[tr])
            pred[te] = model.predict(X.iloc[te])

        mask = np.isfinite(pred)
        mae = float(mean_absolute_error(y[mask], pred[mask])) if mask.any() else np.nan
        rmse = float(np.sqrt(mean_squared_error(y[mask], pred[mask]))) if mask.any() else np.nan
        r2 = float(r2_score(y[mask], pred[mask])) if mask.any() else np.nan

        pred_df = df[["period_date", "delta_bp"]].copy()
        pred_df["pred_delta_bp"] = pred

        # --- Fit final on all data ---
        model.fit(X, y)

        # --- Coef extraction (word tfidf only) ---
        coef_df = pd.DataFrame()
        try:
            # pipeline -> prep -> 'w' vectorizer feature names
            reg = model.named_steps["reg"]
            prep = model.named_steps["prep"]
            w_vec = prep.named_transformers_["w"]
            w_names = np.array(w_vec.get_feature_names_out(), dtype=object)

            # coef vector = [word_feats, char_feats, numeric_feats] birleşik.
            # word boyutu:
            n_w = len(w_names)
            coefs = np.asarray(reg.coef_).ravel()
            w_coef = coefs[:n_w]

            coef_df = pd.DataFrame({"term": w_names, "coef": w_coef})
            coef_df = coef_df.replace([np.inf, -np.inf], np.nan).dropna()
        except Exception:
            coef_df = pd.DataFrame()

        return {
            "model": model,
            "metrics": {"mae": mae, "rmse": rmse, "r2": r2, "n": int(len(df))},
            "pred_df": pred_df,
            "coef_df": coef_df
        }


    def predict_textasdata_hybrid_cpi(model_pack: dict, df_td: pd.DataFrame, text: str) -> dict:
        """
        Tek metin için delta_bp tahmin eder.
        Numeric side: df_td'nin son satırındaki (policy_rate, cpi lag vs.) değerleri kullanır.
        """
        if not model_pack or "model" not in model_pack:
            return {"pred_delta_bp": 0.0}

        model = model_pack["model"]
        if df_td is None or df_td.empty:
            last = {}
        else:
            last = df_td.sort_values("period_date").iloc[-1].to_dict()

        num_cols = ["policy_rate", "prev_delta_bp", "prev_policy_rate", "cpi_yoy", "cpi_mom", "cpi_yoy_l1", "cpi_mom_l1"]
        row = {"text": normalize_tr_text(text)}
        for c in num_cols:
            v = last.get(c, 0.0)
            try:
                row[c] = float(v) if np.isfinite(float(v)) else 0.0
            except Exception:
                row[c] = 0.0

        X_one = pd.DataFrame([row])
        pred = float(model.predict(X_one)[0])
        return {"pred_delta_bp": pred}




with tab_textdata:
    _render_tab_text_as_data()
with tab6:
    st.header("☁️ Kelime Bulutu (WordCloud)")
    # WordCloud sekmesi, TF-IDF modeline bağlı olmamalı; burada veriyi yeniden çekip normalize ediyoruz.
    df_all = utils.fetch_all_data()
    if df_all is None:
        df_all = pd.DataFrame()
    if not df_all.empty:
        df_all = df_all.copy()
        df_all["period_date"] = pd.to_datetime(df_all.get("period_date", None), errors="coerce")
        df_all = df_all.dropna(subset=["period_date"]).sort_values("period_date", ascending=False).reset_index(drop=True)
        df_all["Donem"] = df_all["period_date"].dt.strftime("%Y-%m")

    if not df_all.empty:
        st.text_input("🚫 Buluttan Çıkarılacak Kelimeler (Enter)", key="cloud_stop_in", on_change=add_cloud_stop)
        if st.session_state['stop_words_cloud']:
            st.write("Filtreler:")
            cols = st.columns(8)
            for i, word in enumerate(st.session_state['stop_words_cloud']):
                if cols[i % 8].button(f"{word} ✖", key=f"del_cloud_{word}"):
                    st.session_state['stop_words_cloud'].remove(word)
                    st.rerun()
        st.divider()
        dates = df_all['Donem'].tolist()
        sel_cloud_date = st.selectbox("Dönem Seçin:", ["Tüm Zamanlar"] + dates)
        if st.button("Bulutu Oluştur", type="primary"):
            if sel_cloud_date == "Tüm Zamanlar": text_cloud = " ".join(df_all['text_content'].astype(str).tolist())
            else: text_cloud = df_all[df_all['Donem'] == sel_cloud_date].iloc[0]['text_content']
            fig_wc = utils.generate_wordcloud_img(text_cloud, st.session_state['stop_words_cloud'])
            if fig_wc: st.pyplot(fig_wc)
            else: st.error("Kütüphane eksik veya metin boş.")
    else: st.info("Veri yok.")

with tab7:
    st.header("📜 Apel, Blix ve Grimaldi (2019) Analizi")
    st.info("Bu yöntem, kelimeleri 'enflasyon', 'büyüme', 'istihdam' gibi kategorilere ayırarak, yanlarındaki sıfatlara göre 'Şahin' veya 'Güvercin' olarak puanlar.")
    df_abg_source = utils.fetch_all_data()
    if not df_abg_source.empty:
        df_abg_source = df_abg_source.copy()
        df_abg_source['period_date'] = pd.to_datetime(df_abg_source['period_date'])
        df_abg_source['Donem'] = df_abg_source['period_date'].dt.strftime('%Y-%m')
        abg_df = utils.calculate_abg_scores(df_abg_source)
        fig_abg = go.Figure()
        fig_abg.add_trace(go.Scatter(x=abg_df['period_date'], y=abg_df['abg_index'], name="ABF Net Hawkishness", line=dict(color='purple', width=3), marker=dict(size=8)))
        fig_abg.add_shape(type="line", x0=abg_df['period_date'].min(), x1=abg_df['period_date'].max(), y0=1, y1=1, line=dict(color="gray", dash="dash"))
        fig_abg.update_layout(title="ABF (2019) Endeksi Zaman Serisi (Nötr=1.0)", yaxis_title="Hawkishness Index (0 - 2)", hovermode="x unified")
        st.plotly_chart(fig_abg, use_container_width=True)
        st.divider()
        st.subheader("🔍 Dönem Bazlı Detaylar")
        sel_abg_period = st.selectbox("İncelenecek Dönem:", abg_df['Donem'].tolist())
        if sel_abg_period:
            subset = df_abg_source[df_abg_source['Donem'] == sel_abg_period]
            if not subset.empty:
                text_abg = subset.iloc[0]['text_content']
                
                # Analiz fonksiyonunu çağır
                res = utils.analyze_hawk_dove(
                    text_abg, 
                    DICT=utils.DICT, 
                    window_words=10, 
                    dedupe_within_term_window=True, 
                    nearest_only=True
                )
                
                net_h = res.get('net_hawkishness', 0)
                h_cnt = res.get('hawk_count', 0)
                d_cnt = res.get('dove_count', 0)
                details = res.get('match_details', [])
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Net Endeks", f"{net_h:.4f}")
                c2.metric("🦅 Şahin Eşleşme", h_cnt)
                c3.metric("🕊️ Güvercin Eşleşme", d_cnt)
                
                if "topic_counts" in res:
                      with st.expander("Detaylı Kırılım (Topic Counts)"):
                          st.json(res["topic_counts"])

                with st.expander("📝 Detaylı Eşleşme Tablosu (Cümle Bağlamı)", expanded=True):
                    if details:
                        detail_data = []
                        for m in details:
                            detail_data.append({"Tip": "🦅 ŞAHİN" if m['type'] == "HAWK" else "🕊️ GÜVERCİN", "Eşleşen Terim": m['term'], "Cümle": m['sentence']})
                        st.dataframe(pd.DataFrame(detail_data), use_container_width=True, hide_index=True)
                    else: 
                        st.info("Bu metinde herhangi bir ABF sözlük eşleşmesi bulunamadı.")
                
                with st.expander("Metin Önizleme"): st.write(text_abg)
            else: st.error("Seçilen dönem için metin bulunamadı.")
    else: st.info("Analiz için veri yok.")

# ==============================================================================
# TAB ROBERTA: CB-RoBERTa
# ==============================================================================

def _render_tab_roberta():
    st.header("🧠 CentralBankRoBERTa (Yapay Zeka Analizi)")

    if not utils.HAS_TRANSFORMERS:
        st.error("Kütüphaneler eksik. (transformers/torch)")
        return

    # -------------------------
    # 1) GENEL TREND
    # -------------------------
    st.subheader("📈 Tarihsel Trend (Calib + EMA + Hysteresis)")

    st.radio(
        "Grafik adımı (tıkla, özellik ekle):",
        options=[
            "0) Ham sinyal (Diff)",
            "1) + Kalibrasyon (robust z + tanh)",
            "2) + EMA (yumuşatma)",
            "3) + Histerezis (rejim etiketi)"
        ],
        index=int(st.session_state.get("ai_step", 3)),
        key="ai_step_radio",
        on_change=lambda: st.session_state.update({"ai_step": int(str(st.session_state.get("ai_step_radio")).split(")")[0])})
    )


    # Trend hesapla / göster
    if st.session_state.get("ai_trend_df") is not None and not st.session_state["ai_trend_df"].empty:
        df_tr = st.session_state["ai_trend_df"]

        fig_trend = None
        if hasattr(utils, "create_ai_trend_chart"):
            fig_trend = utils.create_ai_trend_chart_step(df_tr, step=st.session_state.get('ai_step', 3)) if hasattr(utils, 'create_ai_trend_chart_step') else utils.create_ai_trend_chart(df_tr)

        if fig_trend:
            st.plotly_chart(fig_trend, use_container_width=True, key="ai_chart_roberta")
        else:
            st.warning("Grafik oluşturulamadı.")

        cbtn1, cbtn2 = st.columns([1, 3])
        with cbtn1:
            if st.button("🔄 Tekrar Hesapla", key="btn_ai_recalc"):
                st.session_state["ai_trend_df"] = None
                st.rerun()

    else:
        st.info("Tarihsel trend için tüm metinler taranır. (Biraz zaman alabilir)")
        if st.button("🚀 Tüm Geçmişi Analiz Et", type="primary", key="btn_ai_run_all"):
            with st.spinner("Model tüm geçmişi tarıyor..."):
                df_all_rob = utils.fetch_all_data()
                res_df = utils.calculate_ai_trend_series(df_all_rob)

            if res_df is None or res_df.empty:
                st.error("Analiz sonucu boş geldi. (DB boş olabilir veya model hata vermiş olabilir)")
            else:
                st.session_state["ai_trend_df"] = res_df
                st.rerun()

    # Açıklama kutusu
    with st.expander("ℹ️ Bu grafik nasıl hesaplanıyor?", expanded=False):
        st.markdown("""
    Bu grafik, modelin verdiği **3 sınıf olasılığından** (Şahin / Güvercin / Nötr) türetilmiş bir **endeks**tir.

    1) Her metin için `P(HAWK)`, `P(DOVE)`, `P(NEUT)` alınır.  
    2) Ham fark: **diff = P(HAWK) − P(DOVE)**  
    3) Serinin kendi dağılımına göre **robust kalibrasyon** yapılır (median + MAD → robust z-score)  
    4) `tanh` ile skor **−100..+100** bandına sıkıştırılır  
    5) **EMA (span=7)** ile yumuşatılır  
    6) Rejim etiketinde hızlı flip olmasın diye **histerezis** uygulanır (±25 giriş eşiği + ±15 nötr bant + işaret değişiminde hızlı nötre dönüş)

    Bu yüzden, model 3 sınıf üretse bile grafikteki çizgi “süreklilik” gösterir: bu bir **türetilmiş duruş endeksi**dir.
        """)

    st.divider()

    # -------------------------
    # 2) TEKİL DÖNEM ANALİZİ
    # -------------------------
    st.subheader("🔍 Tekil Dönem Detay Analizi")

    df_all_rob = utils.fetch_all_data()
    if df_all_rob is None or df_all_rob.empty:
        st.info("Tekil analiz için veritabanında kayıt yok.")
        return

    df_all_rob = df_all_rob.copy()
    df_all_rob["period_date"] = pd.to_datetime(df_all_rob["period_date"], errors="coerce")
    df_all_rob = df_all_rob.dropna(subset=["period_date"]).sort_values("period_date", ascending=False)
    df_all_rob["Donem"] = df_all_rob["period_date"].dt.strftime("%Y-%m")

    sel_rob_period = st.selectbox(
        "İncelenecek Dönem:",
        df_all_rob["Donem"].tolist(),
        index=0,
        key="rob_single_sel"
    )

    row_rob = df_all_rob[df_all_rob["Donem"] == sel_rob_period].iloc[0]
    txt_input = str(row_rob.get("text_content", "") or "")

    with st.expander("Metni Gör", expanded=False):
        st.write(txt_input)

    if st.button("🧪 Bu Metni Analiz Et", type="secondary", key="btn_ai_single"):
        with st.spinner("Analiz ediliyor..."):
            roberta_res = utils.analyze_with_roberta(txt_input)

        if not isinstance(roberta_res, dict):
            st.error(f"Model hata döndürdü: {roberta_res}")
        else:
            scores = roberta_res.get("scores_map", {}) or {}
            h = float(scores.get("HAWK", 0.0))
            d = float(scores.get("DOVE", 0.0))
            n = float(scores.get("NEUT", 0.0))
            diff = float(roberta_res.get("diff", h - d))
            stance = str(roberta_res.get("stance", ""))
            stance_sent = roberta_res.get("stance_sentence")
            doc_diff_mean = roberta_res.get("doc_diff_mean")
            net_push = roberta_res.get("net_push")

            # Eğer trend serisinde EMA skor varsa, bu dönemin EMA skorunu da yakalayalım
            ema_score = None
            if st.session_state.get("ai_trend_df") is not None and not st.session_state["ai_trend_df"].empty:
                tmp = st.session_state["ai_trend_df"]
                hit = tmp[tmp["Dönem"] == sel_rob_period]
                if not hit.empty and "AI Score (EMA)" in hit.columns:
                    ema_score = float(hit.iloc[0]["AI Score (EMA)"])

            c1, c2, c3 = st.columns(3)
            # Öncelik: cümle-özet duruşu (varsa) — full-text tek parça tahmin bazen şişebiliyor
            primary_stance = str(stance_sent or stance)
            c1.metric("Duruş", primary_stance)
            c2.metric("Diff (H-D)", f"{diff:.3f}")
            if ema_score is not None:
                c3.metric("AI Score (EMA)", f"{ema_score:.1f}")
            else:
                c3.metric("AI Score (EMA)", "—")

            st.write("Sınıf Skorları:")
            st.json({"HAWK": h, "DOVE": d, "NEUT": n})

            if stance_sent is not None:
                st.caption(
                    f"Cümle-özet: {stance_sent} | diff_mean={doc_diff_mean:+.3f} | net_push={net_push:+.2f} — "
                    f"Full-text ham: {roberta_res.get('stance_full_raw', stance)}"
                )

            # Debug
            with st.expander("DEBUG (ham çıktı)", expanded=False):
                st.json(roberta_res)

                        # Cümle bazlı analiz
            st.markdown("---")
            st.subheader("🧩 Cümle Bazlı Ayrıştırma (RoBERTa)")

            if hasattr(utils, "analyze_sentences_with_roberta"):
                df_sent = utils.analyze_sentences_with_roberta(txt_input)
            
                # ✅ Action etiketi (CUT/HIKE/HOLD)
                action = utils.detect_policy_action(txt_input) if hasattr(utils, "detect_policy_action") else "UNKNOWN"

                # 📐 Gerçek delta bp (metinden) -> varsa aksiyonu override et (en güvenilir kaynak)
                real_delta_bp = utils.extract_delta_bp_from_text(txt_input) if hasattr(utils, "extract_delta_bp_from_text") else None
                if real_delta_bp is not None:
                    if real_delta_bp > 0:
                        action = "HIKE"
                    elif real_delta_bp < 0:
                        action = "CUT"
                    else:
                        action = "HOLD"
            
                # ✅ Sayım + ağırlıklı özet
                summary = utils.summarize_sentence_roberta(df_sent, full_text=txt_input) if hasattr(utils, "summarize_sentence_roberta") else {}
            
                cA, cB, cC, cD = st.columns(4)
                pa = summary.get("policy_action", action)
                if (pa or "").upper() == "UNKNOWN" and action != "UNKNOWN":
                    pa = action
                cA.metric("Policy Action", pa)
                cB.metric("🦅 Şahin cümle", summary.get("hawk_n", 0))
                cC.metric("🕊️ Güvercin cümle", summary.get("dove_n", 0))
                cD.metric("⚖️ Nötr cümle", summary.get("neut_n", 0))
            
                c1, c2, c3 = st.columns(3)
                c1.metric("Diff ortalama", f"{summary.get('diff_mean', np.nan):.3f}" if summary.get("n", 0) else "—")
                c2.metric("Pozitif toplam (hawk itişi)", f"{summary.get('pos_sum', np.nan):.2f}" if summary.get("n", 0) else "—")
                c3.metric("Negatif toplam (dove itişi)", f"{summary.get('neg_sum', np.nan):.2f}" if summary.get("n", 0) else "—")

                # 📐 Gerçek delta bp (metinden)
                real_delta_bp = utils.extract_delta_bp_from_text(txt_input) if hasattr(utils, "extract_delta_bp_from_text") else None
                if real_delta_bp is not None:
                    st.metric("📐 Gerçek Delta BP", f"{real_delta_bp:+.0f} bp")
                else:
                    st.caption("📐 Gerçek delta bp metinden çıkarılamadı (örn: 'from X percent to Y percent' kalıbı yok).")

                # 🔥 Aksiyon sinyali (RoBERTa) — bu bir bp değildir, modelin dil/şiddet skorudur
                ap = float(summary.get("action_points", 0.0) or 0.0)
                aw = float(summary.get("action_weight", 0.0) or 0.0)
                aw_local = float(summary.get("action_weight_local", 0.0) or 0.0)
                al = str(summary.get("action_label", "—") or "—")
                a_sent = summary.get("action_sentence", "—")

                r1, r2 = st.columns(2)
                if al == "HIKE":
                    r1.metric("📈 Rate hike puanı", f"{ap:.1f}")
                    r2.metric("⚖️ Rate hike ağırlığı (lokal)", f"{aw_local:.2%}")
                    st.caption(f"Global ağırlık: {aw:.2%}")
                    if a_sent and a_sent != "—":
                        st.caption(f"Rate hike cümlesi: {a_sent}")
                elif al == "CUT":
                    r1.metric("✂️ Rate cut puanı", f"{ap:.1f}")
                    r2.metric("⚖️ Rate cut ağırlığı (lokal)", f"{aw_local:.2%}")
                    st.caption(f"Global ağırlık: {aw:.2%}")
                    if a_sent and a_sent != "—":
                        st.caption(f"Rate cut cümlesi: {a_sent}")
                else:
                    r1.metric("Aksiyon puanı", f"{ap:.1f}")
                    r2.metric("Aksiyon ağırlığı (lokal)", f"{aw_local:.2%}")
                    st.caption(f"Global ağırlık: {aw:.2%}")
                    if a_sent and a_sent != "—":
                        st.caption(f"Aksiyon cümlesi: {a_sent}")

                st.caption("Not: Net duruş, cümle sayısından değil **Diff (H−D) ağırlıklarından** geliyor. Bu puanlar **bp değildir**; modelin cümlenin ‘rate hike/cut’ dili taşıdığına dair şiddet skorudur.")

            
                if df_sent is None or df_sent.empty:
                    st.info("Metinden ayrıştırılabilir cümle bulunamadı.")
                else:
                    st.dataframe(df_sent, use_container_width=True)
            
            else:
                st.error("analyze_sentences_with_roberta bulunamadı.")

with tab_roberta:
    _render_tab_roberta()
