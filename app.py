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

tab1, tab2, tab3, tab4, tab_textdata, tab6, tab7, tab_roberta, tab_tone, tab_imp = st.tabs([
    "📈 Dashboard",
    "📝 Veri Girişi",
    "📊 Veriler",
    "🔍 Frekans",
    "📚 Text as Data (TF-IDF)",
    "☁️ WordCloud",
    "📜 ABG (2019)",
    "🧠 CB-RoBERTa",
    "🗺️ Ton Haritası & Konular",
    "📅 Haberler"
])

ENFLATION_EXPECTATION_COLS = [
    "PKA 12 Ay Enflasyon Beklentisi",
    "İYA 12 Ay Enflasyon Beklentisi",
    "HBA 12 Ay Enflasyon Beklentisi",
]


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
        for col in ENFLATION_EXPECTATION_COLS:
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col], errors='coerce')
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Word Count
        fig.add_trace(go.Bar(x=merged['period_date'], y=merged['word_count'], name="Metin Uzunluğu", marker=dict(color='gray'), opacity=0.10, yaxis="y3", hoverinfo="x+y+name"))
        
        # --- SKORLAR ---
        # NOT: "Şahin/Güvercin-Hibrit" (score_abg_scaled) çizgisi kullanıcı isteği üzerine ana ekrandan kaldırıldı.
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

        # --- AOFM (TP.APIFON4) ---
        # Politika faizi değil, fiilen gerçekleşen fonlama maliyeti. İkisini birlikte
        # görmenin anlamı: ilan edilen faiz ile fiili duruş arasındaki ayrışma.
        if 'AOFM' in merged.columns and merged['AOFM'].notna().any():
            fig.add_trace(go.Scatter(
                x=merged['period_date'], y=merged['AOFM'],
                name="AOFM (%)",
                line=dict(color='darkgoldenrod', dash='solid', width=2),
                yaxis="y"
            ))

        # Türetilmiş fark serisi. Varsayılan olarak KAPALI gelir; legend'dan açılır.
        if 'AOFM-Faiz Farkı' in merged.columns and merged['AOFM-Faiz Farkı'].notna().any():
            fig.add_trace(go.Scatter(
                x=merged['period_date'], y=merged['AOFM-Faiz Farkı'],
                name="AOFM − Faiz (puan)",
                line=dict(color='saddlebrown', dash='dashdot', width=1.5),
                visible='legendonly',
                yaxis="y"
            ))

        expectation_styles = {
            "PKA 12 Ay Enflasyon Beklentisi": dict(color='purple', dash='solid', width=2),
            "İYA 12 Ay Enflasyon Beklentisi": dict(color='mediumpurple', dash='dash', width=2),
            "HBA 12 Ay Enflasyon Beklentisi": dict(color='indigo', dash='dot', width=2),
        }
        for col in ENFLATION_EXPECTATION_COLS:
            if col in merged.columns and merged[col].notna().any():
                fig.add_trace(go.Scatter(
                    x=merged['period_date'],
                    y=merged[col],
                    name=f"{col} (%)",
                    line=expectation_styles.get(col, dict(dash='dash')),
                    yaxis="y"
                ))

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
        # st.plotly_chart yerine temiz-PNG render:
        # Legend'da kapatılan seri (visible='legendonly') yerleşik "toImage" ile
        # indirilen görselde soluk halde görünmeye devam ediyordu. Bu render,
        # modebar'a indirme öncesi o kayıtları legend'dan tamamen kaldıran
        # özel bir kamera butonu ekler. Ayrıntı: utils.render_plotly_clean_png
        utils.render_plotly_clean_png(
            fig,
            div_id="dash_ana_grafik",
            height=620,
            filename="merkez_bankasi_analiz_paneli",
            png_width=1900, png_height=950, png_scale=2,
        )

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
                        # Önce önbellek: model hiç çalışmadan seri kurulur.
                        res_df = utils.trend_series_from_cache()
                        if res_df is None or res_df.empty:
                            # Önbellek boş (ilk kurulum) -> tam tarama
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
        df, err = utils.fetch_market_data_adapter(d1, d2)
        if not df.empty:
            fig_m = go.Figure()
            if 'Yıllık TÜFE' in df.columns: fig_m.add_trace(go.Scatter(x=df['Donem'], y=df['Yıllık TÜFE'], name="Yıllık TÜFE", line=dict(color='red')))
            if 'Aylık TÜFE' in df.columns: fig_m.add_trace(go.Scatter(x=df['Donem'], y=df['Aylık TÜFE'], name="Aylık TÜFE", line=dict(color='blue', dash='dot')))
            if 'PPK Faizi' in df.columns: fig_m.add_trace(go.Scatter(x=df['Donem'], y=df['PPK Faizi'], name="Faiz", line=dict(color='orange')))
            expectation_styles_m = {
                "PKA 12 Ay Enflasyon Beklentisi": dict(color='purple', dash='solid', width=2),
                "İYA 12 Ay Enflasyon Beklentisi": dict(color='mediumpurple', dash='dash', width=2),
                "HBA 12 Ay Enflasyon Beklentisi": dict(color='indigo', dash='dot', width=2),
            }
            for col in ENFLATION_EXPECTATION_COLS:
                if col in df.columns and pd.to_numeric(df[col], errors='coerce').notna().any():
                    fig_m.add_trace(go.Scatter(
                        x=df['Donem'],
                        y=pd.to_numeric(df[col], errors='coerce'),
                        name=col,
                        line=expectation_styles_m.get(col, dict(dash='dash'))
                    ))
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

    # HYBRID + CPI prepare (TÜM tarihçeyle hazırla; lag/rolling feature'lar bozulmasın)
    df_td_full = utils.textasdata_prepare_df_hybrid_cpi(
        df_logs,
        df_market,
        text_col="text_content",
        date_col="period_date",
        y_col="delta_bp",
        rate_col="policy_rate"
    )

    if df_td_full.empty or df_td_full["delta_bp"].notna().sum() < 10:
        st.warning("HYBRID+CPI eğitim için yeterli gözlem yok. (En az ~10 kayıt önerilir)")
        return

    # -------------------------
    # 0) TARİH ARALIĞI & VERİ TAZELEME
    # -------------------------
    # ÜÇ FARKLI "MAX TARİH" KAVRAMI VAR — kullanıcının kafası karışmasın diye hepsini gösteriyoruz:
    #   1) raw_max     : Supabase'deki TÜM kayıtların en yeni period_date'i (etiketli/etiketsiz fark etmez)
    #   2) prep_max    : df_td_full'a giren (yani lag/rolling feature'ları NaN-değil olan) en yeni satır
    #   3) labeled_max : delta_bp'si DOLU olan en yeni satır — backtest grafiği bunda biter
    raw_dates = pd.to_datetime(df_logs["period_date"], errors="coerce").dropna()
    raw_min = raw_dates.min().date()
    raw_max = raw_dates.max().date()

    prep_dates = pd.to_datetime(df_td_full["period_date"], errors="coerce").dropna()
    prep_max = prep_dates.max().date() if not prep_dates.empty else raw_max

    labeled_dates = df_td_full.dropna(subset=["delta_bp"])["period_date"]
    if labeled_dates.empty:
        st.warning("Etiketli (delta_bp dolu) hiçbir karar yok.")
        return
    data_min = pd.to_datetime(labeled_dates.min()).date()
    data_max = pd.to_datetime(labeled_dates.max()).date()

    # Etiketli son karardan SONRA gelen ama eğitime giremeyen kayıtları yakala
    df_logs_after = df_logs[pd.to_datetime(df_logs["period_date"]).dt.date > data_max].copy()
    missing_after = []
    for _, r in df_logs_after.iterrows():
        d = pd.to_datetime(r["period_date"]).date()
        why = []
        if pd.isna(r.get("delta_bp")):
            why.append("delta_bp boş")
        if pd.isna(r.get("policy_rate")):
            why.append("policy_rate boş")
        if not why:
            why.append("lag/rolling feature NaN (CPI veya komşu kayıt eksik)")
        missing_after.append((d, ", ".join(why)))

    cdate1, cdate2, cdate3 = st.columns([2, 2, 1])
    with cdate1:
        auto_extend = st.checkbox(
            "🔄 Son karara kadar otomatik uzat",
            value=True,
            help="Açıkken üst sınır otomatik olarak veritabanındaki en güncel ETİKETLİ (delta_bp dolu) PPK kararına eşitlenir."
        )
    with cdate2:
        st.caption(
            f"📅 Ham veri: **{raw_min} → {raw_max}**  ·  "
            f"Etiketli son karar: **{data_max}**  ·  "
            f"Etiketli karar sayısı: **{int(labeled_dates.notna().sum())}**"
        )
    with cdate3:
        if st.button("♻️ Veriyi Yenile", help="Cache'i temizleyip Supabase + EVDS'den yeniden çeker"):
            try:
                utils.fetch_market_data_adapter.clear()
            except Exception:
                pass
            try:
                st.cache_data.clear()
            except Exception:
                pass
            st.rerun()

    # Eğer DB'de etiketli son karardan daha yeni kayıtlar varsa, kullanıcıyı bilgilendir
    if missing_after:
        lines = "\n".join([f"- **{d}** → _{reason}_" for d, reason in missing_after])
        st.warning(
            f"⚠️ DB'de **{data_max}** sonrası **{len(missing_after)}** kayıt var ama eğitime giremiyor:\n\n{lines}\n\n"
            f"👉 Çözüm: **Veri Girişi** sekmesinden o kararın `delta_bp` ve `policy_rate` alanlarını doldurun, "
            f"sonra burada **♻️ Veriyi Yenile**'ye basın."
        )

    # Date range picker — max_value RAW max'a kadar açık (kullanıcı son etiketsiz kaydı da görsün);
    # ama default ve auto_extend ETİKETLİ son karara (data_max) bağlanır.
    picker_max = max(raw_max, data_max)
    dr_default = (data_min, data_max)
    dr = st.date_input(
        "Backtest tarih aralığı (eğitim ve grafik bu aralıkla sınırlanır)",
        value=dr_default,
        min_value=data_min,
        max_value=picker_max,
        help="Sol uçtan başlangıç, sağ uçtan bitiş tarihini seç. "
             "‘Son karara kadar otomatik uzat’ açıkken sağ uç her zaman en güncel ETİKETLİ karara eşitlenir."
    )

    # date_input bazen tek tarih, bazen tuple döner — güvenli ayrıştırma
    if isinstance(dr, (list, tuple)) and len(dr) == 2:
        sel_start, sel_end = dr
    else:
        sel_start, sel_end = data_min, data_max

    if auto_extend:
        sel_end = data_max  # her durumda etiketli son karara çek

    if sel_start > sel_end:
        st.error("Başlangıç tarihi, bitiş tarihinden büyük olamaz.")
        return

    # df_td = seçili aralığa göre filtrelenmiş veri
    mask_range = (
        (pd.to_datetime(df_td_full["period_date"]).dt.date >= sel_start) &
        (pd.to_datetime(df_td_full["period_date"]).dt.date <= sel_end)
    )
    df_td = df_td_full.loc[mask_range].reset_index(drop=True)

    if df_td.empty or df_td["delta_bp"].notna().sum() < 10:
        st.warning(
            f"Seçili aralıkta ({sel_start} → {sel_end}) yeterli etiketli gözlem yok. "
            f"En az ~10 kayıt önerilir. Aralığı genişletmeyi deneyin."
        )
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
            "0) Ham ton sinyali (saf)",
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

        # Ton × Aksiyon ayrık grafik (Hawkish Cut / Dovish Hike yakalar)
        fig_trend = None
        if hasattr(utils, "create_tone_action_chart"):
            fig_trend = utils.create_tone_action_chart(df_tr, step=st.session_state.get('ai_step', 3))
        elif hasattr(utils, "create_ai_trend_chart_step"):
            fig_trend = utils.create_ai_trend_chart_step(df_tr, step=st.session_state.get('ai_step', 3))
        else:
            fig_trend = utils.create_ai_trend_chart(df_tr)

        if fig_trend:
            st.plotly_chart(fig_trend, use_container_width=True, key="ai_chart_roberta")
            st.caption(
                "Üst panel **ton** (iletişim, saf cümle sinyali); işaret şekli **aksiyonu** gösterir "
                "(▲ hike, ▼ cut, ■ hold). Alt panel **gerçekleşen Δbp**. "
                "Yukarıda duran bir ▼ = **Hawkish Cut**; aşağıda duran bir ▲ = **Dovish Hike**."
            )
        else:
            st.warning("Grafik oluşturulamadı.")

        cbtn1, cbtn2 = st.columns([1, 3])
        with cbtn1:
            if st.button("🔄 Tekrar Hesapla", key="btn_ai_recalc"):
                st.session_state["ai_trend_df"] = None
                st.rerun()

        # --- Rejim özet tablosu (tüm dönemler) ---
        if hasattr(utils, "build_regime_summary_table"):
            tbl = utils.build_regime_summary_table(df_tr)
            if tbl is not None and not tbl.empty:
                st.markdown("#### 🧭 Ton × Aksiyon Rejim Özeti (tüm dönemler)")
                # 'Hawkish Cut' / 'Dovish Hike' satırlarını hafifçe vurgula
                def _hl(row):
                    rg = str(row.get("Rejim", ""))
                    if "Hawkish Cut" in rg:
                        return ["background-color: rgba(192,57,43,0.12)"] * len(row)
                    if "Dovish Hike" in rg:
                        return ["background-color: rgba(36,113,163,0.12)"] * len(row)
                    return [""] * len(row)
                try:
                    st.dataframe(tbl.style.apply(_hl, axis=1), use_container_width=True, hide_index=True)
                except Exception:
                    st.dataframe(tbl, use_container_width=True, hide_index=True)

                # Dikkat çekici rejimleri ayrıca say
                rg_all = tbl["Rejim"].astype(str)
                n_hc = int(rg_all.str.contains("Hawkish Cut").sum())
                n_dh = int(rg_all.str.contains("Dovish Hike").sum())
                if n_hc or n_dh:
                    st.caption(f"⚡ Ton–aksiyon ayrışması: **Hawkish Cut** × {n_hc} · **Dovish Hike** × {n_dh}")

    else:
        st.info("Tarihsel trend için tüm metinler taranır. (Biraz zaman alabilir)")
        if st.button("🚀 Tüm Geçmişi Analiz Et", type="primary", key="btn_ai_run_all"):
            with st.spinner("Model tüm geçmişi tarıyor..."):
                df_all_rob = utils.fetch_all_data()
                # Önce önbellek: model hiç çalışmadan seri kurulur.
                res_df = utils.trend_series_from_cache()
                if res_df is None or res_df.empty:
                    res_df = utils.calculate_ai_trend_series(df_all_rob)

            if res_df is None or res_df.empty:
                st.error("Analiz sonucu boş geldi. (DB boş olabilir veya model hata vermiş olabilir)")
            else:
                st.session_state["ai_trend_df"] = res_df
                st.rerun()

    # Açıklama kutusu
    with st.expander("ℹ️ Bu grafik nasıl hesaplanıyor?", expanded=False):
        st.markdown("""
    Grafik **iki ayrı boyutu** gösterir: **ton** (iletişim) ve **aksiyon** (gerçekleşen faiz kararı).
    Bunları tek sayıya sıkıştırmak **'Hawkish Cut'** gibi ayrışmaları gizler; burada ayrı tutulurlar.

    **TON (üst panel, çizgi + renk):**
    1) Metin **cümlelere** bölünür; her cümle için `P(HAWK)`, `P(DOVE)`, `P(NEUT)` alınır.  
       *(CentralBankRoBERTa cümle düzeyinde eğitilir; metnin tamamı taranır, ilk 512 token'la sınırlı değil.)*  
    2) Her cümlenin katkısı **diff = P(HAWK) − P(DOVE)**'dir; bu zaten **güven-ağırlıklıdır**.  
    3) **Saf ton sinyali** = cümle diff'lerinin ortalaması (karar-ağırlığı **yok**; ton aksiyona bulaşmaz).  
    4) Tek, gerekçeli **deadband** (±{db:.2f}) ile Şahin/Güvercin/Nötr etiketi.  
    5) **Robust kalibrasyon** (median+MAD → z) + `tanh` (−100..+100) + **EMA (span=7)** + **histerezis** (radio ile adım adım açılır).

    **AKSİYON (alt panel, barlar) ve işaret şekli:**
    - Gerçek **Δbp** metinden çıkarılır (*'from X percent to Y percent'*); yoksa **CUT/HIKE/HOLD** etiketine düşülür.  
    - Üstteki işaret şekli aksiyonu kodlar: **▲ hike, ▼ cut, ■ hold**.

    **REJİM = ton × aksiyon:** yukarıda duran bir ▼ → **🦅 Hawkish Cut**; aşağıda duran bir ▲ → **🕊️ Dovish Hike**.

    **Not:** Full-text tek-parça tahmin yalnızca *referans/debug*. Detay paneli ile bu grafik **aynı kanonik tonu** okur.
        """.format(db=utils.DOC_STANCE_DEADBAND))

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
            # 'diff'/'stance' = KANONİK SAF TON. Aksiyon + rejim AYRI.
            doc_signal = float(roberta_res.get("diff", h - d))
            stance = str(roberta_res.get("stance", ""))
            regime = str(roberta_res.get("regime", ""))
            delta_bp = roberta_res.get("delta_bp", None)
            action_label = str(roberta_res.get("action_label", "UNKNOWN"))
            diff_fulltext = float(roberta_res.get("diff_fulltext", h - d))
            stance_full_raw = str(roberta_res.get("stance_full_raw", ""))
            doc_diff_mean = roberta_res.get("doc_diff_mean")
            net_push = roberta_res.get("net_push")

            # Eğer trend serisinde EMA skor varsa, bu dönemin EMA skorunu da yakalayalım
            ema_score = None
            if st.session_state.get("ai_trend_df") is not None and not st.session_state["ai_trend_df"].empty:
                tmp = st.session_state["ai_trend_df"]
                hit = tmp[tmp["Dönem"] == sel_rob_period]
                if not hit.empty and "AI Score (EMA)" in hit.columns:
                    ema_score = float(hit.iloc[0]["AI Score (EMA)"])

            # Rejim öne çıksın (Hawkish Cut / Dovish Hike...)
            st.markdown(f"### {regime if regime else '—'}")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ton (Duruş)", stance)              # iletişim tonu
            c2.metric("Ton Sinyali", f"{doc_signal:+.3f}")
            if delta_bp is not None:
                c3.metric("Aksiyon (Δbp)", f"{float(delta_bp):+.0f} bp")
            else:
                c3.metric("Aksiyon", action_label)
            if ema_score is not None:
                c4.metric("AI Score (EMA)", f"{ema_score:.1f}")
            else:
                c4.metric("AI Score (EMA)", "—")

            st.write("Full-text ham sınıf skorları (yalnızca referans):")
            st.json({"HAWK": h, "DOVE": d, "NEUT": n})

            st.caption(
                "**Ton** = saf cümle sinyali (iletişim); **Aksiyon** = gerçekleşen faiz kararı. "
                "İkisi ayrı tutulur; rejim ikisinin birleşimidir. "
                f"Referans → full-text ham diff = {diff_fulltext:+.3f} ({stance_full_raw})"
                + (f" | cümle diff_mean = {doc_diff_mean:+.3f}" if doc_diff_mean is not None else "")
                + (f" | net_push = {net_push:+.2f}" if net_push is not None else "")
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


# ==============================================================================
# TAB: TON HARİTASI & KONULAR
# ==============================================================================
# İki analiz, tek veri kaynağı (önbellekteki cümle tablosu):
#   1. Cümle bazlı ton haritası     -> tek dokümanın ton mimarisi
#   2. Konu/kesim payları serisi    -> zaman içinde neyden, kimin adına bahsedildiği
#
# Ton etiketi     : mrince/CBRT-RoBERTa-HawkishDovish-Classifier
# İlgili Kesim etiketi : Moritz-Pfeifer/CentralBankRoBERTa-agent-classifier
# Tema etiketi    : deterministik sözlük (model değil — bkz. utils.THEME_PATTERNS)
# ==============================================================================


def _tone_cache_panel(df_logs):
    st.subheader("🗃️ Model Önbelleği")

    diag = utils.diagnose(df_logs)
    if diag.empty:
        st.info("Veritabanında analiz edilecek metin yok.")
        return

    counts = diag["durum"].value_counts().to_dict()
    taze = counts.get("taze", 0)
    bayat = len(diag) - taze

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kayıt", len(diag))
    c2.metric("Taze", taze)
    c3.metric("Eksik", counts.get("eksik", 0))
    c4.metric("Bayat", counts.get("metin_degisti", 0) + counts.get("surum_eskidi", 0))

    with st.expander("Kayıt bazında durum", expanded=False):
        show = diag.copy()
        show["Dönem"] = pd.to_datetime(show["period_date"]).dt.strftime("%Y-%m")
        show = show[["Dönem", "source", "durum", "n_sentences"]].rename(
            columns={"source": "Kaynak", "durum": "Durum", "n_sentences": "Cümle"}
        )
        st.dataframe(show.sort_values("Dönem", ascending=False),
                     use_container_width=True, hide_index=True)

    use_agent = st.checkbox(
        "İlgili kesim (ekonomik aktör) sınıflandırıcısını da çalıştır",
        value=True,
        help="İkinci bir RoBERTa (~500 MB) yükler. Konu payları serisi buna bağlıdır; "
             "kapatırsan yalnızca sözlük tabanlı tema etiketi üretilir.",
        key="rc_use_agent",
    )

    b1, b2, _ = st.columns([1.5, 1.5, 3])
    with b1:
        if st.button(f"⚡ Eksik/bayat {bayat} kaydı hesapla",
                     type="primary", disabled=(bayat == 0), key="rc_sync"):
            _tone_run_sync(df_logs, force=False, with_agent=use_agent)
    with b2:
        if st.button("♻️ Tümünü yeniden hesapla", key="rc_force"):
            _tone_run_sync(df_logs, force=True, with_agent=use_agent)

    st.caption(
        f"Sürüm `{utils.PIPELINE_VERSION}` · Skorlama mantığını değiştirdiğinde "
        "`utils.PIPELINE_VERSION` değerini bump et; tüm kayıtlar otomatik bayatlar. "
        "Metni düzenlersen o kaydın hash'i değişir ve yalnızca o kayıt yeniden hesaplanır."
    )


def _tone_run_sync(df_logs, force, with_agent):
    if not utils.HAS_TRANSFORMERS:
        st.error("transformers/torch yüklü değil.")
        return
    if not utils.supabase:
        st.error("Supabase bağlantısı yok — önbellek yazılamaz.")
        return

    pb = st.progress(0.0)
    lbl = st.empty()

    def _cb(frac, text):
        pb.progress(min(1.0, frac))
        lbl.caption(f"Hesaplanıyor: {text}")

    with st.spinner("Model çalışıyor..."):
        res = utils.sync_cache(df_logs, force=force, with_agent=with_agent, progress_cb=_cb)

    if with_agent:
        utils.release_agent_pipeline()

    pb.empty()
    lbl.empty()
    st.success(f"Tamamlandı — işlenen {res['islenen']}, atlanan {res['atlanan']}, hata {res['hata']}")
    st.rerun()


def _tone_sentence_map(df_sent):
    st.subheader("🧭 Cümle Bazlı Ton Haritası")
    st.caption(
        f"Metin, modelin cümle düzeyi çıktısına göre renklendirilir. Kırmızı = şahin "
        f"(ton ≥ {utils.DOC_STANCE_DEADBAND:.2f}), mavi = güvercin "
        f"(ton ≤ −{utils.DOC_STANCE_DEADBAND:.2f}), gri = nötr bant. Renk eşiği "
        "yukarıdaki sayaçlarla **aynı** eşiktir; ekranda saydığın renk ile metrikteki "
        "sayı birebir tutar. Cümlenin üzerine gelince ham olasılıklar görünür."
    )

    donems = sorted(df_sent["Donem"].dropna().unique().tolist(), reverse=True)
    if not donems:
        st.info("Önbellekte cümle yok.")
        return

    cA, cB = st.columns([1, 2])
    with cA:
        sel = st.selectbox("Dönem", donems, index=0, key="tm_period")
    d = df_sent[df_sent["Donem"] == sel].copy()

    with cB:
        agents = ["(tümü)"] + sorted(d["agent_label"].dropna().unique().tolist())
        agent_f = st.selectbox("İlgili Kesim filtresi", agents, index=0, key="tm_agent")

    d_view = d if agent_f == "(tümü)" else d[d["agent_label"] == agent_f]
    if d_view.empty:
        st.info("Bu filtreyle cümle kalmadı.")
        return

    diffs = pd.to_numeric(d_view["diff"], errors="coerce")
    n_hawk = int((diffs >= utils.DOC_STANCE_DEADBAND).sum())
    n_dove = int((diffs <= -utils.DOC_STANCE_DEADBAND).sum())
    n_neut = int(len(d_view) - n_hawk - n_dove)

    # Nötr de gösterilir ki üç sayı toplamı cümle sayısını versin; aksi halde
    # "8 şahin var ama ekranda 7 görüyorum" tipi belirsizlik doğuyor.
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Cümle", len(d_view))
    m2.metric("Ortalama ton", f"{diffs.mean():+.3f}")
    m3.metric("🦅 Şahin", n_hawk)
    m4.metric("⚪ Nötr", n_neut)
    m5.metric("🕊️ Güvercin", n_dove)

    fig_strip = utils.chart_sentence_strip(d_view, title=f"{sel} — cümle sırasına göre ton")
    if fig_strip:
        st.plotly_chart(fig_strip, use_container_width=True, key=f"strip_{sel}")

    st.markdown(
        utils.sentence_heatmap_html(d_view, show_agent=(agent_f == "(tümü)")),
        unsafe_allow_html=True,
    )

    with st.expander("En uçtaki cümleler", expanded=False):
        srt = d_view.sort_values("diff", ascending=False)
        cols = ["sent_idx", "sentence", "diff", "agent_label", "theme_label"]
        names = {"sent_idx": "#", "sentence": "Cümle", "diff": "Ton",
                 "agent_label": "İlgili Kesim", "theme_label": "Tema"}
        cL, cR = st.columns(2)
        with cL:
            st.markdown("**🦅 En şahin 5**")
            st.dataframe(srt.head(5)[cols].rename(columns=names),
                         use_container_width=True, hide_index=True)
        with cR:
            st.markdown("**🕊️ En güvercin 5**")
            st.dataframe(srt.tail(5)[cols].rename(columns=names),
                         use_container_width=True, hide_index=True)


def _tone_topic_series(df_sent):
    st.subheader("📊 Konu Payları Zaman Serisi")

    c1, c2 = st.columns([2, 1])
    with c1:
        mode = st.radio(
            "Etiket kaynağı",
            ["İlgili Kesim (CentralBankRoBERTa)", "Tema (sözlük)"],
            horizontal=True,
            key="tp_mode",
        )
    is_agent = mode.startswith("İlgili Kesim")
    col = "agent_label" if is_agent else "theme_label"
    colors = utils.AGENT_COLORS if is_agent else None

    if is_agent and df_sent["agent_label"].fillna("Belirsiz").eq("Belirsiz").all():
        st.warning(
            "İlgili kesim etiketi yok — önbellek, ekonomik aktör sınıflandırıcısı "
            "kapalıyken üretilmiş. Yukarıdan seçeneği açıp yeniden hesaplaman gerekiyor."
        )
        return

    # --- Etiketleme modu: tek etiket (kompozisyon) vs çok etiket (kapsam) ------
    multi = False
    if not is_agent:
        with c2:
            multi = st.toggle(
                "Çok etiketli", value=True, key="tp_multi",
                help="Bir cümle birden çok konuya değinebilir. Kapalıyken cümle "
                     "başına tek etiket verilir ve beraberlikte kazananı sözlükteki "
                     "yazım sırası belirler — bu keyfîdir.",
            )

    # --- Güven filtresi (yalnızca model etiketi için) --------------------------
    df_use = df_sent
    if is_agent and "agent_conf" in df_sent.columns:
        with c2:
            min_conf = st.slider("Min. güven", 0.0, 0.95, 0.0, 0.05, key="tp_conf",
                                 help="Model tahmininin altında kaldığı cümleleri "
                                      "analizden çıkarır.")
        if min_conf > 0:
            df_use = df_sent[pd.to_numeric(df_sent["agent_conf"],
                                           errors="coerce").fillna(0) >= min_conf]
            atilan = len(df_sent) - len(df_use)
            if atilan:
                st.caption(f"Güven filtresi {atilan} cümleyi ({atilan/len(df_sent):.0%}) dışarıda bıraktı.")
        if df_use.empty:
            st.info("Filtre sonrası cümle kalmadı.")
            return

    # --- Pay / kapsam serisi ---------------------------------------------------
    df_long = df_use
    if multi:
        df_long = utils.explode_themes(df_use)
        share = utils.coverage_timeseries(df_long, df_use, col)
        st.caption(
            "**Kapsam** okunur: bir dönemde cümlelerin yüzde kaçı o konuya değindi. "
            "Bir cümle birden çok konuya değinebildiği için sütun toplamı %100'ü aşar — "
            "beklenen davranış budur. PPK metni ~13 cümle olduğundan bir cümle ≈ 8 puan; "
            "**8 puandan küçük hareketler kuantizasyon gürültüsüdür**, yorumlanmamalıdır."
        )
    else:
        share = utils.share_timeseries(df_use, col)
        st.caption(
            "**Kompozisyon** okunur: payların toplamı her dönemde %100'dür. Dikkat — "
            "kompozisyonda bir konunun payı, kendisi hiç değişmeden de düşebilir "
            "(başka bir konu büyüdüğü için). Mutlak hacim için dashboard'daki "
            "Metin Uzunluğu serisiyle birlikte oku. Bir cümle ≈ 8 puan; **8 puandan "
            "küçük hareketler gürültüdür**."
        )

    if share.empty:
        st.info("Yeterli veri yok.")
        return

    if is_agent:
        order = [c for c in utils.AGENT_ORDER if c in share.columns]
        order += [c for c in share.columns if c not in order]
        share = share[order]

    if multi:
        fig_share = utils.chart_share_lines(share, "Konu kapsamı (%)", colors=colors)
    else:
        fig_share = utils.chart_share_area(share, "Cümle payları (%)", colors=colors)
    if fig_share:
        st.plotly_chart(fig_share, use_container_width=True, key=f"share_{col}_{multi}")

    # --- Konu × Ton -----------------------------------------------------------
    st.markdown("#### 🌡️ Konu × Ton")

    h1, h2 = st.columns([1, 1])
    with h1:
        min_n = st.slider(
            "Hücre başına en az cümle", 1, 6, 3, key="tp_minn",
            help="Bu sayının altındaki hücreler gizlenir. Tek cümleye dayanan bir "
                 "'ortalama' aslında tek bir model tahminidir; on cümlelik bir "
                 "ortalamayla aynı görsel ağırlıkta görünmesi yanıltıcıdır.",
        )
    with h2:
        centered = st.toggle(
            "Sapma modu", value=True, key="tp_center",
            help="Her konudan kendi ortalamasını çıkarır. Bazı konuların tonu sözcük "
                 "dağarcığı yüzünden sistematik olarak kayıktır (enflasyon dili şahin, "
                 "likidite dili güvercin tınlar) — bu bir bulgu değil, model artefaktıdır. "
                 "Merkezlendiğinde geriye yalnızca zaman içindeki hareket kalır.",
        )

    tm, tc = utils.tone_matrix(df_long, col, min_n=min_n)
    if tm.empty:
        st.info("Yeterli veri yok.")
        return

    gizli = int(tm.isna().sum().sum() - (tc.reindex_like(tm) == 0).sum().sum())
    if centered:
        tm = utils.center_matrix(tm)

    st.caption(
        "Pay bir konuya **ne kadar** yer ayrıldığını, bu ısı haritası ise o konu "
        "hakkında **nasıl** konuşulduğunu gösterir — agregat ton endeksinin sildiği ayrım. "
        "Beyaz = nötr, **gri zemin = veri yok veya eşik altı**. "
        + (f"Bu ayarda {gizli} hücre az cümle nedeniyle gizlendi. " if gizli > 0 else "")
        + ("Değerler sapmadır: 0 = o konunun kendi ortalaması."
           if centered else "Değerler ham tondur; satırlar arası SEVİYE karşılaştırması "
                            "yapma, sözcük dağarcığı kaynaklı kayıklık içerir.")
    )

    fig_tone = utils.chart_tone_heatmap(
        tm, "Dönem × konu " + ("ton sapması" if centered else "ortalama tonu"),
        ylab="İlgili Kesim" if is_agent else "Tema",
        counts=tc, centered=centered,
    )
    if fig_tone:
        st.plotly_chart(fig_tone, use_container_width=True, key=f"tone_{col}_{centered}")

    tbl = utils.divergence_table(df_long, col)
    if not tbl.empty:
        st.markdown("#### Tüm dönemler — konu bazında özet")
        st.caption(
            "«Dönem» sütunu etiketin kaç farklı dönemde göründüğünü verir: tek bir "
            "döneme yığılmış bir etiketin genel ortalaması zaman serisi olarak "
            "yorumlanamaz."
            + (" «Ort. Güven» 0.60 altındaysa o sınıfı ciddiye alma." if is_agent else "")
        )
        st.dataframe(
            tbl.rename(columns={col: ("İlgili Kesim" if is_agent else "Tema")}).round(3),
            use_container_width=True, hide_index=True,
        )


def _tone_position_map(df_sent):
    st.subheader("📐 Metnin Neresinde Şahinleşiyor?")

    p1, p2, p3 = st.columns([1, 1.3, 1])
    with p1:
        bins = st.select_slider("Dilim", options=[3, 5, 10], value=3, key="pm_bins")
    with p2:
        drop_dec = st.toggle(
            "Karar cümlesini hariç tut", value=True, key="pm_drop",
            help="Metnin ilk cümlesi her zaman faiz kararıdır ve model faiz indirimini "
                 "güvercin okur. Yani ilk dilim tonu değil, faizin YÖNÜNÜ izler. "
                 "Hariç tutulduğunda 'karar bir yana, çerçeve metni ne diyor' sorusu "
                 "cevaplanır.",
        )
    with p3:
        min_n = st.slider("Min. cümle", 1, 5, 2, key="pm_minn")

    ort = len(df_sent) / max(1, df_sent["Donem"].nunique())
    per_cell = ort / bins
    st.caption(
        f"Her metin göreli konuma göre {bins} dilime bölünür ve dilim ortalaması alınır. "
        f"Bu korpusta metin başına ortalama {ort:.0f} cümle var → **hücre başına "
        f"≈{per_cell:.1f} cümle**. "
        + ("Bu, ortalama almaya yeterli değil; grafik tek tek model tahminlerinin "
           "yeniden dizilmiş hali olur ve gürültü sönümlenmez."
           if per_cell < 2.5 else
           "Aranan şey DİKEY bir örüntüdür: bir sütunda üst dilimlerin alt dilimlerden "
           "sistematik farklı olması. Yatay örüntüyü zaten dashboard'daki agregat seri gösterir.")
    )

    pm, pc = utils.position_tone_matrix(df_sent, bins=bins,
                                        drop_decision=drop_dec, min_n=min_n)
    if pm.empty:
        st.info("Yeterli veri yok.")
        return

    fig_pos = utils.chart_tone_heatmap(
        pm, "Dönem × metin içi konum" + (" (karar cümlesi hariç)" if drop_dec else ""),
        ylab="Metin içi konum", counts=pc,
    )
    st.plotly_chart(fig_pos, use_container_width=True, key=f"posmap_{bins}_{drop_dec}")


def _render_tab_tone_topics():
    st.header("🗺️ Ton Haritası & Konular")

    if not utils.supabase:
        st.error("Supabase bağlantısı yok.")
        return

    df_logs = utils.fetch_all_data()
    if df_logs is None or df_logs.empty:
        st.info("Veritabanında kayıt yok.")
        return

    _tone_cache_panel(df_logs)
    st.divider()

    df_sent = utils.fetch_sentences()
    if df_sent is None or df_sent.empty:
        st.info("Önbellek boş. Yukarıdaki hesaplama butonunu bir kez çalıştır — "
                "sonrasında bu sekme model çalıştırmadan açılır.")
        return

    _tone_sentence_map(df_sent)
    st.divider()
    _tone_topic_series(df_sent)
    st.divider()
    _tone_position_map(df_sent)

    with st.expander("ℹ️ Etiketler nereden geliyor?", expanded=False):
        st.markdown(f"""
**Ton (şahin/güvercin)** — `{utils.MODEL_HD}`, cümle düzeyinde.
Katkı `diff = P(HAWK) − P(DOVE)`; bu zaten güven-ağırlıklıdır, çıplak sayımdan üstündür.
Dashboard'daki kanonik ton ile **aynı** hesaptır; tek fark burada cümle sırası korunur.

**İlgili Kesim** — `{utils.MODEL_AGENT}` (Pfeifer & Marohl, 2023).
Beş makroekonomik aktör: hanehalkı, firmalar, finansal sektör, kamu, merkez bankası.

⚠️ Bu **hedef kitle değildir**. Model "bu metin kime hitap ediyor" sorusunu değil,
"*bu cümlenin ekonomik içeriği kimin durumuna dair*" sorusunu cevaplar. Yazarların
kendi örneği: "ücretler beklentinin ötesinde artıyor" → ücreti **alan** hanehalkı için
olumlu, ücreti **ödeyen** firmalar için olumsuz. Bu yüzden bir enflasyon cümlesi
çoğunlukla `Households` çıkar (tüketici fiyatlarını ödeyen ve reel geliri aşınan taraf),
faiz kararı cümlesi ise `Central Bank` çıkar.

Etiketten emin olmak için `agent_conf` kolonuna bak; 0.60 altındaki tahminleri
ciddiye alma.

**Tema** — sözlük tabanlı, **model değil**. CentralBankRoBERTa ailesinde bir konu
sınıflandırıcısı yok; ilgili kesim ve duygu başlıkları var. Konu boyutunu uydurma bir
model çıktısı gibi sunmak yerine deterministik ve denetlenebilir bir sözlük katmanı
kullanıldı. Sözlüğü `utils.THEME_PATTERNS` içinden genişletebilirsin.

Tema etiketi **okuma anında** `sentence` metninden yeniden hesaplanır; veritabanındaki
`theme_label` kolonu yok sayılır. Yani sözlükteki her düzeltme anında yansır, model
taraması tekrarlanmaz, veritabanı şeması hiç değişmez.

⚖️ Güvenilirlik sırası: **ton** ve **ilgili kesim** gerçek model tahminleridir;
**tema** benim yazdığım kurallardır. Sunumda bu ayrımı koru.

**Önbellek** — model çıktısı cümle düzeyinde `roberta_sentences` tablosuna yazılır.
Kalibrasyon / EMA / histerezis **saklanmaz**: bunlar tüm serinin dağılımına bağlıdır,
yeni bir kayıt geçmiş skorları da değiştirir. Bu yüzden her yüklemede yeniden hesaplanır.
        """)


with tab_tone:
    _render_tab_tone_topics()
