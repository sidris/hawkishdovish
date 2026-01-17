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

# SEKME YAPILANDIRMASI (TEMİZLENMİŞ HALİ)
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab_roberta, tab_imp = st.tabs([
    "📈 Dashboard", "📝 Veri Girişi", "📊 Veriler", "🔍 Frekans", "🤖 Faiz Tahmini", "☁️ WordCloud", "📜 ABF (2019)", 
    "🧠 CB-RoBERTa", "📅 Haberler"
])


# --- SESSION & STATE --- bloğunun içine ekleyin:
if 'ai_trend_df' not in st.session_state: st.session_state['ai_trend_df'] = None

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
        if 'Yıllık TÜFE' in merged.columns: merged['Yıllık TÜFE'] = pd.to_numeric(merged['Yıllık TÜFE'], errors='coerce')
        if 'PPK Faizi' in merged.columns: merged['PPK Faizi'] = pd.to_numeric(merged['PPK Faizi'], errors='coerce')
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Word Count
        fig.add_trace(go.Bar(x=merged['period_date'], y=merged['word_count'], name="Metin Uzunluğu", marker=dict(color='gray'), opacity=0.10, yaxis="y3", hoverinfo="x+y+name"))
        
        # --- SKORLAR ---
        fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['score_abg_scaled'], name="Şahin/Güvercin-Hibrit", line=dict(color='black', width=2, dash='dot'), marker=dict(size=6, color='black'), yaxis="y"))
        fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['abg_dashboard_val'], name="Şahin/Güvercin ABG 2019", line=dict(color='navy', width=4), yaxis="y"))
        
        if 'Yıllık TÜFE' in merged.columns: fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['Yıllık TÜFE'], name="Yıllık TÜFE (%)", line=dict(color='red', dash='dot'), yaxis="y"))
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
            fig_ai = utils.create_ai_trend_chart(st.session_state['ai_trend_df'])
            
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
            st.plotly_chart(fig_m, use_container_width=True)
            st.dataframe(df, use_container_width=True)
        else: st.error(f"Hata: {err}")

with tab4:
    st.header("🔍 Frekans ve Diff Analizi")
    df_all = utils.fetch_all_data()
    if not df_all.empty:
        df_all['period_date'] = pd.to_datetime(df_all['period_date'])
        df_all['Donem'] = df_all['period_date'].dt.strftime('%Y-%m')
        df_all = df_all.sort_values('period_date', ascending=False)
        st.subheader("📊 En Çok Tekrar Eden Ekonomi Terimleri")
        st.text_input("🚫 Grafikten Çıkarılacak Kelimeler (Enter)", key="deep_stop_in", on_change=add_deep_stop)
        if st.session_state['stop_words_deep']:
            st.write("Filtreler:")
            cols = st.columns(8)
            for i, word in enumerate(st.session_state['stop_words_deep']):
                if cols[i % 8].button(f"{word} ✖", key=f"del_deep_{word}"):
                    st.session_state['stop_words_deep'].remove(word)
                    st.rerun()
        st.divider()
        freq_df, top_terms = utils.get_top_terms_series(df_all, 7, st.session_state['stop_words_deep'])
        if not freq_df.empty:
            fig_freq = go.Figure()
            for term in top_terms:
                fig_freq.add_trace(go.Scatter(x=freq_df['period_date'], y=freq_df[term], name=term, mode='lines+markers'))
            fig_freq.update_layout(title="Kelime Kullanım Sıklığı Trendi", hovermode="x unified", height=400)
            st.plotly_chart(fig_freq, use_container_width=True)
        st.divider()
        st.subheader("🔄 Metin Farkı (Diff) Analizi")
        c_diff1, c_diff2 = st.columns(2)
        with c_diff1: sel_date1 = st.selectbox("Eski Metin:", df_all['Donem'].tolist(), index=min(1, len(df_all)-1))
        with c_diff2: sel_date2 = st.selectbox("Yeni Metin:", df_all['Donem'].tolist(), index=0)
        if st.button("Farkları Göster", type="primary"):
            if sel_date1 and sel_date2:
                t1 = df_all[df_all['Donem'] == sel_date1].iloc[0]['text_content']
                t2 = df_all[df_all['Donem'] == sel_date2].iloc[0]['text_content']
                diff_html = utils.generate_diff_html(t1, t2)
                st.markdown(f"**Kırmızı:** {sel_date1}'den silinenler | **Yeşil:** {sel_date2}'ye eklenenler")
                with st.container(border=True, height=400): st.markdown(diff_html, unsafe_allow_html=True)
    else: st.info("Yeterli veri yok.")

with tab5:
    st.header("🤖 Gelişmiş PPK Faiz Tahmin Modeli")
    st.info("Bu model, veritabanındaki tüm veriyi eğitir ancak aşağıda seçtiğiniz metin üzerinden 'sonraki' adımı tahmin eder.")
    
    with st.expander("ℹ️ Model Detayları", expanded=False):
        st.markdown("""
        * **Algoritma:** Ridge Regresyon (Sürekli Tahmin) + Logistic Regresyon (Yön Tahmini)
        * **Özellikler:** TF-IDF (Metin), Anahtar Kelimeler (Enflasyon, Büyüme vb.), Geçmiş Faiz Değişimleri, Volatilite.
        * **Validasyon:** Walk-Forward Validation (Zaman serisine duyarlı doğrulama).
        * **Güven Aralığı:** Model hatalarına dayalı dinamik bant genişliği.
        """)

    # 1. Veri Hazırlığı
    df_logs = utils.fetch_all_data()
    
    min_d = datetime.date(2020, 1, 1)
    if not df_logs.empty:
        df_logs['period_date'] = pd.to_datetime(df_logs['period_date'], errors='coerce')
        df_logs = df_logs.dropna(subset=['period_date'])
        
        if not df_logs.empty:
            # Tarih aralığı seçicisi için min/max
            min_avail_date = df_logs['period_date'].min().date()
            max_avail_date = df_logs['period_date'].max().date()
            
            # Min_d, market verisi çekmek için
            min_val = df_logs['period_date'].min()
            if isinstance(min_val, pd.Timestamp): min_d = min_val.date()
            elif isinstance(min_val, str): min_d = pd.to_datetime(min_val).date()
            elif isinstance(min_val, datetime.date): min_d = min_val

    # Market Verisi
    df_market, err = utils.fetch_market_data_adapter(min_d, datetime.date.today())
    ml_df = utils.prepare_ml_dataset(df_logs, df_market)

    if not ml_df.empty and len(ml_df) > 10:
        # 2. Modeli Eğit
        predictor = utils.AdvancedMLPredictor()
        status = predictor.train(ml_df)
        
        if status == "OK":
            # 3. KULLANICI SEÇİMİ
            st.markdown("### 📅 Analiz İçin Dönem Seçimi")
            c_d1, c_d2 = st.columns(2)
            start_date_sel = c_d1.date_input("Başlangıç", value=min_avail_date, min_value=min_avail_date, max_value=max_avail_date)
            end_date_sel = c_d2.date_input("Bitiş", value=max_avail_date, min_value=min_avail_date, max_value=max_avail_date)
            
            filtered_logs = df_logs[(df_logs['period_date'].dt.date >= start_date_sel) & (df_logs['period_date'].dt.date <= end_date_sel)].copy()
            filtered_logs = filtered_logs.sort_values("period_date", ascending=False)
            filtered_logs['Dönem'] = filtered_logs['period_date'].dt.strftime('%Y-%m')
            
            period_options = filtered_logs['Dönem'].tolist()
            
            if period_options:
                selected_period = st.selectbox("Analiz Edilecek Toplantıyı Seçin:", period_options, index=0)
                
                target_row = filtered_logs[filtered_logs['Dönem'] == selected_period].iloc[0]
                target_text = target_row['text_content']
                target_source = f"Seçilen Kayıt: {target_row['Dönem']}"
                selected_date_for_chart = target_row['period_date']
                
                st.divider()
                st.subheader(f"Analiz Edilen Metin: {target_source}")
                
                # 4. Tahmin Yap
                prediction = predictor.predict(target_text)
                
                if prediction:
                    c1, c2, c3 = st.columns(3)
                    direction = prediction['pred_direction']
                    color = "green" if direction == "ARTIRIM" else "red" if direction == "İNDİRİM" else "gray"
                    with c1:
                        st.markdown(f"### Yön: :{color}[{direction}]")
                        st.caption(f"Güven Skoru: %{prediction['direction_confidence']*100:.1f}")
                    bps = prediction['pred_change_bps']
                    with c2: st.metric("Tahmini Değişim", f"{bps:.0f} bps")
                    lo = prediction['pred_interval_lo']
                    hi = prediction['pred_interval_hi']
                    with c3: st.metric("Tahmin Aralığı", f"{lo:.0f} / {hi:.0f} bps")
                    
                    st.divider()
                    
                    # 5. Grafik
                    st.subheader("📊 Model Performansı (Geçmiş)")
                    
                    if predictor.df_hist is not None:
                        hist = predictor.df_hist.copy()
                        hist['date'] = pd.to_datetime(hist['date'])
                        
                        chart_start = pd.to_datetime(start_date_sel) - pd.Timedelta(days=90)
                        chart_end = pd.to_datetime(end_date_sel) + pd.Timedelta(days=90)
                        
                        hist_view = hist[(hist['date'] >= chart_start) & (hist['date'] <= chart_end)]
                        
                        if not hist_view.empty:
                            fig = go.Figure()
                            # Gerçekleşen
                            fig.add_trace(go.Bar(
                                x=hist_view['date'], y=hist_view['y_bps'],
                                name="Gerçekleşen Değişim", marker_color='gray', opacity=0.5
                            ))
                            # Geçmiş Tahminler
                            if 'predicted_bps' in hist_view.columns:
                                hist_pred = hist_view.dropna(subset=['predicted_bps'])
                                fig.add_trace(go.Scatter(
                                    x=hist_pred['date'], y=hist_pred['predicted_bps'],
                                    name="Model Geçmiş Tahminleri", 
                                    line=dict(color='blue', width=2, dash='dot')
                                ))
                            
                            # Şu anki tahmin noktası
                            if chart_start <= selected_date_for_chart <= chart_end:
                                fig.add_trace(go.Scatter(
                                    x=[selected_date_for_chart], 
                                    y=[bps],
                                    mode='markers',
                                    marker=dict(color=color, size=15, symbol='star'),
                                    name=f"Seçilen ({target_source}) Tahmini"
                                ))
                            
                            fig.update_layout(hovermode="x unified", title="Faiz Değişimleri ve Tahminler")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Seçilen tarih aralığında görüntülenecek grafik verisi yok.")
                else:
                    st.error("Tahmin üretilemedi.")
            else:
                st.info("Bu tarih aralığında kayıt bulunamadı.")
        else:
            st.warning(f"Model eğitilemedi: {status}")
    else:
        st.warning("Model eğitimi için yeterli veri yok (En az 10 toplantı kaydı ve piyasa verisi gerekli).")

with tab6:
    st.header("☁️ Kelime Bulutu (WordCloud)")
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

with tab_roberta:
    st.header("🧠 CentralBankRoBERTa (Yapay Zeka Analizi)")

    if not utils.HAS_TRANSFORMERS:
        st.error("Kütüphaneler eksik.")
        st.stop()

    # 1) GENEL TREND
    st.subheader("📈 Tarihsel Trend (Yumuşatılmış Skor)")

    if st.session_state.get("ai_trend_df") is not None and not st.session_state["ai_trend_df"].empty:
        fig_trend = utils.create_ai_trend_chart(st.session_state["ai_trend_df"])
        if fig_trend:
            st.plotly_chart(fig_trend, use_container_width=True, key="ai_chart_roberta")
        else:
            st.warning("Grafik oluşturulamadı.")

        if st.button("🔄 Tekrar Hesapla", key="ai_recalc"):
            st.session_state["ai_trend_df"] = None
            st.rerun()
    else:
        if st.button("🚀 Tüm Geçmişi Analiz Et", type="primary", key="ai_run_all"):
            with st.spinner("Model tüm geçmişi tarıyor..."):
                df_all_rob = utils.fetch_all_data()
                res_df = utils.calculate_ai_trend_series(df_all_rob)

                if res_df is None or res_df.empty:
                    st.error("Model hiçbir sonuç döndürmedi. (metinler boş olabilir veya model yüklenememiş olabilir)")
                else:
                    st.session_state["ai_trend_df"] = res_df
                    st.rerun()

    st.divider()

    # 2) HIZLI MODEL TESTİ (debug)
    st.subheader("🧪 Hızlı Test (Debug)")
    tests = {
        "HAWK_TEST": "Monetary policy will be tightened further and additional rate hikes may be delivered.",
        "DOVE_TEST": "Monetary policy easing will begin soon and rate cuts are likely in the coming meetings.",
        "NEUT_TEST": "The committee decided to keep the policy rate unchanged."
    }

    clf = utils.load_roberta_pipeline()
    if clf is None:
        st.error("Model pipeline yüklenemedi.")
    else:
        for k, s in tests.items():
            out = clf(s)
            if isinstance(out, list) and out and isinstance(out[0], list):
                out = out[0]
            best = max(out, key=lambda x: x.get("score", 0.0))
            st.write(k, best)

    st.divider()

    # 3) TEKİL DÖNEM DETAY
    st.subheader("🔍 Tekil Dönem Detay Analizi")

    df_all_rob = utils.fetch_all_data()
    if df_all_rob is None or df_all_rob.empty:
        st.info("Veri yok.")
        st.stop()

    df_all_rob = df_all_rob.copy()
    df_all_rob["period_date"] = pd.to_datetime(df_all_rob["period_date"], errors="coerce")
    df_all_rob = df_all_rob.dropna(subset=["period_date"])
    df_all_rob["Donem"] = df_all_rob["period_date"].dt.strftime("%Y-%m")
    df_all_rob = df_all_rob.sort_values("period_date", ascending=False)

    sel_rob_period = st.selectbox(
        "İncelenecek Dönem:",
        df_all_rob["Donem"].tolist(),
        index=0,
        key="rob_single_sel"
    )

    row_rob = df_all_rob[df_all_rob["Donem"] == sel_rob_period].iloc[0]
    txt_input = str(row_rob.get("text_content", "") or "")

    with st.expander("Metni Gör"):
        st.write(txt_input)

    if st.button("Bu Metni Detaylandır", type="secondary", key="rob_detail_btn"):
        with st.spinner("Analiz ediliyor..."):
            roberta_res = utils.analyze_with_roberta(txt_input)

        if not isinstance(roberta_res, dict):
            st.error(f"Model hata döndürdü: {roberta_res}")
        else:
            scores = roberta_res.get("scores_map", {})
            h = float(scores.get("HAWK", 0.0))
            d = float(scores.get("DOVE", 0.0))
            n = float(scores.get("NEUT", 0.0))

            net = float(roberta_res.get("net_score", 0.0))
            net_raw = float(roberta_res.get("net_score_raw", 0.0))

            c1, c2, c3 = st.columns(3)
            c1.metric("Duruş", roberta_res.get("best_label", ""))
            c2.metric("Güven", f"%{float(roberta_res.get('best_score', 0.0))*100:.1f}")
            c3.metric("Net Skor", f"{net:.2f}")

            st.caption(f"Ham Net: {net_raw:.2f}  |  Yumuşatılmış Net: {net:.2f}")
            st.write("Sınıf Skorları:")
            st.json({"HAWK": h, "DOVE": d, "NEUT": n})

            with st.expander("DEBUG (raw response)"):
                st.json(roberta_res)

    st.info("Not: Cümle bazlı analiz bu sürümde devre dışı bırakıldı.")
