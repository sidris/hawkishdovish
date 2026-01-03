import streamlit as st
import pandas as pd
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import utils 
import uuid

# Sayfa ayarını 'wide' tutuyoruz ama CSS ile daraltacağız
st.set_page_config(page_title="Piyasa Analiz", layout="wide", initial_sidebar_state="collapsed")

# --- MOBİL UYUMLU CSS ---
# Bu blok, uygulamanın mobilde bir 'App' gibi görünmesini sağlar.
st.markdown("""
<style>
    /* Üstteki boşluğu kaldır */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 5rem;
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }
    /* Başlık boyutunu mobilde küçült */
    h1 {
        font-size: 1.8rem !important;
    }
    /* Metrikleri mobilde daha derli toplu göster */
    [data-testid="stMetricValue"] {
        font-size: 1.2rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
    }
    /* Tablo yazı boyutunu ayarla */
    .stDataFrame {
        font-size: 0.8rem;
    }
    /* Butonları tam genişlik yap (Mobilde kolay tıklama için) */
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# --- 0. GÜVENLİK ---
APP_PWD = "SahinGuvercin34"      
ADMIN_PWD = "SahinGuvercin06"    

# --- 1. GİRİŞ EKRANI ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    # Mobilde ortalamak için boşlukları azalttık
    col1, col2, col3 = st.columns([1, 8, 1]) 
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>🔐 Güvenli Giriş</h3>", unsafe_allow_html=True)
        pwd_input = st.text_input("Uygulama Şifresi", type="password")
        if st.button("Giriş Yap", type="primary"):
            if pwd_input == APP_PWD:
                st.session_state['logged_in'] = True
                st.success("Giriş Başarılı!")
                st.rerun()
            else:
                st.error("Hatalı Şifre!")
    st.stop()

# --- 2. SESSION STATE ---
if 'form_data' not in st.session_state:
    st.session_state['form_data'] = {
        'id': None,
        'date': datetime.date.today().replace(day=1),
        'source': "TCMB",
        'text': ""
    }

if 'table_key' not in st.session_state:
    st.session_state['table_key'] = str(uuid.uuid4())

if 'collision_state' not in st.session_state:
    st.session_state['collision_state'] = {'active': False, 'target_id': None, 'pending_text': None, 'target_date': None}
if 'update_state' not in st.session_state:
    st.session_state['update_state'] = {'active': False, 'pending_text': None}

def reset_form():
    st.session_state['form_data'] = {'id': None, 'date': datetime.date.today(), 'source': "TCMB", 'text': ""}
    st.session_state['collision_state'] = {'active': False, 'target_id': None, 'pending_text': None, 'target_date': None}
    st.session_state['update_state'] = {'active': False, 'pending_text': None}
    st.session_state['table_key'] = str(uuid.uuid4())

# --- ARAYÜZ ---
# Başlık ve Çıkış Butonu
c_head1, c_head2 = st.columns([4, 1])
with c_head1: 
    st.markdown("### 🦅 Şahin/Güvercin Paneli")
with c_head2: 
    if st.button("Çıkış"):
        st.session_state['logged_in'] = False
        st.rerun()

tab1, tab2, tab3 = st.tabs(["📈 Dashboard", "📝 Veri Girişi", "📊 Veriler"])

# ==============================================================================
# TAB 1: DASHBOARD
# ==============================================================================
with tab1:
    with st.spinner("Veriler Yükleniyor..."):
        df_logs = utils.fetch_all_data()
    
    if not df_logs.empty:
        df_logs['period_date'] = pd.to_datetime(df_logs['period_date'])
        df_logs['Donem'] = df_logs['period_date'].dt.strftime('%Y-%m')
        
        # --- HESAPLAMALAR ---
        df_logs['word_count'] = df_logs['text_content'].apply(lambda x: len(str(x).split()) if x else 0)
        df_logs['flesch_score'] = df_logs['text_content'].apply(lambda x: utils.calculate_flesch_reading_ease(str(x)))
        df_logs['score_abg_scaled'] = df_logs['score_abg'].apply(lambda x: x*100 if abs(x) <= 1 else x)

        min_d = df_logs['period_date'].min().date()
        max_d = datetime.date.today()
        df_market, err = utils.fetch_market_data_adapter(min_d, max_d)
        
        merged = pd.merge(df_logs, df_market, on="Donem", how="left")
        merged = merged.sort_values("period_date")
        
        # Veri Tipi Dönüşümü ve Max Değer Hesabı
        if 'Yıllık TÜFE' in merged.columns: merged['Yıllık TÜFE'] = pd.to_numeric(merged['Yıllık TÜFE'], errors='coerce')
        if 'PPK Faizi' in merged.columns: merged['PPK Faizi'] = pd.to_numeric(merged['PPK Faizi'], errors='coerce')
        
        market_vals = [80]
        if 'Yıllık TÜFE' in merged.columns: market_vals.append(merged['Yıllık TÜFE'].max())
        if 'PPK Faizi' in merged.columns: market_vals.append(merged['PPK Faizi'].max())
        market_vals = [v for v in market_vals if pd.notna(v)]
        market_max = max(market_vals) + 10

        # --- ANA GRAFİK ---
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # 1. Kelime Sayısı
        fig.add_trace(go.Bar(
            x=merged['period_date'], y=merged['word_count'], name="Metin Uzunluğu",
            marker=dict(color='gray'), opacity=0.15, yaxis="y3", hoverinfo="x+y+name"
        ))

        # 2. Skor Çizgisi
        fig.add_trace(go.Scatter(
            x=merged['period_date'], y=merged['score_abg_scaled'], name="Şahin/Güvercin Skoru", 
            line=dict(color='black', width=3), marker=dict(size=8, color='black')
        ), secondary_y=False)
        
        # 3. Piyasa Verileri
        if 'Yıllık TÜFE' in merged.columns:
            fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['Yıllık TÜFE'], name="Yıllık TÜFE (%)", line=dict(color='red', dash='dot')), secondary_y=True)
        if 'PPK Faizi' in merged.columns:
            fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['PPK Faizi'], name="Faiz (%)", line=dict(color='orange', dash='dot')), secondary_y=True)

        # 4. Okunabilirlik Skoru
        fig.add_trace(go.Scatter(
            x=merged['period_date'], 
            y=merged['flesch_score'], 
            name="Okunabilirlik (Flesch)",
            mode='markers',
            marker=dict(color='teal', size=9, opacity=0.8, line=dict(width=1, color='darkslategrey')),
            hoverinfo="x+y+name"
        ), secondary_y=True)

        # Şekiller ve Etiketler
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
            # Mobilde isimler üst üste binmesin diye font küçüldü
            layout_annotations.append(dict(x=start_date, y=1.02, xref="x", yref="paper", text=f" <b>{name.split()[0][0]}.{name.split()[-1]}</b>", showarrow=False, xanchor="left", font=dict(size=9, color="#555")))

        fig.update_layout(
            title=dict(text="Analiz Paneli", font=dict(size=16)),
            hovermode="x unified", 
            height=500, # Mobilde çok uzun olmasın
            margin=dict(l=10, r=10, t=80, b=10), # Kenar boşlukları azaltıldı
            shapes=layout_shapes, annotations=layout_annotations,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=1),
            yaxis=dict(title="Net Skor", range=[-150, 150], zeroline=False),
            yaxis2=dict(title="Faiz/Enf/Okunabilirlik", overlaying="y", side="right", range=[-market_max, market_max], showgrid=False),
            yaxis3=dict(overlaying="y", side="right", showgrid=False, visible=False, range=[0, merged['word_count'].max() * 2])
        )
        
        # --- ETKİLEŞİMLİ GRAFİK ---
        st.caption("🖱️ Grafikteki noktalara tıklayarak yapay zeka özetini görebilirsiniz.")
        selected_points = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points", key="main_dashboard_chart")
        
        if selected_points and selected_points["selection"]["points"]:
            try:
                point_data = selected_points["selection"]["points"][0]
                clicked_date = point_data.get("x")
                if clicked_date:
                    mask = merged['period_date'].astype(str).str.startswith(str(clicked_date))
                    if mask.any():
                        row = merged[mask].iloc[0]
                        st.divider()
                        st.subheader(f"📅 {row['Donem']} Analiz Özeti")
                        ai_summary = utils.generate_smart_summary(row)
                        with st.chat_message("assistant"): st.markdown(ai_summary)
                        # Metrikler mobilde alt alta sığsın diye columns(2) x 2 satır yapıyoruz
                        m1, m2 = st.columns(2)
                        m1.metric("Skor", f"{row['score_abg_scaled']:.1f}")
                        m2.metric("Enflasyon", f"%{row.get('Yıllık TÜFE', 0)}")
                        m3, m4 = st.columns(2)
                        m3.metric("Faiz", f"%{row.get('PPK Faizi', 0)}")
                        m4.metric("Okunabilirlik", f"{row.get('flesch_score', 0)}")
            except Exception as e: st.error(f"Hata: {e}")

        if st.button("🔄 Yenile"): st.cache_data.clear(); st.rerun()
    else: st.info("Kayıt yok.")

# ==============================================================================
# TAB 2: VERİ GİRİŞİ
# ==============================================================================
with tab2:
    st.subheader("Veri İşlemleri")
    st.info("ℹ️ Aşağıdaki listeden seçim yaparak detayları görebilirsiniz.")

    with st.container():
        df_all = utils.fetch_all_data()
        if not df_all.empty: 
            df_all['period_date'] = pd.to_datetime(df_all['period_date'])
            df_all['date_only'] = df_all['period_date'].dt.date
            current_id = st.session_state['form_data']['id']
    
            with st.container(border=True):
                if st.button("➕ YENİ VERİ GİRİŞİ (Temizle)", type="secondary"):
                    reset_form(); st.rerun()
                st.markdown("---")
                
                # Mobilde inputs alt alta
                val_date = st.session_state['form_data']['date']
                selected_date = st.date_input("Tarih", value=val_date)
                
                val_source = st.session_state['form_data']['source']
                source = st.text_input("Kaynak", value=val_source)
                
                val_text = st.session_state['form_data']['text']
                txt = st.text_area("Metin", value=val_text, height=200, placeholder="Metni buraya yapıştırın...")
                st.markdown("---")
                
                # 1. ÇAKIŞMA
                if st.session_state['collision_state']['active']:
                    st.error(f"⚠️ Bu tarihte kayıt var!")
                    admin_pass = st.text_input("Admin Şifresi", type="password", key="overwrite_pass")
                    if st.button("🚨 Onayla ve Üzerine Yaz", type="primary"):
                        if admin_pass == ADMIN_PWD:
                            p_txt = st.session_state['collision_state']['pending_text']
                            t_id = st.session_state['collision_state']['target_id']
                            s_abg, h_cnt, d_cnt, hawks, doves, h_ctx, d_ctx, flesch = utils.run_full_analysis(p_txt)
                            utils.update_entry(t_id, selected_date, p_txt, source, s_abg, s_abg)
                            st.success("Yazıldı!"); reset_form(); st.rerun()
                        else: st.error("Hatalı Şifre!")
                    if st.button("❌ İptal"):
                        st.session_state['collision_state']['active'] = False; st.rerun()

                # 2. GÜNCELLEME
                elif st.session_state['update_state']['active']:
                    st.warning("Güncelleme Onayı")
                    update_pass = st.text_input("Admin Şifresi", type="password", key="update_pass")
                    if st.button("💾 Onayla ve Güncelle", type="primary"):
                        if update_pass == ADMIN_PWD:
                            p_txt = st.session_state['update_state']['pending_text']
                            s_abg, h_cnt, d_cnt, hawks, doves, h_ctx, d_ctx, flesch = utils.run_full_analysis(p_txt)
                            utils.update_entry(current_id, selected_date, p_txt, source, s_abg, s_abg)
                            st.success("Güncellendi!"); reset_form(); st.rerun()
                        else: st.error("Hatalı Şifre!")
                    if st.button("❌ İptal"):
                        st.session_state['update_state']['active'] = False; st.rerun()

                # 3. NORMAL
                else:
                    btn_label = "💾 Güncelle" if current_id else "💾 Kaydet / Analiz Et"
                    if st.button(btn_label, type="primary"):
                        if txt:
                            collision_record = None
                            if not df_all.empty:
                                mask = df_all['date_only'] == selected_date
                                if mask.any(): collision_record = df_all[mask].iloc[0]
                            
                            is_self_update = current_id and ((collision_record is None) or (collision_record is not None and int(collision_record['id']) == current_id))

                            if is_self_update:
                                st.session_state['update_state'] = {'active': True, 'pending_text': txt}; st.rerun()
                            elif collision_record is not None:
                                st.session_state['collision_state'] = {'active': True, 'target_id': int(collision_record['id']), 'target_date': selected_date, 'pending_text': txt}; st.rerun()
                            else:
                                s_abg, h_cnt, d_cnt, hawks, doves, h_ctx, d_ctx, flesch = utils.run_full_analysis(txt)
                                utils.insert_entry(selected_date, txt, source, s_abg, s_abg)
                                st.success("Eklendi!"); reset_form(); st.rerun()
                        else: st.error("Metin boş.")
                    
                    if current_id:
                        with st.popover("🗑️ Sil"):
                            del_pass = st.text_input("Şifre", type="password", key="del_pass")
                            if st.button("🔥 Sil"):
                                if del_pass == ADMIN_PWD:
                                    utils.delete_entry(current_id); st.success("Silindi!"); reset_form(); st.rerun()
                                else: st.error("Hatalı!")

                # CANLI ANALİZ
                if txt:
                    s_live, h_cnt, d_cnt, h_list, d_list, h_ctx, d_ctx, flesch_live = utils.run_full_analysis(txt)
                    st.markdown("---")
                    c1, c2 = st.columns(2)
                    with c1: st.metric("Şahin", f"{h_cnt}")
                    with c2: st.metric("Güvercin", f"{d_cnt}")
                    st.metric("Okunabilirlik", f"{flesch_live:.1f}")
                    st.caption(f"**Net Skor:** {s_live:.2f}")
                    
                    with st.expander("Detaylar"):
                        st.markdown("**🦅 Şahin**")
                        if h_list:
                            for item in h_list: st.write(f"- {item}")
                        st.markdown("**🕊️ Güvercin**")
                        if d_list:
                            for item in d_list: st.write(f"- {item}")

            st.markdown("### 📋 Kayıtlar")
            df_show = df_all.copy()
            df_show['Dönem'] = df_show['period_date'].dt.strftime('%Y-%m')
            df_show['Görsel Skor'] = df_show['score_abg'].apply(lambda x: x*100 if abs(x)<=1 else x)
            event = st.dataframe(df_show[['id', 'Dönem', 'Görsel Skor']], on_select="rerun", selection_mode="single-row", use_container_width=True, hide_index=True, key=st.session_state['table_key'])
            
            if len(event.selection.rows) > 0:
                sel_id = df_show.iloc[event.selection.rows[0]]['id']
                if st.session_state['collision_state']['active'] or st.session_state['update_state']['active']:
                    st.session_state['collision_state']['active'] = False; st.session_state['update_state']['active'] = False
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
            if 'Yıllık TÜFE' in df.columns: fig_m.add_trace(go.Scatter(x=df['Donem'], y=df['Yıllık TÜFE'], name="TÜFE", line=dict(color='red')))
            if 'PPK Faizi' in df.columns: fig_m.add_trace(go.Scatter(x=df['Donem'], y=df['PPK Faizi'], name="Faiz", line=dict(color='orange')))
            st.plotly_chart(fig_m, use_container_width=True)
            st.dataframe(df, use_container_width=True)
        else: st.error(f"Hata: {err}")
