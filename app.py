import streamlit as st
import pandas as pd
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import utils 
import uuid

st.set_page_config(page_title="Piyasa Analiz", layout="wide")

# --- 0. GÜVENLİK ---
APP_PWD = "SahinGuvercin34"      
ADMIN_PWD = "SahinGuvercin06"    

# --- 1. GİRİŞ EKRANI ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center;'>🔐 Güvenli Giriş</h2>", unsafe_allow_html=True)
        pwd_input = st.text_input("Uygulama Şifresi", type="password")
        if st.button("Giriş Yap", type="primary", use_container_width=True):
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
c1, c2 = st.columns([6, 1])
with c1: st.title("🦅 Şahin/Güvercin Analiz Paneli")
with c2: 
    if st.button("Çıkış Yap"):
        st.session_state['logged_in'] = False
        st.rerun()

tab1, tab2, tab3 = st.tabs(["📈 Dashboard", "📝 Veri Girişi & Yönetimi", "📊 Piyasa Verileri"])

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
        
        # Veri Tipi Dönüşümü ve Max Değer Hesabı (HATA ÖNLEYİCİ)
        if 'Yıllık TÜFE' in merged.columns: merged['Yıllık TÜFE'] = pd.to_numeric(merged['Yıllık TÜFE'], errors='coerce')
        if 'PPK Faizi' in merged.columns: merged['PPK Faizi'] = pd.to_numeric(merged['PPK Faizi'], errors='coerce')
        
        market_vals = [80]
        if 'Yıllık TÜFE' in merged.columns: market_vals.append(merged['Yıllık TÜFE'].max())
        if 'PPK Faizi' in merged.columns: market_vals.append(merged['PPK Faizi'].max())
        market_vals = [v for v in market_vals if pd.notna(v)]
        market_max = max(market_vals) + 10

        # --- ANA GRAFİK ---
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # 1. Kelime Sayısı (Arka Plan Bar - Gizli Eksen Y3)
        fig.add_trace(go.Bar(
            x=merged['period_date'], y=merged['word_count'], name="Metin Uzunluğu",
            marker=dict(color='gray'), opacity=0.15, yaxis="y3", hoverinfo="x+y+name"
        ))

        # 2. Skor Çizgisi (Sol Eksen Y1)
        fig.add_trace(go.Scatter(
            x=merged['period_date'], y=merged['score_abg_scaled'], name="Şahin/Güvercin Skoru", 
            line=dict(color='black', width=3), marker=dict(size=8, color='black')
        ), secondary_y=False)
        
        # 3. Piyasa Verileri (Sağ Eksen Y2)
        if 'Yıllık TÜFE' in merged.columns:
            fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['Yıllık TÜFE'], name="Yıllık TÜFE (%)", line=dict(color='red', dash='dot')), secondary_y=True)
        if 'PPK Faizi' in merged.columns:
            fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['PPK Faizi'], name="Faiz (%)", line=dict(color='orange', dash='dot')), secondary_y=True)

        # 4. Okunabilirlik Skoru (Sağ Eksen Y2 - Yuvarlak Noktalar)
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
            dict(x=0.02, y=130, xref="paper", yref="y", text="🦅 ŞAHİN BÖLGESİ", showarrow=False, font=dict(size=14, color="darkred", weight="bold"), xanchor="left"),
            dict(x=0.02, y=-130, xref="paper", yref="y", text="🕊️ GÜVERCİN BÖLGESİ", showarrow=False, font=dict(size=14, color="darkblue", weight="bold"), xanchor="left")
        ]
        
        governors = [("2020-11-01", "Naci Ağbal"), ("2021-04-01", "Şahap Kavcıoğlu"), ("2023-06-01", "Hafize Gaye Erkan"), ("2024-02-01", "Fatih Karahan")]
        for start_date, name in governors:
            layout_shapes.append(dict(type="line", xref="x", yref="paper", x0=start_date, x1=start_date, y0=0, y1=1, line=dict(color="gray", width=1, dash="longdash"), layer="below"))
            layout_annotations.append(dict(x=start_date, y=1.02, xref="x", yref="paper", text=f" <b>{name}</b>", showarrow=False, xanchor="left", font=dict(size=10, color="#555")))

        fig.update_layout(
            title="Merkez Bankası Analiz Paneli (Ton, Piyasa ve Okunabilirlik)", 
            hovermode="x unified", height=600,
            shapes=layout_shapes, annotations=layout_annotations,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom", 
                y=1.02,
                xanchor="right", 
                x=1
            ),
            yaxis=dict(title="Net Skor (-100 / +100)", range=[-150, 150], zeroline=False),
            yaxis2=dict(
                title="Faiz, Enflasyon & Okunabilirlik", 
                overlaying="y", 
                side="right", 
                range=[-market_max, market_max], 
                showgrid=False
            ),
            yaxis3=dict(title="Kelime", overlaying="y", side="right", showgrid=False, visible=False, range=[0, merged['word_count'].max() * 2])
        )
        
        # --- ETKİLEŞİMLİ GRAFİK VE YAPAY ZEKA ÖZETİ ---
        st.markdown("##### 🖱️ Grafikte bir noktaya tıklayarak Yapay Zeka Özeti alabilirsiniz:")
        
        # 'on_select="rerun"' ve 'selection_mode="points"' kullanarak tıklamayı yakalıyoruz
        selected_points = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points")
        
        if selected_points and selected_points["selection"]["points"]:
            try:
                # Seçilen noktanın verisini bul
                point_data = selected_points["selection"]["points"][0]
                clicked_date = point_data.get("x")
                
                if clicked_date:
                    # Tarihe göre satırı filtrele
                    row = merged[merged['period_date'] == clicked_date].iloc[0]
                    
                    st.divider()
                    st.subheader("🤖 Yapay Zeka Destekli Dönem Analizi")
                    
                    # Akıllı Özet Üret
                    ai_summary = utils.generate_smart_summary(row)
                    
                    with st.chat_message("assistant"):
                        st.write(ai_summary)
                        
                    # Ekstra metrikler (Özetin altında şık dursun)
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Skor", f"{row['score_abg_scaled']:.1f}")
                    c2.metric("Enflasyon", f"%{row.get('Yıllık TÜFE', 0)}")
                    c3.metric("Faiz", f"%{row.get('PPK Faizi', 0)}")
                    c4.metric("Okunabilirlik", f"{row.get('flesch_score', 0)}")
                    
            except Exception as e:
                st.error(f"Analiz sırasında hata oluştu: {e}")

        if st.button("🔄 Yenile"): st.cache_data.clear(); st.rerun()
    else: st.info("Kayıt yok.")

# ==============================================================================
# TAB 2: VERİ GİRİŞİ
# ==============================================================================
with tab2:
    st.subheader("Veri İşlemleri")
    
    st.info("ℹ️ **BİLGİ:** Aşağıdaki geçmiş kayıtlar listesinden istediğiniz dönemi seçerek, hangi cümlelerin hesaplamaya alındığını görebilirsiniz.")

    # --- GEÇMİŞ KAYITLAR (KONTEYNER İÇİNDE - HATA ÖNLEYİCİ) ---
    with st.container():
        df_all = utils.fetch_all_data()
        if not df_all.empty: 
            df_all['period_date'] = pd.to_datetime(df_all['period_date'])
            df_all['date_only'] = df_all['period_date'].dt.date
            
            # Form ID kontrolü
            current_id = st.session_state['form_data']['id']
    
            with st.container(border=True):
                if st.button("➕ YENİ VERİ GİRİŞİ (Ekranı Temizle)", type="secondary", use_container_width=True):
                    reset_form(); st.rerun()
                st.markdown("---")
                c1, c2 = st.columns([1, 2])
                with c1:
                    val_date = st.session_state['form_data']['date']
                    selected_date = st.date_input("Tarih", value=val_date)
                    val_source = st.session_state['form_data']['source']
                    source = st.text_input("Kaynak", value=val_source)
                    st.caption(f"Dönem: **{selected_date.strftime('%Y-%m')}**")
                with c2:
                    val_text = st.session_state['form_data']['text']
                    txt = st.text_area("Metin", value=val_text, height=200, placeholder="Metni buraya yapıştırın...")
                st.markdown("---")
                
                # 1. ÇAKIŞMA
                if st.session_state['collision_state']['active']:
                    col_alert, col_act = st.columns([2, 2])
                    with col_alert:
                        t_date = st.session_state['collision_state']['target_date']
                        st.error(f"⚠️ **ÇAKIŞMA:** {t_date} tarihinde kayıt var!")
                    with col_act:
                        admin_pass = st.text_input("Admin Şifresi", type="password", key="overwrite_pass")
                        c_b1, c_b2 = st.columns(2)
                        with c_b1:
                            if st.button("🚨 Onayla ve Üzerine Yaz", type="primary", use_container_width=True):
                                if admin_pass == ADMIN_PWD:
                                    p_txt = st.session_state['collision_state']['pending_text']
                                    t_id = st.session_state['collision_state']['target_id']
                                    s_abg, h_cnt, d_cnt, hawks, doves, h_ctx, d_ctx, flesch = utils.run_full_analysis(p_txt)
                                    utils.update_entry(t_id, selected_date, p_txt, source, s_abg, s_abg)
                                    st.success("Başarıyla üzerine yazıldı!"); reset_form(); st.rerun()
                                else: st.error("Şifre Hatalı!")
                        with c_b2:
                            if st.button("❌ İptal", use_container_width=True):
                                st.session_state['collision_state']['active'] = False; st.rerun()

                # 2. GÜNCELLEME
                elif st.session_state['update_state']['active']:
                    col_alert, col_act = st.columns([2, 2])
                    with col_alert:
                        st.warning("Değişikliği kaydetmek için **Admin Şifresi** giriniz.")
                    with col_act:
                        update_pass = st.text_input("Admin Şifresi", type="password", key="update_pass")
                        c_b1, c_b2 = st.columns(2)
                        with c_b1:
                            if st.button("💾 Onayla ve Güncelle", type="primary", use_container_width=True):
                                if update_pass == ADMIN_PWD:
                                    p_txt = st.session_state['update_state']['pending_text']
                                    s_abg, h_cnt, d_cnt, hawks, doves, h_ctx, d_ctx, flesch = utils.run_full_analysis(p_txt)
                                    utils.update_entry(current_id, selected_date, p_txt, source, s_abg, s_abg)
                                    st.success("Kayıt güncellendi!"); reset_form(); st.rerun()
                                else: st.error("Şifre Hatalı!")
                        with c_b2:
                            if st.button("❌ İptal", use_container_width=True):
                                st.session_state['update_state']['active'] = False; st.rerun()

                # 3. NORMAL
                else:
                    col_b1, col_b2, col_b3 = st.columns([2, 1, 1])
                    with col_b1:
                        btn_label = "💾 Güncelle" if current_id else "💾 Kaydet / Analiz Et"
                        if st.button(btn_label, type="primary"):
                            if txt:
                                collision_record = None
                                if not df_all.empty:
                                    mask = df_all['date_only'] == selected_date
                                    if mask.any(): collision_record = df_all[mask].iloc[0]
                                
                                is_self_update = current_id and ((collision_record is None) or (collision_record is not None and int(collision_record['id']) == current_id))

                                if is_self_update:
                                    st.session_state['update_state'] = {'active': True, 'pending_text': txt}
                                    st.rerun()
                                elif collision_record is not None:
                                    st.session_state['collision_state'] = {
                                        'active': True, 'target_id': int(collision_record['id']), 'target_date': selected_date, 'pending_text': txt
                                    }
                                    st.rerun()
                                else:
                                    s_abg, h_cnt, d_cnt, hawks, doves, h_ctx, d_ctx, flesch = utils.run_full_analysis(txt)
                                    utils.insert_entry(selected_date, txt, source, s_abg, s_abg)
                                    st.success("Yeni kayıt eklendi!"); reset_form(); st.rerun()
                            else: st.error("Metin alanı boş.")
                    with col_b2:
                        if st.button("Temizle"): reset_form(); st.rerun()
                    with col_b3:
                        if current_id:
                            with st.popover("🗑️ Sil"):
                                st.write("Admin şifresi:"); del_pass = st.text_input("Şifre", type="password", key="del_pass")
                                if st.button("🔥 Sil"):
                                    if del_pass == ADMIN_PWD:
                                        utils.delete_entry(current_id); st.success("Silindi!"); reset_form(); st.rerun()
                                    else: st.error("Şifre Hatalı!")

                # CANLI ANALİZ
                if txt:
                    s_live, h_cnt, d_cnt, h_list, d_list, h_ctx, d_ctx, flesch_live = utils.run_full_analysis(txt)
                    st.markdown("---")
                    met1, met2, met3 = st.columns(3)
                    with met1: st.metric("Şahin", f"{h_cnt} Kelime")
                    with met2: st.metric("Güvercin", f"{d_cnt} Kelime")
                    with met3: 
                        d_col = "normal" if flesch_live > 60 else "inverse" if flesch_live < 30 else "off"
                        st.metric("Okunabilirlik", f"{flesch_live:.1f}", delta_color=d_col)
                    
                    st.caption(f"**Net Skor:** {s_live:.2f} (Ölçek: -100 / +100)")
                    
                    exp = st.expander("🔍 Detaylar", expanded=True)
                    with exp:
                        k1, k2 = st.columns(2)
                        with k1:
                            st.markdown("**🦅 Şahin**"); 
                            if h_list: 
                                for item in h_list:
                                    t = item.split(' (')[0]; st.write(f"🔹 **{item}**")
                                    if t in h_ctx: 
                                        for s in h_ctx[t]: st.caption(f"📝 ...{s}...")
                            else: st.write("-")
                        with k2:
                            st.markdown("**🕊️ Güvercin**"); 
                            if d_list: 
                                for item in d_list:
                                    t = item.split(' (')[0]; st.write(f"🔹 **{item}**")
                                    if t in d_ctx: 
                                        for s in d_ctx[t]: st.caption(f"📝 ...{s}...")
                            else: st.write("-")

            st.markdown("### 📋 Geçmiş Kayıtlar")
            
            df_show = df_all.copy()
            df_show['Dönem'] = df_show['period_date'].dt.strftime('%Y-%m')
            df_show['Görsel Skor'] = df_show['score_abg'].apply(lambda x: x*100 if abs(x)<=1 else x)
            event = st.dataframe(df_show[['id', 'Dönem', 'period_date', 'source', 'Görsel Skor']].sort_values('period_date', ascending=False), on_select="rerun", selection_mode="single-row", use_container_width=True, hide_index=True, key=st.session_state['table_key'])
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
