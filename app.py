import streamlit as st
import pandas as pd
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import utils 
import uuid

st.set_page_config(page_title="Piyasa Analiz", layout="wide")

# --- GÜVENLİK ---
APP_PWD = "SahinGuvercin34"      
ADMIN_PWD = "SahinGuvercin06"    

# --- GİRİŞ EKRANI ---
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

# --- SESSION STATE ---
if 'form_data' not in st.session_state:
    st.session_state['form_data'] = {'id': None, 'date': datetime.date.today().replace(day=1), 'source': "TCMB", 'text': ""}
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

# YENİ: 4. Sekme Eklendi "Derin Analiz"
tab1, tab2, tab3, tab4 = st.tabs(["📈 Dashboard", "📝 Veri Girişi & Yönetimi", "📊 Piyasa Verileri", "🔍 Derin Analiz"])

# ==============================================================================
# TAB 1: DASHBOARD
# ==============================================================================
with tab1:
    with st.spinner("Veriler Yükleniyor..."):
        df_logs = utils.fetch_all_data()
    
    if not df_logs.empty:
        df_logs['period_date'] = pd.to_datetime(df_logs['period_date'])
        df_logs['Donem'] = df_logs['period_date'].dt.strftime('%Y-%m')
        
        # Hesaplamalar
        df_logs['word_count'] = df_logs['text_content'].apply(lambda x: len(str(x).split()) if x else 0)
        df_logs['flesch_score'] = df_logs['text_content'].apply(lambda x: utils.calculate_flesch_reading_ease(str(x)))
        df_logs['score_abg_scaled'] = df_logs['score_abg'].apply(lambda x: x*100 if abs(x) <= 1 else x)

        min_d = df_logs['period_date'].min().date()
        max_d = datetime.date.today()
        df_market, err = utils.fetch_market_data_adapter(min_d, max_d)
        
        merged = pd.merge(df_logs, df_market, on="Donem", how="left")
        merged = merged.sort_values("period_date")
        
        if 'Yıllık TÜFE' in merged.columns: merged['Yıllık TÜFE'] = pd.to_numeric(merged['Yıllık TÜFE'], errors='coerce')
        if 'PPK Faizi' in merged.columns: merged['PPK Faizi'] = pd.to_numeric(merged['PPK Faizi'], errors='coerce')
        
        market_vals = [80]
        if 'Yıllık TÜFE' in merged.columns: market_vals.append(merged['Yıllık TÜFE'].max())
        if 'PPK Faizi' in merged.columns: market_vals.append(merged['PPK Faizi'].max())
        market_vals = [v for v in market_vals if pd.notna(v)]
        market_max = max(market_vals) + 10

        # --- GRAFİK ---
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(go.Bar(
            x=merged['period_date'], y=merged['word_count'], name="Metin Uzunluğu",
            marker=dict(color='gray'), opacity=0.10, yaxis="y3", hoverinfo="x+y+name"
        ))

        fig.add_trace(go.Scatter(
            x=merged['period_date'], y=merged['score_abg_scaled'], name="Şahin/Güvercin Skoru", 
            line=dict(color='black', width=3), marker=dict(size=8, color='black'), yaxis="y"
        ))
        
        if 'Yıllık TÜFE' in merged.columns:
            fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['Yıllık TÜFE'], name="Yıllık TÜFE (%)", line=dict(color='red', dash='dot'), yaxis="y"))
        if 'PPK Faizi' in merged.columns:
            fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['PPK Faizi'], name="Faiz (%)", line=dict(color='orange', dash='dot'), yaxis="y"))

        fig.add_trace(go.Scatter(
            x=merged['period_date'], y=merged['flesch_score'], name="Okunabilirlik (Flesch)",
            mode='markers', marker=dict(color='teal', size=8, opacity=0.8), yaxis="y"
        ))

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
            title="Merkez Bankası Analiz Paneli", hovermode="x unified", height=650,
            shapes=layout_shapes, annotations=layout_annotations, showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
            yaxis=dict(title="Skor & Oranlar", range=[-150, 150], zeroline=False),
            yaxis2=dict(visible=False, overlaying="y", side="right"),
            yaxis3=dict(title="Kelime", overlaying="y", side="right", showgrid=False, visible=False, range=[0, merged['word_count'].max() * 2])
        )
        st.plotly_chart(fig, use_container_width=True)

        if st.button("🔄 Yenile"): st.cache_data.clear(); st.rerun()
    else: st.info("Kayıt yok.")

# ==============================================================================
# TAB 2: VERİ GİRİŞİ
# ==============================================================================
with tab2:
    st.subheader("Veri İşlemleri")
    st.info("ℹ️ Kayıt seçerek düzenleme yapabilirsiniz.")

    with st.container():
        df_all = utils.fetch_all_data()
        if not df_all.empty: 
            df_all['period_date'] = pd.to_datetime(df_all['period_date'])
            df_all['date_only'] = df_all['period_date'].dt.date
            current_id = st.session_state['form_data']['id']
    
            with st.container(border=True):
                if st.button("➕ YENİ VERİ GİRİŞİ (Temizle)", type="secondary"): reset_form(); st.rerun()
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
                    st.error("⚠️ Kayıt Çakışması")
                    admin_pass = st.text_input("Admin Şifresi", type="password", key="overwrite_pass")
                    if st.button("🚨 Üzerine Yaz", type="primary"):
                        if admin_pass == ADMIN_PWD:
                            p_txt = st.session_state['collision_state']['pending_text']
                            t_id = st.session_state['collision_state']['target_id']
                            s_abg, h_cnt, d_cnt, hawks, doves, h_ctx, d_ctx, flesch = utils.run_full_analysis(p_txt)
                            utils.update_entry(t_id, selected_date, p_txt, source, s_abg, s_abg)
                            st.success("Başarılı!"); reset_form(); st.rerun()
                        else: st.error("Hatalı Şifre!")
                    if st.button("❌ İptal"): st.session_state['collision_state']['active'] = False; st.rerun()

                # 2. GÜNCELLEME
                elif st.session_state['update_state']['active']:
                    st.warning("Güncelleme Onayı")
                    update_pass = st.text_input("Admin Şifresi", type="password", key="update_pass")
                    if st.button("💾 Güncelle", type="primary"):
                        if update_pass == ADMIN_PWD:
                            p_txt = st.session_state['update_state']['pending_text']
                            s_abg, h_cnt, d_cnt, hawks, doves, h_ctx, d_ctx, flesch = utils.run_full_analysis(p_txt)
                            utils.update_entry(current_id, selected_date, p_txt, source, s_abg, s_abg)
                            st.success("Güncellendi!"); reset_form(); st.rerun()
                        else: st.error("Hatalı Şifre!")
                    if st.button("❌ İptal"): st.session_state['update_state']['active'] = False; st.rerun()

                # 3. NORMAL
                else:
                    btn_label = "💾 Güncelle" if current_id else "💾 Kaydet"
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
                                if del_pass == ADMIN_PWD: utils.delete_entry(current_id); st.success("Silindi!"); reset_form(); st.rerun()
                                else: st.error("Hatalı!")

                # CANLI ANALİZ
                if txt:
                    s_live, h_cnt, d_cnt, h_list, d_list, h_ctx, d_ctx, flesch_live = utils.run_full_analysis(txt)
                    st.markdown("---")
                    c1, c2 = st.columns(2)
                    with c1: st.metric("Şahin", f"{h_cnt}")
                    with c2: st.metric("Güvercin", f"{d_cnt}")
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

# ==============================================================================
# TAB 4: DERİN ANALİZ (DİFF & FREKANS)
# ==============================================================================
with tab4:
    st.header("🔍 Derin Analiz")
    
    df_all = utils.fetch_all_data()
    if not df_all.empty:
        df_all['period_date'] = pd.to_datetime(df_all['period_date'])
        df_all['Donem'] = df_all['period_date'].dt.strftime('%Y-%m')
        df_all = df_all.sort_values('period_date', ascending=False)
        
        # --- BÖLÜM 1: METİN FARKI (DIFF) ANALİZİ ---
        st.subheader("1. Metin Farkı Analizi (Diff)")
        st.markdown("İki farklı toplantı metnini karşılaştırarak nelerin değiştiğini (eklenen/çıkarılan cümleler) görün.")
        
        c_diff1, c_diff2 = st.columns(2)
        with c_diff1:
            # Varsayılan olarak en son kaydı seç
            date1_opts = df_all['Donem'].tolist()
            sel_date1 = st.selectbox("Eski Metin (Referans):", date1_opts, index=min(1, len(date1_opts)-1))
        
        with c_diff2:
            # Varsayılan olarak bir önceki kaydı seç
            date2_opts = df_all['Donem'].tolist()
            sel_date2 = st.selectbox("Yeni Metin (Karşılaştırılan):", date2_opts, index=0)
            
        if st.button("Farkları Göster", type="primary"):
            if sel_date1 and sel_date2:
                text1 = df_all[df_all['Donem'] == sel_date1].iloc[0]['text_content']
                text2 = df_all[df_all['Donem'] == sel_date2].iloc[0]['text_content']
                
                diff_html = utils.generate_diff_html(text1, text2)
                
                st.markdown("#### Analiz Sonucu:")
                st.caption(f"**Kırmızı (Üstü Çizili):** {sel_date1} metninden çıkarılanlar. | **Yeşil:** {sel_date2} metninde eklenenler.")
                
                with st.container(border=True, height=400):
                    st.markdown(diff_html, unsafe_allow_html=True)
        
        st.divider()
        
        # --- BÖLÜM 2: KELİME FREKANSI ZAMAN SERİSİ ---
        st.subheader("2. Kelime Frekansı Zaman Serisi")
        st.markdown("Belirli bir kelimenin (örn: 'enflasyon', 'sıkılaştırma') zaman içindeki kullanım sıklığını inceleyin.")
        
        search_term = st.text_input("Aranacak Kelime:", value="enflasyon")
        
        if search_term:
            freq_df = utils.get_word_frequency_series(df_all, search_term)
            
            if not freq_df.empty:
                # Toplam kullanım
                total_usage = freq_df['count'].sum()
                st.metric(f"Toplam '{search_term}' Geçişi", total_usage)
                
                # Grafik
                fig_freq = go.Figure()
                fig_freq.add_trace(go.Bar(
                    x=freq_df['period_date'], 
                    y=freq_df['count'],
                    name="Frekans",
                    marker_color='teal'
                ))
                
                fig_freq.update_layout(
                    title=f"'{search_term}' Kelimesinin Zaman İçindeki Sıklığı",
                    xaxis_title="Tarih",
                    yaxis_title="Geçiş Sayısı",
                    hovermode="x unified",
                    height=400
                )
                st.plotly_chart(fig_freq, use_container_width=True)
            else:
                st.warning("Veri bulunamadı.")
                
    else:
        st.info("Analiz için yeterli veri yok.")
