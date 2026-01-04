import streamlit as st
import pandas as pd
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import utils 
import uuid

st.set_page_config(page_title="Piyasa Analiz", layout="wide")

# --- CSS İYİLEŞTİRMELERİ ---
st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 5rem; }
    h1 { font-size: 1.8rem !important; }
    .stDataFrame { font-size: 0.8rem; }
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

# --- SESSION ---
if 'form_data' not in st.session_state: st.session_state['form_data'] = {'id': None, 'date': datetime.date.today().replace(day=1), 'source': "TCMB", 'text': ""}
if 'table_key' not in st.session_state: st.session_state['table_key'] = str(uuid.uuid4())
if 'collision_state' not in st.session_state: st.session_state['collision_state'] = {'active': False, 'target_id': None, 'pending_text': None, 'target_date': None}
if 'update_state' not in st.session_state: st.session_state['update_state'] = {'active': False, 'pending_text': None}

def reset_form():
    st.session_state['form_data'] = {'id': None, 'date': datetime.date.today(), 'source': "TCMB", 'text': ""}
    st.session_state['collision_state'] = {'active': False, 'target_id': None, 'pending_text': None, 'target_date': None}
    st.session_state['update_state'] = {'active': False, 'pending_text': None}
    st.session_state['table_key'] = str(uuid.uuid4())

# --- HEADER ---
c_head1, c_head2 = st.columns([6, 1])
with c_head1: st.title("🦅 Şahin/Güvercin Paneli")
with c_head2: 
    if st.button("Çıkış"): st.session_state['logged_in'] = False; st.rerun()

# --- TABS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Dashboard", 
    "📝 Veri Girişi", 
    "📊 Veriler", 
    "🔍 Derin Analiz",
    "🤖 Faiz Tahmini",
    "☁️ WordCloud"
])

# ==============================================================================
# TAB 1: DASHBOARD
# ==============================================================================
with tab1:
    with st.spinner("Veriler Yükleniyor..."):
        df_logs = utils.fetch_all_data()
    
    if not df_logs.empty:
        df_logs['period_date'] = pd.to_datetime(df_logs['period_date'])
        df_logs['Donem'] = df_logs['period_date'].dt.strftime('%Y-%m')
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

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=merged['period_date'], y=merged['word_count'], name="Metin Uzunluğu", marker=dict(color='gray'), opacity=0.10, yaxis="y3", hoverinfo="x+y+name"))
        fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['score_abg_scaled'], name="Şahin/Güvercin Skoru", line=dict(color='black', width=3), marker=dict(size=8, color='black'), yaxis="y"))
        
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

        fig.update_layout(
            title="Merkez Bankası Analiz Paneli", hovermode="x unified", height=600,
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
# TAB 2: VERİ GİRİŞİ (DETAYLAR GERİ GELDİ!)
# ==============================================================================
with tab2:
    st.subheader("Veri İşlemleri")
    # --- GERİ GELEN UYARI ---
    st.info("ℹ️ **BİLGİ:** Aşağıdaki geçmiş kayıtlar listesinden istediğiniz dönemi seçerek, hangi cümlelerin hesaplamaya alındığını görebilirsiniz.")
    
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
                    val_date = st.session_state['form_data']['date']; selected_date = st.date_input("Tarih", value=val_date)
                    val_source = st.session_state['form_data']['source']; source = st.text_input("Kaynak", value=val_source)
                    st.caption(f"Dönem: **{selected_date.strftime('%Y-%m')}**")
                with c2:
                    val_text = st.session_state['form_data']['text']; txt = st.text_area("Metin", value=val_text, height=200, placeholder="Metni buraya yapıştırın...")
                st.markdown("---")
                
                # ONAY BUTONLARI (GÜVENLİK)
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

                # --- GERİ GELEN DETAYLAR ---
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
            if 'Yıllık TÜFE' in df.columns: fig_m.add_trace(go.Scatter(x=df['Donem'], y=df['Yıllık TÜFE'], name="TÜFE", line=dict(color='red')))
            if 'PPK Faizi' in df.columns: fig_m.add_trace(go.Scatter(x=df['Donem'], y=df['PPK Faizi'], name="Faiz", line=dict(color='orange')))
            st.plotly_chart(fig_m, use_container_width=True)
            st.dataframe(df, use_container_width=True)
        else: st.error(f"Hata: {err}")

# ==============================================================================
# TAB 4: DERİN ANALİZ (DIFF & TREND)
# ==============================================================================
with tab4:
    st.header("🔍 Derin Analiz ve Metin Madenciliği")
    df_all = utils.fetch_all_data()
    if not df_all.empty:
        df_all['period_date'] = pd.to_datetime(df_all['period_date'])
        df_all['Donem'] = df_all['period_date'].dt.strftime('%Y-%m')
        df_all = df_all.sort_values('period_date', ascending=False)
        
        # --- KELİME FREKANSI TRENDİ ---
        st.subheader("📊 En Çok Tekrar Eden Ekonomi Terimleri")
        st.caption("Seçilen dönemdeki metinlerde en sık geçen ekonomi terimlerinin zaman içindeki değişimi.")
        
        top_n = st.slider("Gösterilecek Kelime Sayısı", 3, 10, 5)
        freq_df, top_terms = utils.get_top_terms_series(df_all, top_n)
        
        if not freq_df.empty:
            fig_freq = go.Figure()
            for term in top_terms:
                fig_freq.add_trace(go.Scatter(x=freq_df['period_date'], y=freq_df[term], name=term, mode='lines+markers'))
            fig_freq.update_layout(title="Kelime Kullanım Sıklığı Trendi", hovermode="x unified", height=400)
            st.plotly_chart(fig_freq, use_container_width=True)
        
        st.divider()
        
        # --- DIFF ANALİZİ ---
        st.subheader("🔄 Metin Farkı (Diff) Analizi")
        st.caption("İki farklı PPK metnini karşılaştırarak nelerin değiştiğini görün.")
        c_diff1, c_diff2 = st.columns(2)
        with c_diff1:
            sel_date1 = st.selectbox("Eski Metin (Referans):", df_all['Donem'].tolist(), index=min(1, len(df_all)-1))
        with c_diff2:
            sel_date2 = st.selectbox("Yeni Metin (Karşılaştırılan):", df_all['Donem'].tolist(), index=0)
            
        if st.button("Farkları Göster", type="primary"):
            if sel_date1 and sel_date2:
                t1 = df_all[df_all['Donem'] == sel_date1].iloc[0]['text_content']
                t2 = df_all[df_all['Donem'] == sel_date2].iloc[0]['text_content']
                diff_html = utils.generate_diff_html(t1, t2)
                st.markdown(f"**Kırmızı:** {sel_date1}'den silinenler | **Yeşil:** {sel_date2}'ye eklenenler")
                with st.container(border=True, height=400):
                    st.markdown(diff_html, unsafe_allow_html=True)
    else: st.info("Yeterli veri yok.")

# ==============================================================================
# TAB 5: FAİZ TAHMİNİ (TEXT AS DATA)
# ==============================================================================
with tab5:
    st.header("🤖 Text-as-Data: Faiz Tahmini")
    st.info("Bu modül, geçmiş metinlerin 'Şahinlik Skoru' ile bir sonraki aydaki 'Faiz Değişimi' arasındaki ilişkiyi öğrenerek (Lineer Regresyon), girilen metne göre faiz kararı tahmini yapar.")
    
    # Veri Hazırla
    if 'merged' in locals() and not merged.empty: # Tab 1'den gelen merged data
        # Güncel metin girişi veya son kayıt
        if st.session_state['form_data']['text']:
            target_text = st.session_state['form_data']['text']
            target_source = "Giriş Alanındaki Metin"
        elif not df_all.empty:
            target_text = df_all.iloc[0]['text_content']
            target_source = f"Son Kayıt ({df_all.iloc[0]['Donem']})"
        else:
            target_text = None
            
        if target_text:
            # Anlık Skorla
            s_live, _, _, _, _, _, _, _ = utils.run_full_analysis(target_text)
            
            # Modeli Eğit ve Tahmin Et
            result, error = utils.train_and_predict_rate(merged, s_live)
            
            if result:
                st.subheader(f"Analiz Kaynağı: {target_source}")
                
                col_pred1, col_pred2 = st.columns(2)
                with col_pred1:
                    change_bps = result['prediction'] * 100 # Baz puan
                    direction = "ARTIRIM" if change_bps > 25 else "İNDİRİM" if change_bps < -25 else "SABİT"
                    color = "red" if direction == "ARTIRIM" else "blue" if direction == "İNDİRİM" else "gray"
                    
                    st.markdown(f"### Tahmin: :{color}[{direction}]")
                    st.metric("Beklenen Değişim (Baz Puan)", f"{change_bps:.0f} bps")
                
                with col_pred2:
                    st.write("📊 **Model İstatistikleri**")
                    st.write(f"- Eğitim Verisi: {result['sample_size']} Toplantı")
                    st.write(f"- Korelasyon: {result['correlation']:.2f}")
                    st.caption("*Not: Bu sadece istatistiksel bir modellemedir, yatırım tavsiyesi değildir.*")
                    
                # Regresyon Grafiği
                fig_reg = go.Figure()
                # Geçmiş veriler
                fig_reg.add_trace(go.Scatter(
                    x=merged['score_abg_scaled'], 
                    y=merged['PPK Faizi'].shift(-1) - merged['PPK Faizi'],
                    mode='markers', name='Geçmiş Kararlar'
                ))
                # Tahmin noktası
                fig_reg.add_trace(go.Scatter(
                    x=[s_live], y=[result['prediction']],
                    mode='markers', marker=dict(color='red', size=15, symbol='star'),
                    name='Şu Anki Tahmin'
                ))
                fig_reg.update_layout(title="Skor vs. Faiz Değişimi İlişkisi", xaxis_title="Metin Skoru", yaxis_title="Sonraki Ay Faiz Değişimi")
                st.plotly_chart(fig_reg, use_container_width=True)
            else:
                st.warning(f"Tahmin yapılamadı: {error}")
        else:
            st.warning("Lütfen Veri Girişi sekmesinden bir metin girin veya geçmiş kayıt yükleyin.")
    else:
        st.warning("Modeli eğitmek için Dashboard sekmesinin yüklenmesi ve yeterli piyasa verisi olması gerekir.")

# ==============================================================================
# TAB 6: WORDCLOUD
# ==============================================================================
with tab6:
    st.header("☁️ Kelime Bulutu (WordCloud)")
    if not df_all.empty:
        dates = df_all['Donem'].tolist()
        sel_cloud_date = st.selectbox("Dönem Seçin:", ["Tüm Zamanlar"] + dates)
        
        if st.button("Bulutu Oluştur", type="primary"):
            if sel_cloud_date == "Tüm Zamanlar":
                text_cloud = " ".join(df_all['text_content'].astype(str).tolist())
            else:
                text_cloud = df_all[df_all['Donem'] == sel_cloud_date].iloc[0]['text_content']
            
            fig_wc = utils.generate_wordcloud_img(text_cloud)
            if fig_wc:
                st.pyplot(fig_wc)
            else:
                st.error("WordCloud kütüphanesi eksik veya metin boş.")
    else:
        st.info("Veri yok.")
