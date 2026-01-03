import streamlit as st
import pandas as pd
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# transformers (FinBERT) importu kaldırıldı
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

# --- ARAYÜZ ---
st.title("🦅 Şahin/Güvercin Analiz Paneli")
tab1, tab2, tab3 = st.tabs(["📈 Dashboard", "📝 Veri Girişi & Yönetimi", "📊 Piyasa Verileri"])

# ==============================================================================
# TAB 1: DASHBOARD (GÖRSEL DÜZELTME - GRADIENT KALDIRILDI)
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
        
        # ABG Skoru Çizgisi
        fig.add_trace(go.Scatter(
            x=merged['period_date'], 
            y=merged['score_abg'], 
            name="Şahin/Güvercin Skoru (ABG)", 
            line=dict(color='black', width=3),
            marker=dict(size=8, color='black')
        ), secondary_y=False)
        
        if 'Yıllık TÜFE' in merged.columns:
            fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['Yıllık TÜFE'], name="Yıllık TÜFE (%)", line=dict(color='red', dash='dot')), secondary_y=True)
        if 'PPK Faizi' in merged.columns:
            fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['PPK Faizi'], name="Faiz (%)", line=dict(color='orange', dash='dot')), secondary_y=True)

        # --- GÖRSEL AYARLAR (DÜZELTİLDİ) ---
        fig.update_layout(
            title="Merkez Bankası Tonu ve Piyasa Verileri",
            hovermode="x unified", 
            height=600,
            # Şekiller (Gradient yerine düz şeffaf renkler kullanıldı)
            shapes=[
                # 1. Kırmızı Bölge (Şahin - Üst)
                dict(
                    type="rect", xref="paper", yref="y",
                    x0=0, x1=1, y0=0, y1=1.5, # 0'dan yukarı
                    fillcolor="rgba(255, 0, 0, 0.08)", # Çok hafif kırmızı
                    line_width=0, layer="below"
                ),
                # 2. Mavi Bölge (Güvercin - Alt)
                dict(
                    type="rect", xref="paper", yref="y",
                    x0=0, x1=1, y0=-1.5, y1=0, # 0'dan aşağı
                    fillcolor="rgba(0, 0, 255, 0.08)", # Çok hafif mavi
                    line_width=0, layer="below"
                ),
                # 3. Kalın Sıfır Çizgisi
                dict(
                    type="line", xref="paper", yref="y",
                    x0=0, x1=1, y0=0, y1=0,
                    line=dict(color="black", width=3), layer="below"
                ),
            ],
            annotations=[
                dict(
                    x=0.01, y=0.95, xref="paper", yref="y",
                    text="🦅 ŞAHİN BÖLGESİ", showarrow=False,
                    font=dict(size=14, color="darkred", weight="bold")
                ),
                dict(
                    x=0.01, y=-0.95, xref="paper", yref="y",
                    text="🕊️ GÜVERCİN BÖLGESİ", showarrow=False,
                    font=dict(size=14, color="darkblue", weight="bold")
                )
            ]
        )
        
        # Y Eksenleri
        fig.update_yaxes(title_text="Şahin (+) / Güvercin (-) Skoru", range=[-1.1, 1.1], secondary_y=False, zeroline=False)
        fig.update_yaxes(title_text="Faiz & Enflasyon (%)", secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)
        if st.button("🔄 Yenile"): st.cache_data.clear(); st.rerun()
    else: st.info("Kayıt yok.")

# ==============================================================================
# TAB 2: VERİ GİRİŞİ
# ==============================================================================
with tab2:
    st.subheader("Veri İşlemleri")
    
    df_all = utils.fetch_all_data()
    if not df_all.empty: 
        df_all['period_date'] = pd.to_datetime(df_all['period_date'])
        df_all['date_only'] = df_all['period_date'].dt.date
    
    current_id = st.session_state['form_data']['id']
    
    with st.container(border=True):
        c1, c2 = st.columns([1, 2])
        with c1:
            val_date = st.session_state['form_data']['date']
            selected_date = st.date_input("Tarih", value=val_date)
            
            val_source = st.session_state['form_data']['source']
            source = st.text_input("Kaynak", value=val_source)
            st.caption(f"Dönem: **{selected_date.strftime('%Y-%m')}**")
            
            collision_record = None
            if not df_all.empty:
                mask = df_all['date_only'] == selected_date
                if mask.any():
                    collision_record = df_all[mask].iloc[0]
            
            if collision_record is not None and (current_id != collision_record['id']):
                st.warning(f"⚠️ **DİKKAT:** {selected_date} tarihinde zaten bir kayıt var!")
                st.markdown(f"*Kaydet tuşuna basarsanız mevcut verinin **üzerine yazılacaktır**.*")

        with c2:
            val_text = st.session_state['form_data']['text']
            txt = st.text_area("Metin", value=val_text, height=200)
        
        col_b1, col_b2, col_b3 = st.columns([2, 1, 1])
        with col_b1:
            btn_text = "💾 Kaydet / Analiz Et"
            if collision_record is not None and (current_id != collision_record['id']):
                btn_text = "⚠️ Üzerine Yaz ve Kaydet"
            elif current_id:
                btn_text = "💾 Güncelle"

            if st.button(btn_text, type="primary"):
                if txt:
                    # Analiz
                    s_abg, h_cnt, d_cnt, hawks, doves, h_ctx, d_ctx = utils.run_full_analysis(txt)
                    
                    # DB Kayıt
                    if current_id:
                        utils.update_entry(current_id, selected_date, txt, source, s_abg, s_abg)
                        st.success("Kayıt güncellendi!")
                        
                    elif collision_record is not None:
                        target_id = int(collision_record['id'])
                        utils.update_entry(target_id, selected_date, txt, source, s_abg, s_abg)
                        st.warning(f"{selected_date} tarihli eski kayıt güncellendi.")
                        
                    else:
                        utils.insert_entry(selected_date, txt, source, s_abg, s_abg)
                        st.success("Yeni kayıt eklendi!")
                    
                    st.session_state['form_data'] = {'id': None, 'date': datetime.date.today(), 'source': "TCMB", 'text': ""}
                    st.rerun()
                else:
                    st.error("Metin giriniz.")

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

        # CANLI ANALİZ GÖSTERİMİ
        if txt:
            s_live, h_cnt, d_cnt, h_list, d_list, h_ctx, d_ctx = utils.run_full_analysis(txt)
            
            total_sigs = h_cnt + d_cnt
            if total_sigs > 0:
                h_pct = (h_cnt / total_sigs) * 100
                d_pct = (d_cnt / total_sigs) * 100
                tone_label = "ŞAHİN" if h_pct > d_pct else "GÜVERCİN" if d_pct > h_pct else "DENGELİ"
            else:
                h_pct = 0; d_pct = 0
                tone_label = "NÖTR"
            
            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1: st.metric("Şahin (Hawkish)", f"%{h_pct:.1f}", f"{h_cnt} Sinyal")
            with c2: st.metric("Güvercin (Dovish)", f"%{d_pct:.1f}", f"{d_cnt} Sinyal")
            
            st.progress(h_pct / 100)
            st.caption(f"Genel Ton: **{tone_label}** | Skor: {s_live:.2f}")

            exp = st.expander("🔍 Kelime ve Cümle Detayları", expanded=True)
            with exp:
                k1, k2 = st.columns(2)
                with k1:
                    st.markdown("**🦅 Şahin İfadeler**")
                    if h_list:
                        for item in h_list:
                            term = item.split(' (')[0]
                            st.write(f"🔹 **{item}**")
                            if term in h_ctx:
                                for s in h_ctx[term]:
                                    st.caption(f"📝 ...{s}...")
                    else: st.write("- Yok")
                
                with k2:
                    st.markdown("**🕊️ Güvercin İfadeler**")
                    if d_list:
                        for item in d_list:
                            term = item.split(' (')[0]
                            st.write(f"🔹 **{item}**")
                            if term in d_ctx:
                                for s in d_ctx[term]:
                                    st.caption(f"📝 ...{s}...")
                    else: st.write("- Yok")

    # LİSTE
    st.markdown("### 📋 Geçmiş Kayıtlar")
    if not df_all.empty:
        df_show = df_all.copy()
        df_show['Dönem'] = df_show['period_date'].dt.strftime('%Y-%m')
        
        event = st.dataframe(
            df_show[['id', 'Dönem', 'period_date', 'source', 'score_abg']].sort_values('period_date', ascending=False),
            on_select="rerun", selection_mode="single-row", use_container_width=True, hide_index=True
        )
        
        if len(event.selection.rows) > 0:
            sel_idx = event.selection.rows[0]
            sel_id = df_show.iloc[sel_idx]['id']
            if st.session_state['form_data']['id'] != sel_id:
                orig = df_all[df_all['id'] == sel_id].iloc[0]
                st.session_state['form_data'] = {
                    'id': int(orig['id']),
                    'date': pd.to_datetime(orig['period_date']).date(),
                    'source': orig['source'],
                    'text': orig['text_content']
                }
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
