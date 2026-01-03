import streamlit as st
import pandas as pd
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import utils 

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
# Form Verileri
if 'form_data' not in st.session_state:
    st.session_state['form_data'] = {
        'id': None,
        'date': datetime.date.today().replace(day=1),
        'source': "TCMB",
        'text': ""
    }

# Çakışma Yönetimi (Collision State)
if 'collision_state' not in st.session_state:
    st.session_state['collision_state'] = {
        'active': False,       # Çakışma ekranı açık mı?
        'target_id': None,     # Üzerine yazılacak ID
        'pending_text': None,  # Kaydedilmeyi bekleyen metin
        'target_date': None    # Çakışan tarih
    }

def reset_form():
    """Formu ve çakışma durumunu sıfırlar"""
    st.session_state['form_data'] = {'id': None, 'date': datetime.date.today(), 'source': "TCMB", 'text': ""}
    st.session_state['collision_state'] = {'active': False, 'target_id': None, 'pending_text': None, 'target_date': None}

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
        
        min_d = df_logs['period_date'].min().date()
        max_d = datetime.date.today()
        df_market, err = utils.fetch_market_data_adapter(min_d, max_d)
        
        merged = pd.merge(df_logs, df_market, on="Donem", how="left")
        merged = merged.sort_values("period_date")
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(go.Scatter(
            x=merged['period_date'], y=merged['score_abg'], name="Şahin/Güvercin Skoru", 
            line=dict(color='black', width=3), marker=dict(size=8, color='black')
        ), secondary_y=False)
        
        if 'Yıllık TÜFE' in merged.columns:
            fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['Yıllık TÜFE'], name="Yıllık TÜFE (%)", line=dict(color='red', dash='dot')), secondary_y=True)
        if 'PPK Faizi' in merged.columns:
            fig.add_trace(go.Scatter(x=merged['period_date'], y=merged['PPK Faizi'], name="Faiz (%)", line=dict(color='orange', dash='dot')), secondary_y=True)

        fig.update_layout(
            title="Merkez Bankası Tonu ve Piyasa Verileri", hovermode="x unified", height=600,
            shapes=[
                dict(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=0, y1=1.5, fillcolor="rgba(255, 0, 0, 0.08)", line_width=0, layer="below"),
                dict(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=-1.5, y1=0, fillcolor="rgba(0, 0, 255, 0.08)", line_width=0, layer="below"),
                dict(type="line", xref="paper", yref="y", x0=0, x1=1, y0=0, y1=0, line=dict(color="black", width=3), layer="below"),
            ],
            annotations=[
                dict(x=0.01, y=0.95, xref="paper", yref="y", text="🦅 ŞAHİN BÖLGESİ", showarrow=False, font=dict(size=14, color="darkred", weight="bold")),
                dict(x=0.01, y=-0.95, xref="paper", yref="y", text="🕊️ GÜVERCİN BÖLGESİ", showarrow=False, font=dict(size=14, color="darkblue", weight="bold"))
            ]
        )
        fig.update_yaxes(title_text="Skor", range=[-1.1, 1.1], secondary_y=False, zeroline=False)
        fig.update_yaxes(title_text="Faiz & Enflasyon (%)", secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)
        if st.button("🔄 Yenile"): st.cache_data.clear(); st.rerun()
    else: st.info("Kayıt yok.")

# ==============================================================================
# TAB 2: VERİ GİRİŞİ (KESİN KONTROLLÜ AKIŞ)
# ==============================================================================
with tab2:
    st.subheader("Veri İşlemleri")
    
    # DB Verilerini Çek (Kontrol için)
    df_all = utils.fetch_all_data()
    if not df_all.empty: 
        df_all['period_date'] = pd.to_datetime(df_all['period_date'])
        df_all['date_only'] = df_all['period_date'].dt.date
    
    # Şu anki form durumunu al
    current_id = st.session_state['form_data']['id']
    
    with st.container(border=True):
        c1, c2 = st.columns([1, 2])
        with c1:
            val_date = st.session_state['form_data']['date']
            # Tarih değişince sadece UI güncellenir, DB'ye dokunulmaz
            selected_date = st.date_input("Tarih", value=val_date)
            
            val_source = st.session_state['form_data']['source']
            source = st.text_input("Kaynak", value=val_source)
            st.caption(f"Dönem: **{selected_date.strftime('%Y-%m')}**")

        with c2:
            val_text = st.session_state['form_data']['text']
            txt = st.text_area("Metin", value=val_text, height=200, placeholder="Metni buraya yapıştırın...")
        
        # --- BUTON VE MANTIK ALANI ---
        st.markdown("---")
        
        # DURUM 1: ÇAKIŞMA TESPİT EDİLDİ, ONAY BEKLİYOR
        if st.session_state['collision_state']['active']:
            col_alert, col_act = st.columns([2, 2])
            
            with col_alert:
                t_date = st.session_state['collision_state']['target_date']
                st.error(f"⚠️ **DİKKAT:** {t_date} tarihinde veritabanında zaten kayıt var!")
                st.info("Bu işlemi onaylarsanız eski kayıt silinecek ve yerine bu yeni metin yazılacaktır.")
            
            with col_act:
                admin_pass = st.text_input("Onay için Admin Şifresi:", type="password", key="overwrite_pass")
                
                c_b1, c_b2 = st.columns(2)
                with c_b1:
                    if st.button("🚨 Onayla ve Üzerine Yaz", type="primary", use_container_width=True):
                        if admin_pass == ADMIN_PWD:
                            # Bekleyen verileri al
                            pending_txt = st.session_state['collision_state']['pending_text']
                            t_id = st.session_state['collision_state']['target_id']
                            
                            # Analizi tekrar yap (Gerekirse)
                            s_abg, h_cnt, d_cnt, hawks, doves, h_ctx, d_ctx = utils.run_full_analysis(pending_txt)
                            
                            # ÜZERİNE YAZMA (UPDATE)
                            utils.update_entry(t_id, selected_date, pending_txt, source, s_abg, s_abg)
                            
                            st.success("Kayıt başarıyla üzerine yazıldı!")
                            reset_form()
                            st.rerun()
                        else:
                            st.error("Şifre Hatalı!")
                
                with c_b2:
                    if st.button("❌ İptal Et", use_container_width=True):
                        st.warning("İşlem iptal edildi.")
                        st.session_state['collision_state']['active'] = False
                        st.rerun()

        # DURUM 2: NORMAL GÖRÜNÜM (HENÜZ BUTONA BASILMADI)
        else:
            col_b1, col_b2, col_b3 = st.columns([2, 1, 1])
            with col_b1:
                # Buton her zaman standart görünür
                btn_label = "💾 Kaydet / Analiz Et"
                if current_id: btn_label = "💾 Güncelle" # Sadece görsel olarak güncelle yazar
                
                if st.button(btn_label, type="primary"):
                    if txt:
                        # 1. BUTONA BASILDI -> ŞİMDİ KONTROL ET
                        collision_record = None
                        if not df_all.empty:
                            mask = df_all['date_only'] == selected_date
                            if mask.any(): collision_record = df_all[mask].iloc[0]
                        
                        # A) LİSTEDEN SEÇİLİ KAYITSA (KENDİ KENDİNİ GÜNCELLİYOR)
                        if current_id and (collision_record is not None) and (int(collision_record['id']) == current_id):
                            s_abg, h_cnt, d_cnt, hawks, doves, h_ctx, d_ctx = utils.run_full_analysis(txt)
                            utils.update_entry(current_id, selected_date, txt, source, s_abg, s_abg)
                            st.success("Kayıt güncellendi!")
                            reset_form()
                            st.rerun()

                        # B) ÇAKIŞMA VAR (BAŞKA BİR ID BU TARİHTE VAR)
                        elif collision_record is not None:
                            # HİÇBİR ŞEY KAYDETME. SADECE ÇAKIŞMA MODUNU AÇ.
                            st.session_state['collision_state'] = {
                                'active': True,
                                'target_id': int(collision_record['id']),
                                'target_date': selected_date,
                                'pending_text': txt # Metni hafızaya al
                            }
                            st.rerun() # Ekranı yenile, yukarıdaki 'DURUM 1' bloğu çalışsın
                        
                        # C) TERTEMİZ (KAYIT YOK)
                        else:
                            s_abg, h_cnt, d_cnt, hawks, doves, h_ctx, d_ctx = utils.run_full_analysis(txt)
                            utils.insert_entry(selected_date, txt, source, s_abg, s_abg)
                            st.success("Yeni kayıt eklendi!")
                            reset_form()
                            st.rerun()
                    else:
                        st.error("Lütfen metin giriniz.")

            with col_b2:
                if st.button("Temizle"):
                    reset_form()
                    st.rerun()

            with col_b3:
                if current_id:
                    with st.popover("🗑️ Sil"):
                        st.write("Admin şifresi girin:")
                        del_pass = st.text_input("Şifre", type="password", key="del_pass")
                        if st.button("🔥 Sil"):
                            if del_pass == ADMIN_PWD:
                                utils.delete_entry(current_id)
                                st.success("Silindi!")
                                reset_form()
                                st.rerun()
                            else:
                                st.error("Şifre Hatalı!")

        # CANLI ANALİZ GÖSTERİMİ (HER ZAMAN AKTİF)
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
                                for s in h_ctx[term]: st.caption(f"📝 ...{s}...")
                    else: st.write("- Yok")
                with k2:
                    st.markdown("**🕊️ Güvercin İfadeler**")
                    if d_list:
                        for item in d_list:
                            term = item.split(' (')[0]
                            st.write(f"🔹 **{item}**")
                            if term in d_ctx:
                                for s in d_ctx[term]: st.caption(f"📝 ...{s}...")
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
            
            # Eğer çakışma modu açıksa kapat (kullanıcı listeden başka bir şey seçti)
            if st.session_state['collision_state']['active']:
                st.session_state['collision_state']['active'] = False
            
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
