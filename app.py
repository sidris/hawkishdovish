import streamlit as st
import pandas as pd
from supabase import create_client, Client
from transformers import pipeline
from collections import Counter
import re
import plotly.express as px
import datetime

# -----------------------------------------------------------------------------
# 1. AYARLAR VE BAĞLANTILAR
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Piyasa Analiz Sistemi", layout="wide")

# Supabase Bağlantısı
@st.cache_resource
def init_supabase():
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        return create_client(url, key)
    except KeyError:
        st.error("Supabase sırları bulunamadı. Lütfen Streamlit Secrets ayarlarını kontrol edin.")
        return None

supabase = init_supabase()

# FinBERT Modelini Yükle
@st.cache_resource
def load_finbert():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

try:
    classifier = load_finbert()
except Exception as e:
    st.warning("Model yükleniyor... (İlk açılış yavaş olabilir)")
    classifier = None

# -----------------------------------------------------------------------------
# 2. ALGORİTMALAR
# -----------------------------------------------------------------------------

def analyze_with_dictionary(text):
    text = text.lower()
    tokens = re.findall(r"[a-z']+", text)
    hawkish = ["high", "rising", "elevated", "strong", "tightening", "inflation", "risk", "hike", "upside"]
    dovish = ["low", "falling", "weak", "slow", "easing", "cut", "stimulus", "decline", "downside"]
    
    c = Counter(tokens)
    h_score = sum(c[t] for t in hawkish)
    d_score = sum(c[t] for t in dovish)
    total = h_score + d_score
    
    if total == 0: return 0
    return (h_score - d_score) / total

def analyze_with_finbert(text):
    if not classifier: return 0, "neutral"
    res = classifier(text[:512])[0]
    score = res['score']
    label = res['label']
    final_score = score if label == "positive" else -score if label == "negative" else 0
    return final_score, label

# -----------------------------------------------------------------------------
# 3. VERİTABANI İŞLEMLERİ (YENİ EKLENEN FONKSİYONLAR)
# -----------------------------------------------------------------------------

def fetch_all_data():
    """Veritabanındaki tüm kayıtları çeker"""
    response = supabase.table("market_logs").select("*").order("period_date", desc=True).execute()
    return pd.DataFrame(response.data)

def delete_entry(record_id):
    """Kaydı siler"""
    supabase.table("market_logs").delete().eq("id", record_id).execute()

def update_entry(record_id, date, text, source):
    """Kaydı günceller ve analizi tekrar yapar"""
    # 1. Yeni metni tekrar analiz et
    dict_score = analyze_with_dictionary(text)
    fb_score, fb_label = analyze_with_finbert(text)
    
    # 2. Güncelleme verisi
    update_data = {
        "period_date": str(date),
        "text_content": text,
        "source": source,
        "score_dict": dict_score,
        "score_finbert": fb_score,
        "finbert_label": fb_label
    }
    
    # 3. Supabase Update
    supabase.table("market_logs").update(update_data).eq("id", record_id).execute()

# -----------------------------------------------------------------------------
# 4. ARAYÜZ
# -----------------------------------------------------------------------------

st.title("🦅 Şahin/Güvercin Analiz Paneli")

# Sekmeleri 3'e çıkardık
tab1, tab2, tab3 = st.tabs(["📝 Yeni Veri Girişi", "✏️ Kayıtları Düzenle/Sil", "📈 Dashboard"])

# --- TAB 1: YENİ VERİ GİRİŞİ ---
with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Yeni Dönem")
        year = st.selectbox("Yıl", range(2020, 2030), index=5)
        month = st.selectbox("Ay", range(1, 13))
        source = st.text_input("Kaynak", "TCMB")
    with col2:
        text_input = st.text_area("Metin", height=200, placeholder="Yeni metni buraya girin...")
        if st.button("Kaydet ve Analiz Et", type="primary"):
            if text_input:
                with st.spinner("Analiz ediliyor..."):
                    d_score = analyze_with_dictionary(text_input)
                    fb_score, fb_label = analyze_with_finbert(text_input)
                    period_date = f"{year}-{month:02d}-01"
                    
                    data = {
                        "period_date": period_date, "text_content": text_input, "source": source,
                        "score_dict": d_score, "score_finbert": fb_score, "finbert_label": fb_label
                    }
                    supabase.table("market_logs").insert(data).execute()
                    st.success("✅ Kaydedildi!")
            else:
                st.error("Metin boş olamaz.")

# --- TAB 2: DÜZENLEME VE SİLME (YENİ BÖLÜM) ---
with tab2:
    st.header("Kayıt Yönetimi")
    
    # Tüm verileri çek
    df = fetch_all_data()
    
    if not df.empty:
        # Seçim Kutusu Oluştur (Kullanıcının hangisini düzenleyeceğini seçmesi için)
        # Görünen isim formatı: "ID: 5 | 2025-01-01 | TCMB"
        record_options = df.apply(lambda x: f"ID: {x['id']} | {x['period_date']} | {x['source']}", axis=1)
        selected_option = st.selectbox("Düzenlenecek Kaydı Seçin:", record_options)
        
        # Seçilen ID'yi bul
        selected_id = int(selected_option.split("|")[0].replace("ID:", "").strip())
        
        # Seçilen satırın verilerini al
        selected_row = df[df['id'] == selected_id].iloc[0]
        
        st.markdown("---")
        
        with st.form("edit_form"):
            col_edit1, col_edit2 = st.columns(2)
            
            with col_edit1:
                # Tarih objesine çeviriyoruz ki date_input kabul etsin
                current_date = pd.to_datetime(selected_row['period_date']).date()
                new_date = st.date_input("Dönem", value=current_date)
                new_source = st.text_input("Kaynak", value=selected_row['source'])
                
            with col_edit2:
                # Mevcut metni getir
                new_text = st.text_area("Metin İçeriği", value=selected_row['text_content'], height=200)
            
            # Butonlar
            c1, c2 = st.columns([1,4])
            with c1:
                update_btn = st.form_submit_button("💾 Değişiklikleri Kaydet")
            with c2:
                # Silme işlemi form içinde riskli olabilir, form dışında checkbox ile onaylatacağız
                pass

        if update_btn:
            with st.spinner("Güncelleniyor ve Tekrar Analiz Ediliyor..."):
                update_entry(selected_id, new_date, new_text, new_source)
                st.success("✅ Kayıt başarıyla güncellendi!")
                st.rerun() # Sayfayı yenile ki liste güncellensin

        # Silme Bölümü (Form Dışında Güvenlik İçin)
        with st.expander("🗑️ Bu Kaydı Sil"):
            st.warning("Bu işlem geri alınamaz.")
            if st.button("Evet, Sil"):
                delete_entry(selected_id)
                st.success("Kayıt silindi.")
                st.rerun()

    else:
        st.info("Düzenlenecek kayıt bulunamadı.")

# --- TAB 3: DASHBOARD ---
with tab3:
    st.header("Analiz Grafikleri")
    if st.button("Grafikleri Yenile"):
        df = fetch_all_data()
        if not df.empty:
            df['period_date'] = pd.to_datetime(df['period_date'])
            chart_df = df.melt(id_vars=['period_date', 'source'], value_vars=['score_dict', 'score_finbert'], var_name='Algoritma', value_name='Skor')
            
            fig = px.line(chart_df, x='period_date', y='Skor', color='Algoritma', markers=True, title="Şahin/Güvercin Trendi")
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df)
        else:
            st.warning("Veri yok.")
