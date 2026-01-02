# --- 1. AYARLAR VE BAĞLANTI ---
try:
    # Zorunlu alanlar (Bunlar yoksa uygulama çalışmaz)
    # Hem st.secrets["ANAHTAR"] hem de st.secrets["bolum"]["ANAHTAR"] formatını dener
    if "supabase" in st.secrets:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
    else:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]

    # Opsiyonel alanlar (Bunlar yoksa hata vermez, None döner)
    APP_PASSWORD = st.secrets.get("APP_PASSWORD", None)
    EVDS_API_KEY = st.secrets.get("EVDS_KEY", None)
    
    supabase: Client = create_client(url, key)

except KeyError as e:
    st.error(f"Kritik Hata: Supabase bağlantı bilgileri (URL veya Key) eksik! Detay: {e}")
    st.stop()
except Exception as e:
    st.error(f"Beklenmeyen bir hata oluştu: {e}")
    st.stop()
