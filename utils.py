import streamlit as st
from supabase import create_client, Client
import pandas as pd
import requests
import io
import datetime
import smtplib
from email.mime.text import MIMEText

# --- 1. AYARLAR VE BAĞLANTI ---
# Bu bölüm kütüphaneler yüklendikten SONRA gelmelidir.
try:
    # 1. Supabase Bağlantı Bilgilerini Al (Hem secrets.toml hem Cloud uyumlu)
    if "supabase" in st.secrets:
        # Localde [supabase] başlığı altındaysa
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
    else:
        # Cloud'da direkt ana dizindeyse
        url = st.secrets.get("SUPABASE_URL")
        key = st.secrets.get("SUPABASE_KEY")

    # 2. Opsiyonel Anahtarlar (Hata verdirmemesi için .get kullanıyoruz)
    APP_PASSWORD = st.secrets.get("APP_PASSWORD", None)
    EVDS_API_KEY = st.secrets.get("EVDS_KEY", None)
    
    # 3. Bağlantıyı Kur
    if url and key:
        supabase: Client = create_client(url, key)
    else:
        supabase = None
        # Burada stop() demiyoruz, belki sadece veri çekmek için kullanılıyordur.
        # Ancak app.py içinde kontrol edilmeli.

except Exception as e:
    st.error(f"Ayarlar yüklenirken hata oluştu: {e}")
    st.stop()

# Tablo Adları
TABLE_TAHMIN = "beklentiler_takip"
TABLE_KATILIMCI = "katilimcilar"

# Sabitler
EVDS_BASE = "https://evds2.tcmb.gov.tr/service/evds"

# --- 2. YARDIMCI FONKSİYONLAR ---

def get_period_list():
    years = range(2024, 2033)
    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    return [f"{y}-{m}" for y in years for m in months]

def clean_and_sort_data(df):
    if df.empty: return df
    numeric_cols = [
        "tahmin_ppk_faiz", "min_ppk_faiz", "max_ppk_faiz", 
        "tahmin_yilsonu_faiz", "min_yilsonu_faiz", "max_yilsonu_faiz", 
        "tahmin_aylik_enf", "min_aylik_enf", "max_aylik_enf", 
        "tahmin_yillik_enf", "min_yillik_enf", "max_yillik_enf", 
        "tahmin_yilsonu_enf", "min_yilsonu_enf", "max_yilsonu_enf", 
        "katilimci_sayisi"
    ]
    for col in numeric_cols:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    if "tahmin_tarihi" in df.columns:
        df["tahmin_tarihi"] = pd.to_datetime(df["tahmin_tarihi"], errors='coerce')
    return df

def upsert_tahmin(user, hedef_donemi, category, forecast_date, link, data_dict):
    if not supabase: return False, "Veritabanı bağlantısı yok."
    
    if isinstance(forecast_date, str):
        date_obj = pd.to_datetime(forecast_date)
        date_str = forecast_date
    else:
        date_obj = pd.to_datetime(forecast_date)
        date_str = forecast_date.strftime("%Y-%m-%d")
    
    anket_donemi = date_obj.strftime("%Y-%m")
    new_input_data = {k: v for k, v in data_dict.items() if v is not None and v != ""}
    
    final_data = {
        "kullanici_adi": user,
        "kategori": category,
        "anket_donemi": anket_donemi, 
        "hedef_donemi": hedef_donemi, 
        "tahmin_tarihi": date_str,
    }
    if link: final_data["kaynak_link"] = link
    final_data.update(new_input_data)

    try:
        supabase.table(TABLE_TAHMIN).upsert(final_data, on_conflict="kullanici_adi, anket_donemi, hedef_donemi").execute()
        return True, "Kayıt Başarılı"
    except Exception as e:
        return False, str(e)

def sync_participants_from_forecasts():
    if not supabase: return 0, "DB yok."
    res_t = supabase.table(TABLE_TAHMIN).select("kullanici_adi, kategori").execute()
    df_t = pd.DataFrame(res_t.data)
    if df_t.empty: return 0, "Tahmin verisi yok."
    res_k = supabase.table(TABLE_KATILIMCI).select("ad_soyad").execute()
    existing_users = set([r['ad_soyad'] for r in res_k.data])
    unique_forecast_users = df_t.drop_duplicates(subset=['kullanici_adi'])
    added_count = 0
    for _, row in unique_forecast_users.iterrows():
        user = row['kullanici_adi']
        cat = row.get('kategori')
        if not cat: cat = "Bireysel"
        if user not in existing_users:
            try:
                supabase.table(TABLE_KATILIMCI).insert({"ad_soyad": user, "kategori": cat}).execute()
                added_count += 1
            except: pass
    return added_count, f"{added_count} yeni kişi eklendi."

def update_participant(old_name, new_name, new_category, row_id):
    if not supabase: return False, "DB yok."
    try:
        supabase.table(TABLE_KATILIMCI).update({"ad_soyad": new_name, "kategori": new_category}).eq("id", row_id).execute()
        if old_name != new_name:
            supabase.table(TABLE_TAHMIN).update({"kullanici_adi": new_name}).eq("kullanici_adi", old_name).execute()
        return True, "Güncellendi"
    except Exception as e:
        return False, str(e)

# --- 3. VERİ ÇEKME (CACHE) ---

@st.cache_data(ttl=600)
def get_all_forecasts():
    if not supabase: return pd.DataFrame()
    res = supabase.table(TABLE_TAHMIN).select("*").order("tahmin_tarihi", desc=True).limit(5000).execute()
    return clean_and_sort_data(pd.DataFrame(res.data))

def get_participants():
    if not supabase: return pd.DataFrame()
    res = supabase.table(TABLE_KATILIMCI).select("*").order("ad_soyad").execute()
    return pd.DataFrame(res.data)

def check_login():
    if 'giris_yapildi' not in st.session_state:
        st.session_state['giris_yapildi'] = False
    return st.session_state['giris_yapildi']

# --- 4. EVDS VE BIS FONKSİYONLARI (API ENTEGRASYONU) ---

@st.cache_data(ttl=600)
def fetch_evds_tufe_monthly_yearly(api_key, start_date, end_date):
    """
    TCMB EVDS Sisteminden Enflasyon Verilerini Çeker.
    YÖNTEM GÜNCELLEMESİ: Endeks yerine doğrudan değişim oranları çekiliyor.
    """
    if not api_key: return pd.DataFrame(), "EVDS_KEY eksik."
    try:
        # DOĞRUDAN ENFLASYON SERİLERİ
        # TP.FE.OKTG01: Yıllık TÜFE (% Değişim)
        # TP.FE.OKTG02: Aylık TÜFE (% Değişim)
        series_code = "TP.FE.OKTG01-TP.FE.OKTG02"
        
        s = start_date.strftime("%d-%m-%Y")
        e = end_date.strftime("%d-%m-%Y")
        
        # URL oluştur (Formül parametresini kaldırdık, direct veri çekiyoruz)
        url = f"{EVDS_BASE}/series={series_code}&startDate={s}&endDate={e}&type=json"
        
        headers = {"key": api_key, "User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=25)
        
        if r.status_code != 200: 
            return pd.DataFrame(), f"EVDS Hata: {r.status_code}"
            
        js = r.json()
        items = js.get("items", [])
        
        if not items: 
            return pd.DataFrame(), "Veri bulunamadı (Boş yanıt)."
            
        df = pd.DataFrame(items)
        
        # Tarih İşlemleri
        if "Tarih" not in df.columns: 
            return pd.DataFrame(), "Tarih kolonu yok."
            
        df["Tarih_dt"] = pd.to_datetime(df["Tarih"], dayfirst=True, errors="coerce")
        # Format bazen YYYY-MM gelebilir, onu da dene
        if df["Tarih_dt"].isnull().all():
            df["Tarih_dt"] = pd.to_datetime(df["Tarih"], format="%Y-%m", errors="coerce")
            
        df = df.dropna(subset=["Tarih_dt"]).sort_values("Tarih_dt")
        df["Donem"] = df["Tarih_dt"].dt.strftime("%Y-%m")
        
        # Kolon Eşleştirme (API'den gelen kodlar _ (alt çizgi) ile gelir)
        rename_map = {
            "TP_FE_OKTG01": "TUFE_Yillik",
            "TP_FE_OKTG02": "TUFE_Aylik"
        }
        
        # Sadece ilgili kolonları al
        available_cols = [c for c in rename_map.keys() if c in df.columns]
        
        if not available_cols:
            return pd.DataFrame(), "Beklenen enflasyon serileri (OKTG01/02) yanıtta yok."
            
        # Sayısal çevrim ve isim değişikliği
        final_df = df[["Tarih_dt", "Donem"] + available_cols].copy()
        final_df = final_df.rename(columns=rename_map)
        
        for col in ["TUFE_Yillik", "TUFE_Aylik"]:
            if col in final_df.columns:
                final_df[col] = pd.to_numeric(final_df[col], errors="coerce")
        
        # Tarih formatını string yap (Merge için)
        final_df["Tarih"] = final_df["Tarih_dt"].dt.strftime("%d-%m-%Y")
        
        # DataFrame ve None (Hata yok) döndür
        return final_df[["Donem", "Tarih", "TUFE_Yillik", "TUFE_Aylik"]], None

    except Exception as e: 
        return pd.DataFrame(), str(e)

@st.cache_data(ttl=600)
def fetch_bis_cbpol_tr(start_date, end_date):
    """
    BIS (Bank for International Settlements) üzerinden Türkiye Politika Faizi
    """
    try:
        s = start_date.strftime("%Y-%m-%d")
        e = end_date.strftime("%Y-%m-%d")
        url = f"https://stats.bis.org/api/v1/data/WS_CBPOL/D.TR?format=csv&startPeriod={s}&endPeriod={e}"
        r = requests.get(url, timeout=25)
        if r.status_code >= 400: return pd.DataFrame(), f"BIS HTTP {r.status_code}"
        
        df = pd.read_csv(io.StringIO(r.content.decode("utf-8", errors="ignore")))
        df.columns = [c.strip().upper() for c in df.columns]
        
        out = df[["TIME_PERIOD", "OBS_VALUE"]].copy()
        out["TIME_PERIOD"] = pd.to_datetime(out["TIME_PERIOD"], errors="coerce")
        out = out.dropna(subset=["TIME_PERIOD"])
        
        out["Donem"] = out["TIME_PERIOD"].dt.strftime("%Y-%m")
        out["REPO_RATE"] = pd.to_numeric(out["OBS_VALUE"], errors="coerce")
        
        # Ay bazında son değeri alalım (Günlük veri geliyor olabilir)
        out = out.sort_values("TIME_PERIOD").groupby("Donem").last().reset_index()
        
        return out[["Donem", "REPO_RATE"]], None
    except Exception as e: return pd.DataFrame(), str(e)

def fetch_market_data_adapter(start_date, end_date):
    """
    Dashboard tarafından çağrılan ana fonksiyon. 
    Global EVDS_API_KEY'i kullanır.
    """
    # 1. EVDS'den Enflasyon
    df_inf, err1 = fetch_evds_tufe_monthly_yearly(EVDS_API_KEY, start_date, end_date)
    
    # 2. BIS'den Faiz
    df_pol, err2 = fetch_bis_cbpol_tr(start_date, end_date)
    
    combined = pd.DataFrame()
    
    # Birleştirme Mantığı
    if not df_inf.empty and not df_pol.empty:
        combined = pd.merge(df_inf, df_pol, on="Donem", how="outer")
    elif not df_inf.empty: 
        combined = df_inf
        combined['REPO_RATE'] = None
    elif not df_pol.empty: 
        combined = df_pol
        combined['TUFE_Aylik'] = None
        combined['TUFE_Yillik'] = None
        
    if combined.empty:
        errors = f"Enflasyon Hatası: {err1} | Faiz Hatası: {err2}"
        return pd.DataFrame(), errors

    # Kolon İsimlerini Standartlaştır (Grafik İçin)
    combined = combined.rename(columns={
        'REPO_RATE': 'PPK Faizi', 
        'TUFE_Aylik': 'Aylık TÜFE', 
        'TUFE_Yillik': 'Yıllık TÜFE'
    })
    
    # Tarih kolonu yoksa Donem'den üret (Grafik X ekseni için)
    if 'Tarih' not in combined.columns and 'Donem' in combined.columns:
        combined['Tarih'] = pd.to_datetime(combined['Donem'] + "-01")
    elif 'Tarih' in combined.columns:
        combined['Tarih'] = pd.to_datetime(combined['Tarih'], dayfirst=True)
        
    return combined, None
