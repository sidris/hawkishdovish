import streamlit as st
from supabase import create_client, Client
import pandas as pd
import requests
import io
import datetime

# --- 1. GÜÇLENDİRİLMİŞ AYARLAR VE BAĞLANTI ---
# Anahtarları her ihtimale karşı farklı yerlerde arayacağız.
try:
    # 1. Supabase Ayarları
    if "supabase" in st.secrets:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
    else:
        url = st.secrets.get("SUPABASE_URL")
        key = st.secrets.get("SUPABASE_KEY")

    # 2. EVDS Anahtarı (En önemli kısım burası)
    # Önce en dışta arıyoruz, yoksa supabase başlığı altına bakıyoruz.
    EVDS_API_KEY = st.secrets.get("EVDS_KEY")
    if not EVDS_API_KEY and "supabase" in st.secrets:
         EVDS_API_KEY = st.secrets["supabase"].get("EVDS_KEY")
         
    # 3. Bağlantıyı Kur
    if url and key:
        supabase: Client = create_client(url, key)
    else:
        supabase = None

except Exception as e:
    st.error(f"Ayarlar yüklenirken hata: {e}")
    st.stop()

# Sabitler
TABLE_TAHMIN = "beklentiler_takip"
TABLE_KATILIMCI = "katilimcilar"
EVDS_BASE = "https://evds2.tcmb.gov.tr/service/evds"
EVDS_TUFE_SERIES = "TP.FG.J0" # TÜFE Endeks Kodu

# --- 2. YARDIMCI FONKSİYONLAR ---

@st.cache_data(ttl=600)
def fetch_evds_tufe_monthly_yearly(api_key, start_date, end_date):
    """
    TCMB'den TÜFE verisini çeker. 
    Formula 1 (Aylık) ve Formula 3 (Yıllık) uygular.
    """
    if not api_key: 
        return pd.DataFrame(), "EVDS_KEY bulunamadı. Secrets dosyasını kontrol et."
    
    try:
        results = []
        # 1: Aylık, 3: Yıllık
        for formula, col_name in [(1, "Aylık TÜFE"), (3, "Yıllık TÜFE")]:
            s = start_date.strftime("%d-%m-%Y")
            e = end_date.strftime("%d-%m-%Y")
            
            # Endeks verisine formül uygulayarak çekiyoruz
            url = f"{EVDS_BASE}/series={EVDS_TUFE_SERIES}&startDate={s}&endDate={e}&type=json&formulas={formula}"
            headers = {"key": api_key, "User-Agent": "Mozilla/5.0"}
            
            r = requests.get(url, headers=headers, timeout=25)
            if r.status_code == 200:
                js = r.json()
                items = js.get("items", [])
                if items:
                    df = pd.DataFrame(items)
                    # Tarih düzeltme
                    df["Tarih_dt"] = pd.to_datetime(df["Tarih"], dayfirst=True, errors="coerce")
                    if df["Tarih_dt"].isnull().all():
                        df["Tarih_dt"] = pd.to_datetime(df["Tarih"], format="%Y-%m", errors="coerce")
                    
                    df = df.dropna(subset=["Tarih_dt"])
                    df["Donem"] = df["Tarih_dt"].dt.strftime("%Y-%m")
                    
                    # Değer kolonunu bul (API ismi TP_FG_J0 gibi gelir)
                    val_col = [c for c in df.columns if "TP" in c and c not in ["Tarih", "UNIXTIME"]]
                    if val_col:
                        clean = df[["Donem", val_col[0]]].rename(columns={val_col[0]: col_name})
                        clean[col_name] = pd.to_numeric(clean[col_name], errors="coerce")
                        results.append(clean)
        
        if not results:
            return pd.DataFrame(), "TCMB verisi boş döndü."
            
        # Birleştirme
        final_df = results[0]
        if len(results) > 1:
            final_df = pd.merge(final_df, results[1], on="Donem", how="outer")
            
        # Tarih sütunu ekle (Grafik için)
        final_df["Tarih"] = pd.to_datetime(final_df["Donem"] + "-01")
        
        return final_df, None

    except Exception as e:
        return pd.DataFrame(), f"EVDS Hatası: {str(e)}"

@st.cache_data(ttl=600)
def fetch_bis_cbpol_tr(start_date, end_date):
    """
    BIS üzerinden Politika Faizi (Geri Getirildi!)
    """
    try:
        s = start_date.strftime("%Y-%m-%d")
        e = end_date.strftime("%Y-%m-%d")
        # BIS CSV formatı
        url = f"https://stats.bis.org/api/v1/data/WS_CBPOL/D.TR?format=csv&startPeriod={s}&endPeriod={e}"
        
        r = requests.get(url, timeout=25)
        if r.status_code >= 400: 
            return pd.DataFrame(), f"BIS HTTP Hatası: {r.status_code}"
        
        # CSV okuma
        df = pd.read_csv(io.StringIO(r.content.decode("utf-8", errors="ignore")))
        df.columns = [c.strip().upper() for c in df.columns]
        
        if "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
            return pd.DataFrame(), "BIS veri formatı değişmiş."

        out = df[["TIME_PERIOD", "OBS_VALUE"]].copy()
        out["TIME_PERIOD"] = pd.to_datetime(out["TIME_PERIOD"], errors="coerce")
        out = out.dropna(subset=["TIME_PERIOD"])
        
        out["Donem"] = out["TIME_PERIOD"].dt.strftime("%Y-%m")
        out["REPO_RATE"] = pd.to_numeric(out["OBS_VALUE"], errors="coerce")
        
        # Günlük veriyi aya çevir (Son günün verisi o ayın faizidir)
        out = out.sort_values("TIME_PERIOD").groupby("Donem").last().reset_index()
        
        return out[["Donem", "REPO_RATE"]], None
        
    except Exception as e: 
        return pd.DataFrame(), f"BIS Hatası: {str(e)}"

def fetch_market_data_adapter(start_date, end_date):
    # 1. Enflasyon (TCMB EVDS)
    df_inf, err1 = fetch_evds_tufe_monthly_yearly(EVDS_API_KEY, start_date, end_date)
    
    # 2. Faiz (BIS)
    df_pol, err2 = fetch_bis_cbpol_tr(start_date, end_date)
    
    # Birleştirme Mantığı
    combined = pd.DataFrame()
    
    if not df_inf.empty and not df_pol.empty:
        combined = pd.merge(df_inf, df_pol, on="Donem", how="outer")
    elif not df_inf.empty:
        combined = df_inf
        combined['REPO_RATE'] = None
    elif not df_pol.empty:
        combined = df_pol
        combined['Aylık TÜFE'] = None
        combined['Yıllık TÜFE'] = None
    
    if combined.empty:
        # Hata varsa döndür
        errors = []
        if err1: errors.append(f"TCMB: {err1}")
        if err2: errors.append(f"BIS: {err2}")
        return pd.DataFrame(), " | ".join(errors)

    # İsimlendirme
    combined = combined.rename(columns={'REPO_RATE': 'PPK Faizi'})
    
    # Tarih yoksa Donem'den üret
    if 'Tarih' not in combined.columns and 'Donem' in combined.columns:
        combined['Tarih'] = pd.to_datetime(combined['Donem'] + "-01")
        
    return combined, None

# --- DB Fonksiyonları (Aynı kaldı) ---
def fetch_all_data():
    if not supabase: return pd.DataFrame()
    res = supabase.table("market_logs").select("*").order("period_date", desc=True).execute()
    return pd.DataFrame(res.data)

def delete_entry(record_id):
    if supabase: supabase.table("market_logs").delete().eq("id", record_id).execute()

def update_entry(record_id, date, text, source, score_dict, score_abg, score_fb, fb_label):
    if supabase:
        data = {
            "period_date": str(date), "text_content": text, "source": source,
            "score_dict": score_dict, "score_abg": score_abg, 
            "score_finbert": score_fb, "finbert_label": fb_label
        }
        supabase.table("market_logs").update(data).eq("id", record_id).execute()

def insert_entry(date, text, source, score_dict, score_abg, score_fb, fb_label):
    if supabase:
        data = {
            "period_date": str(date), "text_content": text, "source": source,
            "score_dict": score_dict, "score_abg": score_abg, 
            "score_finbert": score_fb, "finbert_label": fb_label
        }
        supabase.table("market_logs").insert(data).execute()
