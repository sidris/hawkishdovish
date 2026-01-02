import streamlit as st
from supabase import create_client, Client
import pandas as pd
import requests
import io
import datetime

# --- 1. AYARLAR VE BAĞLANTI ---
try:
    if "supabase" in st.secrets:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
    else:
        url = st.secrets.get("SUPABASE_URL")
        key = st.secrets.get("SUPABASE_KEY")

    # Opsiyonel Anahtarlar
    APP_PASSWORD = st.secrets.get("APP_PASSWORD", None)
    EVDS_API_KEY = st.secrets.get("EVDS_KEY", None)
    
    if url and key:
        supabase: Client = create_client(url, key)
    else:
        supabase = None

except Exception as e:
    st.error(f"Ayarlar yüklenirken hata oluştu: {e}")
    st.stop()

# Sabitler
TABLE_TAHMIN = "beklentiler_takip"
TABLE_KATILIMCI = "katilimcilar"
EVDS_BASE = "https://evds2.tcmb.gov.tr/service/evds"
EVDS_TUFE_SERIES = "TP.FG.J0"  # TÜFE Genel Endeksi

# --- 2. YARDIMCI FONKSİYONLAR ---

def fetch_evds_tufe_monthly_yearly(api_key, start_date, end_date):
    """
    TP.FG.J0 (Endeks) serisini çeker.
    Formula 1 = Aylık Değişim
    Formula 3 = Yıllık Değişim
    """
    if not api_key: return pd.DataFrame(), "EVDS_KEY eksik."
    
    try:
        results = []
        # Hem Aylık (1) hem Yıllık (3) veriyi ayrı ayrı çekip birleştireceğiz
        # 1: Aylık Yüzde Değişim, 3: Yıllık Yüzde Değişim
        for formula, col_name in [(1, "Aylık TÜFE"), (3, "Yıllık TÜFE")]:
            s = start_date.strftime("%d-%m-%Y")
            e = end_date.strftime("%d-%m-%Y")
            
            # API URL: series=TP.FG.J0&formulas=1 veya 3
            url = f"{EVDS_BASE}/series={EVDS_TUFE_SERIES}&startDate={s}&endDate={e}&type=json&formulas={formula}"
            
            headers = {"key": api_key, "User-Agent": "Mozilla/5.0"}
            r = requests.get(url, headers=headers, timeout=25)
            
            if r.status_code == 200:
                js = r.json()
                items = js.get("items", [])
                if items:
                    temp_df = pd.DataFrame(items)
                    # Tarih düzeltme
                    temp_df["Tarih_dt"] = pd.to_datetime(temp_df["Tarih"], dayfirst=True, errors="coerce")
                    # Format YYYY-MM ise
                    if temp_df["Tarih_dt"].isnull().all():
                        temp_df["Tarih_dt"] = pd.to_datetime(temp_df["Tarih"], format="%Y-%m", errors="coerce")
                    
                    temp_df = temp_df.dropna(subset=["Tarih_dt"])
                    temp_df["Donem"] = temp_df["Tarih_dt"].dt.strftime("%Y-%m")
                    
                    # Değer kolonu (Genelde TP_FG_J0 ismiyle gelir)
                    val_col = [c for c in temp_df.columns if "TP_FG_J0" in c or "TP.FG.J0" in c]
                    if val_col:
                        # Sadece Donem ve Değeri al
                        clean_df = temp_df[["Donem", val_col[0]]].rename(columns={val_col[0]: col_name})
                        clean_df[col_name] = pd.to_numeric(clean_df[col_name], errors="coerce")
                        results.append(clean_df)

        if not results:
            return pd.DataFrame(), "Veri bulunamadı."
        
        # İki tabloyu (Aylık ve Yıllık) Donem üzerinden birleştir
        final_df = results[0]
        if len(results) > 1:
            final_df = pd.merge(final_df, results[1], on="Donem", how="outer")
            
        # Tarih kolonu ekle (Ayın 1'i olarak)
        final_df["Tarih"] = pd.to_datetime(final_df["Donem"] + "-01")
        
        return final_df, None

    except Exception as e:
        return pd.DataFrame(), str(e)

@st.cache_data(ttl=600)
def fetch_bis_cbpol_tr(start_date, end_date):
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
        out = out.sort_values("TIME_PERIOD").groupby("Donem").last().reset_index()
        
        return out[["Donem", "REPO_RATE"]], None
    except Exception as e: return pd.DataFrame(), str(e)

def fetch_market_data_adapter(start_date, end_date):
    # 1. EVDS (TÜFE)
    df_inf, err1 = fetch_evds_tufe_monthly_yearly(EVDS_API_KEY, start_date, end_date)
    # 2. BIS (Faiz)
    df_pol, err2 = fetch_bis_cbpol_tr(start_date, end_date)
    
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
        return pd.DataFrame(), f"Hata: {err1} {err2}"

    combined = combined.rename(columns={'REPO_RATE': 'PPK Faizi'})
    
    # Tarih yoksa oluştur
    if 'Tarih' not in combined.columns and 'Donem' in combined.columns:
        combined['Tarih'] = pd.to_datetime(combined['Donem'] + "-01")
        
    return combined, None

# Diğer gerekli fonksiyonlar (DB işlemleri vb.)
# (Burada fetch_all_data, delete_entry vb. önceki kodlarınızla aynı kalabilir)
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
