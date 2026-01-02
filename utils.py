import streamlit as st
from supabase import create_client, Client
import pandas as pd
import requests
import io
import datetime

# --- 1. BAĞLANTI AYARLARI ---
try:
    if "supabase" in st.secrets:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        EVDS_API_KEY = st.secrets["supabase"].get("EVDS_KEY") or st.secrets.get("EVDS_KEY")
    else:
        url = st.secrets.get("SUPABASE_URL")
        key = st.secrets.get("SUPABASE_KEY")
        EVDS_API_KEY = st.secrets.get("EVDS_KEY")

    if url and key:
        supabase: Client = create_client(url, key)
    else:
        supabase = None
except Exception as e:
    st.error(f"Ayarlar hatası: {e}")
    st.stop()

# Sabitler
EVDS_BASE = "https://evds2.tcmb.gov.tr/service/evds"
EVDS_TUFE_SERIES = "TP.FG.J0"

# --- 2. VERİ ÇEKME FONKSİYONLARI ---

@st.cache_data(ttl=600)
def fetch_market_data_adapter(start_date, end_date):
    """
    TCMB (Enflasyon) ve BIS (Faiz) verilerini çeker.
    Hepsini 'Donem' (YYYY-MM) bazında birleştirir.
    """
    if not EVDS_API_KEY:
        return pd.DataFrame(), "EVDS Anahtarı Eksik."

    # --- A. ENFLASYON (TCMB) ---
    df_inf = pd.DataFrame()
    try:
        s = start_date.strftime("%d-%m-%Y"); e = end_date.strftime("%d-%m-%Y")
        # Aylık (1) ve Yıllık (3) değişim
        for form, col in [(1, "Aylık TÜFE"), (3, "Yıllık TÜFE")]:
            url = f"{EVDS_BASE}/series={EVDS_TUFE_SERIES}&startDate={s}&endDate={e}&type=json&formulas={form}"
            r = requests.get(url, headers={"key": EVDS_API_KEY}, timeout=20)
            if r.status_code == 200 and r.json().get("items"):
                temp = pd.DataFrame(r.json()["items"])
                # Tarih İşleme
                temp["dt"] = pd.to_datetime(temp["Tarih"], dayfirst=True, errors="coerce")
                if temp["dt"].isnull().all(): temp["dt"] = pd.to_datetime(temp["Tarih"], format="%Y-%m", errors="coerce")
                temp = temp.dropna(subset=["dt"])
                
                # Dönem Sütunu (Anahtar)
                temp["Donem"] = temp["dt"].dt.strftime("%Y-%m")
                
                # Değer Sütunu
                val_c = [c for c in temp.columns if "TP" in c][0]
                temp = temp.rename(columns={val_c: col})[["Donem", col]]
                
                if df_inf.empty: df_inf = temp
                else: df_inf = pd.merge(df_inf, temp, on="Donem", how="outer")
    except Exception as e: return pd.DataFrame(), f"TÜFE Hatası: {e}"

    # --- B. FAİZ (BIS) ---
    df_pol = pd.DataFrame()
    try:
        s_bis = start_date.strftime("%Y-%m-%d"); e_bis = end_date.strftime("%Y-%m-%d")
        url_bis = f"https://stats.bis.org/api/v1/data/WS_CBPOL/D.TR?format=csv&startPeriod={s_bis}&endPeriod={e_bis}"
        r_bis = requests.get(url_bis, timeout=20)
        if r_bis.status_code == 200:
            temp_bis = pd.read_csv(io.StringIO(r_bis.content.decode("utf-8")), usecols=["TIME_PERIOD", "OBS_VALUE"])
            temp_bis["dt"] = pd.to_datetime(temp_bis["TIME_PERIOD"])
            temp_bis["Donem"] = temp_bis["dt"].dt.strftime("%Y-%m")
            temp_bis["PPK Faizi"] = pd.to_numeric(temp_bis["OBS_VALUE"], errors="coerce")
            
            # Ayın son kararını al
            df_pol = temp_bis.sort_values("dt").groupby("Donem").last().reset_index()[["Donem", "PPK Faizi"]]
    except Exception as e: return pd.DataFrame(), f"BIS Hatası: {e}"

    # --- C. BİRLEŞTİRME ---
    # Ortak anahtar: "Donem"
    master_df = pd.DataFrame()
    if not df_inf.empty and not df_pol.empty:
        master_df = pd.merge(df_inf, df_pol, on="Donem", how="outer")
    elif not df_inf.empty: master_df = df_inf
    elif not df_pol.empty: master_df = df_pol

    if master_df.empty: return pd.DataFrame(), "Veri bulunamadı."
    
    # Grafik için tarih (Ayın 1'i temsili)
    master_df["Tarih"] = pd.to_datetime(master_df["Donem"] + "-01")
    return master_df.sort_values("Donem"), None

# --- 3. VERİTABANI İŞLEMLERİ ---

def fetch_all_data():
    if not supabase: return pd.DataFrame()
    res = supabase.table("market_logs").select("*").order("period_date", desc=True).execute()
    return pd.DataFrame(res.data)

def insert_entry(date, text, source, s_dict, s_abg, s_fb, l_fb):
    if not supabase: return
    # Tarihi string olarak kaydet
    data = {
        "period_date": str(date), 
        "text_content": text, 
        "source": source,
        "score_dict": s_dict, "score_abg": s_abg, "score_finbert": s_fb, "finbert_label": l_fb
    }
    supabase.table("market_logs").insert(data).execute()

def update_entry(rid, date, text, source, s_dict, s_abg, s_fb, l_fb):
    if not supabase: return
    data = {
        "period_date": str(date), 
        "text_content": text, "source": source,
        "score_dict": s_dict, "score_abg": s_abg, "score_finbert": s_fb, "finbert_label": l_fb
    }
    supabase.table("market_logs").update(data).eq("id", rid).execute()

def delete_entry(rid):
    if supabase: supabase.table("market_logs").delete().eq("id", rid).execute()
