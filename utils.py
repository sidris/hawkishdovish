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
        EVDS_API_KEY = st.secrets["supabase"].get("EVDS_KEY") or st.secrets.get("EVDS_KEY")
    else:
        url = st.secrets.get("SUPABASE_URL")
        key = st.secrets.get("SUPABASE_KEY")
        EVDS_API_KEY = st.secrets.get("EVDS_KEY")

    APP_PASSWORD = st.secrets.get("APP_PASSWORD", None)
    
    if url and key:
        supabase: Client = create_client(url, key)
    else:
        supabase = None

except Exception as e:
    st.error(f"Secrets/Ayarlar yüklenirken hata: {e}")
    st.stop()

# Sabitler
TABLE_TAHMIN = "beklentiler_takip"
TABLE_KATILIMCI = "katilimcilar"
EVDS_BASE = "https://evds2.tcmb.gov.tr/service/evds"
EVDS_TUFE_SERIES = "TP.FG.J0"

# --- 2. YARDIMCI FONKSİYONLAR (DB) ---
# (Bu kısımlar değişmedi, app.py uyumu için tutuyoruz)

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
    if not supabase: return False, "DB Bağlantısı Yok"
    if isinstance(forecast_date, str):
        date_str = forecast_date
        date_obj = pd.to_datetime(forecast_date)
    else:
        date_str = forecast_date.strftime("%Y-%m-%d")
        date_obj = pd.to_datetime(forecast_date)
    
    anket_donemi = date_obj.strftime("%Y-%m")
    new_input_data = {k: v for k, v in data_dict.items() if v is not None and v != ""}
    final_data = {
        "kullanici_adi": user, "kategori": category, "anket_donemi": anket_donemi, 
        "hedef_donemi": hedef_donemi, "tahmin_tarihi": date_str,
    }
    if link: final_data["kaynak_link"] = link
    final_data.update(new_input_data)
    try:
        supabase.table(TABLE_TAHMIN).upsert(final_data, on_conflict="kullanici_adi, anket_donemi, hedef_donemi").execute()
        return True, "Kayıt Başarılı"
    except Exception as e: return False, str(e)

def sync_participants_from_forecasts():
    if not supabase: return 0, "DB Yok"
    res_t = supabase.table(TABLE_TAHMIN).select("kullanici_adi, kategori").execute()
    df_t = pd.DataFrame(res_t.data)
    if df_t.empty: return 0, "Tahmin verisi yok."
    res_k = supabase.table(TABLE_KATILIMCI).select("ad_soyad").execute()
    existing_users = set([r['ad_soyad'] for r in res_k.data])
    unique_forecast_users = df_t.drop_duplicates(subset=['kullanici_adi'])
    added_count = 0
    for _, row in unique_forecast_users.iterrows():
        user = row['kullanici_adi']; cat = row.get('kategori')
        if not cat: cat = "Bireysel"
        if user not in existing_users:
            try:
                supabase.table(TABLE_KATILIMCI).insert({"ad_soyad": user, "kategori": cat}).execute()
                added_count += 1
            except: pass
    return added_count, f"{added_count} yeni kişi eklendi."

def update_participant(old_name, new_name, new_category, row_id):
    if not supabase: return False, "DB Yok"
    try:
        supabase.table(TABLE_KATILIMCI).update({"ad_soyad": new_name, "kategori": new_category}).eq("id", row_id).execute()
        if old_name != new_name:
            supabase.table(TABLE_TAHMIN).update({"kullanici_adi": new_name}).eq("kullanici_adi", old_name).execute()
        return True, "Güncellendi"
    except Exception as e: return False, str(e)

# --- 3. DB OKUMA (CACHE) ---
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
    if 'giris_yapildi' not in st.session_state: st.session_state['giris_yapildi'] = False
    return st.session_state['giris_yapildi']

# --- APP.PY İÇİN DB FONKSİYONLARI ---
def fetch_all_data():
    if not supabase: return pd.DataFrame()
    res = supabase.table("market_logs").select("*").order("period_date", desc=True).execute()
    return pd.DataFrame(res.data)

def delete_entry(record_id):
    if supabase: supabase.table("market_logs").delete().eq("id", record_id).execute()

def update_entry(record_id, date, text, source, score_dict, score_abg, score_fb, fb_label):
    if supabase:
        data = {"period_date": str(date), "text_content": text, "source": source,
            "score_dict": score_dict, "score_abg": score_abg, "score_finbert": score_fb, "finbert_label": fb_label}
        supabase.table("market_logs").update(data).eq("id", record_id).execute()

def insert_entry(date, text, source, score_dict, score_abg, score_fb, fb_label):
    if supabase:
        data = {"period_date": str(date), "text_content": text, "source": source,
            "score_dict": score_dict, "score_abg": score_abg, "score_finbert": score_fb, "finbert_label": fb_label}
        supabase.table("market_logs").insert(data).execute()

# --- 4. EVDS VE BIS (GÜNCELLENMİŞ TARİH MANTIĞI) ---

def _evds_headers(api_key: str) -> dict: return {"key": api_key, "User-Agent": "Mozilla/5.0"}

def _evds_url_single(series_code, start_date, end_date, formulas):
    s = start_date.strftime("%d-%m-%Y"); e = end_date.strftime("%d-%m-%Y")
    url = f"{EVDS_BASE}/series={series_code}&startDate={s}&endDate={e}&type=json"
    if formulas is not None: url += f"&formulas={int(formulas)}"
    return url

@st.cache_data(ttl=600)
def fetch_evds_tufe_monthly_yearly(api_key, start_date, end_date):
    if not api_key: return pd.DataFrame(), "EVDS_KEY eksik."
    try:
        results = {}
        for formulas, out_col in [(1, "TUFE_Aylik"), (3, "TUFE_Yillik")]:
            url = _evds_url_single(EVDS_TUFE_SERIES, start_date, end_date, formulas=formulas)
            r = requests.get(url, headers=_evds_headers(api_key), timeout=25)
            if r.status_code != 200: continue
            js = r.json(); items = js.get("items", [])
            if not items: continue
            df = pd.DataFrame(items)
            if "Tarih" not in df.columns: continue
            
            df["Tarih_dt"] = pd.to_datetime(df["Tarih"], dayfirst=True, errors="coerce")
            if df["Tarih_dt"].isnull().all(): df["Tarih_dt"] = pd.to_datetime(df["Tarih"], format="%Y-%m", errors="coerce")
            
            df = df.dropna(subset=["Tarih_dt"]).sort_values("Tarih_dt")
            df["Donem"] = df["Tarih_dt"].dt.strftime("%Y-%m")
            
            val_cols = [c for c in df.columns if c not in ["Tarih", "UNIXTIME", "Tarih_dt", "Donem"]]
            if not val_cols: continue
            
            # Enflasyon genellikle ayın başı veya sonu olarak temsil edilir. 
            # EVDS'den gelen tarihi koruyoruz (genelde ayın 1'i)
            results[out_col] = pd.DataFrame({
                "Tarih_Enflasyon": df["Tarih_dt"], # Özel tarih kolonu
                "Donem": df["Donem"], 
                out_col: pd.to_numeric(df[val_cols[0]], errors="coerce")
            })
            
        df_m = results.get("TUFE_Aylik", pd.DataFrame())
        df_y = results.get("TUFE_Yillik", pd.DataFrame())
        
        if df_m.empty and df_y.empty: return pd.DataFrame(), "Veri bulunamadı."
        if df_m.empty: out = df_y
        elif df_y.empty: out = df_m
        else: out = pd.merge(df_m, df_y, on=["Tarih_Enflasyon", "Donem"], how="outer")
        
        return out.sort_values(["Donem"]), None
    except Exception as e: return pd.DataFrame(), str(e)

@st.cache_data(ttl=600)
def fetch_bis_cbpol_tr(start_date, end_date):
    """
    BIS Faiz Verisi. 
    YENİLİK: Tarih bilgisini kaybetmiyoruz (groupby last yerine tail kullanıyoruz)
    Böylece 11 Aralık'taki veri 11 Aralık olarak kalıyor.
    """
    try:
        s = start_date.strftime("%Y-%m-%d"); e = end_date.strftime("%Y-%m-%d")
        url = f"https://stats.bis.org/api/v1/data/WS_CBPOL/D.TR?format=csv&startPeriod={s}&endPeriod={e}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=25)
        
        if r.status_code >= 400: return pd.DataFrame(), f"BIS HTTP {r.status_code}"
        
        df = pd.read_csv(io.StringIO(r.content.decode("utf-8", errors="ignore")))
        df.columns = [c.strip().upper() for c in df.columns]
        
        if "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
             return pd.DataFrame(), "BIS veri formatı hatası."

        out = df[["TIME_PERIOD", "OBS_VALUE"]].copy()
        out["TIME_PERIOD"] = pd.to_datetime(out["TIME_PERIOD"], errors="coerce")
        out = out.dropna(subset=["TIME_PERIOD"])
        
        out["Donem"] = out["TIME_PERIOD"].dt.strftime("%Y-%m")
        out["REPO_RATE"] = pd.to_numeric(out["OBS_VALUE"], errors="coerce")
        
        # YENİ MANTIK: Her ayın son gözlemini al ama TARİHİ KORU.
        # Böylece 11.12.2025 verisi, 01.12.2025'e dönüşmez.
        out = out.sort_values("TIME_PERIOD").groupby("Donem").tail(1).reset_index(drop=True)
        
        return out[["Donem", "TIME_PERIOD", "REPO_RATE"]], None # TIME_PERIOD'u döndürüyoruz
    except Exception as e: return pd.DataFrame(), str(e)

def fetch_market_data_adapter(start_date, end_date):
    """
    Dashboard tarafından çağrılan ana fonksiyon. 
    """
    # 1. Enflasyon (Tarih_Enflasyon içeriyor)
    df_inf, err1 = fetch_evds_tufe_monthly_yearly(EVDS_API_KEY, start_date, end_date)
    # 2. Faiz (TIME_PERIOD içeriyor)
    df_pol, err2 = fetch_bis_cbpol_tr(start_date, end_date)
    
    combined = pd.DataFrame()
    
    if not df_inf.empty and not df_pol.empty:
        # Donem üzerinden birleştir
        combined = pd.merge(df_inf, df_pol, on="Donem", how="outer")
    elif not df_inf.empty: 
        combined = df_inf
        combined['REPO_RATE'] = None; combined['TIME_PERIOD'] = None
    elif not df_pol.empty: 
        combined = df_pol.rename(columns={'REPO_RATE': 'REPO_RATE'})
        combined['TUFE_Aylik'] = None; combined['TUFE_Yillik'] = None; combined['Tarih_Enflasyon'] = None
        
    if combined.empty:
         return pd.DataFrame(), f"Hata: {err1} | {err2}"

    # Kolon İsimlerini Standartlaştır
    combined = combined.rename(columns={'REPO_RATE': 'PPK Faizi', 'TUFE_Aylik': 'Aylık TÜFE', 'TUFE_Yillik': 'Yıllık TÜFE'})
    
    # TARİH MANTIĞI:
    # Grafikte X ekseni için tek bir 'Tarih' kolonu lazım.
    # Eğer o ay Faiz kararı varsa (örn: 11 Aralık), o tarihi kullan.
    # Yoksa Enflasyon tarihini (Örn: 1 Aralık) kullan.
    
    # 1. TIME_PERIOD (Faiz Tarihi) varsa onu al
    # 2. Yoksa Tarih_Enflasyon (Enflasyon Tarihi) al
    # 3. O da yoksa Dönem başını (01) oluştur
    
    if 'TIME_PERIOD' in combined.columns:
        combined['Tarih'] = combined['TIME_PERIOD']
    else:
        combined['Tarih'] = pd.NaT

    if 'Tarih_Enflasyon' in combined.columns:
        combined['Tarih'] = combined['Tarih'].fillna(combined['Tarih_Enflasyon'])
        
    # Hala boş olanlar varsa Donem'den üret
    if 'Donem' in combined.columns:
        # Donem YYYY-MM formatında, sonuna -01 ekle
        donem_dates = pd.to_datetime(combined['Donem'] + "-01")
        combined['Tarih'] = combined['Tarih'].fillna(donem_dates)

    # Hala boş olan varsa düşür veya bugün yap (Nadir)
    combined = combined.dropna(subset=['Tarih']).sort_values('Tarih')
        
    return combined, None
