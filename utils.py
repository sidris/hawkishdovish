import streamlit as st
from supabase import create_client, Client
import pandas as pd
import requests
import io
import datetime
import re
import difflib
import gc
import hashlib
import html as _htmllib
from collections import Counter
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional, Callable, Iterable

# --- 1. KÜTÜPHANE KONTROLLERİ VE GLOBAL FLAGLER ---
HAS_ML_DEPS = False
HAS_TRANSFORMERS = False

# ML Kütüphaneleri
try:
    import sklearn
    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, FunctionTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report, confusion_matrix
    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.pyplot as plt
    HAS_ML_DEPS = True
except ImportError:
    HAS_ML_DEPS = False

# RoBERTa / Transformers Kontrolü
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# --- 2. AYARLAR VE BAĞLANTI ---
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
except Exception:
    supabase = None

# TÜFE serileri: eski (2003=100) + yeni (2025=100)
EVDS_TUFE_OLD = "TP.FE.OKTG01"        # 2003=100, geçmiş veriler
EVDS_TUFE_NEW = "TP.TUKFIY2025.GENEL" # 2025=100, 2026+ veriler

# Enflasyon beklentisi serileri
# Not: EVDS seri kodları resmi ekranda noktalı biçimde geçiyor.
# API dönüşlerinde kolon adları alt çizgili gelebileceği için okuma fonksiyonları iki biçimi de destekler.
EVDS_EXPECTATION_SERIES = {
    "PKA 12 Ay Enflasyon Beklentisi": "TP.ENFBEK.PKA12ENF",
    "İYA 12 Ay Enflasyon Beklentisi": "TP.ENFBEK.IYA12ENF",
    "HBA 12 Ay Enflasyon Beklentisi": "TP.ENFBEK.HBA12ENF",
}

# TCMB Ağırlıklı Ortalama Fonlama Maliyeti (AOFM)
# DİKKAT: Bu politika faizi DEĞİLDİR; fiilen gerçekleşen fonlama maliyetidir.
# 2018-2021 arasında ilan edilen faizden ciddi biçimde ayrışmıştı, 2023 sonrasında
# üst üste bindi. Paneldeki değeri de buradan gelir: "ilan edilen faiz" ile
# "fiilen uygulanan duruş" arasındaki farkı görünür kılar.
EVDS_AOFM = "TP.APIFON4"

# =============================================================================
# 3. VERİTABANI İŞLEMLERİ
# =============================================================================

def fetch_all_data():
    if not supabase: return pd.DataFrame()
    try:
        res = supabase.table("market_logs").select("*").order("period_date", desc=True).execute()
        data = getattr(res, 'data', []) if res else []
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()

def insert_entry(date, text, source, s_dict, s_abg):
    if not supabase: return
    try:
        data = {"period_date": str(date), "text_content": text, "source": source,
            "score_dict": s_dict, "score_abg": s_abg}
        supabase.table("market_logs").insert(data).execute()
    except Exception: pass

def update_entry(rid, date, text, source, s_dict, s_abg):
    if not supabase: return
    try:
        data = {"period_date": str(date), "text_content": text, "source": source,
            "score_dict": s_dict, "score_abg": s_abg}
        supabase.table("market_logs").update(data).eq("id", rid).execute()
    except Exception: pass

def delete_entry(rid):
    if supabase: 
        try:
            supabase.table("market_logs").delete().eq("id", rid).execute()
        except Exception: pass

def fetch_events():
    if not supabase: return pd.DataFrame()
    try:
        res = supabase.table("event_logs").select("*").order("event_date", desc=True).execute()
        data = getattr(res, 'data', []) if res else []
        return pd.DataFrame(data)
    except Exception: return pd.DataFrame()

def add_event(date, links):
    if not supabase: return
    try:
        data = {"event_date": str(date), "links": links}
        supabase.table("event_logs").insert(data).execute()
    except Exception: pass

def delete_event(rid):
    if supabase:
        try: supabase.table("event_logs").delete().eq("id", rid).execute()
        except Exception: pass

def fetch_ppk_text_rate_data(source_filter: str = "TCMB PPK Kararı") -> pd.DataFrame:
    """
    Text-as-Data için: text_content + policy_rate + delta_bp çek.
    """
    if not supabase:
        return pd.DataFrame()

    try:
        res = (
            supabase.table("market_logs")
            .select("id, period_date, source, text_content, policy_rate, delta_bp")
            .eq("source", source_filter)
            .order("period_date", desc=False)
            .execute()
        )
        data = getattr(res, "data", []) if res else []
        df = pd.DataFrame(data)
        if df.empty:
            return df

        df["period_date"] = pd.to_datetime(df["period_date"], errors="coerce")
        df = df.dropna(subset=["period_date"])

        df["policy_rate"] = pd.to_numeric(df.get("policy_rate"), errors="coerce")
        df["delta_bp"] = pd.to_numeric(df.get("delta_bp"), errors="coerce")

        # text boş olanları at
        df["text_content"] = df["text_content"].fillna("").astype(str)
        df = df[df["text_content"].str.len() >= 20]

        return df.sort_values("period_date").reset_index(drop=True)

    except Exception:
        return pd.DataFrame()


# --- MARKET DATA ---
def _evds_to_pct(evds_client, series_code, fetch_start, fetch_end):
    """Verilen seri kodunu çekip aylık/yıllık pct_change hesaplar, Donem kolonlu df döner."""
    try:
        raw = evds_client.get_data(
            [series_code],
            startdate=fetch_start,
            enddate=fetch_end,
            frequency=5
        )
        if raw is None or raw.empty:
            return pd.DataFrame()
        raw["dt"] = pd.to_datetime(raw["Tarih"], format="%Y-%m", errors="coerce")
        raw = raw.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
        val_col = [c for c in raw.columns if c not in ("Tarih", "dt")][0]
        raw[val_col] = pd.to_numeric(raw[val_col], errors="coerce")
        raw = raw.dropna(subset=[val_col])
        raw["Aylık TÜFE"]  = raw[val_col].pct_change(1)  * 100
        raw["Yıllık TÜFE"] = raw[val_col].pct_change(12) * 100
        raw["Donem"] = raw["dt"].dt.strftime("%Y-%m")
        return raw[["Donem", "Aylık TÜFE", "Yıllık TÜFE"]].copy()
    except Exception:
        return pd.DataFrame()


def _clean_evds_numeric(series):
    """EVDS değerlerini güvenli biçimde sayıya çevirir; hem nokta hem virgül ondalığı destekler."""
    s = series.astype(str).str.strip()

    # Virgül varsa TR biçimi kabul et: 22,17 veya 1.234,56
    has_comma = s.str.contains(",", regex=False)
    out = pd.Series(np.nan, index=s.index, dtype="float64")

    if has_comma.any():
        out.loc[has_comma] = pd.to_numeric(
            s.loc[has_comma]
             .str.replace(".", "", regex=False)
             .str.replace(",", ".", regex=False),
            errors="coerce"
        )

    if (~has_comma).any():
        out.loc[~has_comma] = pd.to_numeric(s.loc[~has_comma], errors="coerce")

    return out


def _parse_evds_months(date_series):
    """EVDS Tarih kolonunu aylık dönemlere çevirir."""
    dt = pd.to_datetime(date_series, format="%Y-%m", errors="coerce")
    if dt.isna().all():
        dt = pd.to_datetime(date_series, dayfirst=True, errors="coerce")
    return dt


def _pick_evds_value_col(raw: pd.DataFrame, series_code: str):
    """EVDS dönüşünde gerçek seri kolonunu bulur; Tarih/UNIXTIME gibi yardımcı kolonları dışarıda bırakır."""
    if raw is None or raw.empty:
        return None

    ignored = {"Tarih", "TARIH", "DATE", "date", "dt", "UNIXTIME", "unixTime", "UNIX_TIME"}
    candidates = [c for c in raw.columns if c not in ignored]
    if not candidates:
        return None

    normalized_target = series_code.replace(".", "_").replace("-", "_").upper()
    for c in candidates:
        normalized_col = str(c).replace(".", "_").replace("-", "_").upper()
        if normalized_col == normalized_target:
            return c

    # İlk sayısallaştırılabilen kolonu seç. Böylece kolon adı beklenenden farklı gelse de veri kaybolmaz.
    for c in candidates:
        vals = _clean_evds_numeric(raw[c])
        if vals.notna().any():
            return c

    return candidates[0]


def _evds_direct_request(series_code, fetch_start, fetch_end, aggregation="avg"):
    """evds paketinden sonuç gelmezse resmi web servise doğrudan istek atar."""
    if not EVDS_API_KEY:
        return pd.DataFrame()

    params = {
        "series": series_code,
        "startDate": fetch_start,
        "endDate": fetch_end,
        "type": "json",
        "frequency": "5",
        "formulas": "0",
        "aggregationTypes": aggregation,
    }

    urls = [
        "https://evds2.tcmb.gov.tr/service/evds",
        "https://evds2.tcmb.gov.tr/service/evds/",
    ]

    for url in urls:
        try:
            r = requests.get(url, params=params, headers={"key": EVDS_API_KEY}, timeout=20)
            if r.status_code != 200:
                continue
            payload = r.json()
            items = payload.get("items", []) if isinstance(payload, dict) else []
            if items:
                return pd.DataFrame(items)
        except Exception:
            continue

    return pd.DataFrame()


def _evds_to_monthly_level(evds_client, series_code, column_name, fetch_start, fetch_end,
                           aggregation="avg"):
    """Verilen EVDS serisini aylık frekansta çekip Donem + değer kolonu olarak döner.

    aggregation: günlük serilerde ay içi toplulaştırma tipi.
      "avg"  -> ay ortalaması (endeks/beklenti serileri için uygun)
      "last" -> ay sonu değeri (faiz/oran serileri için uygun; ay içi faiz
                değişimini bulanıklaştırmaz ve PPK karar tarihiyle hizalanır)
    """
    raw = pd.DataFrame()

    # 1) Önce mevcut evds paketiyle dene.
    try:
        raw = evds_client.get_data(
            [series_code],
            startdate=fetch_start,
            enddate=fetch_end,
            frequency=5,
            aggregation_types=aggregation
        )
    except Exception:
        try:
            raw = evds_client.get_data(
                [series_code],
                startdate=fetch_start,
                enddate=fetch_end,
                frequency=5
            )
        except Exception:
            raw = pd.DataFrame()

    # 2) Sonuç gelmezse alt çizgili varyantı dene.
    if raw is None or raw.empty:
        try:
            raw = evds_client.get_data(
                [series_code.replace(".", "_")],
                startdate=fetch_start,
                enddate=fetch_end,
                frequency=5
            )
        except Exception:
            raw = pd.DataFrame()

    # 3) Hâlâ boşsa web servise doğrudan git.
    if raw is None or raw.empty:
        raw = _evds_direct_request(series_code, fetch_start, fetch_end, aggregation=aggregation)

    if raw is None or raw.empty or "Tarih" not in raw.columns:
        return pd.DataFrame()

    try:
        raw = raw.copy()
        raw["dt"] = _parse_evds_months(raw["Tarih"])
        raw = raw.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)

        val_col = _pick_evds_value_col(raw, series_code)
        if val_col is None:
            return pd.DataFrame()

        raw[column_name] = _clean_evds_numeric(raw[val_col])
        raw = raw.dropna(subset=[column_name])
        if raw.empty:
            return pd.DataFrame()

        raw["Donem"] = raw["dt"].dt.strftime("%Y-%m")
        return raw.sort_values("dt").groupby("Donem").last().reset_index()[["Donem", column_name]]
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600)
def fetch_market_data_adapter(start_date, end_date):
    expectation_cols = list(EVDS_EXPECTATION_SERIES.keys())
    # Veri gelmezse 0 basılmayacak, NaN kalacak kolonlar (yanlış sıfır çizgisi olmasın)
    level_cols = ["AOFM", "AOFM-Faiz Farkı"]
    base_cols = ["Donem", "Aylık TÜFE", "Yıllık TÜFE", "PPK Faizi"]
    empty_df = pd.DataFrame(columns=base_cols + expectation_cols + level_cols + ["SortDate"])
    ts_start = pd.Timestamp(start_date)
    ts_end = pd.Timestamp(end_date)

    if not EVDS_API_KEY:
        dates = pd.date_range(start=start_date, end=end_date, freq='ME')
        if len(dates) == 0: return empty_df, "Tarih Yok"
        fallback = pd.DataFrame({
            'Donem': dates.strftime('%Y-%m'),
            'Aylık TÜFE': [0]*len(dates),
            'Yıllık TÜFE': [0]*len(dates),
            'PPK Faizi': [0]*len(dates),
            'SortDate': dates
        })
        for col in expectation_cols + level_cols:
            fallback[col] = np.nan
        return fallback, "API Key Yok"

    # --- TÜFE: iki seriden hibrit ---
    # Eski seri (2003=100): 2025 sonuna kadar — geçmiş yıllıklar buradan
    # Yeni seri (2025=100): 2025 Ocak'tan başlar — 2026+ yıllıklar buradan
    df_inf = pd.DataFrame()
    try:
        from evds import evdsAPI
        evds_client = evdsAPI(EVDS_API_KEY)

        # Eski seri: başlangıçtan 13 ay öncesi → 2025 sonu
        fetch_start_old = (ts_start - pd.DateOffset(months=13)).replace(day=1).strftime("%d-%m-%Y")
        fetch_end_old   = "01-12-2025"
        df_old = _evds_to_pct(evds_client, EVDS_TUFE_OLD, fetch_start_old, fetch_end_old)

        # Yeni seri: 2025 Ocak'tan bitiş tarihine kadar (yıllık için 12 ay lazım)
        fetch_end_new = ts_end.replace(day=1).strftime("%d-%m-%Y")
        df_new = _evds_to_pct(evds_client, EVDS_TUFE_NEW, "01-01-2025", fetch_end_new)
        # Yeni seriden sadece 2026+ dönemleri al (2025 yıllıklar eski seriden gelsin)
        if not df_new.empty:
            df_new = df_new[df_new["Donem"] >= "2026-01"].copy()

        # Birleştir
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=["Donem"], keep="last")
        df_combined = df_combined.sort_values("Donem").reset_index(drop=True)

        # İstenen aralığa kırp
        cutoff = ts_start.strftime("%Y-%m")
        end_cutoff = ts_end.strftime("%Y-%m")
        df_inf = df_combined[
            (df_combined["Donem"] >= cutoff) & (df_combined["Donem"] <= end_cutoff)
        ].copy()
        df_inf["Aylık TÜFE"]  = pd.to_numeric(df_inf["Aylık TÜFE"],  errors="coerce").round(2)
        df_inf["Yıllık TÜFE"] = pd.to_numeric(df_inf["Yıllık TÜFE"], errors="coerce").round(2)
        df_inf = df_inf.dropna(subset=["Aylık TÜFE", "Yıllık TÜFE"]).reset_index(drop=True)
    except Exception:
        pass

    # --- Enflasyon beklentileri: EVDS ---
    df_exp = pd.DataFrame()
    try:
        from evds import evdsAPI
        evds_client = evdsAPI(EVDS_API_KEY)
        fetch_start_exp = ts_start.replace(day=1).strftime("%d-%m-%Y")
        fetch_end_exp = ts_end.replace(day=1).strftime("%d-%m-%Y")

        for col_name, series_code in EVDS_EXPECTATION_SERIES.items():
            tmp_exp = _evds_to_monthly_level(evds_client, series_code, col_name, fetch_start_exp, fetch_end_exp)
            if tmp_exp.empty:
                continue
            if df_exp.empty:
                df_exp = tmp_exp
            else:
                df_exp = pd.merge(df_exp, tmp_exp, on="Donem", how="outer")

        if not df_exp.empty:
            df_exp = df_exp.sort_values("Donem").reset_index(drop=True)
    except Exception:
        pass

    # --- AOFM (TP.APIFON4): TCMB Ağırlıklı Ortalama Fonlama Maliyeti ---
    # Günlük seri; "last" ile ay sonu değeri alınır. Ay ortalaması, ay içinde
    # yapılan faiz değişimini bulanıklaştırır ve PPK karar tarihiyle hizalanmaz.
    df_aofm = pd.DataFrame()
    try:
        from evds import evdsAPI
        evds_client = evdsAPI(EVDS_API_KEY)
        fetch_start_aofm = ts_start.replace(day=1).strftime("%d-%m-%Y")
        fetch_end_aofm = ts_end.replace(day=1).strftime("%d-%m-%Y")
        df_aofm = _evds_to_monthly_level(
            evds_client, EVDS_AOFM, "AOFM",
            fetch_start_aofm, fetch_end_aofm, aggregation="last"
        )
    except Exception:
        pass

    # --- PPK Faizi: BIS ---
    df_pol = pd.DataFrame()
    try:
        s_bis = pd.Timestamp(start_date).strftime("%Y-%m-%d")
        e_bis = pd.Timestamp(end_date).strftime("%Y-%m-%d")
        url_bis = f"https://stats.bis.org/api/v1/data/WS_CBPOL/D.TR?format=csv&startPeriod={s_bis}&endPeriod={e_bis}"
        r_bis = requests.get(url_bis, timeout=20)
        if r_bis.status_code == 200:
            temp_bis = pd.read_csv(io.StringIO(r_bis.content.decode("utf-8")), usecols=["TIME_PERIOD", "OBS_VALUE"])
            temp_bis["dt"] = pd.to_datetime(temp_bis["TIME_PERIOD"])
            temp_bis["Donem"] = temp_bis["dt"].dt.strftime("%Y-%m")
            temp_bis["PPK Faizi"] = pd.to_numeric(temp_bis["OBS_VALUE"], errors="coerce")
            df_pol = temp_bis.sort_values("dt").groupby("Donem").last().reset_index()[["Donem", "PPK Faizi"]]
    except Exception:
        pass

    # --- Birleştir ---
    market_frames = [df for df in [df_inf, df_pol, df_exp, df_aofm] if df is not None and not df.empty]
    master_df = pd.DataFrame()
    for frame in market_frames:
        if master_df.empty:
            master_df = frame.copy()
        else:
            master_df = pd.merge(master_df, frame, on="Donem", how="outer")

    if master_df.empty:
        return empty_df, "Veri Bulunamadı"

    master_df = master_df.sort_values("Donem").reset_index(drop=True)
    if "PPK Faizi" in master_df.columns:
        master_df["PPK Faizi"] = master_df["PPK Faizi"].ffill()

    # Eski kolon davranışını koru: temel seriler yoksa 0.0 ile tamamlanabilir.
    # Beklenti serilerinde ise veri gelmezse 0 basma; NaN kalsın ki grafikte yanlış sıfır çizgisi görünmesin.
    for c in ["Aylık TÜFE", "Yıllık TÜFE", "PPK Faizi"]:
        if c not in master_df.columns:
            master_df[c] = 0.0
        master_df[c] = pd.to_numeric(master_df[c], errors="coerce")

    for c in expectation_cols + level_cols:
        if c not in master_df.columns:
            master_df[c] = np.nan
        master_df[c] = pd.to_numeric(master_df[c], errors="coerce")

    # Türetilmiş: fiili fonlama maliyeti ile ilan edilen politika faizi arasındaki fark.
    # Metin tonu ile fiili duruş arasındaki tutarsızlığı yakalayan en doğrudan gösterge.
    if "AOFM" in master_df.columns and "PPK Faizi" in master_df.columns:
        master_df["AOFM-Faiz Farkı"] = master_df["AOFM"] - master_df["PPK Faizi"]

    cutoff = ts_start.strftime("%Y-%m")
    end_cutoff = ts_end.strftime("%Y-%m")
    master_df = master_df[(master_df["Donem"] >= cutoff) & (master_df["Donem"] <= end_cutoff)].copy()
    master_df["SortDate"] = pd.to_datetime(master_df["Donem"] + "-01")
    return master_df.sort_values("SortDate").reset_index(drop=True), None



# =============================================================================
# 4b. PLOTLY: LEGEND'DA KAPATILAN SERİYİ PNG'DE DE GİZLEYEN RENDER
# =============================================================================

_CLEAN_PNG_JS = """
<script>
(function init() {
  var gd  = document.getElementById('__DIV_ID__');
  var btn = document.getElementById('__DIV_ID___btn');
  if (!window.Plotly || !gd || !gd.data || !btn) { return setTimeout(init, 120); }

  // Ekranda çizilenden bağımsız, BOZULMAMIŞ figür tanımı. Baskı kopyası bundan
  // türetilir; gd.layout kullanılsaydı Plotly'nin hesapladığı iç alanlar da
  // kopyaya sızardı.
  var PRISTINE = __FIGJSON__;

  // --- 1) YENİDEN BOYUTLANDIRMA ---------------------------------------------
  // Streamlit grafiği iframe'e koyar. Script çalıştığı anda iframe'in genişliği
  // çoğu zaman nihai genişliği DEĞİLDİR; Plotly legend'ı o dar ölçüye göre
  // yerleştirir, iframe sonradan genişler ve legend bir daha akmaz — tüm girdiler
  // tek satırda üst üste biner. ResizeObserver bunu düzeltir.
  var rt = null;
  var ro = new ResizeObserver(function () {
    clearTimeout(rt);
    rt = setTimeout(function () { try { Plotly.Plots.resize(gd); } catch (e) {} }, 60);
  });
  ro.observe(gd);
  window.addEventListener('load', function () {
    try { Plotly.Plots.resize(gd); } catch (e) {}
  });

  // --- 2) BASKI PROFİLİ ------------------------------------------------------
  // Word/PowerPoint görseli sayfa genişliğine ölçekler. Okunabilirliği belirleyen
  // şey mutlak piksel değil, YAZI BOYUTUNUN GÖRSEL GENİŞLİĞİNE ORANIdır.
  // 11px yazı / 1900px genişlik = %0.58 -> 16cm sayfada ~2.6pt, okunmaz.
  // Bu profil oranı ~%1.6'ya çıkarır (16cm sayfada ~7pt).
  // Not: 'scale' artırmak İŞE YARAMAZ — o yalnızca çözünürlüğü çoğaltır, oranı
  // değiştirmez. Oranı büyütmek için genişliği düşürüp yazıyı büyütmek gerekir.
  function baskiKopyasi(gizli) {
    var f = JSON.parse(JSON.stringify(PRISTINE));

    // Legend'da kapatılmış serileri kaydıyla birlikte kaldır
    gizli.forEach(function (i) { if (f.data[i]) { f.data[i].showlegend = false; } });

    var L = f.layout || (f.layout = {});
    L.font = L.font || {};                 L.font.size = 20;
    L.title = L.title || {};
    L.title.font = L.title.font || {};     L.title.font.size = 28;

    L.legend = L.legend || {};
    L.legend.font = L.legend.font || {};   L.legend.font.size = 22;
    L.legend.entrywidthmode = 'fraction';
    L.legend.entrywidth = 0.33;            // baskıda satır başına 3 girdi
    L.legend.y = -0.07;
    L.legend.orientation = 'h';
    L.legend.xanchor = 'center';           L.legend.x = 0.5;
    L.legend.yanchor = 'top';

    L.margin = L.margin || {};
    L.margin.t = 90; L.margin.b = 330; L.margin.l = 100; L.margin.r = 50;

    ['xaxis', 'yaxis', 'xaxis2', 'yaxis2'].forEach(function (ax) {
      if (!L[ax]) { return; }
      L[ax].tickfont = L[ax].tickfont || {}; L[ax].tickfont.size = 19;
      if (L[ax].title) {
        L[ax].title.font = L[ax].title.font || {};
        L[ax].title.font.size = 21;
      }
    });

    // Vali isimleri, ŞAHİN/GÜVERCİN etiketleri, "Haber" bağlantıları
    (L.annotations || []).forEach(function (an) {
      an.font = an.font || {};
      an.font.size = Math.max(16, Math.round((an.font.size || 12) * 1.6));
    });

    delete L.width; delete L.height;
    return f;
  }

  btn.addEventListener('click', function () {
    var gizli = [];
    gd.data.forEach(function (t, i) { if (t.visible === 'legendonly') { gizli.push(i); } });

    var f = baskiKopyasi(gizli);

    // Ekrandaki grafiğe hiç dokunulmaz: baskı kopyası ekran dışında ayrı bir
    // div'e çizilir. Böylece titreme olmaz ve "geri alma" adımı hiç gerekmez.
    var off = document.createElement('div');
    off.style.cssText = 'position:absolute;left:-99999px;top:0;width:__PW__px;height:__PH__px;';
    document.body.appendChild(off);

    btn.disabled = true;
    var temizle = function () {
      try { Plotly.purge(off); } catch (e) {}
      if (off.parentNode) { off.parentNode.removeChild(off); }
      btn.disabled = false;
    };

    Plotly.newPlot(off, f.data, f.layout, {staticPlot: true})
      .then(function () {
        return Plotly.downloadImage(off, {
          format: 'png', width: __PW__, height: __PH__, scale: __PS__,
          filename: '__FNAME__'
        });
      })
      .then(temizle)
      .catch(temizle);
  });
})();
</script>
"""

_CLEAN_PNG_BTN = """
<div style="text-align:right;margin:0 0 4px 0;">
  <button id="__DIV_ID___btn" style="
      font-size:12px;padding:4px 10px;cursor:pointer;
      border:1px solid rgba(128,128,128,.45);border-radius:6px;
      background:transparent;color:inherit;font-family:inherit;">
    ⬇ PNG indir (belge için — büyük yazı, kapalı seriler hariç)
  </button>
</div>
"""


def render_plotly_clean_png(fig,
                            div_id: str = "ana_grafik",
                            height: int = 660,
                            filename: str = "analiz_paneli",
                            png_width: int = 1400,
                            png_height: int = 1200,
                            png_scale: int = 2):
    """
    Plotly grafiğini basar ve yanına belge-dostu PNG indirme butonu koyar.

    ÇÖZÜLEN İKİ SORUN
    -----------------
    1) Legend'a tıklayınca Plotly trace'i silmez, `visible='legendonly'` yapar.
       Seri çizim alanından kalkar ama legend kaydı soluk halde kalır ve yerleşik
       `toImage` o hali basar. Baskı kopyasında `showlegend=False` uygulanarak
       kayıt tamamen kaldırılır.

    2) Word/PowerPoint görseli sayfa genişliğine ölçekler; okunabilirliği belirleyen
       şey mutlak piksel değil, yazı boyutunun görsel genişliğine ORANIdır. Ekran
       ayarıyla (11px / 1900px) bu oran %0.58'dir ve 16cm'lik bir sayfada ~2.6pt'ye
       düşer — okunmaz. İndirmede ayrı bir "baskı profili" uygulanır: genişlik
       düşürülür, yazılar büyütülür, oran ~%1.6'ya çıkar (~7pt).
       DİKKAT: `png_scale` artırmak bu sorunu ÇÖZMEZ; o yalnızca çözünürlüğü
       çoğaltır, oranı değiştirmez.

    TASARIM NOTU — NEDEN MODEBAR BUTONU DEĞİL
    -----------------------------------------
    Modebar'a özel buton eklemek `Plotly.newPlot` config'ini değiştirmeyi, yani
    grafiği İKİNCİ KEZ çizdirmeyi gerektirir. İkinci çizim, birinciden kalma
    hesaplanmış layout nesnesiyle yapıldığı için legend akışını bozuyor ve 16
    serilik grafikte tüm legend girdileri tek satırda üst üste biniyordu. Bu yüzden
    grafik yalnızca BİR KEZ çizilir; indirme, ekran dışına çizilen ayrı bir kopyayla
    yapılır ve ekrandaki grafiğe hiç dokunulmaz.

    NOT: Bu düzeltme `st.plotly_chart` üzerinden mümkün değildir — legend tıklaması
    Python tarafına hiç ulaşmaz. Grafik iframe içinde render edilir, bu yüzden
    Streamlit temasını miras almaz; figürün kendi renkleri geçerlidir.
    """
    try:
        import streamlit.components.v1 as components
        import plotly.io as pio
    except Exception:
        st.plotly_chart(fig, use_container_width=True)
        return

    try:
        html_fig = pio.to_html(
            fig,
            include_plotlyjs="cdn",
            full_html=False,
            div_id=div_id,
            config={
                "displaylogo": False,
                "responsive": True,
                "modeBarButtonsToRemove": ["toImage", "lasso2d", "select2d"],
            },
        )
        fig_json = pio.to_json(fig).replace("</script", "<\\/script")
        btn = _CLEAN_PNG_BTN.replace("__DIV_ID__", div_id)
        js = (_CLEAN_PNG_JS
              .replace("__FIGJSON__", fig_json)
              .replace("__DIV_ID__", div_id)
              .replace("__PW__", str(int(png_width)))
              .replace("__PH__", str(int(png_height)))
              .replace("__PS__", str(int(png_scale)))
              .replace("__FNAME__", str(filename)))
        components.html(btn + html_fig + js, height=int(height) + 60, scrolling=False)
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# 4. OKUNABİLİRLİK VE FREKANS ANALİZİ
# =============================================================================

def count_syllables_en(word):
    word = word.lower()
    if len(word) <= 3: return 1
    word = re.sub(r'(?:[^laeiouy]|ed|[^laeiouy]e)$', '', word, flags=re.IGNORECASE)
    word = re.sub(r'^y', '', word, flags=re.IGNORECASE)
    syllables = re.findall(r'[aeiouy]{1,2}', word, flags=re.IGNORECASE)
    return len(syllables) if syllables else 1

def calculate_flesch_reading_ease(text):
    if not text: return 0
    lines = text.split('\n')
    filtered_lines = [ln for ln in lines if not re.match(r'^\s*[-•]\s*', ln)]
    filtered_text = ' '.join(filtered_lines)
    cleaned_text = re.sub(r'\d+\.\d+', '', filtered_text)
    
    sentences = re.findall(r'[^\.!\?]+[\.!\?]+', cleaned_text)
    sentence_count = len(sentences) if sentences else 1
    
    words_cleaned = [w for w in re.split(r'\s+', cleaned_text) if w]
    total_words_cleaned = len(words_cleaned)
    average_sentence_length = total_words_cleaned / sentence_count if sentence_count > 0 else 0
    
    words_raw = [w for w in re.split(r'\s+', text) if w]
    total_words_raw = len(words_raw)
    
    if total_words_raw == 0: return 0
    
    total_syllables_raw = sum(count_syllables_en(w) for w in words_raw)
    average_syllables_per_word = total_syllables_raw / total_words_raw
    
    score = 206.835 - (1.015 * average_sentence_length) - (84.6 * average_syllables_per_word)
    return round(score, 2)

def get_terms_series(df: pd.DataFrame,
                     terms: list,
                     text_col: str = "text_content",
                     date_col: str = "period_date") -> pd.DataFrame:
    """
    Verilen terms listesi için dönem bazlı sayım serisi üretir.
    - Tek kelime ve çok kelimeli ifadeleri destekler (örn: "policy rate")
    - Bulunmayan terimler 0 döner
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out_rows = []
    tmp = df.copy()

    # tarih normalize
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col]).sort_values(date_col)

    # metin normalize
    texts = tmp[text_col].fillna("").astype(str).str.lower()

    # terms normalize
    terms_norm = [str(t).strip().lower() for t in (terms or []) if str(t).strip()]
    terms_norm = list(dict.fromkeys(terms_norm))  # uniq

    for i, row in tmp.iterrows():
        txt = str(row.get(text_col, "") or "").lower()

        rec = {
            "period_date": row[date_col],
            "Donem": pd.to_datetime(row[date_col]).strftime("%Y-%m")
        }

        for term in terms_norm:
            # basit substring sayımı (ppk metinleri için yeterli ve hızlı)
            rec[term] = txt.count(term)

        out_rows.append(rec)

    return pd.DataFrame(out_rows)
def build_watch_terms_timeseries(df_all: pd.DataFrame, terms: list) -> pd.DataFrame:
    if df_all is None or df_all.empty or not terms:
        return pd.DataFrame()

    df = df_all.copy()
    if "period_date" not in df.columns:
        return pd.DataFrame()

    df["period_date"] = pd.to_datetime(df["period_date"], errors="coerce")
    df = df.dropna(subset=["period_date"]).sort_values("period_date")
    df["Donem"] = df["period_date"].dt.strftime("%Y-%m")

    rows = []
    for _, r in df.iterrows():
        txt = str(r.get("text_content", "") or "").lower()
        rec = {"period_date": r["period_date"], "Donem": r["Donem"]}
        for t in terms:
            rec[t] = txt.count(str(t).lower())
        rows.append(rec)

    return pd.DataFrame(rows).reset_index(drop=True)


    rows = []
    for _, r in out.iterrows():
        txt = r["text_lc"]
        row = {"period_date": r["period_date"], "Donem": r["Donem"]}
        for t in terms:
            # basit substring sayımı (phrase için de çalışır)
            row[t] = int(txt.count(t))
        rows.append(row)

    return pd.DataFrame(rows)


def generate_diff_html(text1, text2):
    if not text1: text1 = ""
    if not text2: text2 = ""
    a = text1.split()
    b = text2.split()
    matcher = difflib.SequenceMatcher(None, a, b)
    html_output = []
    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        if opcode == 'equal':
            html_output.append(" ".join(a[a0:a1]))
        elif opcode == 'insert':
            html_output.append(f'<span style="background-color: #d4fcbc; color: #376e37; font-weight: bold;">+ {" ".join(b[b0:b1])}</span>')
        elif opcode == 'delete':
            html_output.append(f'<span style="background-color: #fcd4bc; color: #9c4444; text-decoration: line-through;">- {" ".join(a[a0:a1])}</span>')
        elif opcode == 'replace':
            html_output.append(f'<span style="background-color: #fcd4bc; color: #9c4444; text-decoration: line-through;">- {" ".join(a[a0:a1])}</span>')
            html_output.append(f'<span style="background-color: #d4fcbc; color: #376e37; font-weight: bold;">+ {" ".join(b[b0:b1])}</span>')
    return " ".join(html_output)

def get_top_terms_series(df, top_n=7, custom_stops=None):
    if df.empty: return pd.DataFrame(), []
    all_text = " ".join(df['text_content'].astype(str).tolist()).lower()
    words = re.findall(r"\b[a-z]{4,}\b", all_text)
    
    stops = set([
        "that", "with", "this", "from", "have", "which", "will", "been", "were", 
        "market", "central", "bank", "committee", "monetary", "policy", "decision", 
        "percent", "rates", "level", "year", "their", "over", "also", "under", 
        "developments", "conditions", "indicators", "recent", "remain", "remains",
        "period", "has", "are", "for", "and", "the", "decided", "keep", "constant",
        "take", "taking", "account"
    ])
    
    if custom_stops:
        for s in custom_stops: stops.add(s.lower().strip())
        
    filtered_words = [w for w in words if w not in stops]
    common = Counter(filtered_words).most_common(top_n)
    top_terms = [t[0] for t in common]
    
    results = []
    for _, row in df.iterrows():
        txt = str(row['text_content']).lower()
        entry = {'period_date': row['period_date'], 'Donem': row.get('Donem', '')}
        for term in top_terms:
            entry[term] = txt.count(term)
        results.append(entry)
    return pd.DataFrame(results).sort_values('period_date'), top_terms

def generate_wordcloud_img(text, custom_stops=None):
    if not HAS_ML_DEPS or not text: return None
    stopwords = set(STOPWORDS)
    stopwords.update(["central", "bank", "committee", "monetary", "policy", "percent", "decision", "rate", "board", "meeting"])
    if custom_stops:
        for s in custom_stops: stopwords.add(s.lower().strip())
    wc = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
    return fig

# =============================================================================
# 5. ABF (APEL, BLIX, GRIMALDI - 2019) ALGORİTMASI (ORİJİNAL)
# =============================================================================

def M(token_or_phrase: str, wildcard_first: bool = False):
    toks = token_or_phrase.split()
    wild = [False] * len(toks)
    if wildcard_first and toks:
        wild[0] = True
    return {"phrase": toks, "wild": wild, "pattern": token_or_phrase}

DICT = {
   "inflation": [
        {
            "block": "consumer_prices_inflation",
            "terms": ["consumer prices", "inflation"],
            "hawk": [M("accelerat", True), M("boost", True), M("elevated"), M("escalat", True), M("high", True), M("increas", True), M("jump", True), M("pickup"), M("rise", True), M("rose"), M("rising"), M("runup"), M("strong", True), M("surg", True), M("up", True)],
            "dove": [M("decelerat", True), M("declin", True), M("decreas", True), M("down", True), M("drop", True), M("fall", True), M("fell"), M("low", True), M("muted"), M("reduc", True), M("slow", True), M("stable"), M("subdued"), M("weak", True), M("contained")],
        },
        {
            "block": "inflation_pressure",
            "terms": ["inflation pressure"],
            "hawk": [M("accelerat", True), M("boost", True), M("build", True), M("elevat", True), M("emerg", True), M("great", True), M("height", True), M("high", True), M("increas", True), M("intensif", True), M("mount", True), M("pickup"), M("rise", True), M("rose"), M("rising"), M("stok", True), M("strong", True), M("sustain", True)],
            "dove": [M("abat", True), M("contain", True), M("dampen", True), M("decelerat", True), M("declin", True), M("decreas", True), M("dimin", True), M("eas", True), M("fall", True), M("fell"), M("low", True), M("moderat", True), M("reced", True), M("reduc", True), M("subdued"), M("temper", True)],
        },
    ],
   "economic_activity": [
        {
            "block": "consumer_spending",
            "terms": ["consumer spending"],
            "hawk": [M("accelerat", True), M("edg up", True), M("expan", True), M("increas", True), M("pick up", True), M("pickup"), M("soft", True), M("strength", True), M("strong", True), M("weak", True)],
            "dove": [M("contract", True), M("decelerat", True), M("decreas", True), M("drop", True), M("retrench", True), M("slow", True), M("slugg", True), M("soft", True), M("subdued")],
        },
        {
            "block": "economic_activity_growth",
            "terms": ["economic activity", "economic growth"],
            "hawk": [M("accelerat", True), M("buoyant"), M("edg up", True), M("expan", True), M("increas", True), M("high", True), M("pick up", True), M("pickup"), M("rise", True), M("rose"), M("rising"), M("step up", True), M("strength", True), M("strong", True), M("upside")],
            "dove": [M("contract", True), M("curtail", True), M("decelerat", True), M("declin", True), M("decreas", True), M("downside"), M("drop"), M("fall", True), M("fell"), M("low", True), M("moderat", True), M("slow", True), M("slugg", True), M("weak", True)],
        },
        {
            "block": "resource_utilization",
            "terms": ["resource utilization"],
            "hawk": [M("high", True), M("increas", True), M("rise"), M("rising"), M("rose"), M("tight", True)],
            "dove": [M("declin", True), M("fall", True), M("fell"), M("loose", True), M("low", True)],
        },
    ],
   "employment": [
        {
            "block": "employment",
            "terms": ["employment"],
            "hawk": [M("expand", True), M("gain", True), M("improv", True), M("increas", True), M("pick up", True), M("pickup"), M("rais", True), M("rise", True), M("rising"), M("rose"), M("strength", True), M("turn up", True)],
            "dove": [M("slow", True), M("declin", True), M("reduc", True), M("weak", True), M("deteriorat", True), M("shrink", True), M("shrank"), M("fall", True), M("fell"), M("drop", True), M("contract", True), M("sluggish")],
        },
        {
            "block": "labor_market",
            "terms": ["labor market"],
            "hawk": [M("strain", True), M("tight", True)],
            "dove": [M("eased", True), M("easing", True), M("loos", True), M("soft", True), M("weak", True)],
        },
        {
            "block": "unemployment",
            "terms": ["unemployment"],
            "hawk": [M("declin", True), M("fall", True), M("fell"), M("low", True), M("reduc", True)],
            "dove": [M("elevat", True), M("high"), M("increas", True), M("ris", True), M("rose", True)],
        },
    ],
}

def normalize_text(text: str) -> str:
    t = text.lower().replace("’", "'").replace("`", "'")
    t = re.sub(r"(?<=\w)-(?=\w)", " ", t)
    t = re.sub(r"\brun\s+up\b", "runup", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def split_sentences_nlp(text: str):
    text = re.sub(r"\n+", ". ", text)
    sents = re.split(r"(?<=[\.\!\?\;])\s+", text)
    return [s.strip() for s in sents if s.strip()]



def split_sentences_tr(text: str):
    """Basit ve güvenli cümle bölücü (TR/EN karışık metinler için).
    - Yeni satırları nokta gibi ele alır.
    - . ! ? ; sonrası bölmeye çalışır.
    """
    if text is None:
        return []
    t = str(text).strip()
    if not t:
        return []
    t = re.sub(r"\n+", ". ", t)
    sents = re.split(r"(?<=[\.!\?\;])\s+", t)
    # Fallback: yine de tek parça geldiyse, newline bazlı dene
    if len(sents) <= 1:
        sents = [s.strip() for s in str(text).splitlines() if s.strip()]
    return [s.strip() for s in sents if s.strip()]

def tokenize(sent: str):
    return re.findall(r"[a-z]+", sent)

def match_token(tok: str, pat: str, wildcard: bool) -> bool:
    return tok.startswith(pat) if wildcard else tok == pat

def find_phrase_positions(tokens, phrase_tokens, wild_flags):
    m = len(phrase_tokens)
    hits = []
    for i in range(0, len(tokens) - m + 1):
        ok = True
        for j in range(m):
            if not match_token(tokens[i + j], phrase_tokens[j], wild_flags[j]):
                ok = False
                break
        if ok:
           hits.append((i, i + m - 1))
    return hits

def find_term_positions_flex(tokens, term: str):
    tt = term.split()
    m = len(tt)
    hits = []
    for i in range(0, len(tokens) - m + 1):
        window = tokens[i:i+m]
        ok = True
        for j in range(m):
            if window[j] == tt[j]: continue
            if window[j] == tt[j] + "s" or tt[j] == window[j] + "s": continue
            ok = False
            break
        if ok: hits.append((i, i + m - 1))
    return hits

def select_non_overlapping_terms(tokens, term_infos):
    term_infos_sorted = sorted(term_infos, key=lambda x: len(x["term"].split()), reverse=True)
    occupied = set()
    selected = []
    for info in term_infos_sorted:
        for (s, e) in find_term_positions_flex(tokens, info["term"]):
            if any(k in occupied for k in range(s, e + 1)): continue
            occupied.update(range(s, e + 1))
            selected.append({**info, "start": s, "end": e})
    selected.sort(key=lambda x: x["start"])
    return selected

def analyze_hawk_dove(text: str, DICT: dict, window_words: int = 7, dedupe_within_term_window: bool = True, nearest_only: bool = False):
    text_n = normalize_text(text)
    sentences = split_sentences_nlp(text_n)
    
    topic_term_infos = {}
    for topic, blocks in DICT.items():
        infos = []
        for b in blocks:
            for term in b["terms"]:
               infos.append({"topic": topic, "block": b["block"], "term": term})
        topic_term_infos[topic] = infos

    topic_counts = {topic: {"hawk": 0, "dove": 0} for topic in DICT.keys()}
    matches = []

    for sent in sentences:
        tokens = tokenize(sent)
        if not tokens: continue

        for topic, term_infos in topic_term_infos.items():
            selected_terms = select_non_overlapping_terms(tokens, term_infos)
            if not selected_terms: continue

            blocks_by_name = {b["block"]: b for b in DICT[topic]}

            for tinfo in selected_terms:
                block = blocks_by_name[tinfo["block"]]
                ts, te = tinfo["start"], tinfo["end"]
                w0 = max(0, ts - window_words)
                w1 = min(len(tokens) - 1, te + window_words)
                term_found = " ".join(tokens[ts:te + 1])

                hawk_hits = []
                for m in block["hawk"]:
                   for (ms, me) in find_phrase_positions(tokens, m["phrase"], m["wild"]):
                       if me < w0 or ms > w1: continue
                       dist = min(abs(ms - te), abs(ts - me))
                       hawk_hits.append((dist, m, ms, me))

                dove_hits = []
                for m in block["dove"]:
                   for (ms, me) in find_phrase_positions(tokens, m["phrase"], m["wild"]):
                       if me < w0 or ms > w1: continue
                       dist = min(abs(ms - te), abs(ts - me))
                       dove_hits.append((dist, m, ms, me))

                if nearest_only:
                   hawk_hits = sorted(hawk_hits, key=lambda x: x[0])[:1]
                   dove_hits = sorted(dove_hits, key=lambda x: x[0])[:1]

                seen = set()
                def add_hit(direction, dist, m, ms, me):
                    mod_found = " ".join(tokens[ms:me+1])
                    key = (topic, block["block"], ts, te, direction, mod_found)
                    if dedupe_within_term_window and key in seen: return
                    seen.add(key)
                    topic_counts[topic][direction] += 1
                    matches.append({
                        "topic": topic, "block": block["block"], "direction": direction,
                        "term_found": term_found, "modifier_pattern": m["pattern"], "modifier_found": mod_found,
                        "distance": dist, "sentence": sent,
                        "term": tinfo["term"], "type": "HAWK" if direction == "hawk" else "DOVE"
                    })

                for (dist, m, ms, me) in hawk_hits: add_hit("hawk", dist, m, ms, me)
                for (dist, m, ms, me) in dove_hits: add_hit("dove", dist, m, ms, me)

    hawk_total = sum(v["hawk"] for v in topic_counts.values())
    dove_total = sum(v["dove"] for v in topic_counts.values())
    denom = hawk_total + dove_total
    net_hawkishness = 1.0 if denom == 0 else (1.0 + (hawk_total - dove_total) / denom)

    return {
       "topic_counts": topic_counts,
       "matches": matches,
       "match_details": matches,
       "net_hawkishness": net_hawkishness,
       "hawk_count": hawk_total,
       "dove_count": dove_total
    }

# =============================================================================
# 6. ENTEGRASYON VE ML YARDIMCILARI
# =============================================================================

class ABGAnalyzer:
    def analyze(self, text):
        return analyze_hawk_dove(text, DICT=DICT, window_words=10, dedupe_within_term_window=True, nearest_only=True)

def run_full_analysis(text):
    res = analyze_hawk_dove(text, DICT=DICT, window_words=10, dedupe_within_term_window=True, nearest_only=True)
    s_abg = res['net_hawkishness']
    h_cnt = res['hawk_count']
    d_cnt = res['dove_count']
    
    h_list = []
    d_list = []
    h_ctx = {}
    d_ctx = {}
    
    for m in res['matches']:
        item = f"{m['term_found']} ({m['modifier_found']})"
        if m['direction'] == 'hawk':
            h_list.append(item)
            if m['term_found'] not in h_ctx: h_ctx[m['term_found']] = []
            h_ctx[m['term_found']].append(m['sentence'])
        else:
            d_list.append(item)
            if m['term_found'] not in d_ctx: d_ctx[m['term_found']] = []
            d_ctx[m['term_found']].append(m['sentence'])
            
    flesch = calculate_flesch_reading_ease(text)
    return s_abg, h_cnt, d_cnt, h_list, d_list, h_ctx, d_ctx, flesch

def calculate_abg_scores(df):
    if df is None or df.empty: return pd.DataFrame()
    rows = []
    for _, row in df.iterrows():
        res = analyze_hawk_dove(str(row.get('text_content', '')), DICT=DICT, window_words=10, dedupe_within_term_window=True, nearest_only=True)
        donem = row.get("Donem", "")
        if not donem and "period_date" in row:
             try: donem = pd.to_datetime(row["period_date"]).strftime("%Y-%m")
             except: pass
        rows.append({
            "period_date": row.get("period_date"),
            "Donem": donem,
            "abg_index": res['net_hawkishness']
        })
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner=False)
def build_text_as_data_model(df: pd.DataFrame):
    """
    TF-IDF (1-2 gram) + Ridge ile delta_bp tahmin modeli.
    """
    if not HAS_ML_DEPS:
        return None

    if df is None or df.empty:
        return None

    # Eğitim datası: delta_bp dolu olmalı
    d = df.dropna(subset=["delta_bp"]).copy()
    if len(d) < 8:
        return None

    X = d["text_content"].astype(str).values
    y = d["delta_bp"].astype(float).values

    # Not: metinler İngilizce ise stop_words="english" iyi çalışır.
    # Türkçe metin de gelecekse stop_words=None bırakmak daha güvenli.
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        max_features=50000,
        sublinear_tf=True,
        stop_words="english"
    )
    Xv = vec.fit_transform(X)

    reg = Ridge(alpha=10.0, random_state=42)
    reg.fit(Xv, y)

    return {"vectorizer": vec, "model": reg, "train_n": len(d)}


def predict_text_as_data_delta_bp(text: str, bundle):
    if bundle is None or not text:
        return None
    vec = bundle["vectorizer"]
    reg = bundle["model"]
    Xv = vec.transform([str(text)])
    return float(reg.predict(Xv)[0])


def text_as_data_top_terms(bundle, top_k: int = 20):
    """
    Ridge coef -> hangi kelimeler indirime/ artırıma iter?
    + coef: daha çok ARTIRIM (pozitif bps)
    - coef: daha çok İNDİRİM (negatif bps)
    """
    if bundle is None:
        return pd.DataFrame()

    vec = bundle["vectorizer"]
    reg = bundle["model"]

    if not hasattr(reg, "coef_"):
        return pd.DataFrame()

    coefs = reg.coef_.ravel()
    feats = vec.get_feature_names_out()

    df = pd.DataFrame({"term": feats, "coef": coefs})
    df = df.sort_values("coef", ascending=False)

    top_pos = df.head(top_k).copy()
    top_pos["direction"] = "ARTIRIM (+)"

    top_neg = df.tail(top_k).copy().sort_values("coef", ascending=True)
    top_neg["direction"] = "İNDİRİM (-)"

    out = pd.concat([top_pos, top_neg], ignore_index=True)
    return out


def text_as_data_walk_forward(df: pd.DataFrame, min_train: int = 8):
    """
    Basit walk-forward: her adımda geçmişle eğit, bir sonraki noktayı tahmin et.
    """
    if not HAS_ML_DEPS:
        return None

    if df is None or df.empty:
        return None

    d = df.dropna(subset=["delta_bp"]).copy().sort_values("period_date")
    if len(d) < (min_train + 1):
        return None

    preds = []
    actuals = []
    dates = []

    for i in range(min_train, len(d)):
        train = d.iloc[:i].copy()
        test_row = d.iloc[i]

        bundle = build_text_as_data_model(train)
        if bundle is None:
            continue

        yhat = predict_text_as_data_delta_bp(test_row["text_content"], bundle)
        preds.append(yhat)
        actuals.append(float(test_row["delta_bp"]))
        dates.append(test_row["period_date"])

    out = pd.DataFrame({"date": dates, "y_true": actuals, "y_pred": preds})
    if out.empty:
        return None

    out["err"] = out["y_true"] - out["y_pred"]
    return out



# =============================================================================
# 7. ML ALGORİTMASI (Ridge + Logistic)
# =============================================================================


def prepare_next_rate_dataset(df_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Her toplantı metninden bir sonraki toplantının policy_rate'ini tahmin etmek için dataset.
    Çıktı kolonları: date, text, policy_rate, delta_bp, next_policy_rate
    """
    if df_logs is None or df_logs.empty:
        return pd.DataFrame()

    df = df_logs.copy()
    df["period_date"] = pd.to_datetime(df["period_date"], errors="coerce")
    df = df.dropna(subset=["period_date"]).sort_values("period_date")

    if "policy_rate" not in df.columns:
        return pd.DataFrame()

    df["policy_rate"] = pd.to_numeric(df["policy_rate"], errors="coerce")
    df["delta_bp"] = pd.to_numeric(df.get("delta_bp", np.nan), errors="coerce")

    # delta_bp boşsa otomatik üret (fallback)
    if df["delta_bp"].isna().all():
        df["delta_bp"] = df["policy_rate"].diff().fillna(0.0) * 100.0

    df["text"] = df["text_content"].fillna("").apply(normalize_tr_text)

    # hedef = bir sonraki toplantının faizi
    df["next_policy_rate"] = df["policy_rate"].shift(-1)

    # son satırın hedefi yok
    df = df.dropna(subset=["next_policy_rate", "policy_rate", "delta_bp"])

    out = pd.DataFrame({
        "date": df["period_date"],
        "text": df["text"],
        "policy_rate": df["policy_rate"].astype(float),
        "delta_bp": df["delta_bp"].astype(float),
        "next_policy_rate": df["next_policy_rate"].astype(float),
    })
    return out.reset_index(drop=True)




@dataclass
class CFG:
    cap_low: int = -750
    cap_high: int = 750
    token_pattern: str = r"(?u)\b[0-9a-zçğıöşü]{2,}\b"
    word_ngram: Tuple[int,int] = (1, 2)
    min_df: int = 1
    max_df: float = 1.0
    max_features: int = 20000     
    trend_window: int = 6
    max_splits: int = 6
    half_life_days: float = 365.0
    q_lo: float = 0.02
    q_hi: float = 0.98
    vol_factor: float = 1.0
    vol_cap: float = 3.0
    unc_factor: float = 1.5
    blend_cond: float = 0.65
    blend_all: float = 0.35
    fallback_cut_bps: float = -75.0
    fallback_hike_bps: float = 75.0

cfg = CFG()

def normalize_tr_text(s: str) -> str:
    if s is None: return ""
    s = str(s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clip_bps(x, lo=cfg.cap_low, hi=cfg.cap_high):
    return np.clip(x, lo, hi)

def bps_to_direction(y_bps: np.ndarray) -> np.ndarray:
    y = np.asarray(y_bps, dtype=float)
    out = np.zeros_like(y, dtype=int)
    out[y < 0] = -1
    out[y > 0] = 1
    return out

def exp_time_weights(dates: pd.Series, half_life_days: float = cfg.half_life_days) -> np.ndarray:
    d = pd.to_datetime(dates)
    t = (d - d.min()).dt.days.values.astype(float)
    lam = np.log(2.0) / float(half_life_days)
    w = np.exp(lam * t)
    return w / np.mean(w)

def rolling_slope(y: np.ndarray, window: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y, dtype=float)
    for i in range(len(y)):
        j0 = max(0, i - window + 1)
        seg = y[j0:i+1]
        if len(seg) < 3:
            out[i] = 0.0
            continue
        x = np.arange(len(seg), dtype=float)
        out[i] = np.polyfit(x, seg, 1)[0]
    return out

def safe_median_days(dates: pd.Series) -> float:
    if len(dates) <= 1: return 30.0
    diffs = pd.to_datetime(dates).diff().dt.days.dropna()
    return float(diffs.median()) if len(diffs) else 30.0

def choose_splits(n: int) -> int:
    return int(min(cfg.max_splits, max(3, n // 8)))

def rmse_metric(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def add_features(df: pd.DataFrame, trend_window: int = cfg.trend_window) -> pd.DataFrame:
    out = df.copy()
    out["y_bps"] = clip_bps(out["rate_change_bps"].values)
    out["y_dir"] = bps_to_direction(out["y_bps"].values)

    out["prev_change_bps"] = clip_bps(out["y_bps"].shift(1).fillna(0.0).values)
    out["prev_abs_change"] = np.abs(out["prev_change_bps"].values)
    out["prev_sign"] = np.sign(out["prev_change_bps"].values).astype(int)

    streak, cur = [], 0
    for v in out["y_bps"].shift(1).fillna(0.0).values:
        if float(v) == 0.0: cur += 1
        else: cur = 0
        streak.append(cur)
    out["hold_streak"] = np.array(streak, dtype=int)

    out["mean_abs_last3"] = (
        out["y_bps"].shift(1).fillna(0).abs() +
        out["y_bps"].shift(2).fillna(0).abs() +
        out["y_bps"].shift(3).fillna(0).abs()
    ).values / 3.0

    med = safe_median_days(out["date"])
    out["days_since_prev"] = out["date"].diff().dt.days.fillna(med).clip(lower=0).astype(float)

    out["roll_mean_bps"] = out["y_bps"].rolling(trend_window, min_periods=1).mean()
    out["roll_std_bps"] = out["y_bps"].rolling(trend_window, min_periods=1).std().fillna(0.0)
    out["roll_slope_bps"] = rolling_slope(out["y_bps"].values, trend_window)
    out["momentum_bps"] = out["y_bps"] - out["roll_mean_bps"]

    base = float(out["roll_std_bps"].median()) if len(out) else 1.0
    base = base if np.isfinite(base) and base > 0 else 1.0
    out["vol_ratio"] = (out["roll_std_bps"] / base).replace([np.inf, -np.inf], 1.0).fillna(1.0)
    return out

KEYWORDS = [
    "enflasyon", "çekirdek", "fiyat", "beklenti", "talep", "iktisadi faaliyet", "büyüme",
    "kur", "kredi", "risk primi", "finansal koşul", "sıkı", "sıkılaşma", "gevşeme", 
    "kararlı", "ilave", "gerekirse", "dezenflasyon", "inflation", "price", "growth"
]

def keyword_features(text_series: pd.Series) -> np.ndarray:
    X = []
    for t in text_series.fillna("").astype(str).values:
        t = t.lower()
        row = [t.count(kw) for kw in KEYWORDS]
        row.append(len(t))
        X.append(row)
    return np.asarray(X, dtype=float)

kw_transformer = FunctionTransformer(keyword_features, validate=False)

def build_preprocess(numeric_cols: List[str]) -> ColumnTransformer:
    word_tfidf = TfidfVectorizer(
        token_pattern=cfg.token_pattern,
        analyzer="word",
        ngram_range=cfg.word_ngram,
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        max_features=cfg.max_features,
        sublinear_tf=True
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("w", word_tfidf, "text"),
            ("kw", Pipeline([("kw", kw_transformer), ("sc", StandardScaler(with_mean=False))]), "text"),
            ("num", Pipeline([("sc", StandardScaler(with_mean=False))]), numeric_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )
    return preprocess

def build_models(preprocess: ColumnTransformer):
    clf = LogisticRegression(solver="saga", max_iter=5000, class_weight="balanced", C=2.0, random_state=42)
    reg_all  = Ridge(alpha=2.0, random_state=42)
    reg_cut  = Ridge(alpha=2.0, random_state=42)
    reg_hike = Ridge(alpha=2.0, random_state=42)

    clf_pipe = Pipeline([("prep", clone(preprocess)), ("clf", clf)])
    reg_all_pipe  = Pipeline([("prep", clone(preprocess)), ("reg", reg_all)])
    reg_cut_pipe  = Pipeline([("prep", clone(preprocess)), ("reg", reg_cut)])
    reg_hike_pipe = Pipeline([("prep", clone(preprocess)), ("reg", reg_hike)])
    return clf_pipe, reg_cut_pipe, reg_hike_pipe, reg_all_pipe

def walk_forward_fast(X, y_bps, y_dir, dates, clf_pipe, reg_cut_pipe, reg_hike_pipe, reg_all_pipe, n_splits: int):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    y_pred = np.full(len(y_bps), np.nan, dtype=float)
    dir_pred = np.full(len(y_bps), np.nan, dtype=float)
    conf_pred = np.full(len(y_bps), np.nan, dtype=float)
    residuals = []
    residuals_by_dir = {-1: [], 0: [], 1: []}

    for tr, te in tscv.split(X):
        w_tr = exp_time_weights(dates.iloc[tr])
        clf_pipe.fit(X.iloc[tr], y_dir[tr], clf__sample_weight=w_tr)
        d_hat = clf_pipe.predict(X.iloc[te]).astype(int)

        if hasattr(clf_pipe.named_steps["clf"], "predict_proba"):
            conf_te = clf_pipe.predict_proba(X.iloc[te]).max(axis=1)
        else:
            conf_te = np.ones(len(te), dtype=float)

        reg_all_pipe.fit(X.iloc[tr], y_bps[tr], reg__sample_weight=w_tr)
        tr_cut = tr[y_dir[tr] == -1]; tr_hike = tr[y_dir[tr] == 1]
        can_cut = len(tr_cut) >= 8; can_hike = len(tr_hike) >= 8

        if can_cut: reg_cut_pipe.fit(X.iloc[tr_cut], y_bps[tr_cut], reg__sample_weight=exp_time_weights(dates.iloc[tr_cut]))
        if can_hike: reg_hike_pipe.fit(X.iloc[tr_hike], y_bps[tr_hike], reg__sample_weight=exp_time_weights(dates.iloc[tr_hike]))

        for j, idx in enumerate(te):
            d = int(d_hat[j]); conf_pred[idx] = float(conf_te[j])
            pred_all = float(reg_all_pipe.predict(X.iloc[[idx]])[0])
            if d == 0: pred_cond = 0.0
            elif d == -1: pred_cond = float(reg_cut_pipe.predict(X.iloc[[idx]])[0]) if can_cut else cfg.fallback_cut_bps
            else: pred_cond = float(reg_hike_pipe.predict(X.iloc[[idx]])[0]) if can_hike else cfg.fallback_hike_bps

            pred = cfg.blend_cond * pred_cond + cfg.blend_all * pred_all
            pred = float(clip_bps(pred))
            y_pred[idx] = pred; dir_pred[idx] = d
            res = float(y_bps[idx] - pred)
            residuals.append(res); residuals_by_dir[d].append(res)

    return y_pred, dir_pred, conf_pred, residuals, residuals_by_dir

def compute_interval(residuals, residuals_by_dir, q_lo=cfg.q_lo, q_hi=cfg.q_hi):
    def qpair(arr):
        arr = np.asarray(arr, dtype=float)
        if arr.size < 20: return (-250.0, 250.0)
        return (float(np.quantile(arr, q_lo)), float(np.quantile(arr, q_hi)))
    overall = qpair(residuals)
    by_dir = {d: qpair(residuals_by_dir.get(d, np.array([]))) for d in [-1,0,1]}
    return overall, by_dir

def widen_interval(lo, hi, vol_ratio, conf):
    vr = float(vol_ratio) if np.isfinite(vol_ratio) else 1.0
    vr = max(0.5, min(vr, cfg.vol_cap))
    mult_vol = 1.0 + cfg.vol_factor * max(0.0, (vr - 1.0))
    c = float(conf) if np.isfinite(conf) else 1.0
    unc = max(0.0, 1.0 - c)
    mult_unc = 1.0 + cfg.unc_factor * unc
    mult = mult_vol * mult_unc
    return (lo * mult, hi * mult)

def fit_final(X, y_bps, y_dir, dates, clf_pipe, reg_cut_pipe, reg_hike_pipe, reg_all_pipe):
    w_all = exp_time_weights(dates)
    clf_pipe.fit(X, y_dir, clf__sample_weight=w_all)
    reg_all_pipe.fit(X, y_bps, reg__sample_weight=w_all)
    cut_idx = np.where(y_dir == -1)[0]
    hike_idx = np.where(y_dir == 1)[0]
    if len(cut_idx) >= 8: reg_cut_pipe.fit(X.iloc[cut_idx], y_bps[cut_idx], reg__sample_weight=exp_time_weights(dates.iloc[cut_idx]))
    if len(hike_idx) >= 8: reg_hike_pipe.fit(X.iloc[hike_idx], y_bps[hike_idx], reg__sample_weight=exp_time_weights(dates.iloc[hike_idx]))

def build_next_row(df_hist: pd.DataFrame, next_text: str) -> pd.DataFrame:
    last = df_hist.iloc[-1]
    y = df_hist["y_bps"].values.astype(float)
    prev_change_bps = float(clip_bps(last["y_bps"]))
    hold_streak = int(last["hold_streak"] + (1 if prev_change_bps == 0 else 0))
    
    w = cfg.trend_window
    roll_mean = float(pd.Series(y).tail(w).mean())
    roll_std = float(pd.Series(y).tail(w).std(ddof=0)) if len(y) >= 2 else 0.0
    roll_slope = float(rolling_slope(y, w)[-1])
    momentum = float(prev_change_bps - roll_mean)
    
    base = float(df_hist["roll_std_bps"].median()) if len(df_hist) else 1.0
    vol_ratio = float(roll_std / base) if base > 0 else 1.0

    row = pd.DataFrame([{
        "text": normalize_tr_text(next_text),
        "prev_change_bps": prev_change_bps,
        "prev_abs_change": abs(prev_change_bps),
        "prev_sign": int(np.sign(prev_change_bps)),
        "hold_streak": hold_streak,
        "mean_abs_last3": float(np.mean(np.abs(df_hist["y_bps"].tail(3).values))),
        "days_since_prev": float(last["days_since_prev"]),
        "roll_mean_bps": roll_mean,
        "roll_std_bps": roll_std,
        "roll_slope_bps": roll_slope,
        "momentum_bps": momentum,
        "vol_ratio": vol_ratio
    }])
    return row

def predict_next(df_hist, next_text, clf_pipe, reg_cut_pipe, reg_hike_pipe, reg_all_pipe, overall_q, by_dir_q):
    row = build_next_row(df_hist, next_text)
    d_hat = int(clf_pipe.predict(row)[0])
    
    conf = 1.0
    proba_map = {}
    if hasattr(clf_pipe.named_steps["clf"], "predict_proba"):
        proba = clf_pipe.predict_proba(row)[0]
        classes = clf_pipe.named_steps["clf"].classes_
        proba_map = {int(c): float(p) for c,p in zip(classes, proba)}
        conf = float(np.max(proba))

    pred_all = float(reg_all_pipe.predict(row)[0])
    if d_hat == 0: pred_cond = 0.0
    elif d_hat == -1: 
        try: pred_cond = float(reg_cut_pipe.predict(row)[0])
        except: pred_cond = cfg.fallback_cut_bps
    else: 
        try: pred_cond = float(reg_hike_pipe.predict(row)[0])
        except: pred_cond = cfg.fallback_hike_bps

    pred = cfg.blend_cond * pred_cond + cfg.blend_all * pred_all
    pred = float(clip_bps(pred))

    lo_d, hi_d = by_dir_q.get(d_hat, overall_q)
    lo_o, hi_o = overall_q
    lo = min(lo_d, lo_o); hi = max(hi_d, hi_o)
    lo_w, hi_w = widen_interval(lo, hi, vol_ratio=float(row["vol_ratio"].iloc[0]), conf=conf)
    
    return {
        "pred_direction": {-1:"İNDİRİM", 0:"SABİT", 1:"ARTIRIM"}[d_hat],
        "direction_confidence": conf,
        "direction_proba": proba_map,
        "pred_change_bps": pred,
        "pred_interval_lo": float(clip_bps(pred + lo_w)),
        "pred_interval_hi": float(clip_bps(pred + hi_w))
    }

def prepare_ml_dataset(df_logs, df_market):
    if df_logs.empty or df_market.empty: return pd.DataFrame()
    if 'period_date' in df_logs.columns:
        df_logs = df_logs.copy()
        df_logs['period_date'] = pd.to_datetime(df_logs['period_date'])
        df_logs['Donem'] = df_logs['period_date'].dt.strftime('%Y-%m')
    if 'Donem' not in df_market.columns: return pd.DataFrame()
    
    df = pd.merge(df_logs, df_market, on="Donem", how="left")
    
    # Colab mantığına uygun text clean
    df['text'] = df['text_content'].fillna("").apply(normalize_tr_text)
    
    # FAİZ DEĞİŞİMİ HESAPLAMA (PPK Faizi'nden otomatik)
    if 'PPK Faizi' in df.columns:
        df['rate_change_bps'] = df['PPK Faizi'].diff().fillna(0.0) * 100
        # Colab'da kullanılan kolon isimleri: date, text, rate_change_bps
        return pd.DataFrame({
            "date": df['period_date'],
            "text": df['text'],
            "rate_change_bps": df['rate_change_bps']
        }).dropna()
    
    return pd.DataFrame()

class AdvancedMLPredictor:
    def __init__(self):
        self.clf_pipe = None
        self.reg_pipes = {}
        self.intervals = {}
        self.df_hist = None
        self.metrics = {}
        
    def train(self, ml_df):
        if not HAS_ML_DEPS: return "Kütüphane eksik"
        
        df = add_features(ml_df, trend_window=cfg.trend_window)
        self.df_hist = df # Tahmin için lazım
        
        numeric_cols = [
            "prev_change_bps", "prev_abs_change", "prev_sign",
            "hold_streak", "mean_abs_last3", "days_since_prev",
            "roll_mean_bps", "roll_std_bps", "roll_slope_bps", "momentum_bps", "vol_ratio"
        ]
        
        X = df[["text"] + numeric_cols]
        y_bps = df["y_bps"].values.astype(float)
        y_dir = df["y_dir"].values.astype(int)
        dates = df["date"]
        
        preprocess = build_preprocess(numeric_cols)
        clf, r_cut, r_hike, r_all = build_models(preprocess)
        
        # Walk Forward Validation
        n_splits = choose_splits(len(df))
        y_p, d_p, c_p, res, res_dir = walk_forward_fast(X, y_bps, y_dir, dates, clf, r_cut, r_hike, r_all, n_splits)
        
        # Store predictions in df_hist for visualization
        self.df_hist['predicted_bps'] = y_p

        # Metrics
        mask = ~np.isnan(y_p)
        if np.any(mask):
            self.metrics['mae'] = mean_absolute_error(y_bps[mask], y_p[mask])
            self.metrics['rmse'] = rmse_metric(y_bps[mask], y_p[mask])
            self.metrics['acc'] = np.mean(y_dir[mask] == d_p[mask].astype(int))
        
        # Fit Final Models
        overall_q, by_dir_q = compute_interval(res, res_dir)
        self.intervals = {'overall': overall_q, 'by_dir': by_dir_q}
        
        fit_final(X, y_bps, y_dir, dates, clf, r_cut, r_hike, r_all)
        
        self.clf_pipe = clf
        self.reg_pipes = {'cut': r_cut, 'hike': r_hike, 'all': r_all}
        return "OK"

    def predict(self, text):
        if self.df_hist is None or self.clf_pipe is None: return None
        return predict_next(
            self.df_hist, text, 
            self.clf_pipe, self.reg_pipes['cut'], self.reg_pipes['hike'], self.reg_pipes['all'],
            self.intervals['overall'], self.intervals['by_dir']
        )

# =============================================================================
# 7D. TEXT-AS-DATA HYBRID + CPI (ENGLISH TEXT) — delta_bp prediction
#   X = TFIDF(word+char)(text) + lags(policy/delta) + CPI features (lagged)
#   FIXES:
#     - Numeric pipeline: SimpleImputer(median) + StandardScaler(with_mean=False)
#     - Predict row: NaN/Inf-safe construction
# =============================================================================

def _safe_slope(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 3:
        return 0.0
    x = np.arange(len(s), dtype=float)
    y = s.values.astype(float)
    try:
        return float(np.polyfit(x, y, 1)[0])
    except Exception:
        return 0.0


def textasdata_prepare_df_hybrid_cpi(
    df_logs: pd.DataFrame,
    df_market: pd.DataFrame,
    text_col: str = "text_content",
    date_col: str = "period_date",
    y_col: str = "delta_bp",
    rate_col: str = "policy_rate",
) -> pd.DataFrame:
    """
    HYBRID + CPI dataset builder.
    - English texts -> we'll use stop_words='english' in model.
    - CPI columns expected in df_market: 'Yıllık TÜFE' (and optionally 'Aylık TÜFE')
    - IMPORTANT: CPI is lagged (t-1) to avoid leakage.
    """
    if df_logs is None or df_logs.empty:
        return pd.DataFrame()

    df = df_logs.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    # month key
    df["Donem"] = df[date_col].dt.strftime("%Y-%m")

    # text
    df["text"] = (
        df.get(text_col, "")
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # core numeric
    df["policy_rate"] = pd.to_numeric(df.get(rate_col), errors="coerce")
    df["delta_bp"] = pd.to_numeric(df.get(y_col), errors="coerce")

    # --- merge CPI / market ---
    m = df_market.copy() if isinstance(df_market, pd.DataFrame) else pd.DataFrame()

    if not m.empty:
        if "Donem" not in m.columns and "SortDate" in m.columns:
            m["Donem"] = pd.to_datetime(m["SortDate"], errors="coerce").dt.strftime("%Y-%m")

        # normalize numeric
        if "Yıllık TÜFE" in m.columns:
            m["cpi_yoy"] = pd.to_numeric(m["Yıllık TÜFE"], errors="coerce")
        else:
            m["cpi_yoy"] = np.nan

        if "Aylık TÜFE" in m.columns:
            m["cpi_mom"] = pd.to_numeric(m["Aylık TÜFE"], errors="coerce")
        else:
            # fallback: aylık yoksa yoy farkını momentum gibi kullan
            m["cpi_mom"] = m["cpi_yoy"].diff()

        m = m[["Donem", "cpi_yoy", "cpi_mom"]].drop_duplicates(subset=["Donem"])
        df = pd.merge(df, m, on="Donem", how="left")
    else:
        df["cpi_yoy"] = np.nan
        df["cpi_mom"] = np.nan

    # --- CPI features (LAGGED to avoid leakage) ---
    df = df.sort_values(date_col).reset_index(drop=True)
    df["cpi_yoy_lag1"] = df["cpi_yoy"].shift(1)
    df["cpi_mom_lag1"] = df["cpi_mom"].shift(1)
    df["cpi_trend3_lag1"] = df["cpi_yoy"].rolling(3).apply(lambda x: _safe_slope(pd.Series(x)), raw=False).shift(1)

    # --- rate/delta dynamics (lagged) ---
    df["policy_rate_lag1"] = df["policy_rate"].shift(1)
    df["delta_bp_lag1"] = df["delta_bp"].shift(1)
    df["delta_bp_lag3"] = df["delta_bp"].rolling(3).mean().shift(1)
    df["policy_rate_trend"] = df["policy_rate"].rolling(3).apply(lambda x: _safe_slope(pd.Series(x)), raw=False).shift(1)

    # hold streak (how many consecutive holds before this meeting)
    streak = []
    cur = 0
    prev_changes = df["delta_bp"].shift(1).fillna(0.0).values
    for v in prev_changes:
        if float(v) == 0.0:
            cur += 1
        else:
            cur = 0
        streak.append(cur)
    df["hold_streak"] = np.array(streak, dtype=int)

    df["prev_sign"] = np.sign(df["delta_bp_lag1"].fillna(0.0)).astype(int)
    df["mean_abs_last3"] = df["delta_bp"].shift(1).abs().rolling(3).mean()

    # days since prev meeting
    med = float(df[date_col].diff().dt.days.dropna().median()) if len(df) > 2 else 30.0
    df["days_since_prev"] = df[date_col].diff().dt.days.fillna(med).clip(lower=0).astype(float)

    out = df.rename(columns={date_col: "period_date"})[
        [
            "period_date",
            "text",
            "delta_bp",
            "policy_rate",

            # lags
            "policy_rate_lag1", "delta_bp_lag1", "delta_bp_lag3", "policy_rate_trend",
            "hold_streak", "prev_sign", "mean_abs_last3", "days_since_prev",

            # CPI (lagged)
            "cpi_yoy_lag1", "cpi_mom_lag1", "cpi_trend3_lag1",
        ]
    ].copy()

    # drop rows where core features are missing (avoid breaking model)
    need = [
        "policy_rate_lag1", "delta_bp_lag1", "delta_bp_lag3", "policy_rate_trend",
        "hold_streak", "prev_sign", "mean_abs_last3", "days_since_prev",
        "cpi_yoy_lag1", "cpi_mom_lag1", "cpi_trend3_lag1",
    ]
    out = out.dropna(subset=need).reset_index(drop=True)
    return out


def train_textasdata_hybrid_cpi_ridge(
    df_td: pd.DataFrame,
    min_df: int = 2,
    alpha: float = 10.0,
    n_splits: int = 6,
    word_ngram=(1, 2),
    char_ngram=(3, 5),
    max_features_word: int = 12000,
    max_features_char: int = 20000,
) -> dict:
    if not HAS_ML_DEPS:
        return {}
    if df_td is None or df_td.empty:
        return {}

    df = df_td.copy().dropna(subset=["period_date"]).sort_values("period_date").reset_index(drop=True)
    df_train = df.dropna(subset=["delta_bp"]).copy()
    if df_train["delta_bp"].notna().sum() < 10:
        return {}

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # >>> ADD THESE IMPORTS + CLASS
    from sklearn.base import BaseEstimator, TransformerMixin
    import scipy.sparse as sp

    class SparseFiniteFixer(BaseEstimator, TransformerMixin):
        """Replace NaN/Inf in dense or sparse matrices with 0.0 (safety net)."""
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            if sp.issparse(X):
                X = X.tocsr(copy=True)
                data = X.data
                bad = ~np.isfinite(data)
                if bad.any():
                    data[bad] = 0.0
                    X.data = data
                    X.eliminate_zeros()
                return X
            else:
                X = np.array(X, copy=True)
                X[~np.isfinite(X)] = 0.0
                return X
    # <<< END ADD

    num_cols = [
        "policy_rate_lag1", "delta_bp_lag1", "delta_bp_lag3", "policy_rate_trend",
        "hold_streak", "prev_sign", "mean_abs_last3", "days_since_prev",
        "cpi_yoy_lag1", "cpi_mom_lag1", "cpi_trend3_lag1"
    ]

    X = df_train[["text"] + num_cols].copy()
    y = df_train["delta_bp"].astype(float).values

    preprocess = ColumnTransformer(
        transformers=[
            ("w_tfidf", TfidfVectorizer(
                stop_words="english",
                ngram_range=word_ngram,
                min_df=max(1, int(min_df)),
                max_df=0.95,
                max_features=int(max_features_word),
                sublinear_tf=True
            ), "text"),
            ("c_tfidf", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=char_ngram,
                min_df=max(1, int(min_df)),
                max_df=0.95,
                max_features=int(max_features_char),
                sublinear_tf=True
            ), "text"),
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
            ]), num_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )

    # >>> IMPORTANT: add finite fixer between prep and ridge
    pipe = Pipeline([
        ("prep", preprocess),
        ("finite", SparseFiniteFixer()),
        ("ridge", Ridge(alpha=float(alpha), random_state=42)),
    ])

    tscv = TimeSeriesSplit(n_splits=min(int(n_splits), max(2, len(df_train) // 3)))
    pred = np.full(len(df_train), np.nan, dtype=float)

    for tr, te in tscv.split(X):
        pipe.fit(X.iloc[tr], y[tr])
        pred[te] = pipe.predict(X.iloc[te])

    mask = np.isfinite(pred)
    metrics = {"n": int(mask.sum())}
    if mask.sum() >= 3:
        metrics["mae"] = float(mean_absolute_error(y[mask], pred[mask]))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y[mask], pred[mask])))
        metrics["r2"] = float(r2_score(y[mask], pred[mask]))
    else:
        metrics.update({"mae": np.nan, "rmse": np.nan, "r2": np.nan})

    pred_df = df_train[["period_date", "delta_bp", "policy_rate"]].copy()
    pred_df["pred_delta_bp"] = pred

    pipe.fit(X, y)

    coef_df = pd.DataFrame()
    try:
        wvec = pipe.named_steps["prep"].named_transformers_["w_tfidf"]
        feats = wvec.get_feature_names_out()
        coefs = pipe.named_steps["ridge"].coef_.ravel()
        w_dim = len(feats)
        coef_df = pd.DataFrame({"feature": feats, "coef": coefs[:w_dim]})
        coef_df["abs"] = coef_df["coef"].abs()
    except Exception:
        coef_df = pd.DataFrame()

    return {
        "model": pipe,
        "pred_df": pred_df,
        "metrics": metrics,
        "coef_df": coef_df,
        "num_cols": num_cols
    }



# --- helper: NaN/Inf-safe float ---
def _sf(x, default=0.0) -> float:
    try:
        v = float(pd.to_numeric(x, errors="coerce"))
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def predict_textasdata_hybrid_cpi(model_pack: dict, df_hist: pd.DataFrame, text: str) -> dict:
    """
    For single-text prediction we need the latest known macro/history values from df_hist.
    df_hist should be the output of textasdata_prepare_df_hybrid_cpi (sorted).
    """
    if not model_pack or "model" not in model_pack:
        return {}
    pipe = model_pack["model"]

    txt = (text or "").strip()
    if len(txt) < 30:
        return {}

    if df_hist is None or df_hist.empty:
        return {}

    df_hist = df_hist.sort_values("period_date").reset_index(drop=True)
    last = df_hist.iloc[-1]

    # NaN-safe last state
    last_policy = _sf(last.get("policy_rate", np.nan), default=0.0)
    last_delta  = _sf(last.get("delta_bp", np.nan), default=0.0)

    row = pd.DataFrame([{
        "text": txt,

        "policy_rate_lag1": last_policy,
        "delta_bp_lag1": last_delta,
        "delta_bp_lag3": _sf(df_hist["delta_bp"].tail(3).mean(), default=0.0),
        "policy_rate_trend": _sf(_safe_slope(df_hist["policy_rate"].tail(3)), default=0.0),

        "hold_streak": _sf(last.get("hold_streak", 0.0), default=0.0),
        "prev_sign": float(np.sign(last_delta)),
        "mean_abs_last3": _sf(df_hist["delta_bp"].tail(3).abs().mean(), default=0.0),
        "days_since_prev": _sf(last.get("days_since_prev", 30.0), default=30.0),

        "cpi_yoy_lag1": _sf(last.get("cpi_yoy_lag1", np.nan), default=0.0),
        "cpi_mom_lag1": _sf(last.get("cpi_mom_lag1", np.nan), default=0.0),
        "cpi_trend3_lag1": _sf(last.get("cpi_trend3_lag1", np.nan), default=0.0),
    }])

    # final safety sweep
    row = row.replace([np.inf, -np.inf], np.nan)
    for c in row.columns:
        if c != "text":
            row[c] = pd.to_numeric(row[c], errors="coerce")
    row = row.fillna(0.0)

    pred_bp = float(pipe.predict(row)[0])
    return {"pred_delta_bp": pred_bp}



# =============================================================================
# 8. CENTRAL BANK RoBERTa ENTEGRASYONU (mrince / STABLE)
# Model: mrince/CBRT-RoBERTa-HawkishDovish-Classifier
# =============================================================================

import gc
import numpy as np
import pandas as pd
import streamlit as st

# --- MODEL CACHE ---
@st.cache_resource(show_spinner=False)
def load_roberta_pipeline():
    """
    mrince/CBRT-RoBERTa-HawkishDovish-Classifier pipeline
    top_k=None -> tüm sınıf skorlarını döndürür.
    """
    if not HAS_TRANSFORMERS:
        return None
    try:
        from transformers import pipeline
        model_name = "mrince/CBRT-RoBERTa-HawkishDovish-Classifier"
        clf = pipeline("text-classification", model=model_name, top_k=None)
        return clf
    except Exception as e:
        print(f"Model Yükleme Hatası: {e}")
        return None


# --- AUTO LABEL MAP (en kritik fix) ---
@st.cache_resource(show_spinner=False)
def _mrince_label_map():
    """
    Modelin LABEL_* -> {HAWK, DOVE, NEUT} eşlemesini otomatik bulur.
    Çünkü pratikte LABEL_1/2/0 sabit varsayımı bazı durumlarda ters çıkabiliyor.
    """
    clf = load_roberta_pipeline()
    # güvenli fallback (çalışmazsa)
    fallback = {"HAWK": "LABEL_1", "DOVE": "LABEL_2", "NEUT": "LABEL_0"}
    if clf is None:
        return fallback

    tests = {
        "HAWK": "The committee will tighten monetary policy further and deliver additional rate hikes.",
        "DOVE": "The committee will begin monetary easing soon and deliver rate cuts in the coming meetings.",
        "NEUT": "The committee decided to keep the policy rate unchanged."
    }

    def best_label(text: str) -> str:
        out = clf(text)
        if isinstance(out, list) and out and isinstance(out[0], list):
            out = out[0]
        if not isinstance(out, list) or not out:
            return ""
        best = max(out, key=lambda x: float(x.get("score", 0.0)))
        return str(best.get("label", "")).strip()

    hawk_lab = best_label(tests["HAWK"])
    dove_lab = best_label(tests["DOVE"])
    neut_lab = best_label(tests["NEUT"])

    # çakışma olursa fallback’e dön
    labs = [hawk_lab, dove_lab, neut_lab]
    if any(l == "" for l in labs) or len(set(labs)) < 3:
        return fallback

    return {"HAWK": hawk_lab, "DOVE": dove_lab, "NEUT": neut_lab}


def _normalize_label_mrince(raw_label: str) -> str:
    """
    Otomatik çıkarılan label_map ile normalize eder.
    """
    m = _mrince_label_map()
    lbl = str(raw_label).strip()

    if lbl == m.get("HAWK"):
        return "HAWK"
    if lbl == m.get("DOVE"):
        return "DOVE"
    if lbl == m.get("NEUT"):
        return "NEUT"

    # fallback heuristik
    low = lbl.lower()
    if "hawk" in low:
        return "HAWK"
    if "dove" in low:
        return "DOVE"
    if "neut" in low or "neutral" in low:
        return "NEUT"
    if "label_0" in low:
        return "NEUT"
    return "NEUT"


def stance_3class_from_diff(diff: float, deadband: float = 0.15) -> str:
    """
    diff = P(HAWK) - P(DOVE)
    3 etiket: Şahin / Güvercin / Nötr
    """
    if diff >= deadband:
        return "🦅 Şahin"
    if diff <= -deadband:
        return "🕊️ Güvercin"
    return "⚖️ Nötr"


# ==============================================================================
# KANONİK DOKÜMAN SİNYALİ (cümle-bazlı, karar-ağırlıklı)
# ------------------------------------------------------------------------------
# Bilimsel gerekçe:
#  - CentralBankRoBERTa (Pfeifer & Marohl, 2023) CÜMLE düzeyinde eğitilmiştir;
#    duruş endeksleri literatürde (Apel & Blix Grimaldi, Picault & Renault vb.)
#    cümle bazında sınıflandırıp dokümana aggregate edilerek kurulur.
#  - Tek-parça "full-text" tahmin hem dağılım-dışıdır hem de 512 token sınırı
#    nedeniyle metnin yalnızca başını görür. Bu yüzden full-text yalnızca
#    referans/debug amaçlı tutulur; KANONİK sinyal burada üretilir.
#
# Yöntem:
#  - Her cümlenin katkısı diff = P(HAWK)-P(DOVE)'dir; bu zaten GÜVEN-AĞIRLIKLIDIR
#    (düşük güvenli cümlenin |diff|'i küçüktür -> çıplak sayımdan üstündür).
#  - Politika KARAR cümlesi (faiz artış/indirim bağlamı) ekstra ağırlık alır.
#  - Tek, gerekçeli bir deadband (DOC_STANCE_DEADBAND) ile etiketlenir.
# ==============================================================================

DOC_STANCE_DEADBAND = 0.10   # tek, gerekçeli eşik (full-text 0.15 ile cümle 0.05 arası)
DOC_ACTION_WEIGHT = 3.0      # karar cümlesinin diğer cümlelere göre ağırlık çarpanı


def _is_policy_action_sentence(s: str) -> bool:
    """Cümle policy-rate bağlamında bir faiz artış/indirim cümlesi mi?"""
    s = (s or "").lower()
    if not re.search(r"\b(policy rate|interest rate|repo auction rate|one-week repo)\b", s):
        return False
    verbs = ["increase", "increased", "raise", "raised", "hike", "tightening",
             "decrease", "decreased", "lower", "lowered", "cut", "reduce", "reduced", "easing"]
    return any(re.search(rf"\b{re.escape(w)}\b", s) for w in verbs)


def document_signal_from_sentences(df_sent: pd.DataFrame,
                                   full_text: str | None = None,
                                   action_weight_mult: float = DOC_ACTION_WEIGHT,
                                   deadband: float = DOC_STANCE_DEADBAND) -> dict:
    """
    Cümle-bazlı RoBERTa çıktısından TEK kanonik doküman sinyali üretir.

    Dönüş:
      signal     : karar-ağırlıklı, güven-ağırlıklı ortalama diff (~[-1, +1])
      stance     : tek deadband ile etiket (🦅/🕊️/⚖️)
      diff_mean  : ağırlıksız ortalama diff (referans/debug)
      n          : kullanılan cümle sayısı
      hawk_n/dove_n/neut_n, action, real_delta_bp

    Not: 'signal' full-text diff ile AYNI ölçektedir; mevcut postprocess
    (robust z + tanh + EMA) bu sinyalde de sorunsuz çalışır.
    """
    empty = {"signal": 0.0, "stance": "⚖️ Nötr", "diff_mean": 0.0, "n": 0,
             "hawk_n": 0, "dove_n": 0, "neut_n": 0,
             "action": "UNKNOWN", "real_delta_bp": None}

    if df_sent is None or df_sent.empty or "Diff (H-D)" not in df_sent.columns:
        return empty

    d = df_sent.copy()
    d = d[pd.to_numeric(d["Diff (H-D)"], errors="coerce").notna()].copy()
    if d.empty:
        return empty

    diffs = pd.to_numeric(d["Diff (H-D)"], errors="coerce").astype(float).values
    if "Cümle" in d.columns:
        sents = d["Cümle"].astype(str).values
    else:
        sents = np.array([""] * len(diffs))

    # Karar cümlesine ekstra ağırlık -> güven-ağırlıklı diff'ler üzerinde ağırlıklı ortalama
    weights = np.where(
        np.array([_is_policy_action_sentence(s) for s in sents]),
        float(action_weight_mult),
        1.0,
    )
    wsum = float(np.sum(weights)) + 1e-12
    signal = float(np.sum(diffs * weights) / wsum)

    diff_mean = float(np.mean(diffs))
    stance = stance_3class_from_diff(signal, deadband=deadband)

    stance_series = d.get("Duruş", "").astype(str)
    hawk_n = int(stance_series.str.contains("Şahin", na=False).sum())
    dove_n = int(stance_series.str.contains("Güvercin", na=False).sum())
    neut_n = int(stance_series.str.contains("Nötr", na=False).sum())

    try:
        action = detect_policy_action(full_text or "")
    except Exception:
        action = "UNKNOWN"
    try:
        real_delta_bp = extract_delta_bp_from_text(full_text or "")
    except Exception:
        real_delta_bp = None

    return {
        "signal": signal,
        "stance": stance,
        "diff_mean": diff_mean,
        "n": int(len(d)),
        "hawk_n": hawk_n, "dove_n": dove_n, "neut_n": neut_n,
        "action": action,
        "real_delta_bp": real_delta_bp,
    }


def combine_tone_action(tone_signal,
                        delta_bp=None,
                        action_label=None,
                        deadband: float = DOC_STANCE_DEADBAND) -> dict:
    """
    Tonu (iletişim) ve aksiyonu (gerçekleşen faiz kararı) AYRI iki boyut olarak
    birleştirip rejim etiketi üretir: 'Hawkish Cut', 'Dovish Hike', 'Neutral Hold'...

    - Ton yönü: saf cümle sinyalinden (deadband ile).
    - Aksiyon yönü: önce gerçek delta_bp (metinden), yoksa CUT/HIKE/HOLD etiketi.
    """
    # --- Ton yönü ---
    try:
        ts = float(tone_signal)
        if not np.isfinite(ts):
            ts = 0.0
    except Exception:
        ts = 0.0

    if ts >= deadband:
        tone_word, tone_emoji, tone_dir = "Hawkish", "🦅", 1
    elif ts <= -deadband:
        tone_word, tone_emoji, tone_dir = "Dovish", "🕊️", -1
    else:
        tone_word, tone_emoji, tone_dir = "Neutral", "⚖️", 0

    # --- Aksiyon yönü: önce delta_bp, sonra etiket fallback ---
    bp = None
    try:
        if delta_bp is not None and np.isfinite(float(delta_bp)):
            bp = float(delta_bp)
    except Exception:
        bp = None

    action_dir = None
    if bp is not None:
        action_dir = 1 if bp > 0 else (-1 if bp < 0 else 0)
    else:
        lab = (action_label or "").upper()
        if lab == "HIKE":
            action_dir = 1
        elif lab == "CUT":
            action_dir = -1
        elif lab == "HOLD":
            action_dir = 0
        else:
            action_dir = None  # bilinmiyor

    action_word = {1: "Hike", -1: "Cut", 0: "Hold"}.get(action_dir, "?")

    if action_dir is None:
        regime = f"{tone_emoji} {tone_word} (aksiyon ?)"
    else:
        regime = f"{tone_emoji} {tone_word} {action_word}"

    return {
        "regime": regime,
        "tone_word": tone_word,
        "action_word": action_word,
        "tone_dir": tone_dir,
        "action_dir": action_dir,
        "delta_bp": bp,
    }


def analyze_with_roberta(text: str):
    """
    Tek metin için sınıf olasılıkları + diff + basit duruş.
    """
    if not text:
        return None

    clf = load_roberta_pipeline()
    if clf is None:
        return "ERROR"

    truncated_text = str(text)[:1200]  # RAM koruması

    try:
        raw = clf(truncated_text)
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            raw = raw[0]

        scores_map = {"HAWK": 0.0, "DOVE": 0.0, "NEUT": 0.0}
        best_score = -1.0

        for r in (raw or []):
            lbl_raw = r.get("label", "")
            sc = float(r.get("score", 0.0))
            lbl = _normalize_label_mrince(lbl_raw)
            scores_map[lbl] = sc
            best_score = max(best_score, sc)

        h = float(scores_map.get("HAWK", 0.0))
        d = float(scores_map.get("DOVE", 0.0))
        n = float(scores_map.get("NEUT", 0.0))
        diff = h - d

        # === KANONİK TON (saf, karar-ağırlıksız) + AYRI AKSİYON + REJİM ===
        # Ton = cümle diff'lerinin SAF ortalaması (iletişim tonu).
        # Aksiyon = gerçek delta_bp (yoksa CUT/HIKE/HOLD). Rejim = ton × aksiyon.
        # Full-text tek-parça tahmin yalnızca referans/debug.
        diff_fulltext = diff
        stance_fulltext = stance_3class_from_diff(diff_fulltext)

        tone_signal = diff_fulltext      # fallback (cümle çıkmazsa)
        stance_tone = stance_fulltext    # fallback
        doc_diff_mean = None
        net_push = None
        delta_bp = None
        action_label = "UNKNOWN"

        try:
            # cümle analizi TAM metin üzerinde (truncation bias yok)
            df_sent = analyze_sentences_with_roberta(str(text))
            if df_sent is not None and "Diff (H-D)" in df_sent.columns:
                doc = document_signal_from_sentences(df_sent, full_text=str(text))
                if doc.get("n", 0) > 0:
                    tone_signal = float(doc["diff_mean"])   # SAF ton (karar-ağırlıksız)
                    stance_tone = stance_3class_from_diff(tone_signal, deadband=DOC_STANCE_DEADBAND)
                    doc_diff_mean = tone_signal
                    action_label = str(doc.get("action", "UNKNOWN") or "UNKNOWN")
                    delta_bp = doc.get("real_delta_bp", None)

                    summ = summarize_sentence_roberta(df_sent, full_text=str(text))
                    if summ and summ.get("n", 0) > 0:
                        net_push = float((summ.get("pos_sum", 0.0) or 0.0) + (summ.get("neg_sum", 0.0) or 0.0))
        except Exception:
            pass

        ta = combine_tone_action(tone_signal, delta_bp=delta_bp, action_label=action_label)

        return {
            "scores_map": scores_map,
            "best_score": float(best_score),
            # --- KANONİK TON (chart + detay aynı bunu kullanır) ---
            "diff": float(tone_signal),
            "stance": stance_tone,
            # --- AYRI AKSİYON + REJİM ---
            "delta_bp": ta.get("delta_bp"),
            "action_label": action_label,
            "action_dir": ta.get("action_dir"),
            "tone_dir": ta.get("tone_dir"),
            "regime": ta.get("regime"),
            # --- Referans / debug ---
            "diff_fulltext": float(diff_fulltext),
            "stance_full_raw": stance_fulltext,
            "stance_sentence": stance_tone,
            "doc_diff_mean": doc_diff_mean,
            "net_push": net_push,
            "label_map": _mrince_label_map(),
            "h": h, "d": d, "n": n
        }

    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        gc.collect()


def postprocess_ai_series(df: pd.DataFrame,
                          diff_col: str = "Diff (H-D)",
                          span: int = 7,
                          z_scale: float = 2.0,
                          hyst: float = 25.0) -> pd.DataFrame:
    """
    diff -> robust z-score -> tanh -> score
    score -> EMA smoothing
    EMA -> hysteresis regime etiketi
    """
    if df is None or df.empty or diff_col not in df.columns:
        return df

    out = df.copy()
    x = out[diff_col].astype(float)

    med = float(x.median())
    mad = float((x - med).abs().median()) + 1e-6
    z = (x - med) / (1.4826 * mad)

    out["AI Score (Calib)"] = np.tanh(z / float(z_scale)) * 100.0
    out["AI Score (EMA)"] = out["AI Score (Calib)"].ewm(span=span, adjust=False).mean()

    # --- Rejim (Histerezis) ---
    # Not: CB metinlerinde dil yumuşak döner; HOLD dönemlerinde işaret değişimleri
    # sık görülür. Bu nedenle "sign-flip" olduğunda nötre hızlı dönmek gerekir.
    regime: list[str] = []
    prev: str = "⚖️ Nötr"

    neutral_band = float(hyst) * 0.60   # ±15 (hyst=25 iken)
    flip_band = float(hyst) * 0.50      # ±12.5 (hyst=25 iken)

    for v in out["AI Score (EMA)"].values:
        v = float(v)

        # 1) Güçlü rejim girişleri
        if v >= float(hyst):
            prev = "🦅 Şahin"
        elif v <= -float(hyst):
            prev = "🕊️ Güvercin"
        else:
            # 2) Nötr bant: düşük genlikte rejimi nötrle (<= önemli!)
            if abs(v) <= neutral_band:
                prev = "⚖️ Nötr"
            else:
                # 3) İşaret değişimi: önce nötrle (ve yeterince güçlü ise karşı rejime çevir)
                if prev == "🦅 Şahin" and v < 0:
                    prev = "🕊️ Güvercin" if v <= -flip_band else "⚖️ Nötr"
                elif prev == "🕊️ Güvercin" and v > 0:
                    prev = "🦅 Şahin" if v >= flip_band else "⚖️ Nötr"
                # aksi halde prev korunur

        regime.append(prev)

    out["AI Rejim"] = regime
    return out




# =============================================================================
# 8.1 CB-RoBERTa TREND: ADIM ADIM (RAW -> CALIB -> EMA -> HYSTERESIS)
# =============================================================================

def postprocess_ai_series_steps(df: pd.DataFrame,
                               diff_col: str = "Diff (H-D)",
                               span: int = 7,
                               z_scale: float = 2.0,
                               hyst: float = 25.0) -> pd.DataFrame:
    """
    postprocess_ai_series ile aynı mantık, ama ara kolonları da açıkça üretir.

    Üretilen kolonlar:
      - AI Diff (Raw): ham fark (P(HAWK)-P(DOVE))
      - AI z (robust): median+MAD ile robust z-score
      - AI Score (Calib): tanh ile -100..+100 bandına sıkıştırılmış skor
      - AI Score (EMA): EMA ile yumuşatılmış skor
      - AI Rejim: hysteresis ile etiket (🦅/🕊️/⚖️)

    Not: 'postprocess_ai_series' geriye dönük uyumluluk için duruyor.
    """
    if df is None or df.empty or diff_col not in df.columns:
        return df

    out = df.copy()
    x = out[diff_col].astype(float)
    out["AI Diff (Raw)"] = x

    # Robust z-score (median + MAD)
    med = float(x.median())
    mad = float((x - med).abs().median()) + 1e-6
    z = (x - med) / (1.4826 * mad)
    out["AI z (robust)"] = z

    # Calibrasyon: tanh ile band sınırlama (-100..+100)
    out["AI Score (Calib)"] = np.tanh(z / float(z_scale)) * 100.0

    # EMA smoothing
    out["AI Score (EMA)"] = out["AI Score (Calib)"].ewm(span=span, adjust=False).mean()

    # Hysteresis rejim etiketi (postprocess_ai_series ile aynı)
    regime: list[str] = []
    prev: str = "⚖️ Nötr"

    neutral_band = float(hyst) * 0.60
    flip_band = float(hyst) * 0.50

    for v in out["AI Score (EMA)"].values:
        v = float(v)

        if v >= float(hyst):
            prev = "🦅 Şahin"
        elif v <= -float(hyst):
            prev = "🕊️ Güvercin"
        else:
            if abs(v) <= neutral_band:
                prev = "⚖️ Nötr"
            else:
                if prev == "🦅 Şahin" and v < 0:
                    prev = "🕊️ Güvercin" if v <= -flip_band else "⚖️ Nötr"
                elif prev == "🕊️ Güvercin" and v > 0:
                    prev = "🦅 Şahin" if v >= flip_band else "⚖️ Nötr"

        regime.append(prev)

    out["AI Rejim"] = regime
    return out


def create_ai_trend_chart_step(df_res: pd.DataFrame, step: int = 3):
    """
    step:
      0 -> Raw (Diff)
      1 -> + Calib (robust z + tanh)
      2 -> + EMA
      3 -> + Hysteresis (hover'da rejim)
    """
    import plotly.graph_objects as go
    if df_res is None or df_res.empty:
        return None

    df = df_res.copy()

    # Kolon güvenliği: eski df'ler için gerekirse üret
    if "AI Score (EMA)" not in df.columns or "AI Score (Calib)" not in df.columns or "AI Diff (Raw)" not in df.columns:
        df = postprocess_ai_series_steps(df, diff_col="Diff (H-D)", span=7, z_scale=2.0, hyst=25.0)

    step = int(step or 0)
    if step <= 0:
        y_col = "AI Diff (Raw)"
        title = "CB-RoBERTa — Ham Ton Sinyali (cümle-bazlı, saf)"
        yrange = None
    elif step == 1:
        y_col = "AI Score (Calib)"
        title = "CB-RoBERTa — Kalibre Skor (robust z + tanh)"
        yrange = [-110, 110]
    else:
        y_col = "AI Score (EMA)"
        title = "CB-RoBERTa — EMA ile Yumuşatılmış Skor" if step == 2 else "CB-RoBERTa — Duruş Trendi (Calib + EMA + Hysteresis)"
        yrange = [-110, 110]

    y = pd.to_numeric(df.get(y_col), errors="coerce")

    hover_text = None
    if step >= 3 and "AI Rejim" in df.columns and "Duruş" in df.columns:
        hover_text = (df["AI Rejim"].astype(str) + " | " + df["Duruş"].astype(str))
    elif step >= 3 and "AI Rejim" in df.columns:
        hover_text = df["AI Rejim"].astype(str)
    elif "Duruş" in df.columns:
        hover_text = df["Duruş"].astype(str)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Dönem"], y=y,
        mode="lines", name=y_col,
        line=dict(width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df["Dönem"], y=y,
        mode="markers", name="Aylık",
        marker=dict(
            size=11,
            color=y,
            colorscale="RdBu_r",
            cmin=-100 if yrange else None,
            cmax=100 if yrange else None,
            showscale=True,
            colorbar=dict(title=y_col)
        ),
        text=hover_text,
        hovertemplate="<b>%{x}</b><br>Değer: %{y:.2f}<br>%{text}<extra></extra>" if hover_text is not None else "<b>%{x}</b><br>Değer: %{y:.2f}<extra></extra>"
    ))

    fig.add_hline(y=0, line_color="black", opacity=0.25)

    fig.update_layout(
        title=title,
        yaxis=dict(title=y_col, range=yrange),
        height=450,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


def create_tone_action_chart(df_res: pd.DataFrame, step: int = 3):
    """
    Ton × Aksiyon ayrık görünüm:
      - Üst panel: TON çizgisi (radio adımına göre ham/kalibre/EMA), renk = ton.
                   İşaretçi ŞEKLİ = aksiyon (▲ hike, ▼ cut, ■ hold, ● bilinmiyor).
                   Böylece 'Hawkish Cut' = yukarıda duran bir ▼ olarak GÖZE ÇARPAR.
      - Alt panel: gerçek AKSİYON (Δbp) barları.
    Hover'da birleşik rejim etiketi ('🦅 Hawkish Cut') görünür.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if df_res is None or df_res.empty:
        return None

    df = df_res.copy()
    if ("AI Score (EMA)" not in df.columns) or ("AI Diff (Raw)" not in df.columns):
        df = postprocess_ai_series_steps(df, diff_col="Diff (H-D)", span=7, z_scale=2.0, hyst=25.0)

    step = int(step or 0)
    if step <= 0:
        tone_col, tone_lbl, tone_range = "AI Diff (Raw)", "Ham", None
    elif step == 1:
        tone_col, tone_lbl, tone_range = "AI Score (Calib)", "Kalibre", [-110, 110]
    elif step == 2:
        tone_col, tone_lbl, tone_range = "AI Score (EMA)", "EMA", [-110, 110]
    else:
        tone_col, tone_lbl, tone_range = "AI Score (EMA)", "EMA + Histerezis", [-110, 110]

    y_tone = pd.to_numeric(df.get(tone_col), errors="coerce")
    adir = pd.to_numeric(df.get("Aksiyon Yön", np.nan), errors="coerce")
    bp = pd.to_numeric(df.get("Delta BP", np.nan), errors="coerce")
    regime = df.get("Rejim", pd.Series([""] * len(df))).astype(str)

    sym_map = {1: "triangle-up", -1: "triangle-down", 0: "square"}
    symbols = [sym_map.get(int(a), "circle") if pd.notna(a) else "circle" for a in adir]

    hover = []
    for r, t, b in zip(regime, y_tone, bp):
        b_txt = f"{b:+.0f} bp" if pd.notna(b) else "—"
        t_txt = f"{t:.2f}" if pd.notna(t) else "—"
        hover.append(f"{r}<br>Ton={t_txt}<br>Aksiyon={b_txt}")

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.68, 0.32], vertical_spacing=0.07,
        subplot_titles=(f"Ton ({tone_lbl}) — işaret şekli = aksiyon (▲hike ▼cut ■hold)",
                        "Gerçekleşen Aksiyon (Δ bp)")
    )

    # Üst: ton çizgisi
    fig.add_trace(go.Scatter(
        x=df["Dönem"], y=y_tone, mode="lines",
        line=dict(width=2, color="rgba(120,120,120,0.7)"),
        name="Ton", showlegend=False
    ), row=1, col=1)

    # Üst: ton işaretçileri (şekil = aksiyon, renk = ton)
    fig.add_trace(go.Scatter(
        x=df["Dönem"], y=y_tone, mode="markers", name="Dönem", showlegend=False,
        marker=dict(
            size=13, symbol=symbols, color=y_tone, colorscale="RdBu_r",
            cmin=-100 if tone_range else None, cmax=100 if tone_range else None,
            line=dict(width=1, color="white"),
            showscale=True, colorbar=dict(title="Ton", len=0.55, y=0.78)
        ),
        text=hover,
        hovertemplate="<b>%{x}</b><br>%{text}<extra></extra>"
    ), row=1, col=1)
    fig.add_hline(y=0, line_color="black", opacity=0.25, row=1, col=1)

    # Alt: aksiyon barları (Δbp)
    bar_color = ["#c0392b" if (pd.notna(v) and v > 0)
                 else "#2471a3" if (pd.notna(v) and v < 0)
                 else "#95a5a6" for v in bp]
    fig.add_trace(go.Bar(
        x=df["Dönem"], y=bp.fillna(0.0), marker_color=bar_color,
        name="Δ bp", showlegend=False,
        hovertemplate="<b>%{x}</b><br>Δ bp = %{y:+.0f}<extra></extra>"
    ), row=2, col=1)
    fig.add_hline(y=0, line_color="black", opacity=0.25, row=2, col=1)

    fig.update_yaxes(title_text="Ton", range=tone_range, row=1, col=1)
    fig.update_yaxes(title_text="Δ bp", row=2, col=1)
    fig.update_layout(
        title="CB-RoBERTa — Ton × Aksiyon (Hawkish Cut / Dovish Hike yakalama)",
        height=580, bargap=0.55,
        margin=dict(l=20, r=20, t=70, b=20),
        hovermode="x unified"
    )
    return fig


def build_regime_summary_table(df_res: pd.DataFrame) -> pd.DataFrame:
    """Tüm dönemler için Ton + Aksiyon + Rejim özet tablosu (tarihe göre yeni→eski)."""
    if df_res is None or df_res.empty:
        return pd.DataFrame()

    df = df_res.copy()
    out = pd.DataFrame()
    out["Dönem"] = df.get("Dönem")
    out["Ton"] = df.get("Duruş", "")
    out["Ton Sinyali"] = pd.to_numeric(df.get("Diff (H-D)", np.nan), errors="coerce").round(3)
    bp = pd.to_numeric(df.get("Delta BP", np.nan), errors="coerce")
    out["Aksiyon (bp)"] = bp.map(lambda v: f"{v:+.0f}" if pd.notna(v) else "—")
    out["Aksiyon"] = df.get("Aksiyon", "")
    out["Rejim"] = df.get("Rejim", "")

    try:
        out = out.assign(_d=pd.to_datetime(df.get("period_date"), errors="coerce")) \
                 .sort_values("_d", ascending=False).drop(columns=["_d"])
    except Exception:
        pass
    return out.reset_index(drop=True)


def calculate_ai_trend_series(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Tüm geçmişi tarar (mrince) ve sonra postprocess ile trend endeksi üretir.
    """
    if df_all is None or df_all.empty:
        return pd.DataFrame()

    clf = load_roberta_pipeline()
    if clf is None:
        return pd.DataFrame()

    df_all = df_all.copy()
    df_all["period_date"] = pd.to_datetime(df_all["period_date"], errors="coerce")
    df_all = df_all.dropna(subset=["period_date"]).sort_values("period_date")

    results = []
    st.toast("AI analiz başladı...", icon="⏳")
    pb = st.progress(0)
    total = len(df_all)

    for i, (_, row) in enumerate(df_all.iterrows()):
        pb.progress((i + 1) / total)

        txt = str(row.get("text_content", "") or "")
        if len(txt) < 10:
            continue

        res = analyze_with_roberta(txt)
        if not isinstance(res, dict):
            continue

        dt = row["period_date"]
        period = dt.strftime("%Y-%m")

        scores = res.get("scores_map", {}) or {}
        h = float(scores.get("HAWK", 0.0))
        d = float(scores.get("DOVE", 0.0))
        n = float(scores.get("NEUT", 0.0))
        # 'diff' artık KANONİK SAF TON sinyalidir (karar-ağırlıksız).
        # Aksiyon (delta_bp / etiket) AYRI kolonlarda izlenir.
        diff = float(res.get("diff", h - d))
        diff_fulltext = float(res.get("diff_fulltext", h - d))

        _bp = res.get("delta_bp", None)
        _bp = float(_bp) if (_bp is not None) else np.nan
        _adir = res.get("action_dir", None)
        _adir = float(_adir) if (_adir is not None) else np.nan

        results.append({
            "Dönem": period,
            "period_date": dt,
            "Şahin Olasılık": h,        # full-text raw prob (referans)
            "Güvercin Olasılık": d,     # full-text raw prob (referans)
            "Nötr Olasılık": n,
            "Diff (H-D)": diff,                 # KANONİK SAF TON (grafik bunu çizer)
            "Diff (Full-text)": diff_fulltext,  # referans/debug
            "Duruş": str(res.get("stance", "")),     # ton duruşu (Şahin/Güvercin/Nötr)
            "Delta BP": _bp,                          # gerçek aksiyon (bp)
            "Aksiyon": str(res.get("action_label", "UNKNOWN")),
            "Aksiyon Yön": _adir,                     # +1 hike / -1 cut / 0 hold / NaN ?
            "Rejim": str(res.get("regime", "")),      # ör. "🦅 Hawkish Cut"
            "Güven": float(res.get("best_score", 0.0)),
        })

        gc.collect()

    pb.empty()
    st.toast("AI analiz tamamlandı!", icon="✅")

    out = pd.DataFrame(results)
    if out.empty:
        return out

    out = out.sort_values("period_date").reset_index(drop=True)
    out = postprocess_ai_series_steps(out, diff_col="Diff (H-D)", span=7, z_scale=2.0, hyst=25.0)
    return out


def create_ai_trend_chart(df_res: pd.DataFrame):
    import plotly.graph_objects as go
    if df_res is None or df_res.empty:
        return None

    df = df_res.copy()
    y_col = "AI Score (EMA)" if "AI Score (EMA)" in df.columns else "Diff (H-D)"
    y = df[y_col].astype(float)

    hover_text = None
    if "AI Rejim" in df.columns and "Duruş" in df.columns:
        hover_text = (df["AI Rejim"].astype(str) + " | " + df["Duruş"].astype(str))
    elif "AI Rejim" in df.columns:
        hover_text = df["AI Rejim"].astype(str)
    elif "Duruş" in df.columns:
        hover_text = df["Duruş"].astype(str)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Dönem"], y=y,
        mode="lines", name="AI Trend",
        line=dict(width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df["Dönem"], y=y,
        mode="markers", name="Aylık",
        marker=dict(
            size=11,
            color=y,
            colorscale="RdBu_r",
            cmin=-100, cmax=100,
            showscale=True,
            colorbar=dict(title=y_col)
        ),
        text=hover_text,
        hovertemplate="<b>%{x}</b><br>Skor: %{y:.1f}<br>%{text}<extra></extra>"
    ))

    fig.add_hline(y=0, line_color="black", opacity=0.25)

    fig.update_layout(
        title="CB-RoBERTa — Duruş Trendi (Calib + EMA + Hysteresis)",
        yaxis=dict(title=y_col, range=[-110, 110]),
        height=450,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


import re
import gc
import pandas as pd
import numpy as np

def _fallback_sentence_split(text: str) -> list[str]:
    # Basit ama sağlam: . ! ? ; : ve satır sonlarından böl
    t = re.sub(r"\s+", " ", str(text)).strip()
    if not t:
        return []
    parts = re.split(r"(?<=[\.\!\?\;\:])\s+", t)
    # Çok kısa parçaları temizle
    return [p.strip() for p in parts if p and len(p.strip()) >= 10]

import re
import gc
import pandas as pd
import numpy as np

def _fallback_sentence_split(text: str) -> list[str]:
    t = re.sub(r"\s+", " ", str(text)).strip()
    if not t:
        return []
    parts = re.split(r"(?<=[\.\!\?\;\:])\s+", t)
    return [p.strip() for p in parts if p and len(p.strip()) >= 10]

def analyze_sentences_with_roberta(text: str) -> pd.DataFrame:
    """
    Streamlit Cloud için güvenli:
    - split_sentences_nlp boşsa fallback
    - çok cümlede batch çalışır
    - model call patlarsa UI'da görebil diye ERROR satırı döndürür
    - boş dönmek yerine tanılayıcı (diagnostic) satır üretir
    """
    # Tutarlı kolon seti
    cols = ["Cümle", "Duruş", "Diff (H-D)", "HAWK", "DOVE", "NEUT", "Tag"]
    if not text or not str(text).strip():
        return pd.DataFrame([{"Cümle": "Metin boş", "Tag": "ERROR"}], columns=cols)

    clf = load_roberta_pipeline()
    if clf is None:
        return pd.DataFrame([{"Cümle": "Pipeline yüklenemedi (transformers/torch veya model load hatası)", "Tag": "ERROR"}], columns=cols)

    t = str(text).strip()
    if len(t) < 30:
        return pd.DataFrame([{"Cümle": "Metin çok kısa (>=30 karakter önerilir)", "Tag": "WARN"}], columns=cols)

    # 1) Sentence split
    try:
        sentences = split_sentences_nlp(t)
        sentences = [s.strip() for s in (sentences or [])]
    except Exception:
        sentences = []

    if not sentences:
        sentences = _fallback_sentence_split(t)

    # daha yumuşak filtre: en az 2 kelime veya 20 karakter
    sentences = [s for s in sentences if s and (len(s.split()) >= 2 or len(s) >= 20)]
    if not sentences:
        return pd.DataFrame([{"Cümle": "Cümle ayrıştırıldı ama filtre sonrası cümle kalmadı", "Tag": "WARN"}], columns=cols)

    # cümleleri kısalt (transformers truncation’a ek olarak)
    sentences = [s[:500] for s in sentences]

    # çok uzarsa limitle
    max_sent = 160
    if len(sentences) > max_sent:
        sentences = sentences[:max_sent]

    # 2) Predict in batches (Cloud RAM)
    try:
        rows = []
        batch_size = 16

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]

            # truncation/padding burada kritik
            preds = clf(batch, truncation=True)

            for sent, pred in zip(batch, preds):
                # normalize => list-of-dicts
                if isinstance(pred, list) and pred and isinstance(pred[0], list):
                    pred = pred[0]
                if isinstance(pred, dict):
                    pred = [pred]

                scores_map = {"HAWK": 0.0, "DOVE": 0.0, "NEUT": 0.0}
                for r in (pred or []):
                    lbl = _normalize_label_mrince(r.get("label", ""))
                    sc = float(r.get("score", 0.0))
                    scores_map[lbl] = sc

                h = float(scores_map["HAWK"])
                d = float(scores_map["DOVE"])
                n = float(scores_map["NEUT"])
                diff = h - d

                rows.append({
                    "Cümle": sent,
                    "Duruş": stance_3class_from_diff(diff),
                    "Diff (H-D)": float(diff),
                    "HAWK": h,
                    "DOVE": d,
                    "NEUT": n,
                    "Tag": "",
                })

        df = pd.DataFrame(rows, columns=cols)
        if not df.empty:
            df = df.sort_values("Diff (H-D)", ascending=False).reset_index(drop=True)
        else:
            df = pd.DataFrame([{"Cümle": "Model çalıştı ama sonuç üretilmedi (beklenmeyen durum)", "Tag": "WARN"}], columns=cols)

        return df

    except Exception as e:
        return pd.DataFrame([{"Cümle": f"RoBERTa sentence analizi hata: {e}", "Tag": "ERROR"}], columns=cols)


# --- Policy action (CUT/HIKE/HOLD) & Rate-cut özet yardımcıları ---

_RATE_CUT_KWS = [
    "rate cut", "cut rates", "lowered", "lowering", "reduce", "reduced", "easing",
    "faiz indir", "faiz indirim", "indirime", "indirim",
]
_RATE_HIKE_KWS = [
    "rate hike", "hike", "raised", "raising", "increase", "increased", "tightening",
    "faiz art", "faiz artır", "artırım", "sıkılaş",
]
_RATE_HOLD_KWS = [
    "kept", "maintained", "unchanged", "hold", "pause",
    "sabit", "değişiklik", "korun", "aynı seviyede",
]


def detect_policy_action(text: str) -> str:
    """Metinden kaba bir aksiyon etiketi döndürür: CUT / HIKE / HOLD / UNKNOWN.

    Öncelik sırası:
    1) 'from X percent to Y percent' gibi ifadeleri policy rate bağlamında sayısal kıyasla çöz.
    2) Aynı cümlede 'policy rate / interest rate / repo auction rate' + (increase/raise vs lower/reduce) bağlamı.
    3) Daha zayıf anahtar kelime fallback.
    """
    raw = text or ""
    t = raw.lower()

    # 1) Sayısal "from ... to ..." kalıbı (İngilizce metinlerde net)
    #    Sadece faiz bağlamında yakalamaya çalışıyoruz.
    #    ör: "increase the policy rate ... from 8.5 percent to 15 percent"
    from_to = re.search(
        r"(policy rate|interest rate|repo auction rate|one-week repo).*?from\s+(\d+(?:\.\d+)?)\s*percent\s+to\s+(\d+(?:\.\d+)?)\s*percent",
        t,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if from_to:
        a = float(from_to.group(2))
        b = float(from_to.group(3))
        if b > a:
            return "HIKE"
        if b < a:
            return "CUT"
        return "HOLD"

    # 1b) Daha genel 'from X percent to Y percent' -> delta_bp (ama sadece metinde policy bağlamı varsa)
    try:
        if re.search(r"\b(policy rate|interest rate|repo auction rate|one-week repo)\b", t):
            dbp = extract_delta_bp_from_text(raw)
            if dbp is not None:
                if dbp > 0:
                    return "HIKE"
                if dbp < 0:
                    return "CUT"
                return "HOLD"
    except Exception:
        pass

    # 2) Cümle bazlı bağlam taraması
    sents = split_sentences_tr(raw)

    def _has_context(sent: str, action_words: list[str]) -> bool:
        s = sent.lower()
        if not re.search(r"\b(policy rate|interest rate|repo auction rate|one-week repo)\b", s):
            return False
        # action kelimelerini kelime sınırında ara (auction gibi false positive'leri önlemek için)
        return any(re.search(rf"\b{re.escape(w)}\b", s) for w in action_words)

    # yükseliş
    if any(_has_context(s, ["increase", "increased", "raise", "raised", "hike", "tightening"]) for s in sents):
        return "HIKE"

    # düşüş
    if any(_has_context(s, ["decrease", "decreased", "lower", "lowered", "cut", "reduce", "reduced", "easing"]) for s in sents):
        return "CUT"

    # 3) Fallback: (daha zayıf) ama word-boundary kullan
    if any(re.search(rf"\b{re.escape(k)}\b", t) for k in ["unchanged", "maintained", "kept", "hold", "pause"]):
        return "HOLD"
    if any(re.search(rf"\b{re.escape(k)}\b", t) for k in ["increase", "increased", "raised", "hike"]):
        return "HIKE"
    if any(re.search(rf"\b{re.escape(k)}\b", t) for k in ["decrease", "decreased", "lowered", "cut"]):
        return "CUT"

    return "UNKNOWN"




def extract_delta_bp_from_text(text: str) -> float | None:
    """
    Metinden 'from X percent to Y percent' kalıbını yakalayıp gerçek delta_bp döndürür.
    Örn: from 8.5 percent to 15 percent  -> +650.0
    Bulamazsa None.
    """
    raw = text or ""
    t = raw.lower()

    m = re.search(
        r"(policy rate|interest rate|repo auction rate|one-week repo).*?from\s+(\d+(?:\.\d+)?)\s*percent\s+to\s+(\d+(?:\.\d+)?)\s*percent",
        t,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        # bağlam olmadan da deneyelim (bazı metinlerde 'policy rate' ayrı satır olabilir)
        m2 = re.search(
            r"from\s+(\d+(?:\.\d+)?)\s*percent\s+to\s+(\d+(?:\.\d+)?)\s*percent",
            t,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not m2:
            return None
        a = float(m2.group(1)); b = float(m2.group(2))
        return (b - a) * 100.0

    a = float(m.group(2))
    b = float(m.group(3))
    return (b - a) * 100.0



def summarize_sentence_roberta(df_sent: pd.DataFrame, full_text: str | None = None) -> dict:
    """
    analyze_sentences_with_roberta çıktısından özet:
    - sınıf sayıları
    - diff ortalama / pos sum / neg sum
    - aksiyon cümlesi (rate hike / rate cut): puan + ağırlık + cümle

    Not:
    - 'reduce inflation' gibi ifadeler FAİZ İNDİRİMİ demek değildir. Bu yüzden aksiyon tespiti 'policy rate' bağlamı arar.
    """
    if df_sent is None or df_sent.empty or "Diff (H-D)" not in df_sent.columns:
        return {"n": 0}

    d = df_sent.copy()
    d = d[pd.to_numeric(d["Diff (H-D)"], errors="coerce").notna()].copy()
    if d.empty:
        return {"n": 0}

    diffs = pd.to_numeric(d["Diff (H-D)"], errors="coerce").astype(float).values
    abs_sum = float(np.sum(np.abs(diffs))) + 1e-12

    stance = d.get("Duruş", "").astype(str)
    hawk_n = int((stance.str.contains("Şahin", na=False)).sum())
    dove_n = int((stance.str.contains("Güvercin", na=False)).sum())
    neut_n = int((stance.str.contains("Nötr", na=False)).sum())

    diff_mean = float(np.mean(diffs))
    pos_sum = float(np.sum(diffs[diffs > 0]))
    neg_sum = float(np.sum(diffs[diffs < 0]))  # negatif değer (dove itişi)

    action = detect_policy_action(full_text or "")

    # --- Aksiyon cümlesi: policy rate bağlamında "increase/raise" vs "lower/cut" ---
    def is_hike_sentence(s: str) -> bool:
        s = (s or "").lower()
        if not re.search(r"\b(policy rate|interest rate|repo auction rate|one-week repo)\b", s):
            return False
        return any(re.search(rf"\b{re.escape(w)}\b", s) for w in ["increase", "increased", "raise", "raised", "hike", "tightening"])

    def is_cut_sentence(s: str) -> bool:
        s = (s or "").lower()
        if not re.search(r"\b(policy rate|interest rate|repo auction rate|one-week repo)\b", s):
            return False
        return any(re.search(rf"\b{re.escape(w)}\b", s) for w in ["decrease", "decreased", "lower", "lowered", "cut", "reduce", "reduced", "easing"])

    s_lc = d["Cümle"].astype(str)

    hike_rows = d[s_lc.apply(is_hike_sentence)].copy()
    cut_rows = d[s_lc.apply(is_cut_sentence)].copy()

    # Aksiyon cümleleri içindeki toplam |diff| (lokal ağırlık için)
    try:
        _cand_rows = pd.concat([hike_rows, cut_rows], ignore_index=True) if (not hike_rows.empty or not cut_rows.empty) else pd.DataFrame()
        _cand_diffs = pd.to_numeric(_cand_rows.get("Diff (H-D)", pd.Series(dtype=float)), errors="coerce")
        abs_sum_action = float(np.nansum(np.abs(_cand_diffs))) + 1e-12
    except Exception:
        abs_sum_action = 1e-12

    action_points = 0.0
    action_weight = 0.0
    action_weight_local = 0.0
    action_sentence = "—"
    action_label = "—"

    # HIKE => en pozitif diff'li aksiyon cümlesini seç
    if action == "HIKE" and not hike_rows.empty:
        hike_rows["Diff (H-D)"] = pd.to_numeric(hike_rows["Diff (H-D)"], errors="coerce")
        best = hike_rows.sort_values("Diff (H-D)", ascending=False).iloc[0]
        best_diff = float(best["Diff (H-D)"])
        action_points = float(max(0.0, best_diff) * 100.0)
        action_weight = float(abs(best_diff) / abs_sum)
        action_weight_local = float(abs(best_diff) / abs_sum_action)
        action_weight_local = float(abs(best_diff) / abs_sum_action)
        action_sentence = str(best["Cümle"])
        action_label = "HIKE"

    # CUT => en negatif diff'li aksiyon cümlesini seç
    elif action == "CUT" and not cut_rows.empty:
        cut_rows["Diff (H-D)"] = pd.to_numeric(cut_rows["Diff (H-D)"], errors="coerce")
        best = cut_rows.sort_values("Diff (H-D)", ascending=True).iloc[0]
        best_diff = float(best["Diff (H-D)"])
        action_points = float(max(0.0, -best_diff) * 100.0)  # pozitif puan göster
        action_weight = float(abs(best_diff) / abs_sum)
        action_weight_local = float(abs(best_diff) / abs_sum_action)
        action_sentence = str(best["Cümle"])
        action_label = "CUT"

    # Eğer aksiyon UNKNOWN ama aksiyon cümlesi yakalanabiliyorsa heuristik:
    else:
        # iki taraftan en "güçlü" cümleyi seç
        cand = []
        if not hike_rows.empty:
            hike_rows["Diff (H-D)"] = pd.to_numeric(hike_rows["Diff (H-D)"], errors="coerce")
            r = hike_rows.sort_values("Diff (H-D)", ascending=False).iloc[0]
            cand.append(("HIKE", float(r["Diff (H-D)"]), str(r["Cümle"])))
        if not cut_rows.empty:
            cut_rows["Diff (H-D)"] = pd.to_numeric(cut_rows["Diff (H-D)"], errors="coerce")
            r = cut_rows.sort_values("Diff (H-D)", ascending=True).iloc[0]
            cand.append(("CUT", float(r["Diff (H-D)"]), str(r["Cümle"])))
        if cand:
            # mutlak diff en büyük olanı al
            label, diffv, sent = sorted(cand, key=lambda x: abs(x[1]), reverse=True)[0]
            action_label = label
            if label == "HIKE":
                action_points = float(max(0.0, diffv) * 100.0)
            else:
                action_points = float(max(0.0, -diffv) * 100.0)
            action_weight = float(abs(diffv) / abs_sum)
            action_weight_local = float(abs(diffv) / abs_sum_action)
            action_sentence = sent

    return {
        "n": int(len(d)),
        "hawk_n": hawk_n,
        "dove_n": dove_n,
        "neut_n": neut_n,
        "diff_mean": diff_mean,
        "pos_sum": pos_sum,
        "neg_sum": neg_sum,
        "policy_action": action,  # üst seviye aksiyon etiketi
        "action_label": action_label,  # hangi cümle seçildi
        "action_points": action_points,
        "action_weight": action_weight,
        "action_weight_local": action_weight_local,
        "action_sentence": action_sentence,
    }


# =============================================================================
# ==  CB-ROBERTA SONUÇ ÖNBELLEĞİ + CÜMLE/KONU ANALİZLERİ  =====================
# =============================================================================
# Neden var:
#   Model çıktısı şimdiye kadar yalnızca st.session_state'te tutuluyordu; session
#   ölünce kayboluyor, her yenilemede ~60 doküman x ~40 cümle yeniden modelden
#   geçiyordu. Artık cümle düzeyi çıktı Supabase'e yazılır.
#
# Neyin saklandığı (ve neyin SAKLANMADIĞI):
#   [+] Cümle düzeyi model çıktısı (H/D/N)      -> roberta_sentences
#   [+] Doküman özeti (ton, sayım, aksiyon)     -> roberta_doc_cache (türetilmiş)
#   [-] Kalibrasyon / EMA / histerezis          -> SAKLANMAZ
#       Bunlar tüm serinin dağılımına bağlıdır; yeni bir PPK kaydı eklendiğinde
#       geçmiş dönemlerin skorları da değişir. Saklanırsa sessizce yanlış geçmiş
#       üretilir. Her yüklemede postprocess_ai_series_steps ile yeniden hesaplanır.
#
# Geçersiz kılma iki anahtarla:
#   text_hash        -> metin düzenlendi mi
#   PIPELINE_VERSION -> skorlama mantığı değişti mi
#   İkisi de tutuyorsa model bir daha çalışmaz; biri tutmuyorsa YALNIZCA o kayıt
#   yeniden hesaplanır.
# =============================================================================

# =============================================================================
# 10. CB-ROBERTA ÖNBELLEĞİ — SABİTLER
# =============================================================================

#: Yazma sürümü. Yalnızca MODEL çıktısını etkileyen bir değişiklikte bump et
#: (model değişimi, cümle bölme mantığı, olasılık normalizasyonu).
#:
#: DİKKAT: Tema sözlüğünü değiştirmek artık bump GEREKTİRMEZ. Tema etiketi
#: okuma anında `sentence` metninden yeniden hesaplanır (bkz. fetch_sentences),
#: veritabanındaki `theme_label` kolonu yok sayılır. Böylece sözlük düzeltmeleri
#: anında yansır ve pahalı model taraması tekrarlanmaz.
PIPELINE_VERSION = "v2"

#: Model çıktısı birbiriyle AYNI olan sürümler. Bunların hepsi "taze" sayılır ve
#: birlikte okunur. v1 ile v2 arasındaki tek fark tema sözlüğüydü; tema artık
#: okuma anında hesaplandığı için ikisi eşdeğerdir.
COMPATIBLE_VERSIONS = ("v1", "v2")

MODEL_HD = "mrince/CBRT-RoBERTa-HawkishDovish-Classifier"
MODEL_AGENT = "Moritz-Pfeifer/CentralBankRoBERTa-agent-classifier"

TBL_SENT = "roberta_sentences"
TBL_DOC = "roberta_doc_cache"

#: Supabase tek istekte varsayılan 1000 satır döndürür — sayfalama şart.
_PAGE = 1000
#: Insert chunk (payload boyutu için)
_CHUNK = 400

AGENT_ORDER = ["Households", "Firms", "Financial Sector", "Government", "Central Bank"]

AGENT_TR = {
    "Households": "Hanehalkı",
    "Firms": "Firmalar",
    "Financial Sector": "Finansal Sektör",
    "Government": "Kamu",
    "Central Bank": "Merkez Bankası",
}

# Şahin = kırmızı (üst), Güvercin = mavi (alt) — ana dashboard ile aynı konvansiyon
DIVERGING = [[0.0, "#1f4e9c"], [0.5, "#f2f2f2"], [1.0, "#c0392b"]]

AGENT_COLORS = {
    "Households": "#2e86c1",
    "Firms": "#28b463",
    "Financial Sector": "#8e44ad",
    "Government": "#e67e22",
    "Central Bank": "#c0392b",
    "Belirsiz": "#95a5a6",
}


# =============================================================================
# 11. TEMA SÖZLÜĞÜ (deterministik, model değil)
# =============================================================================
# İlgili Kesim sınıflandırıcısı "bu cümle kimin durumuna dair" sorusunu cevaplar;
# "neyi söylüyor" sorusu için ayrı bir katman gerekiyor. Bu katman kasten sözlük tabanlıdır:
# denetlenebilir, tekrar üretilebilir ve TCMB metinlerinin dar sözcük dağarcığında
# kümeleme tabanlı konu modellerinden daha kararlıdır.

# Kalıplar kasten DAR tutulur. Jenerik kelimeler (committee, price, bank, growth)
# TCMB metninde her yerde geçtiği için sözlüğü işe yaramaz hale getiriyordu:
# her cümle her temaya eşleşiyor, kazananı da rastgele bir sayım farkı belirliyordu.
THEME_PATTERNS: dict[str, list[str]] = {
    "Enflasyon Görünümü": [
        # "price" tek başına ARANMAZ: "price stability" politika hedefi,
        # "pricing behavior" beklenti temasıdır.
        r"\binflation\b", r"\bdisinflation\b", r"\bcpi\b", r"consumer prices",
        r"price (level|increase|pressure|development)", r"underlying trend",
        r"enflasyon", r"dezenflasyon", r"fiyat(lar|ları|ında)\b",
        r"fiyat (artış|gelişme|baskı)",
    ],
    "Talep & Aktivite": [
        # "credit growth" / "loan growth" kredi temasıdır, talep değil.
        r"domestic demand", r"aggregate demand", r"economic activity",
        r"(?<!credit\s)(?<!loan\s)\bgrowth\b", r"output gap", r"\bemployment\b",
        r"labou?r market", r"\bconsumption\b", r"\binvestment\b",
        r"\btalep\b", r"\bbüyüme\b", r"istihdam", r"iktisadi faaliyet",
    ],
    "Kur & Dış Denge": [
        # "import" -> "important" eşleşmesini engellemek için kelime sınırı şart.
        r"exchange rate", r"\blira\b", r"\bcurrency\b", r"current account",
        r"external (demand|balance|financ)", r"\bexports?\b", r"\bimports?\b",
        r"\breserves?\b", r"döviz kur\w*", r"\bkur(u|un)?\b",
        r"cari (açık|denge|işlemler)", r"ihracat", r"ithalat", r"rezerv",
    ],
    "Kredi & Aktarım": [
        # DİKKAT: burada "bank" tek başına ARANMAZ. Önceki sürümde
        # r"bank(ing)? (sector|system)?" kalıbı, parantezler opsiyonel olduğu için
        # çıplak "Bank" kelimesine de eşleşiyordu; "Central Bank" ifadesi hemen her
        # TCMB cümlesinde geçtiği için faiz kararı cümleleri yanlışlıkla bu temaya
        # düşüyordu.
        r"\bcredit\b", r"\bloans?\b", r"\bbanks\b", r"\bbanking (sector|system)\b",
        r"\bliquidity\b", r"monetary transmission", r"\bfunding\b", r"\bdeposits?\b",
        r"kredi", r"likidite", r"mevduat", r"aktarım",
    ],
    "Beklentiler & İletişim": [
        # "forward" tek başına ARANMAZ ("going forward", "brought forward").
        r"expectation", r"pricing behavio", r"\banchor\w*", r"forward guidance",
        r"communicat\w+", r"\bsurvey\b",
        # Karar çerçevesi/iletişim ilkesi cümleleri ("predictable, data-driven and
        # transparent framework") aksi halde hiçbir temaya düşmüyordu.
        r"\bpredictab\w+", r"data-driven", r"\btransparen\w+", r"\bframework\b",
        r"beklenti", r"fiyatlama davranış", r"çıpa", r"öngörülebilir", r"şeffaf",
    ],
    "Politika Duruşu": [
        # "committee" ARANMAZ: konuşan tarafı gösterir, konuyu değil — neredeyse
        # her cümlede geçtiği için tüm cümleleri bu temaya çekiyordu.
        r"policy rate", r"monetary policy stance", r"monetary tightness",
        r"\btighten(ing|ed)?\b", r"\beasing\b", r"macroprudential",
        r"price stability", r"decided to", r"meeting-by-meeting", r"step size",
        r"politika faizi", r"para politikası", r"\bsıkı(laş\w*|lık)?\b",
        r"makroihtiyati", r"fiyat istikrar\w*",
    ],
    "Finansal İstikrar & Risk": [
        r"financial stability", r"risk premium", r"volatilit", r"uncertaint",
        r"geopolitical", r"global financial", r"finansal istikrar", r"risk primi",
        r"belirsizlik", r"oynaklık",
    ],
}

_THEME_COMPILED = {
    k: [re.compile(p, re.IGNORECASE) for p in v] for k, v in THEME_PATTERNS.items()
}

THEME_ORDER = list(THEME_PATTERNS.keys()) + ["Diğer"]

# Karar cümlesi önceliği.
# Faiz kararını bildiren cümle, içinde başka temalara ait kelimeler geçse bile
# ("Central Bank", "deposit rate", "overnight lending") her zaman politika duruşudur.
# Sayım yarışına sokmak yerine deterministik olarak öne alınır.
_DECISION_OVERRIDE = re.compile(
    r"("
    r"decided to|has decided|"
    # "...lending rate ... at/to/from 40 percent" — sabit tutma kararı da karardır
    r"\b(policy|lending|borrowing|repo|deposit|funding)\s+rate[s]?\b[^.]{0,120}"
    r"\b(from|to|at)\b\s*\d|"
    r"\b(maintained|kept|raised|lowered|reduced|increased|cut)\b[^.]{0,60}\brate[s]?\b|"
    r"politika faizini|faiz(i|ini|ler)?\s*(oranını)?\s*(artır|indir|düşür|yükselt|sabit)"
    r")",
    re.IGNORECASE,
)


def theme_hits(sentence: str) -> dict[str, int]:
    """Cümlede her temanın kaç kalıbının eşleştiğini döndürür (denetim için)."""
    s = str(sentence or "")
    if not s.strip():
        return {}
    return {t: c for t, c in
            ((th, sum(1 for p in pats if p.search(s)))
             for th, pats in _THEME_COMPILED.items())
            if c > 0}


def assign_themes(sentence: str) -> list[str]:
    """
    ÇOK ETİKETLİ tema ataması: cümlenin değindiği TÜM temaları döndürür.

    Neden tek etiket yetmiyor:
      "The tight monetary policy stance ... until price stability is achieved ...
       through demand, exchange rate, and expectation channels."
    Bu cümle aynı anda politika duruşu, talep, kur ve beklenti hakkındadır. Tek
    etiket zorunluluğunda üçü siliniyor ve kazananı sözlükteki YAZIM SIRASI
    belirliyordu (2-2 beraberlikte ilk yazılan kazanır) — yani etiket, cümlenin
    anlamına değil benim dosyadaki satır sıramla belirleniyordu. Nesnel değil.

    Çok etikette paylar toplamı %100'ü aşar; okuma da değişir:
      tek etiket -> "metnin yüzde kaçı bu konuya AYRILDI"   (kompozisyon)
      çok etiket -> "cümlelerin yüzde kaçı bu konuya DEĞİNDİ" (kapsam)

    İstisna: saf karar cümlesi ("faizi %40.5'ten %39.5'e indirdi") yalnızca
    politika duruşudur, orada çoklu etiket anlamsızdır.
    """
    s = str(sentence or "")
    if not s.strip():
        return ["Diğer"]
    if _DECISION_OVERRIDE.search(s):
        return ["Politika Duruşu"]
    hits = theme_hits(s)
    return sorted(hits.keys()) if hits else ["Diğer"]


def assign_theme(sentence: str) -> str:
    """
    TEK etiket ataması (kompozisyon görünümü için).

    Sıra:
      1) Karar cümlesi mi? -> doğrudan "Politika Duruşu"
      2) Değilse en çok kalıp eşleşen tema kazanır
      3) Beraberlikte THEME_ORDER sırası belirler — bu KEYFİDİR, bu yüzden
         analizde çok etiketli görünüm tercih edilmelidir.
      4) Hiç eşleşme yoksa "Diğer"
    """
    s = str(sentence or "")
    if not s.strip():
        return "Diğer"
    if _DECISION_OVERRIDE.search(s):
        return "Politika Duruşu"
    hits = theme_hits(s)
    if not hits:
        return "Diğer"
    mx = max(hits.values())
    tied = [t for t, c in hits.items() if c == mx]
    for t in THEME_ORDER:
        if t in tied:
            return t
    return tied[0]


# =============================================================================
# 12. İLGİLİ KESİM (ECONOMIC AGENT) SINIFLANDIRICISI — CentralBankRoBERTa
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_agent_pipeline():
    """
    Moritz-Pfeifer/CentralBankRoBERTa-agent-classifier

    NE ÖLÇER: "Bu cümlenin ekonomik içeriği kimin durumuna dair?"
    NE ÖLÇMEZ: "Bu metin kime hitap ediyor / hedef kitlesi kim?"

    Model, Pfeifer & Marohl (2023) tarafından "ne, kimin için iyi" mantığıyla
    eğitilmiştir. Kendi verdikleri örnek: "ücretler beklentinin ötesinde artıyor"
    cümlesi, ücreti ALAN hanehalkı için olumlu, ücreti ÖDEYEN firmalar için
    olumsuzdur. Dolayısıyla bir enflasyon cümlesi çoğunlukla "Households" çıkar:
    tüketici fiyatlarını ödeyen ve reel geliri aşınan taraf hanehalkıdır. Faiz
    kararı ve duruş taahhüdü cümleleri ise "Central Bank" çıkar.

    Bu yüzden panelde bu boyut "İlgili Kesim" olarak adlandırılır; "muhatap"
    (hedef kitle) anlamına GELMEZ.

    Not: Bu, mrince modelinden AYRI ikinci bir RoBERTa (~500 MB). Streamlit Cloud'un
    ücretsiz katmanında iki modeli aynı anda RAM'de tutmak sınırda kalabilir; bu
    yüzden yalnızca senkron sırasında yüklenir ve sonrasında serbest bırakılabilir
    (`release_agent_pipeline`).
    """
    if not HAS_TRANSFORMERS:
        return None
    try:
        from transformers import pipeline
        return pipeline("text-classification", model=MODEL_AGENT, top_k=None)
    except Exception as e:  # pragma: no cover
        print(f"[agent-classifier] yüklenemedi: {e}")
        return None


def release_agent_pipeline() -> None:
    """Senkron bittikten sonra RAM'i geri ver."""
    try:
        load_agent_pipeline.clear()
    except Exception:
        pass
    gc.collect()


_AGENT_PROBES = {
    "Households": "Real incomes of households declined as consumer prices increased faster than wages.",
    "Firms": "Firms reported weaker demand and scaled back their investment plans for the coming quarter.",
    "Financial Sector": "Banks' capital ratios remain strong and credit growth in the banking sector has moderated.",
    "Government": "The government's budget deficit widened and public debt issuance increased this year.",
    "Central Bank": "The Committee decided to raise the policy rate and will maintain a tight monetary policy stance.",
}


@st.cache_resource(show_spinner=False)
def _agent_label_map() -> dict[str, str]:
    """
    Ham model etiketini (LABEL_0..LABEL_4 ya da açık isim) kanonik isme eşler.
    Önce config.id2label denenir; anlamlı değilse prob cümleleriyle otomatik çıkarım.
    _mrince_label_map ile aynı savunmacı desen.
    """
    clf = load_agent_pipeline()
    if clf is None:
        return {}

    # 1) Modelin kendi id2label'ı anlamlıysa onu kullan
    try:
        id2label = getattr(clf.model.config, "id2label", {}) or {}
        vals = [str(v) for v in id2label.values()]
        if vals and not all(v.upper().startswith("LABEL_") for v in vals):
            out = {}
            for v in vals:
                key = v.strip().replace("_", " ").title()
                key = {"Financial": "Financial Sector", "Cb": "Central Bank"}.get(key, key)
                out[v] = key if key in AGENT_ORDER else v
            if len(set(out.values()) & set(AGENT_ORDER)) >= 3:
                return out
    except Exception:
        pass

    # 2) Prob cümleleriyle otomatik eşleme
    def _best(text: str) -> str:
        try:
            out = clf(text, truncation=True)
        except Exception:
            return ""
        if isinstance(out, list) and out and isinstance(out[0], list):
            out = out[0]
        if isinstance(out, dict):
            out = [out]
        if not out:
            return ""
        return str(max(out, key=lambda x: float(x.get("score", 0.0))).get("label", "")).strip()

    mapping: dict[str, str] = {}
    for canon, probe in _AGENT_PROBES.items():
        raw = _best(probe)
        if raw and raw not in mapping:
            mapping[raw] = canon
    return mapping


def _normalize_agent_label(raw_label: str) -> str:
    m = _agent_label_map()
    lbl = str(raw_label).strip()
    if lbl in m:
        return m[lbl]
    low = lbl.lower()
    for canon in AGENT_ORDER:
        if canon.lower().split()[0] in low:
            return canon
    return "Belirsiz"


# =============================================================================
# 13. CÜMLE SKORLAMA (sıra korunarak)
# =============================================================================
# analyze_sentences_with_roberta çıktıyı diff'e göre SIRALAR ve 160 cümlede
# keser. Ton haritası için cümlenin metindeki YERİ gerektiğinden burada sırayı
# koruyan ayrı bir skorlayıcı var. Etiket normalizasyonu utils'ten devralınır,
# böylece iki yol aynı kanonik tanımı kullanır.

def _split_ordered(text: str) -> list[str]:
    t = str(text or "").strip()
    if not t:
        return []
    try:
        sents = split_sentences_nlp(t) or []
    except Exception:
        sents = []
    if not sents:
        sents = _fallback_sentence_split(t)
    sents = [s.strip() for s in sents if s and (len(s.split()) >= 2 or len(s) >= 20)]
    return [s[:500] for s in sents]


def score_sentences_ordered(text: str, with_agent: bool = True) -> pd.DataFrame:
    """
    Tek dokümanı cümlelere böler, sırayı koruyarak skorlar.

    Kolonlar: sent_idx, sent_total, sentence, hawk, dove, neut, diff,
              agent_label, agent_conf, theme_label
    """
    cols = ["sent_idx", "sent_total", "sentence", "hawk", "dove", "neut",
            "diff", "agent_label", "agent_conf", "theme_label"]

    clf = load_roberta_pipeline()
    sents = _split_ordered(text)
    if clf is None or not sents:
        return pd.DataFrame(columns=cols)

    agent_clf = load_agent_pipeline() if with_agent else None
    total = len(sents)
    rows: list[dict] = []
    bs = 16

    for i in range(0, total, bs):
        batch = sents[i:i + bs]

        try:
            preds = clf(batch, truncation=True)
        except Exception:
            preds = [[] for _ in batch]

        agent_preds: list = [[] for _ in batch]
        if agent_clf is not None:
            try:
                agent_preds = agent_clf(batch, truncation=True)
            except Exception:
                agent_preds = [[] for _ in batch]

        for j, sent in enumerate(batch):
            # --- şahin/güvercin ---
            p = preds[j] if j < len(preds) else []
            if isinstance(p, list) and p and isinstance(p[0], list):
                p = p[0]
            if isinstance(p, dict):
                p = [p]
            sm = {"HAWK": 0.0, "DOVE": 0.0, "NEUT": 0.0}
            for r in (p or []):
                sm[_normalize_label_mrince(r.get("label", ""))] = float(r.get("score", 0.0))

            # --- ilgili kesim (ekonomik aktör) ---
            a_label, a_conf = None, None
            ap = agent_preds[j] if j < len(agent_preds) else []
            if isinstance(ap, list) and ap and isinstance(ap[0], list):
                ap = ap[0]
            if isinstance(ap, dict):
                ap = [ap]
            if ap:
                best = max(ap, key=lambda x: float(x.get("score", 0.0)))
                a_label = _normalize_agent_label(best.get("label", ""))
                a_conf = float(best.get("score", 0.0))

            rows.append({
                "sent_idx": i + j,
                "sent_total": total,
                "sentence": sent,
                "hawk": sm["HAWK"],
                "dove": sm["DOVE"],
                "neut": sm["NEUT"],
                "diff": sm["HAWK"] - sm["DOVE"],
                "agent_label": a_label,
                "agent_conf": a_conf,
                "theme_label": assign_theme(sent),
            })

        gc.collect()

    return pd.DataFrame(rows, columns=cols)


# =============================================================================
# 14. ÖNBELLEK G/Ç
# =============================================================================

def text_fingerprint(text: str) -> str:
    """Metnin normalize edilmiş SHA-256'sı. Boşluk/biçim değişikliği hash'i bozmaz."""
    t = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    return hashlib.sha256(t.encode("utf-8")).hexdigest()[:32]


def _paged_select(table: str, columns: str = "*", flt: Optional[Callable] = None) -> pd.DataFrame:
    """Supabase 1000 satır limitini sayfalayarak aşar."""
    if not supabase:
        return pd.DataFrame()
    out: list[dict] = []
    start = 0
    while True:
        try:
            q = supabase.table(table).select(columns)
            if flt is not None:
                q = flt(q)
            res = q.range(start, start + _PAGE - 1).execute()
        except Exception:
            break
        data = getattr(res, "data", []) or []
        out.extend(data)
        if len(data) < _PAGE:
            break
        start += _PAGE
    return pd.DataFrame(out)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_doc_cache() -> pd.DataFrame:
    df = _paged_select(TBL_DOC)
    if df.empty:
        return df
    df["period_date"] = pd.to_datetime(df["period_date"], errors="coerce")
    return df


@st.cache_data(ttl=300, show_spinner=False)
def fetch_sentences(version: str = None) -> pd.DataFrame:
    """
    Cümle önbelleğini çeker.

    İki tasarım kararı:

    1) SÜRÜM BİRLEŞTİRME — COMPATIBLE_VERSIONS içindeki tüm sürümler okunur ve
       (log_id, sent_idx) başına en yeni sürüm tutulur. Böylece v1 ile v2 satırları
       karışık durursa panel boş görünmez.

    2) TEMA OKUMA ANINDA HESAPLANIR — veritabanındaki `theme_label` kolonu YOK
       SAYILIR; etiket, saklanan `sentence` metninden yeniden üretilir. Sonuç:
       sözlükteki her düzeltme anında yansır, pahalı model taraması tekrarlanmaz
       ve veritabanı şeması hiç değişmez. `theme_labels` kolonu çok etiketli
       görünüm için liste olarak eklenir.
    """
    versions = COMPATIBLE_VERSIONS if version is None else (version,)
    df = _paged_select(
        TBL_SENT,
        flt=lambda q: q.in_("pipeline_version", list(versions))
                       .order("period_date").order("sent_idx"),
    )
    if df.empty:
        return df

    # Aynı cümlenin birden çok sürümü varsa en yenisini tut
    if "pipeline_version" in df.columns:
        order = {v: i for i, v in enumerate(COMPATIBLE_VERSIONS)}
        df["_vrank"] = df["pipeline_version"].map(order).fillna(-1)
        df = (df.sort_values("_vrank")
                .drop_duplicates(subset=["log_id", "sent_idx"], keep="last")
                .drop(columns=["_vrank"]))

    df["period_date"] = pd.to_datetime(df["period_date"], errors="coerce")
    df["Donem"] = df["period_date"].dt.strftime("%Y-%m")
    for c in ("hawk", "dove", "neut", "diff", "agent_conf"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("sent_idx", "sent_total"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df["agent_label"] = df.get("agent_label", pd.Series(dtype=object)).fillna("Belirsiz")

    # Tema: saklanan değeri değil, güncel sözlüğü kullan
    sents = df["sentence"].astype(str)
    df["theme_labels"] = sents.map(assign_themes)
    df["theme_label"] = sents.map(assign_theme)

    return df.sort_values(["period_date", "sent_idx"]).reset_index(drop=True)


def explode_themes(df_sent: pd.DataFrame) -> pd.DataFrame:
    """
    Çok etiketli görünüm için uzun format: bir cümle, değindiği her tema için
    ayrı satır. Toplam satır sayısı cümle sayısından fazladır — bu beklenen
    davranıştır, "pay" yerine "kapsam" okunur.
    """
    if df_sent is None or df_sent.empty or "theme_labels" not in df_sent.columns:
        return pd.DataFrame()
    d = df_sent.explode("theme_labels").copy()
    d["theme_label"] = d["theme_labels"].astype(str)
    return d.drop(columns=["theme_labels"])


def invalidate_cache_reads() -> None:
    """Yazma sonrası okuma cache'lerini düşür."""
    try:
        fetch_doc_cache.clear()
        fetch_sentences.clear()
    except Exception:
        pass


def _write_sentences(log_id, period_date, text_hash: str, df_sent: pd.DataFrame) -> None:
    if not supabase or df_sent is None or df_sent.empty:
        return
    # Aynı log için eski sürümü temizle (idempotent yeniden hesap)
    try:
        (supabase.table(TBL_SENT)
         .delete()
         .eq("log_id", int(log_id))
         .eq("pipeline_version", PIPELINE_VERSION)
         .execute())
    except Exception:
        pass

    pdate = pd.Timestamp(period_date).strftime("%Y-%m-%d")
    payload = []
    for _, r in df_sent.iterrows():
        payload.append({
            "log_id": int(log_id),
            "period_date": pdate,
            "sent_idx": int(r["sent_idx"]),
            "sent_total": int(r["sent_total"]),
            "sentence": str(r["sentence"])[:2000],
            "hawk": float(r["hawk"]),
            "dove": float(r["dove"]),
            "neut": float(r["neut"]),
            "diff": float(r["diff"]),
            "agent_label": (None if pd.isna(r["agent_label"]) else str(r["agent_label"])),
            "agent_conf": (None if pd.isna(r["agent_conf"]) else float(r["agent_conf"])),
            "theme_label": str(r["theme_label"]),
            "text_hash": text_hash,
            "pipeline_version": PIPELINE_VERSION,
            "model_hd": MODEL_HD,
            "model_agent": MODEL_AGENT,
        })

    for i in range(0, len(payload), _CHUNK):
        try:
            supabase.table(TBL_SENT).insert(payload[i:i + _CHUNK]).execute()
        except Exception as e:  # pragma: no cover
            print(f"[cache] cümle yazma hatası: {e}")


def _write_doc(row: dict) -> None:
    if not supabase:
        return
    try:
        supabase.table(TBL_DOC).upsert(row, on_conflict="log_id").execute()
    except Exception as e:  # pragma: no cover
        print(f"[cache] doküman yazma hatası: {e}")


# =============================================================================
# 15. SENKRON: neyi yeniden hesaplayacağına karar ver
# =============================================================================

def diagnose(df_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Her kayıt için önbellek durumu döndürür.
    durum ∈ {taze, eksik, metin_degisti, surum_eskidi}
    """
    if df_logs is None or df_logs.empty:
        return pd.DataFrame()

    d = df_logs.copy()
    d["period_date"] = pd.to_datetime(d["period_date"], errors="coerce")
    d = d.dropna(subset=["period_date"])
    d["text_content"] = d["text_content"].fillna("").astype(str)
    d = d[d["text_content"].str.len() >= 30]
    d["text_hash"] = d["text_content"].map(text_fingerprint)

    cache = fetch_doc_cache()
    if cache.empty:
        d["durum"] = "eksik"
        d["n_sentences"] = np.nan
        return d[["id", "period_date", "source", "text_hash", "durum", "n_sentences"]]

    c = cache[["log_id", "text_hash", "pipeline_version", "n_sentences"]].copy()
    c = c.rename(columns={"text_hash": "cached_hash"})
    m = d.merge(c, left_on="id", right_on="log_id", how="left")

    def _status(r):
        if pd.isna(r.get("log_id")):
            return "eksik"
        if str(r.get("pipeline_version")) not in COMPATIBLE_VERSIONS:
            return "surum_eskidi"
        if str(r.get("cached_hash")) != str(r.get("text_hash")):
            return "metin_degisti"
        return "taze"

    m["durum"] = m.apply(_status, axis=1)
    return m[["id", "period_date", "source", "text_hash", "durum", "n_sentences"]]


def missing_periods(df_logs: pd.DataFrame) -> list:
    """
    Önbellekte taze karşılığı OLMAYAN kayıtların dönemleri ("YYYY-MM" listesi).

    Yeni bir PPK kaydı eklendiğinde önbellek otomatik güncellenmez — model
    çalıştırmak pahalı olduğu için bu bilinçli bir tercih. Ama sessizce eksik
    kalması kötü: grafik hiçbir uyarı vermeden bir önceki döneme kadar çizilir.
    Bu fonksiyon o boşluğu görünür kılar.
    """
    try:
        diag = diagnose(df_logs)
    except Exception:
        return []
    if diag is None or diag.empty:
        return []
    bad = diag[diag["durum"] != "taze"]
    if bad.empty:
        return []
    return sorted(pd.to_datetime(bad["period_date"], errors="coerce")
                  .dropna().dt.strftime("%Y-%m").unique().tolist())


def sync_cache(df_logs: pd.DataFrame,
               only_ids: Optional[Iterable] = None,
               force: bool = False,
               with_agent: bool = True,
               progress_cb: Optional[Callable[[float, str], None]] = None) -> dict:
    """
    Bayat/eksik kayıtları hesaplayıp önbelleğe yazar.

    force=True      → durumdan bağımsız hepsini yeniden hesaplar
    only_ids        → yalnızca bu log id'leri
    with_agent=False→ ilgili kesim sınıflandırıcısını atlar (RAM tasarrufu)

    Dönüş: {"islenen": n, "atlanan": n, "hata": n}
    """
    diag = diagnose(df_logs)
    if diag.empty:
        return {"islenen": 0, "atlanan": 0, "hata": 0}

    todo = diag if force else diag[diag["durum"] != "taze"]
    if only_ids is not None:
        todo = todo[todo["id"].isin(list(only_ids))]

    atlanan = int(len(diag) - len(todo))
    if todo.empty:
        return {"islenen": 0, "atlanan": atlanan, "hata": 0}

    lookup = df_logs.set_index("id")["text_content"].to_dict()
    islenen, hata = 0, 0
    total = len(todo)

    for k, (_, r) in enumerate(todo.iterrows()):
        log_id = r["id"]
        text = str(lookup.get(log_id, "") or "")
        label = pd.Timestamp(r["period_date"]).strftime("%Y-%m")
        if progress_cb:
            progress_cb((k + 1) / total, f"{label} ({k + 1}/{total})")

        try:
            df_sent = score_sentences_ordered(text, with_agent=with_agent)
            if df_sent.empty:
                hata += 1
                continue

            # --- doküman düzeyi: mevcut utils mantığını AYNEN kullan ---
            df_for_doc = df_sent.rename(columns={"sentence": "Cümle", "diff": "Diff (H-D)"}).copy()
            df_for_doc["Duruş"] = df_for_doc["Diff (H-D)"].map(stance_3class_from_diff)
            doc = document_signal_from_sentences(df_for_doc, full_text=text)

            # full-text referans skoru (tek ekstra çağrı, debug/karşılaştırma için)
            hp = dp = np_ = bs = np.nan
            diff_full = np.nan
            try:
                clf = load_roberta_pipeline()
                raw = clf(str(text)[:1200], truncation=True)
                if isinstance(raw, list) and raw and isinstance(raw[0], list):
                    raw = raw[0]
                sm = {"HAWK": 0.0, "DOVE": 0.0, "NEUT": 0.0}
                for x in (raw or []):
                    sm[_normalize_label_mrince(x.get("label", ""))] = float(x.get("score", 0.0))
                hp, dp, np_ = sm["HAWK"], sm["DOVE"], sm["NEUT"]
                bs = max(sm.values())
                diff_full = hp - dp
            except Exception:
                pass

            ta = combine_tone_action(
                doc.get("diff_mean", 0.0),
                delta_bp=doc.get("real_delta_bp"),
                action_label=doc.get("action"),
            )

            _write_sentences(log_id, r["period_date"], r["text_hash"], df_sent)
            _write_doc({
                "log_id": int(log_id),
                "period_date": pd.Timestamp(r["period_date"]).strftime("%Y-%m-%d"),
                "n_sentences": int(len(df_sent)),
                "diff_mean": float(doc.get("diff_mean", 0.0)),
                "signal": float(doc.get("signal", 0.0)),
                "diff_fulltext": (None if pd.isna(diff_full) else float(diff_full)),
                "hawk_p": (None if pd.isna(hp) else float(hp)),
                "dove_p": (None if pd.isna(dp) else float(dp)),
                "neut_p": (None if pd.isna(np_) else float(np_)),
                "best_score": (None if pd.isna(bs) else float(bs)),
                "stance": str(doc.get("stance", "")),
                "hawk_n": int(doc.get("hawk_n", 0)),
                "dove_n": int(doc.get("dove_n", 0)),
                "neut_n": int(doc.get("neut_n", 0)),
                "action_label": str(doc.get("action", "UNKNOWN")),
                "real_delta_bp": (None if doc.get("real_delta_bp") is None
                                  else float(doc["real_delta_bp"])),
                "regime": str(ta.get("regime", "")),
                "text_hash": str(r["text_hash"]),
                "pipeline_version": PIPELINE_VERSION,
                "model_hd": MODEL_HD,
                "model_agent": MODEL_AGENT if with_agent else None,
            })
            islenen += 1
        except Exception as e:  # pragma: no cover
            print(f"[cache] {log_id} hata: {e}")
            hata += 1
        finally:
            gc.collect()

    invalidate_cache_reads()
    return {"islenen": islenen, "atlanan": atlanan, "hata": hata}


# =============================================================================
# 16. ÖNBELLEKTEN TREND SERİSİ (model çağrısı YOK)
# =============================================================================

def trend_series_from_cache(span: int = 7, z_scale: float = 2.0, hyst: float = 25.0) -> pd.DataFrame:
    """
    calculate_ai_trend_series ile aynı kolon setini üretir; fark: model
    hiç çalışmaz, her şey önbellekten okunur. Kalibrasyon/EMA/histerezis burada
    (yani çalışma anında) hesaplanır — bunlar bilerek saklanmaz.
    """
    cache = fetch_doc_cache()
    if cache.empty:
        return pd.DataFrame()

    # DİKKAT: burada tek bir sürüme eşitlik ARANMAZ. fetch_sentences ve diagnose
    # COMPATIBLE_VERSIONS kullanırken burası yalnızca PIPELINE_VERSION'a bakıyordu;
    # önbellek eski sürümde yazılmışsa seri sessizce boş dönüyor ya da satır
    # kaybediyordu.
    c = cache[cache["pipeline_version"].isin(COMPATIBLE_VERSIONS)].copy()
    if c.empty:
        return pd.DataFrame()

    # Aynı kayıt birden çok sürümde varsa en yenisini tut
    order = {v: i for i, v in enumerate(COMPATIBLE_VERSIONS)}
    c["_vrank"] = c["pipeline_version"].map(order).fillna(-1)
    c = (c.sort_values("_vrank")
           .drop_duplicates(subset=["log_id"], keep="last")
           .drop(columns=["_vrank"]))

    c = c.sort_values("period_date").reset_index(drop=True)
    out = pd.DataFrame({
        "Dönem": c["period_date"].dt.strftime("%Y-%m"),
        "period_date": c["period_date"],
        "Şahin Olasılık": pd.to_numeric(c.get("hawk_p"), errors="coerce"),
        "Güvercin Olasılık": pd.to_numeric(c.get("dove_p"), errors="coerce"),
        "Nötr Olasılık": pd.to_numeric(c.get("neut_p"), errors="coerce"),
        "Diff (H-D)": pd.to_numeric(c["diff_mean"], errors="coerce"),
        "Diff (Full-text)": pd.to_numeric(c.get("diff_fulltext"), errors="coerce"),
        "Duruş": c.get("stance", "").astype(str),
        "Delta BP": pd.to_numeric(c.get("real_delta_bp"), errors="coerce"),
        "Aksiyon": c.get("action_label", "").astype(str),
        "Rejim": c.get("regime", "").astype(str),
        "Güven": pd.to_numeric(c.get("best_score"), errors="coerce"),
    })
    out["Aksiyon Yön"] = out["Delta BP"].map(
        lambda x: np.nan if pd.isna(x) else (1.0 if x > 0 else (-1.0 if x < 0 else 0.0))
    )
    return postprocess_ai_series_steps(
        out, diff_col="Diff (H-D)", span=span, z_scale=z_scale, hyst=hyst
    )


# =============================================================================
# 17. AGREGASYONLAR
# =============================================================================

def share_timeseries(df_sent: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    Dönem × etiket → pay (%). TEK etiketli veri için: sütun toplamı %100'dür,
    yani KOMPOZİSYON okunur ("metnin yüzde kaçı bu konuya ayrıldı").
    """
    if df_sent is None or df_sent.empty:
        return pd.DataFrame()
    g = (df_sent.groupby(["Donem", label_col]).size().rename("n").reset_index())
    tot = g.groupby("Donem")["n"].transform("sum")
    g["pay"] = 100.0 * g["n"] / tot
    return g.pivot(index="Donem", columns=label_col, values="pay").fillna(0.0).sort_index()


def coverage_timeseries(df_long: pd.DataFrame, df_sent: pd.DataFrame,
                        label_col: str = "theme_label") -> pd.DataFrame:
    """
    Dönem × etiket → KAPSAM (%): o dönemde cümlelerin yüzde kaçı bu konuya değindi.

    Payda, dönemdeki ÖZGÜN cümle sayısıdır (uzun formdaki satır sayısı değil),
    bu yüzden değerler %0–100 arasındadır ama sütun toplamı %100'ü aşabilir —
    bir cümle birden çok konuya değinebildiği için bu beklenen davranıştır.
    """
    if df_long is None or df_long.empty or df_sent is None or df_sent.empty:
        return pd.DataFrame()
    denom = df_sent.groupby("Donem").size()
    g = (df_long.drop_duplicates(subset=["Donem", "log_id", "sent_idx", label_col])
                .groupby(["Donem", label_col]).size().rename("n").reset_index())
    g["pay"] = 100.0 * g["n"] / g["Donem"].map(denom)
    return g.pivot(index="Donem", columns=label_col, values="pay").fillna(0.0).sort_index()


def tone_matrix(df_sent: pd.DataFrame, label_col: str, min_n: int = 1):
    """
    Dönem × etiket → ortalama ton, ve aynı boyutta cümle sayısı matrisi.

    min_n: bu sayıdan az cümleye dayanan hücreler NaN yapılır. Tek cümleye
    dayanan bir "ortalama" aslında tek bir model tahminidir; renklendirildiğinde
    on cümlelik bir ortalamayla görsel olarak eşit ağırlıkta görünür ve okuyucuyu
    sistematik biçimde yanıltır. PPK metni ~13 cümle olduğu için bu risk yüksektir.

    Dönüş: (ton_matrisi, sayı_matrisi)
    """
    if df_sent is None or df_sent.empty:
        return pd.DataFrame(), pd.DataFrame()
    g = df_sent.groupby(["Donem", label_col])["diff"].agg(["mean", "count"])
    m = g["mean"].unstack(label_col).sort_index()
    c = g["count"].unstack(label_col).sort_index().fillna(0).astype(int)
    if min_n > 1:
        m = m.where(c >= min_n)
    return m, c


def center_matrix(df_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Her sütundan (etiketten) kendi ortalamasını çıkarır: "sapma modu".

    Neden gerekli: bazı temaların tonu sözcük dağarcığı yüzünden sistematik olarak
    kayıktır — enflasyon dili modelde şahin, likidite/destek dili güvercin tınlar.
    Bu taban çizgisi bir bulgu değil, model artefaktıdır. Merkezlendiğinde geriye
    yalnızca zaman içindeki HAREKET kalır; yorumlanabilir olan da odur.
    """
    if df_matrix is None or df_matrix.empty:
        return df_matrix
    return df_matrix.sub(df_matrix.mean(axis=0), axis=1)


def position_tone_matrix(df_sent: pd.DataFrame, bins: int = 3,
                         drop_decision: bool = False, min_n: int = 1):
    """
    Dönem × metin-içi göreli konum → ortalama ton.

    bins varsayılanı 3'tür (giriş / gövde / kapanış). TCMB'nin İngilizce PPK metni
    ~13 cümledir; 10 dilimde hücre başına ~1,3 cümle düşer, yani ortalama alma
    diye bir şey olmaz ve grafik tek tek model tahminlerinin yeniden dizilmiş
    hali haline gelir. 3 dilimde hücre başına ~4 cümle düşer.

    drop_decision=True: karar cümlelerini (faiz kararını bildiren) hariç tutar ve
    konumu kalan cümleler üzerinden YENİDEN hesaplar. Böylece "faiz kararı bir yana,
    çerçeve metni ne diyor" sorusu cevaplanır — ilk satırın faiz yönünü izlemesi
    sorunu ortadan kalkar.

    Dönüş: (ton_matrisi, sayı_matrisi)
    """
    if df_sent is None or df_sent.empty:
        return pd.DataFrame(), pd.DataFrame()
    d = df_sent.copy()

    if drop_decision:
        mask = d["sentence"].astype(str).map(lambda s: bool(_DECISION_OVERRIDE.search(s)))
        d = d[~mask].copy()
        if d.empty:
            return pd.DataFrame(), pd.DataFrame()
        # Konumu kalan cümleler üzerinden yeniden numaralandır
        d = d.sort_values(["Donem", "sent_idx"])
        d["sent_idx"] = d.groupby("Donem").cumcount()
        d["sent_total"] = d.groupby("Donem")["sent_idx"].transform("size")

    d["sent_total"] = pd.to_numeric(d["sent_total"], errors="coerce")
    d = d[d["sent_total"] > 1]
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()

    pos = pd.to_numeric(d["sent_idx"], errors="coerce") / (d["sent_total"] - 1)
    d["dilim"] = np.clip((pos * bins).astype(int), 0, bins - 1)
    g = d.groupby(["Donem", "dilim"])["diff"].agg(["mean", "count"])
    m = g["mean"].unstack("dilim").sort_index()
    c = g["count"].unstack("dilim").sort_index().fillna(0).astype(int)
    if min_n > 1:
        m = m.where(c >= min_n)

    if bins == 3:
        names = {0: "Giriş", 1: "Gövde", 2: "Kapanış"}
        cols = [names.get(x, str(x)) for x in m.columns]
    else:
        cols = [f"%{int(100 * x / bins)}–{int(100 * (x + 1) / bins)}" for x in m.columns]
    m.columns = cols
    c.columns = cols
    return m, c


def divergence_table(df_sent: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    Etiket bazında ton ve hacim özeti.

    "Dönem" sütunu, etiketin kaç FARKLI dönemde göründüğünü verir: tek bir dönemde
    yoğunlaşmış bir etiketin genel ortalaması zaman serisi olarak yorumlanamaz.
    Model tahmini olan etiketlerde ortalama güven de gösterilir; 0.60 altındaki
    sınıfları ciddiye almamak gerekir.
    """
    if df_sent is None or df_sent.empty:
        return pd.DataFrame()
    agg = {"Cümle": ("diff", "count"), "Ort. Ton": ("diff", "mean"),
           "Std": ("diff", "std"), "Dönem": ("Donem", "nunique")}
    if label_col == "agent_label" and "agent_conf" in df_sent.columns:
        agg["Ort. Güven"] = ("agent_conf", "mean")
    g = df_sent.groupby(label_col).agg(**agg).reset_index()
    g["Duruş"] = g["Ort. Ton"].map(stance_3class_from_diff)
    return g.sort_values("Ort. Ton", ascending=False).reset_index(drop=True)


# =============================================================================
# 18. GÖRSELLER
# =============================================================================

def _sym_limit(values) -> float:
    v = pd.Series(values).abs().max()
    if not np.isfinite(v) or v == 0:
        return 0.5
    return float(min(1.0, max(0.15, v)))


def chart_share_area(df_share: pd.DataFrame, title: str, colors: Optional[dict] = None):
    """Yığılmış alan grafiği — pay (%) zaman serisi."""
    if df_share is None or df_share.empty:
        return None
    fig = go.Figure()
    for col in df_share.columns:
        fig.add_trace(go.Scatter(
            x=df_share.index, y=df_share[col], name=str(col),
            mode="lines", stackgroup="one", groupnorm="percent",
            line=dict(width=0.5, color=(colors or {}).get(col)),
            hovertemplate="%{x}<br>" + str(col) + ": %{y:.1f}%<extra></extra>",
        ))
    fig.update_layout(
        title=title, height=430, hovermode="x unified",
        yaxis=dict(title="Cümle payı (%)", range=[0, 100], ticksuffix="%"),
        xaxis=dict(title=""),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        margin=dict(t=50, b=40),
    )
    return fig


def chart_tone_heatmap(df_matrix: pd.DataFrame, title: str, ylab: str = "",
                       counts: Optional[pd.DataFrame] = None,
                       centered: bool = False):
    """
    Dönem × etiket ton ısı haritası.

    Boşluk/nötr ayrımı: nötr hücreler artık BEYAZ, veri olmayan hücreler ise
    grafiğin GRİ zeminidir. Önceki sürümde renk skalasının ortası da açık gri
    olduğu için "o dönem bu konuya değinilmedi" ile "değinildi ama tonu nötr"
    görsel olarak ayırt edilemiyordu — bu, ısı haritasında en sık yapılan okuma
    hatasıdır.

    counts verilirse her hücrenin kaç cümleye dayandığı hover'da görünür.
    """
    if df_matrix is None or df_matrix.empty:
        return None

    z = df_matrix.T.values.astype(float)
    lim = _sym_limit(df_matrix.values.ravel())

    # Nötr = beyaz (zeminin grisinden ayrışsın)
    scale = [[0.0, "#1f4e9c"], [0.5, "#ffffff"], [1.0, "#c0392b"]]

    hover = "%{x} · %{y}<br>ton: %{z:.3f}"
    cdata = None
    if counts is not None and not counts.empty:
        aligned = counts.reindex(index=df_matrix.index, columns=df_matrix.columns)
        cdata = aligned.T.values
        hover += "<br>%{customdata} cümle"
    hover += "<extra></extra>"

    fig = go.Figure(go.Heatmap(
        z=z,
        x=list(df_matrix.index),
        y=[str(c) for c in df_matrix.columns],
        customdata=cdata,
        colorscale=scale, zmid=0, zmin=-lim, zmax=lim,
        hovertemplate=hover,
        colorbar=dict(title=("Sapma<br>(ton − ort.)" if centered else "Ton<br>(H−D)"),
                      thickness=12),
        hoverongaps=False,
        xgap=1, ygap=1,
    ))
    fig.update_layout(
        title=title, height=max(320, 40 * len(df_matrix.columns) + 140),
        xaxis=dict(title=""), yaxis=dict(title=ylab, autorange="reversed"),
        margin=dict(t=50, b=40, l=10),
        plot_bgcolor="#d9dade",   # veri YOK -> gri zemin
    )
    return fig


def chart_share_lines(df_share: pd.DataFrame, title: str,
                      colors: Optional[dict] = None, ylab: str = "Kapsam (%)"):
    """
    Çok etiketli kapsam serisi için çizgi grafiği.

    Yığılmış alan kullanılmaz: çok etiketli veride sütun toplamı %100'ü aştığı
    için yığma matematiksel olarak yanlış olur ve payları olduğundan küçük gösterir.
    """
    if df_share is None or df_share.empty:
        return None
    fig = go.Figure()
    for col in df_share.columns:
        fig.add_trace(go.Scatter(
            x=df_share.index, y=df_share[col], name=str(col), mode="lines",
            line=dict(width=2, color=(colors or {}).get(col)),
            hovertemplate="%{x}<br>" + str(col) + ": %{y:.0f}%<extra></extra>",
        ))
    fig.update_layout(
        title=title, height=430, hovermode="x unified",
        yaxis=dict(title=ylab, ticksuffix="%", rangemode="tozero"),
        xaxis=dict(title=""),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        margin=dict(t=50, b=40),
    )
    return fig


def chart_sentence_strip(df_one: pd.DataFrame, title: str = "Cümle sırasına göre ton"):
    """Tek dokümanın cümle cümle ton profili (bar). Metnin ton mimarisi burada görünür."""
    if df_one is None or df_one.empty:
        return None
    d = df_one.sort_values("sent_idx")
    diffs = pd.to_numeric(d["diff"], errors="coerce").fillna(0.0)
    # Renkler metin haritasıyla ve metriklerle aynı eşiği kullanır: nötr bant gri.
    colors = [
        "#c0392b" if v >= DOC_STANCE_DEADBAND
        else ("#1f4e9c" if v <= -DOC_STANCE_DEADBAND else "#b8bcc0")
        for v in diffs
    ]
    wrapped = d["sentence"].astype(str).str.slice(0, 160).str.replace(r"(.{60})", r"\1<br>", regex=True)
    fig = go.Figure(go.Bar(
        x=d["sent_idx"], y=diffs, marker=dict(color=colors),
        customdata=np.stack([wrapped, d["agent_label"].astype(str), d["theme_label"].astype(str)], axis=-1),
        hovertemplate="#%{x} · ton %{y:.3f}<br>%{customdata[1]} · %{customdata[2]}<br>%{customdata[0]}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color="black", width=1))
    fig.update_layout(
        title=title, height=300, bargap=0.15,
        xaxis=dict(title="Cümle sırası"), yaxis=dict(title="Ton (H−D)"),
        margin=dict(t=50, b=40),
    )
    return fig


def _tone_rgba(diff: float, deadband: float = None) -> str:
    """
    Cümle tonunu arka plan rengine çevirir.

    ÖNEMLİ: Renk eşiği, metriklerin kullandığı DOC_STANCE_DEADBAND ile AYNIDIR.
    Önceki sürümde renk sıfır noktasında dönüyordu (d >= 0 -> kırmızı) ama sayım
    0.10 eşiğini kullanıyordu. Bu iki ayrı sınır iki yönlü uyuşmazlık üretiyordu:

      * 0.00 <= diff < 0.10  -> soluk KIRMIZI görünüyor ama NÖTR sayılıyordu
                                (gözle fazla şahin sayarsın)
      * diff = 0.10 (tam eşik) -> alfa 0.135, yani neredeyse görünmez beyaz;
                                sayılıyor ama gözle seçilemiyordu
                                (gözle eksik şahin sayarsın)

    Artık: nötr bant düz gri, eşiği geçen her cümle en az 0.25 alfa ile
    kesinlikle görünür. Yani metrikteki sayı ile ekranda saydığın renk birebir
    tutar.
    """
    db = DOC_STANCE_DEADBAND if deadband is None else float(deadband)
    d = float(np.clip(diff, -1.0, 1.0))

    # Nötr bant: renk yok, düz gri. "Değinilmiş ama yön yok" demek.
    if abs(d) < db:
        return "rgba(128,128,128,0.10)"

    # Eşiği geçenler: [db, 1] araligi -> [0.25, 0.65] alfa
    t = (abs(d) - db) / max(1e-9, 1.0 - db)
    a = 0.25 + 0.40 * min(1.0, t)
    return f"rgba(192,57,43,{a:.3f})" if d > 0 else f"rgba(31,78,156,{a:.3f})"


def sentence_heatmap_html(df_one: pd.DataFrame,
                          show_agent: bool = True,
                          max_height: int = 520) -> str:
    """
    Dokümanı okunabilir biçimde, her cümle tonuna göre renklenmiş olarak döndürür.
    Kırmızı = şahin, mavi = güvercin, renksiz = nötr. Üzerine gelince olasılıklar.
    """
    if df_one is None or df_one.empty:
        return "<p><i>Bu dönem için önbellekte cümle yok.</i></p>"

    parts: list[str] = []
    for _, r in df_one.sort_values("sent_idx").iterrows():
        diff = float(r.get("diff", 0.0) or 0.0)
        sent = _htmllib.escape(str(r.get("sentence", "")))
        tip = (f"#{int(r['sent_idx'])} · ton {diff:+.3f} · "
               f"H {float(r.get('hawk', 0)):.2f} / D {float(r.get('dove', 0)):.2f} / "
               f"N {float(r.get('neut', 0)):.2f}")
        badge = ""
        if show_agent and r.get("agent_label"):
            badge = (f"<sup style='font-size:.62em;opacity:.65;white-space:nowrap;'>"
                     f"&nbsp;{_htmllib.escape(str(r['agent_label']))}</sup>")
        parts.append(
            f"<span title='{_htmllib.escape(tip)}' "
            f"style='background:{_tone_rgba(diff)};padding:2px 3px;border-radius:3px;'>"
            f"{sent}{badge}</span>"
        )

    body = " ".join(parts)

    # Küçük renk anahtarı — ekrandaki rengin hangi eşiğe karşılık geldiğini gösterir,
    # böylece metriklerdeki sayı ile gözle sayım arasında şüphe kalmaz.
    db = DOC_STANCE_DEADBAND
    legend = (
        "<div style='margin-top:8px;font-size:.78rem;opacity:.72;"
        "display:flex;gap:14px;flex-wrap:wrap;align-items:center;'>"
        f"<span><span style='background:{_tone_rgba(1.0)};padding:1px 10px;"
        "border-radius:3px;'>&nbsp;</span> şahin (ton &ge; "
        f"{db:.2f})</span>"
        f"<span><span style='background:{_tone_rgba(0.0)};padding:1px 10px;"
        "border-radius:3px;'>&nbsp;</span> nötr (|ton| &lt; "
        f"{db:.2f})</span>"
        f"<span><span style='background:{_tone_rgba(-1.0)};padding:1px 10px;"
        "border-radius:3px;'>&nbsp;</span> güvercin (ton &le; -"
        f"{db:.2f})</span>"
        "<span style='opacity:.8;'>Renk yoğunluğu |ton| ile artar. "
        "Cümlenin üzerine gelince ham skor görünür.</span>"
        "</div>"
    )

    return (
        f"<div style='max-height:{max_height}px;overflow-y:auto;line-height:2.05;"
        f"font-size:0.94rem;padding:14px 16px;border:1px solid rgba(128,128,128,.28);"
        f"border-radius:8px;'>{body}</div>{legend}"
    )
