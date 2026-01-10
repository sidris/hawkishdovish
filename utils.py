import streamlit as st
from supabase import create_client, Client
import pandas as pd
import requests
import io
import datetime
import re
import difflib
from collections import Counter
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional

# --- 1. EK KÜTÜPHANELER ---
try:
    import sklearn
    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, FunctionTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.pyplot as plt
    HAS_ML_DEPS = True
except ImportError:
    HAS_ML_DEPS = False

# VADER & FINBERT KONTROLLERİ
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except ImportError:
    HAS_VADER = False

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    HAS_FINBERT = True
except ImportError:
    HAS_FINBERT = False

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

EVDS_BASE = "https://evds2.tcmb.gov.tr/service/evds"
EVDS_TUFE_SERIES = "TP.FG.J0"

# =============================================================================
# 3. KULLANICININ VERDİĞİ ORİJİNAL ALGORİTMA (ABF)
# =============================================================================

def M(token_or_phrase: str, wildcard_first: bool = False):
    toks = token_or_phrase.split()
    wild = [False] * len(toks)
    if wildcard_first and toks:
        wild[0] = True
    return {"phrase": toks, "wild": wild, "pattern": token_or_phrase}

# SİZİN VERDİĞİNİZ SÖZLÜK (ORİJİNAL - DEĞİŞTİRİLMEDİ)
DICT = {
   "inflation": [
        {
           "block": "consumer_prices_inflation",
           "terms": ["consumer prices", "inflation"],
           "hawk": [
               M("accelerat", True), M("boost", True), M("elevated"),
               M("escalat", True), M("high", True), M("increas", True),
               M("jump", True), M("pickup"), M("rise", True),
               M("rose"), M("rising"), M("runup"),
               M("strong", True), M("surg", True), M("up", True)
            ],
           "dove": [
               M("decelerat", True), M("declin", True), M("decreas", True),
               M("down", True), M("drop", True), M("fall", True),
               M("fell"), M("low", True), M("muted"),
               M("reduc", True), M("slow", True), M("stable"),
               M("subdued"), M("weak", True), M("contained")
            ],
        },
        {
           "block": "inflation_pressure",
           "terms": ["inflation pressure"],
           "hawk": [
               M("accelerat", True), M("boost", True), M("build", True),
               M("elevat", True), M("emerg", True), M("great", True),
               M("height", True), M("high", True), M("increas", True),
               M("intensif", True), M("mount", True), M("pickup"),
               M("rise"), M("rose"), M("rising"),
               M("stok", True), M("strong", True), M("sustain", True)
            ],
           "dove": [
               M("abat", True), M("contain", True), M("dampen", True),
               M("decelerat", True), M("declin", True), M("decreas", True),
               M("dimin", True), M("eas", True), M("fall", True),
               M("fell"), M("low", True), M("moderat", True),
               M("reced", True), M("reduc", True), M("subdued"),
               M("temper", True)
            ],
        },
    ],
   "economic_activity": [
        {
           "block": "consumer_spending",
           "terms": ["consumer spending"],
           "hawk": [
               M("accelerat", True), M("edg up", True), M("expan", True),
               M("increas", True), M("pick up", True), M("pickup"),
               M("soft", True), M("strength", True), M("strong", True),
               M("weak", True),
            ],
           "dove": [
               M("contract", True), M("decelerat", True), M("decreas", True),
               M("drop", True), M("retrench", True), M("slow", True),
               M("slugg", True), M("soft", True), M("subdued"),
            ],
        },
        {
           "block": "economic_activity_growth",
           "terms": ["economic activity", "economic growth"],
           "hawk": [
               M("accelerat", True), M("buoyant"), M("edg up", True),
               M("expan", True), M("increas", True), M("high", True),
               M("pick up", True), M("pickup"), M("rise", True),
               M("rose"), M("rising"), M("step up", True),
               M("strength", True), M("strong", True), M("upside"),
            ],
           "dove": [
               M("contract", True), M("curtail", True), M("decelerat", True),
               M("declin", True), M("decreas", True), M("downside"),
               M("drop"), M("fall", True), M("fell"),
               M("low", True), M("moderat", True), M("slow", True),
               M("slugg", True), M("weak", True),
            ],
        },
        {
           "block": "resource_utilization",
           "terms": ["resource utilization"],
           "hawk": [
               M("high", True), M("increas", True), M("rise"),
               M("rising"), M("rose"), M("tight", True),
            ],
           "dove": [
               M("declin", True), M("fall", True), M("fell"),
               M("loose", True), M("low", True),
            ],
        },
    ],
   "employment": [
        {
           "block": "employment",
           "terms": ["employment"],
           "hawk": [
               M("expand", True), M("gain", True), M("improv", True),
               M("increas", True), M("pick up", True), M("pickup"),
               M("rais", True), M("rise", True), M("rising"),
               M("rose"), M("strength", True), M("turn up", True),
            ],
           "dove": [
               M("slow", True), M("declin", True), M("reduc", True),
               M("weak", True), M("deteriorat", True), M("shrink", True),
               M("shrank"), M("fall", True), M("fell"),
               M("drop", True), M("contract", True), M("sluggish"),
            ],
        },
        {
           "block": "labor_market",
           "terms": ["labor market"],
           "hawk": [M("strain", True), M("tight", True)],
           "dove": [M("eased"), M("easing"), M("loos", True), M("soft", True), M("weak", True)],
        },
        {
           "block": "unemployment",
           "terms": ["unemployment"],
           "hawk": [M("declin", True), M("fall", True), M("fell"), M("low", True), M("reduc", True)],
           "dove": [M("elevat", True), M("high"), M("increas", True), M("ris", True), M("rose", True)],
        },
    ],
}

# --- NLP FONKSİYONLARI ---

def normalize_text(text: str) -> str:
    if not text: return ""
    t = text.lower().replace("’", "'").replace("`", "'")
    t = re.sub(r"(?<=\w)-(?=\w)", " ", t)
    t = re.sub(r"\brun\s+up\b", "runup", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def split_sentences(text: str):
    if not text: return []
    text = re.sub(r"\n+", ". ", text)
    sents = re.split(r"(?<=[\.\!\?\;])\s+", text)
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
            if window[j] == tt[j]:
               continue
            if window[j] == tt[j] + "s" or tt[j] == window[j] + "s":
               continue
            ok = False
            break
        if ok:
           hits.append((i, i + m - 1))
    return hits

def select_non_overlapping_terms(tokens, term_infos):
    term_infos_sorted = sorted(term_infos, key=lambda x: len(x["term"].split()), reverse=True)
    occupied = set()
    selected = []
    for info in term_infos_sorted:
        for (s, e) in find_term_positions_flex(tokens, info["term"]):
            if any(k in occupied for k in range(s, e + 1)):
               continue
            occupied.update(range(s, e + 1))
            selected.append({**info, "start": s, "end": e})
    selected.sort(key=lambda x: x["start"])
    return selected

def analyze_hawk_dove(
    text: str,
    DICT: dict = DICT,
    window_words: int = 10,
    verbose: bool = False,
    dedupe_within_term_window: bool = True,
    nearest_only: bool = True
):
    text_n = normalize_text(text)
    sentences = split_sentences(text_n)

    # term infos
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
        if not tokens:
            continue

        for topic, term_infos in topic_term_infos.items():
            selected_terms = select_non_overlapping_terms(tokens, term_infos)
            if not selected_terms:
               continue

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
                       if me < w0 or ms > w1:
                           continue
                       dist = min(abs(ms - te), abs(ts - me))
                       hawk_hits.append((dist, m, ms, me))

                dove_hits = []
                for m in block["dove"]:
                   for (ms, me) in find_phrase_positions(tokens, m["phrase"], m["wild"]):
                       if me < w0 or ms > w1:
                           continue
                       dist = min(abs(ms - te), abs(ts - me))
                       dove_hits.append((dist, m, ms, me))

                if nearest_only:
                   hawk_hits = sorted(hawk_hits, key=lambda x: x[0])[:1]
                   dove_hits = sorted(dove_hits, key=lambda x: x[0])[:1]

                seen = set()

                def add_hit(direction, dist, m, ms, me):
                   mod_found = " ".join(tokens[ms:me+1])
                   key = (topic, block["block"], ts, te, direction, mod_found)
                   if dedupe_within_term_window and key in seen:
                       return
                   seen.add(key)

                   topic_counts[topic][direction] += 1
                   matches.append({
                       "topic": topic,
                       "block": block["block"],
                       "direction": direction,
                       "term_dict": tinfo["term"],
                       "term_found": term_found,
                       "modifier_pattern": m["pattern"],
                       "modifier_found": mod_found,
                       "distance": dist,
                       "sentence": sent,
                       # app.py için gerekli ek alanlar
                       "term": tinfo["term"],
                       "type": "HAWK" if direction == "hawk" else "DOVE"
                   })

                for (dist, m, ms, me) in hawk_hits:
                   add_hit("hawk", dist, m, ms, me)
                for (dist, m, ms, me) in dove_hits:
                   add_hit("dove", dist, m, ms, me)

    hawk_total = sum(v["hawk"] for v in topic_counts.values())
    dove_total = sum(v["dove"] for v in topic_counts.values())
    denom = hawk_total + dove_total

    hawk_pct = 0.0 if denom == 0 else 100.0 * hawk_total / denom
    dove_pct = 0.0 if denom == 0 else 100.0 * dove_total / denom
    net_hawkishness = 1.0 if denom == 0 else (1.0 + (hawk_total - dove_total) / denom)

    df = pd.DataFrame(matches)

    return {
       "topic_counts": topic_counts,
       "matches": matches,
       "match_details": matches, # APP.PY UYUMLULUĞU
       "net_hawkishness": net_hawkishness,
       "hawk_pct": hawk_pct,
       "dove_pct": dove_pct,
       "hawk_count": hawk_total,
       "dove_count": dove_total,
       "df": df
    }

# =============================================================================
# 4. APP.PY İÇİN YARDIMCI FONKSİYONLAR & DATA FETCH (ROBUST)
# =============================================================================

@st.cache_data(ttl=600)
def fetch_market_data_adapter(start_date, end_date):
    """
    EVDS'den veri çeker. Hata durumunda veya API keysiz durumda
    boş ama doğru kolonlara sahip DataFrame döner.
    """
    # Her durumda dönecek boş yapı (KeyError önlemek için)
    empty_df = pd.DataFrame(columns=["Donem", "Yıllık TÜFE", "PPK Faizi", "SortDate"])
    
    if not EVDS_API_KEY: 
        # Dummy data
        dates = pd.date_range(start=start_date, end=end_date, freq='M')
        if len(dates) == 0: return empty_df, "Tarih Yok"
        
        return pd.DataFrame({
            'Donem': dates.strftime('%Y-%m'),
            'Yıllık TÜFE': [0]*len(dates),
            'PPK Faizi': [0]*len(dates),
            'SortDate': dates
        }), "API Key Yok - Dummy"

    # API VARSA VERİ ÇEKMEYİ DENE
    df_inf = pd.DataFrame()
    try:
        s = start_date.strftime("%d-%m-%Y")
        e = end_date.strftime("%d-%m-%Y")
        for form, col in [(1, "Aylık TÜFE"), (3, "Yıllık TÜFE")]:
            url = f"{EVDS_BASE}/series={EVDS_TUFE_SERIES}&startDate={s}&endDate={e}&type=json&formulas={form}"
            r = requests.get(url, headers={"key": EVDS_API_KEY}, timeout=20)
            if r.status_code == 200 and r.json().get("items"):
                temp = pd.DataFrame(r.json()["items"])
                temp["dt"] = pd.to_datetime(temp["Tarih"], dayfirst=True, errors="coerce")
                if temp["dt"].isnull().all(): temp["dt"] = pd.to_datetime(temp["Tarih"], format="%Y-%m", errors="coerce")
                temp = temp.dropna(subset=["dt"])
                temp["Donem"] = temp["dt"].dt.strftime("%Y-%m")
                val_c = [c for c in temp.columns if "TP" in c][0]
                temp = temp.rename(columns={val_c: col})[["Donem", col]]
                if df_inf.empty: df_inf = temp
                else: df_inf = pd.merge(df_inf, temp, on="Donem", how="outer")
    except Exception: pass

    df_pol = pd.DataFrame()
    try:
        s_bis = start_date.strftime("%Y-%m-%d")
        e_bis = end_date.strftime("%Y-%m-%d")
        url_bis = f"https://stats.bis.org/api/v1/data/WS_CBPOL/D.TR?format=csv&startPeriod={s_bis}&endPeriod={e_bis}"
        r_bis = requests.get(url_bis, timeout=20)
        if r_bis.status_code == 200:
            temp_bis = pd.read_csv(io.StringIO(r_bis.content.decode("utf-8")), usecols=["TIME_PERIOD", "OBS_VALUE"])
            temp_bis["dt"] = pd.to_datetime(temp_bis["TIME_PERIOD"])
            temp_bis["Donem"] = temp_bis["dt"].dt.strftime("%Y-%m")
            temp_bis["PPK Faizi"] = pd.to_numeric(temp_bis["OBS_VALUE"], errors="coerce")
            df_pol = temp_bis.sort_values("dt").groupby("Donem").last().reset_index()[["Donem", "PPK Faizi"]]
    except Exception: pass

    master_df = pd.DataFrame()
    if not df_inf.empty and not df_pol.empty: master_df = pd.merge(df_inf, df_pol, on="Donem", how="outer")
    elif not df_inf.empty: master_df = df_inf
    elif not df_pol.empty: master_df = df_pol

    if master_df.empty: return empty_df, "Veri Bulunamadı"
    
    # Eksik kolonları tamamla
    for c in ["Yıllık TÜFE", "PPK Faizi"]:
        if c not in master_df.columns: master_df[c] = 0.0
        
    master_df["SortDate"] = pd.to_datetime(master_df["Donem"] + "-01")
    return master_df.sort_values("SortDate"), None

def fetch_all_data():
    if not supabase: return pd.DataFrame()
    try:
        res = supabase.table("market_logs").select("*").order("period_date", desc=True).execute()
        data = getattr(res, 'data', []) if res else []
        return pd.DataFrame(data)
    except Exception: return pd.DataFrame()

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
        try: supabase.table("market_logs").delete().eq("id", rid).execute()
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

def run_full_analysis(text):
    res = analyze_hawk_dove(text, DICT=DICT)
    s_abg = res['net_hawkishness']
    h_cnt = res['hawk_count']
    d_cnt = res['dove_count']
    
    h_list = []
    d_list = []
    h_ctx = {}
    d_ctx = {}
    
    if res['matches']:
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
    results = []
    for _, row in df.iterrows():
        res = analyze_hawk_dove(str(row.get('text_content', '')), DICT=DICT)
        donem = row.get("Donem", "")
        if not donem and "period_date" in row:
             try: donem = pd.to_datetime(row["period_date"]).strftime("%Y-%m")
             except: pass
        results.append({
            "period_date": row.get("period_date"),
            "Donem": donem,
            "abg_index": res['net_hawkishness']
        })
    return pd.DataFrame(results)

def calculate_flesch_reading_ease(text):
    if not text: return 0
    sentences = split_sentences(text)
    words = re.findall(r'\w+', text)
    num_sentences = len(sentences)
    num_words = len(words)
    if num_words == 0 or num_sentences == 0: return 0
    num_syllables = sum(count_syllables(w) for w in words)
    score = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)
    return score

def count_syllables(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if len(word) == 0: return 0
    if word[0] in vowels: count += 1
    for i in range(1, len(word)):
        if word[i] in vowels and word[i - 1] not in vowels:
            count += 1
    if word.endswith("e"): count -= 1
    if count == 0: count += 1
    return count

def generate_diff_html(text1, text2):
    if not text1: text1 = ""
    if not text2: text2 = ""
    d = difflib.HtmlDiff()
    return d.make_file(text1.splitlines(), text2.splitlines())

def get_top_terms_series(df, top_n=5, stop_words=None):
    if df.empty: return pd.DataFrame(), []
    all_words = []
    for txt in df['text_content']:
        all_words.extend(re.findall(r'\w+', normalize_text(str(txt))))
    if stop_words:
        all_words = [w for w in all_words if w not in stop_words and len(w)>2]
    freq = pd.Series(all_words).value_counts().head(top_n)
    top_terms = freq.index.tolist()
    res_data = []
    for _, row in df.iterrows():
        row_txt = normalize_text(str(row['text_content']))
        counts = {t: len(re.findall(rf'\b{t}\b', row_txt)) for t in top_terms}
        counts['period_date'] = row['period_date']
        res_data.append(counts)
    return pd.DataFrame(res_data), top_terms

def generate_wordcloud_img(text, stop_words):
    if not HAS_ML_DEPS or not text: return None
    try:
        wc = WordCloud(width=800, height=400, background_color='white', stopwords=set(stop_words) if stop_words else None).generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        return fig
    except: return None

# --- ML PLACEHOLDERS ---
@dataclass
class CFG:
    trend_window: int = 6
    cap_low: int = -750
    cap_high: int = 750
    max_splits: int = 6
    half_life_days: float = 365.0
    token_pattern: str = r"(?u)\b[0-9a-zçğıöşü]{2,}\b"
    word_ngram: Tuple[int,int] = (1, 2)
    min_df: int = 1
    max_df: float = 1.0
    max_features: int = 20000
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

def prepare_ml_dataset(df_logs, df_market):
    if df_logs.empty or df_market.empty: return pd.DataFrame()
    if 'period_date' in df_logs.columns:
        df_logs = df_logs.copy()
        df_logs['period_date'] = pd.to_datetime(df_logs['period_date'])
        df_logs['Donem'] = df_logs['period_date'].dt.strftime('%Y-%m')
    if 'Donem' not in df_market.columns: return pd.DataFrame()
    df = pd.merge(df_logs, df_market, on="Donem", how="left")
    if 'PPK Faizi' in df.columns:
        df['rate_change_bps'] = df['PPK Faizi'].diff().fillna(0.0) * 100
        return pd.DataFrame({
            "date": df['period_date'],
            "text": df['text_content'],
            "rate_change_bps": df['rate_change_bps']
        }).dropna()
    return pd.DataFrame()

class AdvancedMLPredictor:
    def __init__(self): self.df_hist = None
    def train(self, df):
        if not HAS_ML_DEPS: return "Kütüphane Eksik"
        self.df_hist = df.copy()
        self.df_hist['y_bps'] = self.df_hist['rate_change_bps']
        self.df_hist['predicted_bps'] = self.df_hist['rate_change_bps'].shift(1).fillna(0)
        return "OK"
    def predict(self, text):
        return {
            "pred_direction": "SABİT", "direction_confidence": 0.5,
            "pred_change_bps": 0.0, "pred_interval_lo": -50.0, "pred_interval_hi": 50.0
        }

def calculate_vader_series(df): return pd.DataFrame()
def calculate_finbert_series(df): return pd.DataFrame()
