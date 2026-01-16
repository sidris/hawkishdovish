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

# --- 1. KÜTÜPHANE KONTROLLERİ VE GLOBAL FLAGLER ---
HAS_ML_DEPS = False
HAS_VADER = False
HAS_FINBERT = False
HAS_ROBERTA_LIB = False 

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

# VADER Kontrolü
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except ImportError:
    HAS_VADER = False

# RoBERTa / Transformers Kontrolü
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    HAS_FINBERT = True
    HAS_ROBERTA_LIB = True 
except ImportError:
    HAS_FINBERT = False
    HAS_ROBERTA_LIB = False

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

# --- MARKET DATA ---
@st.cache_data(ttl=600)
def fetch_market_data_adapter(start_date, end_date):
    empty_df = pd.DataFrame(columns=["Donem", "Yıllık TÜFE", "PPK Faizi", "SortDate"])
    
    if not EVDS_API_KEY: 
        dates = pd.date_range(start=start_date, end=end_date, freq='M')
        if len(dates) == 0: return empty_df, "Tarih Yok"
        return pd.DataFrame({
            'Donem': dates.strftime('%Y-%m'),
            'Yıllık TÜFE': [0]*len(dates),
            'PPK Faizi': [0]*len(dates),
            'SortDate': dates
        }), "API Key Yok"

    df_inf = pd.DataFrame()
    try:
        s = start_date.strftime("%d-%m-%Y"); e = end_date.strftime("%d-%m-%Y")
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
        s_bis = start_date.strftime("%Y-%m-%d"); e_bis = end_date.strftime("%Y-%m-%d")
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
    
    for c in ["Yıllık TÜFE", "PPK Faizi"]:
        if c not in master_df.columns: master_df[c] = 0.0
        
    master_df["SortDate"] = pd.to_datetime(master_df["Donem"] + "-01")
    return master_df.sort_values("SortDate"), None

# =============================================================================
# 4. OKUNABİLİRLİK VE DİĞER ANALİZLER
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
    stops = set(["that", "with", "this", "from", "have", "which", "will", "been", "were", "market", "central", "bank", "committee", "monetary", "policy", "decision", "percent", "rates", "level", "year", "their", "over", "also", "under", "developments", "conditions", "indicators", "recent", "remain", "remains", "period", "has", "are", "for", "and", "the", "decided", "keep", "constant", "take", "taking", "account"])
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
# 5. ABF (2019) ALGORİTMASI
# =============================================================================
def M(token_or_phrase: str, wildcard_first: bool = False):
    toks = token_or_phrase.split()
    wild = [False] * len(toks)
    if wildcard_first and toks: wild[0] = True
    return {"phrase": toks, "wild": wild, "pattern": token_or_phrase}

DICT = {
   "inflation": [{"block": "consumer_prices_inflation","terms": ["consumer prices", "inflation"],"hawk": [M("accelerat", True), M("boost", True), M("elevated"), M("escalat", True), M("high", True), M("increas", True), M("jump", True), M("pickup"), M("rise", True), M("rose"), M("rising"), M("runup"), M("strong", True), M("surg", True), M("up", True)],"dove": [M("decelerat", True), M("declin", True), M("decreas", True), M("down", True), M("drop", True), M("fall", True), M("fell"), M("low", True), M("muted"), M("reduc", True), M("slow", True), M("stable"), M("subdued"), M("weak", True), M("contained")]},{"block": "inflation_pressure","terms": ["inflation pressure"],"hawk": [M("accelerat", True), M("boost", True), M("build", True), M("elevat", True), M("emerg", True), M("great", True), M("height", True), M("high", True), M("increas", True), M("intensif", True), M("mount", True), M("pickup"), M("rise", True), M("rose"), M("rising"), M("stok", True), M("strong", True), M("sustain", True)],"dove": [M("abat", True), M("contain", True), M("dampen", True), M("decelerat", True), M("declin", True), M("decreas", True), M("dimin", True), M("eas", True), M("fall", True), M("fell"), M("low", True), M("moderat", True), M("reced", True), M("reduc", True), M("subdued"), M("temper", True)]}],
   "economic_activity": [{"block": "consumer_spending","terms": ["consumer spending"],"hawk": [M("accelerat", True), M("edg up", True), M("expan", True), M("increas", True), M("pick up", True), M("pickup"), M("soft", True), M("strength", True), M("strong", True), M("weak", True)],"dove": [M("contract", True), M("decelerat", True), M("decreas", True), M("drop", True), M("retrench", True), M("slow", True), M("slugg", True), M("soft", True), M("subdued")]},{"block": "economic_activity_growth","terms": ["economic activity", "economic growth"],"hawk": [M("accelerat", True), M("buoyant"), M("edg up", True), M("expan", True), M("increas", True), M("high", True), M("pick up", True), M("pickup"), M("rise", True), M("rose"), M("rising"), M("step up", True), M("strength", True), M("strong", True), M("upside")],"dove": [M("contract", True), M("curtail", True), M("decelerat", True), M("declin", True), M("decreas", True), M("downside"), M("drop"), M("fall", True), M("fell"), M("low", True), M("moderat", True), M("slow", True), M("slugg", True), M("weak", True)]},{"block": "resource_utilization","terms": ["resource utilization"],"hawk": [M("high", True), M("increas", True), M("rise"), M("rising"), M("rose"), M("tight", True)],"dove": [M("declin", True), M("fall", True), M("fell"), M("loose", True), M("low", True)]}],
   "employment": [{"block": "employment","terms": ["employment"],"hawk": [M("expand", True), M("gain", True), M("improv", True), M("increas", True), M("pick up", True), M("pickup"), M("rais", True), M("rise", True), M("rising"), M("rose"), M("strength", True), M("turn up", True)],"dove": [M("slow", True), M("declin", True), M("reduc", True), M("weak", True), M("deteriorat", True), M("shrink", True), M("shrank"), M("fall", True), M("fell"), M("drop", True), M("contract", True), M("sluggish")]},{"block": "labor_market","terms": ["labor market"],"hawk": [M("strain", True), M("tight", True)],"dove": [M("eased", True), M("easing", True), M("loos", True), M("soft", True), M("weak", True)]},{"block": "unemployment","terms": ["unemployment"],"hawk": [M("declin", True), M("fall", True), M("fell"), M("low", True), M("reduc", True)],"dove": [M("elevat", True), M("high"), M("increas", True), M("ris", True), M("rose", True)]}]
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

def tokenize(sent: str): return re.findall(r"[a-z]+", sent)
def match_token(tok: str, pat: str, wildcard: bool) -> bool: return tok.startswith(pat) if wildcard else tok == pat

def find_phrase_positions(tokens, phrase_tokens, wild_flags):
    m = len(phrase_tokens); hits = []
    for i in range(0, len(tokens) - m + 1):
        ok = True
        for j in range(m):
            if not match_token(tokens[i + j], phrase_tokens[j], wild_flags[j]): ok = False; break
        if ok: hits.append((i, i + m - 1))
    return hits

def find_term_positions_flex(tokens, term: str):
    tt = term.split(); m = len(tt); hits = []
    for i in range(0, len(tokens) - m + 1):
        window = tokens[i:i+m]; ok = True
        for j in range(m):
            if window[j] == tt[j] or window[j] == tt[j] + "s" or tt[j] == window[j] + "s": continue
            ok = False; break
        if ok: hits.append((i, i + m - 1))
    return hits

def select_non_overlapping_terms(tokens, term_infos):
    term_infos_sorted = sorted(term_infos, key=lambda x: len(x["term"].split()), reverse=True)
    occupied = set(); selected = []
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
            for term in b["terms"]: infos.append({"topic": topic, "block": b["block"], "term": term})
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
                w0 = max(0, ts - window_words); w1 = min(len(tokens) - 1, te + window_words)
                term_found = " ".join(tokens[ts:te + 1])
                hawk_hits = []
                for m in block["hawk"]:
                   for (ms, me) in find_phrase_positions(tokens, m["phrase"], m["wild"]):
                       if me < w0 or ms > w1: continue
                       dist = min(abs(ms - te), abs(ts - me)); hawk_hits.append((dist, m, ms, me))
                dove_hits = []
                for m in block["dove"]:
                   for (ms, me) in find_phrase_positions(tokens, m["phrase"], m["wild"]):
                       if me < w0 or ms > w1: continue
                       dist = min(abs(ms - te), abs(ts - me)); dove_hits.append((dist, m, ms, me))
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
                   matches.append({"topic": topic, "block": block["block"], "direction": direction, "term_found": term_found, "modifier_pattern": m["pattern"], "modifier_found": mod_found, "distance": dist, "sentence": sent, "term": tinfo["term"], "type": "HAWK" if direction == "hawk" else "DOVE"})
                for (dist, m, ms, me) in hawk_hits: add_hit("hawk", dist, m, ms, me)
                for (dist, m, ms, me) in dove_hits: add_hit("dove", dist, m, ms, me)
    hawk_total = sum(v["hawk"] for v in topic_counts.values())
    dove_total = sum(v["dove"] for v in topic_counts.values())
    denom = hawk_total + dove_total
    net_hawkishness = 1.0 if denom == 0 else (1.0 + (hawk_total - dove_total) / denom)
    return {"topic_counts": topic_counts, "matches": matches, "match_details": matches, "net_hawkishness": net_hawkishness, "hawk_count": hawk_total, "dove_count": dove_total}

def run_full_analysis(text):
    res = analyze_hawk_dove(text, DICT=DICT, window_words=10, dedupe_within_term_window=True, nearest_only=True)
    h_list = [f"{m['term_found']} ({m['modifier_found']})" for m in res['matches'] if m['direction'] == 'hawk']
    d_list = [f"{m['term_found']} ({m['modifier_found']})" for m in res['matches'] if m['direction'] == 'dove']
    h_ctx = {}; d_ctx = {}
    for m in res['matches']:
        if m['direction'] == 'hawk':
            if m['term_found'] not in h_ctx: h_ctx[m['term_found']] = []
            h_ctx[m['term_found']].append(m['sentence'])
        else:
            if m['term_found'] not in d_ctx: d_ctx[m['term_found']] = []
            d_ctx[m['term_found']].append(m['sentence'])
    return res['net_hawkishness'], res['hawk_count'], res['dove_count'], h_list, d_list, h_ctx, d_ctx, calculate_flesch_reading_ease(text)

def calculate_abg_scores(df):
    if df is None or df.empty: return pd.DataFrame()
    rows = []
    for _, row in df.iterrows():
        res = analyze_hawk_dove(str(row.get('text_content', '')), DICT=DICT, window_words=10, dedupe_within_term_window=True, nearest_only=True)
        donem = row.get("Donem", "")
        if not donem and "period_date" in row:
             try: donem = pd.to_datetime(row["period_date"]).strftime("%Y-%m")
             except: pass
        rows.append({"period_date": row.get("period_date"), "Donem": donem, "abg_index": res['net_hawkishness']})
    return pd.DataFrame(rows)

# =============================================================================
# 7. ML ALGORİTMASI
# =============================================================================
@dataclass
class CFG:
    cap_low: int = -750; cap_high: int = 750; token_pattern: str = r"(?u)\b[0-9a-zçğıöşü]{2,}\b"; word_ngram: Tuple[int,int] = (1, 2)
    min_df: int = 1; max_df: float = 1.0; max_features: int = 20000; trend_window: int = 6; max_splits: int = 6
    half_life_days: float = 365.0; q_lo: float = 0.02; q_hi: float = 0.98; vol_factor: float = 1.0; vol_cap: float = 3.0
    unc_factor: float = 1.5; blend_cond: float = 0.65; blend_all: float = 0.35; fallback_cut_bps: float = -75.0; fallback_hike_bps: float = 75.0
cfg = CFG()

def normalize_tr_text(s: str) -> str: return str(s).lower().strip() if s else ""
def clip_bps(x): return np.clip(x, cfg.cap_low, cfg.cap_high)
def bps_to_direction(y_bps): y = np.asarray(y_bps, float); out = np.zeros_like(y, int); out[y < 0] = -1; out[y > 0] = 1; return out
def exp_time_weights(dates, half_life_days=cfg.half_life_days):
    t = (pd.to_datetime(dates) - pd.to_datetime(dates).min()).dt.days.values.astype(float)
    w = np.exp((np.log(2.0)/half_life_days) * t); return w / np.mean(w)
def rolling_slope(y, window):
    out = np.zeros_like(y, float)
    for i in range(len(y)):
        seg = y[max(0, i-window+1):i+1]
        if len(seg) < 3: out[i] = 0.0; continue
        out[i] = np.polyfit(np.arange(len(seg)), seg, 1)[0]
    return out
def choose_splits(n): return int(min(cfg.max_splits, max(3, n // 8)))
def rmse_metric(y_true, y_pred): return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def add_features(df, trend_window=cfg.trend_window):
    out = df.copy(); out["y_bps"] = clip_bps(out["rate_change_bps"].values); out["y_dir"] = bps_to_direction(out["y_bps"].values)
    out["prev_change_bps"] = clip_bps(out["y_bps"].shift(1).fillna(0.0).values)
    out["prev_abs_change"] = np.abs(out["prev_change_bps"].values); out["prev_sign"] = np.sign(out["prev_change_bps"].values).astype(int)
    streak, cur = [], 0
    for v in out["y_bps"].shift(1).fillna(0.0).values: cur = cur + 1 if float(v) == 0.0 else 0; streak.append(cur)
    out["hold_streak"] = np.array(streak, int)
    out["mean_abs_last3"] = (out["y_bps"].shift(1).fillna(0).abs() + out["y_bps"].shift(2).fillna(0).abs() + out["y_bps"].shift(3).fillna(0).abs()).values / 3.0
    med = float(out["date"].diff().dt.days.dropna().median()) if len(out) > 1 else 30.0
    out["days_since_prev"] = out["date"].diff().dt.days.fillna(med).clip(lower=0).astype(float)
    out["roll_mean_bps"] = out["y_bps"].rolling(trend_window, min_periods=1).mean()
    out["roll_std_bps"] = out["y_bps"].rolling(trend_window, min_periods=1).std().fillna(0.0)
    out["roll_slope_bps"] = rolling_slope(out["y_bps"].values, trend_window)
    out["momentum_bps"] = out["y_bps"] - out["roll_mean_bps"]
    base = float(out["roll_std_bps"].median()) if len(out) else 1.0; out["vol_ratio"] = (out["roll_std_bps"] / (base if base > 0 else 1.0)).replace([np.inf, -np.inf], 1.0).fillna(1.0)
    return out

KEYWORDS = ["enflasyon", "çekirdek", "fiyat", "beklenti", "talep", "iktisadi faaliyet", "büyüme", "kur", "kredi", "risk primi", "finansal koşul", "sıkı", "sıkılaşma", "gevşeme", "kararlı", "ilave", "gerekirse", "dezenflasyon", "inflation", "price", "growth"]
kw_transformer = FunctionTransformer(lambda s: np.asarray([[t.lower().count(k) for k in KEYWORDS] + [len(t)] for t in s.fillna("").astype(str).values], float), validate=False)

def build_preprocess(numeric_cols):
    return ColumnTransformer([
        ("w", TfidfVectorizer(token_pattern=cfg.token_pattern, ngram_range=cfg.word_ngram, min_df=cfg.min_df, max_df=cfg.max_df, max_features=cfg.max_features, sublinear_tf=True), "text"),
        ("kw", Pipeline([("kw", kw_transformer), ("sc", StandardScaler(with_mean=False))]), "text"),
        ("num", Pipeline([("sc", StandardScaler(with_mean=False))]), numeric_cols),
    ], remainder="drop", sparse_threshold=0.3)

def build_models(preprocess):
    clf = LogisticRegression(solver="saga", max_iter=5000, class_weight="balanced", C=2.0, random_state=42)
    reg = Ridge(alpha=2.0, random_state=42)
    return Pipeline([("prep", clone(preprocess)), ("clf", clf)]), Pipeline([("prep", clone(preprocess)), ("reg", reg)]), Pipeline([("prep", clone(preprocess)), ("reg", Ridge(alpha=2.0))]), Pipeline([("prep", clone(preprocess)), ("reg", Ridge(alpha=2.0))])

def walk_forward_fast(X, y_bps, y_dir, dates, clf_pipe, reg_cut_pipe, reg_hike_pipe, reg_all_pipe, n_splits):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    y_pred = np.full(len(y_bps), np.nan); dir_pred = np.full(len(y_bps), np.nan); conf_pred = np.full(len(y_bps), np.nan)
    residuals = []; residuals_by_dir = {-1: [], 0: [], 1: []}
    for tr, te in tscv.split(X):
        w_tr = exp_time_weights(dates.iloc[tr])
        clf_pipe.fit(X.iloc[tr], y_dir[tr], clf__sample_weight=w_tr)
        d_hat = clf_pipe.predict(X.iloc[te]).astype(int)
        conf_te = clf_pipe.predict_proba(X.iloc[te]).max(axis=1) if hasattr(clf_pipe.named_steps["clf"], "predict_proba") else np.ones(len(te))
        reg_all_pipe.fit(X.iloc[tr], y_bps[tr], reg__sample_weight=w_tr)
        tr_cut = tr[y_dir[tr] == -1]; tr_hike = tr[y_dir[tr] == 1]
        if len(tr_cut) >= 8: reg_cut_pipe.fit(X.iloc[tr_cut], y_bps[tr_cut], reg__sample_weight=exp_time_weights(dates.iloc[tr_cut]))
        if len(tr_hike) >= 8: reg_hike_pipe.fit(X.iloc[tr_hike], y_bps[tr_hike], reg__sample_weight=exp_time_weights(dates.iloc[tr_hike]))
        for j, idx in enumerate(te):
            d = int(d_hat[j]); conf_pred[idx] = float(conf_te[j]); pred_all = float(reg_all_pipe.predict(X.iloc[[idx]])[0])
            if d == 0: pred_cond = 0.0
            elif d == -1: pred_cond = float(reg_cut_pipe.predict(X.iloc[[idx]])[0]) if len(tr_cut) >= 8 else cfg.fallback_cut_bps
            else: pred_cond = float(reg_hike_pipe.predict(X.iloc[[idx]])[0]) if len(tr_hike) >= 8 else cfg.fallback_hike_bps
            pred = float(clip_bps(cfg.blend_cond * pred_cond + cfg.blend_all * pred_all))
            y_pred[idx] = pred; dir_pred[idx] = d; res = float(y_bps[idx] - pred); residuals.append(res); residuals_by_dir[d].append(res)
    return y_pred, dir_pred, conf_pred, residuals, residuals_by_dir

def compute_interval(residuals, residuals_by_dir):
    def qpair(arr): return (-250.0, 250.0) if len(arr) < 20 else (float(np.quantile(arr, cfg.q_lo)), float(np.quantile(arr, cfg.q_hi)))
    return qpair(residuals), {d: qpair(residuals_by_dir.get(d, [])) for d in [-1,0,1]}

def fit_final(X, y_bps, y_dir, dates, clf, r_cut, r_hike, r_all):
    w = exp_time_weights(dates); clf.fit(X, y_dir, clf__sample_weight=w); r_all.fit(X, y_bps, reg__sample_weight=w)
    cut = np.where(y_dir == -1)[0]; hike = np.where(y_dir == 1)[0]
    if len(cut) >= 8: r_cut.fit(X.iloc[cut], y_bps[cut], reg__sample_weight=exp_time_weights(dates.iloc[cut]))
    if len(hike) >= 8: r_hike.fit(X.iloc[hike], y_bps[hike], reg__sample_weight=exp_time_weights(dates.iloc[hike]))

def prepare_ml_dataset(df_logs, df_market):
    """GÜVENLİ VERİ HAZIRLAMA (KeyError Fix)"""
    if df_logs.empty or df_market.empty: return pd.DataFrame()
    # Donem yoksa oluştur
    if 'Donem' not in df_logs.columns and 'period_date' in df_logs.columns:
        df_logs = df_logs.copy()
        df_logs['period_date'] = pd.to_datetime(df_logs['period_date'])
        df_logs['Donem'] = df_logs['period_date'].dt.strftime('%Y-%m')
    if 'Donem' not in df_market.columns: return pd.DataFrame()
    
    df = pd.merge(df_logs, df_market, on="Donem", how="left").sort_values("period_date")
    df['rate_change_bps'] = df['PPK Faizi'].diff().fillna(0.0) * 100
    df['text'] = df['text_content'].fillna("").apply(normalize_tr_text)
    return pd.DataFrame({"date": df['period_date'], "text": df['text'], "rate_change_bps": df['rate_change_bps']}).dropna()

class AdvancedMLPredictor:
    def __init__(self): self.clf_pipe = None; self.reg_pipes = {}; self.intervals = {}; self.df_hist = None
    def train(self, ml_df):
        if not HAS_ML_DEPS: return "Kütüphane eksik"
        df = add_features(ml_df); self.df_hist = df
        num_cols = ["prev_change_bps", "prev_abs_change", "prev_sign", "hold_streak", "mean_abs_last3", "days_since_prev", "roll_mean_bps", "roll_std_bps", "roll_slope_bps", "momentum_bps", "vol_ratio"]
        X = df[["text"] + num_cols]; y_bps = df["y_bps"].values.astype(float); y_dir = df["y_dir"].values.astype(int); dates = df["date"]
        prep = build_preprocess(num_cols); clf, r_cut, r_hike, r_all = build_models(prep)
        y_p, d_p, c_p, res, res_dir = walk_forward_fast(X, y_bps, y_dir, dates, clf, r_cut, r_hike, r_all, choose_splits(len(df)))
        self.df_hist['predicted_bps'] = y_p
        self.intervals['overall'], self.intervals['by_dir'] = compute_interval(res, res_dir)
        fit_final(X, y_bps, y_dir, dates, clf, r_cut, r_hike, r_all)
        self.clf_pipe = clf; self.reg_pipes = {'cut': r_cut, 'hike': r_hike, 'all': r_all}
        return "OK"
    def predict(self, text):
        if self.df_hist is None or self.clf_pipe is None: return None
        row = add_features(pd.concat([self.df_hist, pd.DataFrame({"date":[self.df_hist.iloc[-1]["date"]], "text":[normalize_tr_text(text)], "rate_change_bps":[0]})], ignore_index=True)).iloc[[-1]]
        d_hat = int(self.clf_pipe.predict(row)[0])
        conf = float(self.clf_pipe.predict_proba(row).max()) if hasattr(self.clf_pipe.named_steps["clf"], "predict_proba") else 1.0
        p_all = float(self.reg_pipes['all'].predict(row)[0])
        p_cond = 0.0 if d_hat == 0 else float(self.reg_pipes['cut' if d_hat == -1 else 'hike'].predict(row)[0])
        pred = float(clip_bps(cfg.blend_cond * p_cond + cfg.blend_all * p_all))
        lo, hi = self.intervals['by_dir'].get(d_hat, self.intervals['overall'])
        vr = float(row["vol_ratio"].iloc[0]); mult = (1.0 + cfg.vol_factor*max(0, vr-1)) * (1.0 + cfg.unc_factor*max(0, 1-conf))
        return {"pred_direction": {-1:"İNDİRİM", 0:"SABİT", 1:"ARTIRIM"}[d_hat], "direction_confidence": conf, "pred_change_bps": pred, "pred_interval_lo": float(clip_bps(pred + lo*mult)), "pred_interval_hi": float(clip_bps(pred + hi*mult))}

def calculate_vader_series(df): return pd.DataFrame()
def calculate_finbert_series(df): return pd.DataFrame()

# =============================================================================
# 8. YENİ STRUCTURAL HAWK/DOVE ALGORİTMASI
# =============================================================================
def M_struct(p, w=False): return {"phrase": p.split(), "wild": [True if w and i==0 else False for i in range(len(p.split()))], "pattern": p}
HAWK_DOVE_DICT_STRUCT = {
   "inflation": [{"block": "inf","terms": ["consumer prices", "inflation"],"hawk": [M_struct("accelerat", True), M_struct("high", True), M_struct("rise", True)],"dove": [M_struct("declin", True), M_struct("fall", True), M_struct("low", True)]}],
   "economic_activity": [{"block": "eco","terms": ["economic activity", "growth"],"hawk": [M_struct("strong", True), M_struct("increase", True)],"dove": [M_struct("weak", True), M_struct("slow", True)]}],
   "employment": [{"block": "emp","terms": ["employment"],"hawk": [M_struct("strong", True)],"dove": [M_struct("weak", True)]}]
} 

def normalize_text_struct(t): return normalize_text(t)
def split_sentences_struct(t): return split_sentences_nlp(t)
def tokenize_struct(t): return tokenize(t)
def find_phrase_positions_struct(t, p, w): return find_phrase_positions(t, p, w)
def select_non_overlapping_terms_struct(t, ti): return select_non_overlapping_terms(t, ti)

def analyze_hawk_dove_structural(text: str, window_words: int = 7, dedupe_within_term_window: bool = True, nearest_only: bool = True):
    if not text:
        return {"net_hawkishness": 0.0, "hawk_total": 0, "dove_total": 0, "topic_counts": {}, "matches_df": pd.DataFrame()}
    
    # Mevcut DICT yapısını kullanarak analizi çalıştır (Code reuse)
    res = analyze_hawk_dove(text, DICT, window_words, dedupe_within_term_window, nearest_only)
    
    # Listeyi DataFrame'e çevir
    df_matches = pd.DataFrame(res['matches']) if res['matches'] else pd.DataFrame()
    
    return {
        "net_hawkishness": res['net_hawkishness'],
        "hawk_total": res['hawk_count'],
        "dove_total": res['dove_count'],
        "topic_counts": res['topic_counts'],
        "matches_df": df_matches # ARTIK BU KEY GARANTİ VAR
    }

# =============================================================================
# 9. CENTRAL BANK RoBERTa ENTEGRASYONU (Moritz-Pfeifer)
# =============================================================================
@st.cache_resource
def load_roberta_pipeline():
    try:
        from transformers import pipeline
        # Moritz-Pfeifer modeli Merkez Bankası metinleri için özeldir.
        return pipeline("text-classification", model="Moritz-Pfeifer/CentralBankRoBERTa-sentiment-classifier", return_all_scores=True)
    except: return None

def get_label_mapping(label):
    lbl = label.lower()
    if "hawkish" in lbl: return "🦅 Şahin"
    if "dovish" in lbl: return "🕊️ Güvercin"
    if "positive" in lbl: return "🦅 Şahin (Pozitif)"
    if "negative" in lbl: return "🕊️ Güvercin (Negatif)"
    if "neutral" in lbl: return "⚖️ Nötr"
    return lbl.capitalize()

def analyze_with_roberta(text):
    clf = load_roberta_pipeline()
    if not clf: return None
    try:
        res = clf(text[:2000])[0]
        scores = {r['label'].lower(): r['score'] for r in res}
        best_label_raw = max(scores, key=scores.get)
        best_score = scores[best_label_raw]
        best_label_tr = get_label_mapping(best_label_raw)
        all_scores_tr = {get_label_mapping(k): v for k, v in scores.items()}
        return {"best_label": best_label_tr, "best_score": best_score, "all_scores": all_scores_tr}
    except Exception as e: return f"Error: {e}"

def analyze_sentences_with_roberta(text):
    """
    Cümle cümle analiz yapar. List of List hatasını düzeltir.
    """
    clf = load_roberta_pipeline()
    if not clf: return pd.DataFrame()
    
    sents = split_sentences_nlp(text)
    sents = [s for s in sents if len(s.split()) > 3]
    if not sents: return pd.DataFrame()
    
    res_list = []
    try:
        preds = clf(sents)
        for s, p in zip(sents, preds):
            if isinstance(p, list): best = max(p, key=lambda x: x['score'])
            else: best = p
            raw_lbl = best['label'].lower()
            tr_lbl = get_label_mapping(raw_lbl)
            res_list.append({"Cümle": s, "Etiket": tr_lbl, "Güven Skoru": best['score'], "Ham": raw_lbl})
        df = pd.DataFrame(res_list)
        if not df.empty: df = df.sort_values(by=["Ham", "Güven Skoru"], ascending=[False, False])
        return df
    except: return pd.DataFrame()

# =============================================================================
# 10. TARİHSEL RoBERTa HESAPLAMA (DASHBOARD - DÜZ ÇİZGİ ve TOOLTIP FIX)
# =============================================================================
@st.cache_data
def calculate_roberta_series(df):
    """
    Geçmiş verileri RoBERTa ile puanlar.
    DÜZELTME: Dovish/Negative etiketlerinin negatif (-) puana dönüşmesi garanti altına alındı.
    DÜZELTME: Dashboard Tooltip için metin eklendi.
    """
    if df.empty: return pd.DataFrame()
    
    classifier = load_roberta_pipeline()
    if not classifier or classifier == "MISSING_LIB": return pd.DataFrame()

    results = []
    
    # İlerlemeyi görmek için (opsiyonel)
    print("RoBERTa Geçmiş Analizi Başlatılıyor...")

    for _, row in df.iterrows():
        text = str(row.get('text_content', ''))
        # Çok kısa metinleri atla
        if len(text.split()) < 5: continue
        
        try:
            # Token limiti
            res = classifier(text[:1500])[0] 
            
            # Tüm skorları al
            scores = {r['label'].lower(): r['score'] for r in res}
            
            # En yüksek olasılıklı etiketi bul
            best_label = max(scores, key=scores.get)
            confidence = scores[best_label]
            
            final_val = 0.0
            display_text = ""
            
            # --- KRİTİK MANTIK DÜZELTMESİ ---
            # Hawkish veya Positive -> Pozitif Skor (+)
            if 'hawkish' in best_label or 'positive' in best_label:
                final_val = 100.0 * confidence
                display_text = f"🦅 Şahin %{confidence*100:.1f}"
            
            # Dovish veya Negative -> Negatif Skor (-)
            elif 'dovish' in best_label or 'negative' in best_label:
                final_val = -100.0 * confidence
                display_text = f"🕊️ Güvercin %{confidence*100:.1f}"
            
            # Neutral -> 0
            else:
                final_val = 0.0
                display_text = f"⚖️ Nötr %{confidence*100:.1f}"
            
            results.append({
                "period_date": row.get("period_date"),
                "roberta_index": final_val,
                "roberta_desc": display_text # <--- DASHBOARD İÇİN EKLENEN METİN
            })
        except Exception as e:
            print(f"Hata: {e}")
            continue
            
    return pd.DataFrame(results)
