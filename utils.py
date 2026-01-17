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

# =============================================================================
# 7. ML ALGORİTMASI (Ridge + Logistic)
# =============================================================================

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
# 8. CENTRAL BANK RoBERTa ENTEGRASYONU (GÜÇLENDİRİLMİŞ VERSİYON)
# =============================================================================

@st.cache_resource
def load_roberta_pipeline():
    try:
        from transformers import pipeline
        model_name = "gtfintechlab/FOMC-RoBERTa"
        # 'top_k=None' tüm skorları döndür demektir (yeni versiyonlarda return_all_scores yerine geçer)
        classifier = pipeline("text-classification", model=model_name, top_k=None)
        return classifier
    except ImportError:
        return "MISSING_LIB"
    except Exception as e:
        print(f"⚠️ Model Yükleme Hatası: {e}")
        return None

def normalize_label(raw_label):
    """
    Modelden gelen ham etiketi standart formata çevirir.
    gtfintechlab genelde: 0->Dovish, 1->Hawkish, 2->Neutral kullanır.
    """
    lbl = raw_label.lower().strip()
    
    # 1. Kelime bazlı kontrol
    if "hawkish" in lbl or "positive" in lbl: return "HAWK"
    if "dovish" in lbl or "negative" in lbl: return "DOVE"
    if "neutral" in lbl: return "NEUT"
    
    # 2. Label ID bazlı kontrol (gtfintechlab/FOMC özelinde)
    # LABEL_0 -> Dovish, LABEL_1 -> Hawkish, LABEL_2 -> Neutral (Genel varsayım)
    if "label_1" in lbl: return "HAWK"
    if "label_0" in lbl: return "DOVE"
    if "label_2" in lbl: return "NEUT"
    
    return "NEUT" # Hiçbiri uymazsa Nötr kabul et

def analyze_with_roberta(text):
    if not text: return None
        
    classifier = load_roberta_pipeline()
    if classifier == "MISSING_LIB": return "MISSING_LIB"
    if classifier is None: return "ERROR"

    # Token limiti
    truncated_text = text[:2000] 
    
    try:
        # Pipeline çıktısı genelde [[{'label': 'X', 'score': 0.9}, ...]] şeklindedir
        raw_results = classifier(truncated_text)
        
        # Eğer iç içe liste gelirse düzelt
        if isinstance(raw_results, list) and isinstance(raw_results[0], list):
            results = raw_results[0]
        else:
            results = raw_results

        # Skorları toplayacağımız sözlük
        scores_map = {"HAWK": 0.0, "DOVE": 0.0, "NEUT": 0.0}
        
        best_score = -1
        best_raw_label = ""
        
        for r in results:
            lbl_raw = str(r['label'])
            score = float(r['score'])
            
            # Etiketi normalize et
            std_lbl = normalize_label(lbl_raw)
            scores_map[std_lbl] = score
            
            if score > best_score:
                best_score = score
                best_raw_label = lbl_raw
        
        # Konsola bilgi bas (Hata ayıklamak için)
        print(f"DEBUG AI -> Raw: {best_raw_label} | Scores: H={scores_map['HAWK']:.3f} D={scores_map['DOVE']:.3f}")

        # En yüksek skora sahip olanın insan dostu ismini bul
        std_best = normalize_label(best_raw_label)
        human_label = "⚖️ Nötr"
        if std_best == "HAWK": human_label = "🦅 Şahin"
        elif std_best == "DOVE": human_label = "🕊️ Güvercin"
        
        # Dönüş formatı (Standartlaştırılmış anahtarlar kullanıyoruz)
        return {
            "best_label": human_label,
            "best_score": best_score,
            "scores_map": scores_map # Özel standart harita
        }

    except Exception as e:
        print(f"⚠️ Analiz Hatası: {e}")
        return f"Error: {str(e)}"

def analyze_sentences_with_roberta(text):
    if not text: return pd.DataFrame()
    classifier = load_roberta_pipeline()
    if not classifier or classifier == "MISSING_LIB": return pd.DataFrame()

    sentences = split_sentences_nlp(text)
    sentences = [s for s in sentences if len(s.split()) > 3]
    if not sentences: return pd.DataFrame()
    
    results_list = []
    
    try:
        predictions = classifier(sentences)
        
        for sent, pred in zip(sentences, predictions):
            # En yüksek skorlu tahmini al
            if isinstance(pred, list):
                best_pred = max(pred, key=lambda x: x['score'])
            else:
                best_pred = pred

            lbl_raw = str(best_pred['label'])
            score = best_pred['score']
            
            # Normalize et
            std_lbl = normalize_label(lbl_raw)
            
            label_tr = "⚖️ Nötr"
            if std_lbl == "HAWK": label_tr = "🦅 Şahin"
            elif std_lbl == "DOVE": label_tr = "🕊️ Güvercin"
            
            results_list.append({
                "Cümle": sent,
                "Etiket": label_tr,
                "Güven Skoru": score,
                "Ham Etiket": lbl_raw
            })
            
        df = pd.DataFrame(results_list)
        if not df.empty:
            sorter = {"🦅 Şahin": 1, "🕊️ Güvercin": 2, "⚖️ Nötr": 3}
            df['sort_key'] = df['Etiket'].map(sorter).fillna(4)
            df = df.sort_values(by=['sort_key', 'Güven Skoru'], ascending=[True, False]).drop(columns=['sort_key'])
            
        return df

    except Exception as e:
        print(f"⚠️ Cümle Analizi Hatası: {e}")
        return pd.DataFrame()

def calculate_ai_trend_series(df_all):
    """
    Tüm veri setini tarayıp AI skorlarını hesaplar.
    """
    if not HAS_TRANSFORMERS or df_all.empty:
        return pd.DataFrame()

    df_all = df_all.copy()
    df_all['period_date'] = pd.to_datetime(df_all['period_date'])
    df_all = df_all.sort_values('period_date')
    
    results = []
    
    print("--- AI TREND ANALİZİ BAŞLIYOR ---")
    
    for i, row in df_all.iterrows():
        text = row['text_content']
        date_str = row['period_date'].strftime('%Y-%m')
        
        ai_res = analyze_with_roberta(text)
        
        net_score = 0.0
        hawk_prob = 0.0
        dove_prob = 0.0
        
        if isinstance(ai_res, dict) and 'scores_map' in ai_res:
            scores = ai_res['scores_map']
            
            # ARTIK STANDART KEY'LER KULLANIYORUZ
            hawk_prob = scores.get("HAWK", 0.0)
            dove_prob = scores.get("DOVE", 0.0)
            
            # Net Skor
            net_score = (hawk_prob - dove_prob) * 100
        
        results.append({
            "Dönem": date_str,
            "period_date": row['period_date'],
            "Net Skor": net_score,
            "Şahin Olasılık": hawk_prob,
            "Güvercin Olasılık": dove_prob
        })
    
    print("--- AI TREND ANALİZİ BİTTİ ---")
    return pd.DataFrame(results)

# =============================================================================
# 8. CENTRAL BANK ANALİZİ (KARARLI SÜRÜM / STABLE VERSION)
# Model: mrince/CBRT-RoBERTa-Large-HawkishDovish-Classifier
# =============================================================================
import gc
import time

# Modeli önbelleğe alıyoruz ki her defasında RAM şişirmesin
@st.cache_resource(show_spinner=False)
def load_roberta_pipeline():
    try:
        from transformers import pipeline
        # İSTEDİĞİN MODEL (LINKTEKİ)
        model_name = "mrince/CBRT-RoBERTa-Large-HawkishDovish-Classifier"
        classifier = pipeline("text-classification", model=model_name, top_k=None, truncation=True)
        return classifier
    except Exception as e:
        print(f"Model Yükleme Hatası: {e}")
        return None

def normalize_label_mrince(raw_label):
    """
    DOĞRU LABEL MAP:
      LABEL_0 = neutral
      LABEL_1 = hawkish
      LABEL_2 = dovish
    """
    lbl = str(raw_label).lower().strip()

    if "label_0" in lbl or "neutral" in lbl:
        return "NEUT"
    if "label_1" in lbl or "hawkish" in lbl:
        return "HAWK"
    if "label_2" in lbl or "dovish" in lbl:
        return "DOVE"
    return "NEUT"

def analyze_with_roberta(text):
    if not text:
        return None

    classifier = load_roberta_pipeline()
    if classifier is None:
        return "ERROR"

    # RAM KORUMASI
    truncated_text = str(text)[:1500]

    try:
        raw_results = classifier(truncated_text)

        # Sonuç formatını düzelt
        if isinstance(raw_results, list) and raw_results and isinstance(raw_results[0], list):
            results = raw_results[0]
        else:
            results = raw_results

        scores_map = {"HAWK": 0.0, "DOVE": 0.0, "NEUT": 0.0}

        for r in results:
            lbl_raw = str(r.get('label', ''))
            score = float(r.get('score', 0.0))
            std_lbl = normalize_label_mrince(lbl_raw)
            scores_map[std_lbl] = score

        # app.py iki farklı format bekliyor gibi; ikisini de sağlıyoruz:
        best_std = max(scores_map, key=lambda k: scores_map[k])
        best_label = "⚖️ Nötr"
        if best_std == "HAWK":
            best_label = "🦅 Şahin"
        elif best_std == "DOVE":
            best_label = "🕊️ Güvercin"

        return {
            "best_label": best_label,
            "best_score": float(scores_map[best_std]),
            "scores_map": scores_map,
            # app.py'de bazen all_scores aranıyor, uyumluluk için ekliyoruz:
            "all_scores": {
                "🦅 Şahin (Hawkish)": float(scores_map["HAWK"]),
                "🕊️ Güvercin (Dovish)": float(scores_map["DOVE"]),
                "⚖️ Nötr (Neutral)": float(scores_map["NEUT"]),
            }
        }

    except Exception as e:
        return f"Error: {str(e)}"

def calculate_ai_trend_series(df_all):
    if df_all is None or df_all.empty:
        return pd.DataFrame()

    df_all = df_all.copy()
    df_all['period_date'] = pd.to_datetime(df_all['period_date'])
    df_all = df_all.sort_values('period_date')

    results = []

    st.toast("Analiz başladı...", icon="⏳")
    progress_bar = st.progress(0)
    total_rows = len(df_all)

    for i, (idx, row) in enumerate(df_all.iterrows()):
        text = str(row.get('text_content', ''))
        date_str = row['period_date'].strftime('%Y-%m')

        progress_bar.progress((i + 1) / total_rows, text=f"İşleniyor: {date_str}")

        if len(text) < 10:
            continue

        ai_res = analyze_with_roberta(text)

        gc.collect()

        if isinstance(ai_res, str):
            continue

        scores = ai_res.get('scores_map', {}) if isinstance(ai_res, dict) else {}
        hawk_prob = float(scores.get("HAWK", 0.0))
        dove_prob = float(scores.get("DOVE", 0.0))
        net_score = (hawk_prob - dove_prob) * 100

        results.append({
            "Dönem": date_str,
            "period_date": row['period_date'],
            "Net Skor": net_score,
            "Şahin Olasılık": hawk_prob,
            "Güvercin Olasılık": dove_prob
        })

    progress_bar.empty()
    st.toast("Analiz tamamlandı!", icon="✅")

    return pd.DataFrame(results)

def create_ai_trend_chart(df_res):
    import plotly.graph_objects as go
    if df_res is None or df_res.empty:
        return None

    fig_trend = go.Figure()

    fig_trend.add_trace(go.Scatter(
        x=df_res['Dönem'],
        y=df_res['Net Skor'],
        mode='lines',
        name='Trend',
        line=dict(color='gray', width=1, dash='dot')
    ))

    fig_trend.add_trace(go.Scatter(
        x=df_res['Dönem'],
        y=df_res['Net Skor'],
        mode='markers',
        name='Aylık Durum',
        marker=dict(
            size=14,
            color=df_res['Net Skor'],
            colorscale='RdBu_r',
            cmin=-100,
            cmax=100,
            showscale=True,
            colorbar=dict(title="Duruş")
        ),
        hovertemplate="<b>%{x}</b><br>Net Skor: %{y:.1f}<extra></extra>"
    ))

    fig_trend.add_hline(y=0, line_color="black", opacity=0.3)

    fig_trend.update_layout(
        title="🇹🇷 TCMB Metin Analizi (mrince Model)",
        yaxis=dict(title="Net Skor", range=[-110, 110]),
        height=450,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig_trend
