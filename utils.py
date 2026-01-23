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

        return {
            "scores_map": scores_map,
            "best_score": float(best_score),
            "diff": float(diff),
            "stance": stance_3class_from_diff(diff),
            "label_map": _mrince_label_map(),  # debug için (istersen UI'da göster)
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

    regime = []
    prev = "⚖️ Nötr"
    for v in out["AI Score (EMA)"].values:
        v = float(v)
        if prev in ["⚖️ Nötr", "🦅 Şahin"] and v >= hyst:
            prev = "🦅 Şahin"
        elif prev in ["⚖️ Nötr", "🕊️ Güvercin"] and v <= -hyst:
            prev = "🕊️ Güvercin"
        elif abs(v) < hyst * 0.6:
            prev = "⚖️ Nötr"
        regime.append(prev)

    out["AI Rejim"] = regime
    return out


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
        diff = float(res.get("diff", h - d))

        results.append({
            "Dönem": period,
            "period_date": dt,
            "Şahin Olasılık": h,
            "Güvercin Olasılık": d,
            "Nötr Olasılık": n,
            "Diff (H-D)": diff,
            "Duruş": str(res.get("stance", "")),
            "Güven": float(res.get("best_score", 0.0)),
        })

        gc.collect()

    pb.empty()
    st.toast("AI analiz tamamlandı!", icon="✅")

    out = pd.DataFrame(results)
    if out.empty:
        return out

    out = out.sort_values("period_date").reset_index(drop=True)
    out = postprocess_ai_series(out, diff_col="Diff (H-D)", span=7, z_scale=2.0, hyst=25.0)
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
        title="mrince RoBERTa — AI Duruş Trendi (Calib + EMA + Hysteresis)",
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

