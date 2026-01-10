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
    from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report, confusion_matrix
    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.pyplot as plt
    HAS_ML_DEPS = True
except ImportError:
    HAS_ML_DEPS = False

# VADER SENTIMENT KONTROLÜ
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except ImportError:
    HAS_VADER = False

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
except Exception as e:
    st.error(f"Ayarlar hatası: {e}")
    st.stop()

EVDS_BASE = "https://evds2.tcmb.gov.tr/service/evds"
EVDS_TUFE_SERIES = "TP.FG.J0"

# =============================================================================
# 3. VERİTABANI VE PİYASA VERİSİ
# =============================================================================

def fetch_all_data():
    if not supabase: return pd.DataFrame()
    try:
        res = supabase.table("market_logs").select("*").order("period_date", desc=True).execute()
        data = getattr(res, 'data', []) if res else []
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Veri çekme hatası: {e}")
        return pd.DataFrame()

def insert_entry(date, text, source, s_dict, s_abg):
    if not supabase: return
    try:
        data = {"period_date": str(date), "text_content": text, "source": source,
            "score_dict": s_dict, "score_abg": s_abg}
        supabase.table("market_logs").insert(data).execute()
    except Exception as e: st.error(f"Kayıt hatası: {e}")

def update_entry(rid, date, text, source, s_dict, s_abg):
    if not supabase: return
    try:
        data = {"period_date": str(date), "text_content": text, "source": source,
            "score_dict": s_dict, "score_abg": s_abg}
        supabase.table("market_logs").update(data).eq("id", rid).execute()
    except Exception as e: st.error(f"Güncelleme hatası: {e}")

def delete_entry(rid):
    if supabase: 
        try:
            supabase.table("market_logs").delete().eq("id", rid).execute()
        except Exception as e: st.error(f"Silme hatası: {e}")

# --- EKLENEN FONKSİYONLAR (EVENT LOGS) ---
def fetch_events():
    if not supabase: return pd.DataFrame()
    try:
        res = supabase.table("event_logs").select("*").order("event_date", desc=True).execute()
        data = getattr(res, 'data', []) if res else []
        return pd.DataFrame(data)
    except Exception as e:
        # Tablo yoksa hata vermemesi için sessiz geçebilir veya loglayabiliriz
        return pd.DataFrame()

def add_event(date, links):
    if not supabase: return
    try:
        data = {"event_date": str(date), "links": links}
        supabase.table("event_logs").insert(data).execute()
    except Exception as e: st.error(f"Olay ekleme hatası: {e}")

def delete_event(rid):
    if supabase:
        try:
            supabase.table("event_logs").delete().eq("id", rid).execute()
        except Exception as e: st.error(f"Olay silme hatası: {e}")

@st.cache_data(ttl=600)
def fetch_market_data_adapter(start_date, end_date):
    if not EVDS_API_KEY: return pd.DataFrame(), "EVDS Anahtarı Eksik."
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
    except Exception as e: return pd.DataFrame(), f"TÜFE Hatası: {e}"

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
    except Exception as e: return pd.DataFrame(), f"BIS Hatası: {e}"

    master_df = pd.DataFrame()
    if not df_inf.empty and not df_pol.empty: master_df = pd.merge(df_inf, df_pol, on="Donem", how="outer")
    elif not df_inf.empty: master_df = df_inf
    elif not df_pol.empty: master_df = df_pol

    if master_df.empty: return pd.DataFrame(), "Veri bulunamadı."
    master_df["SortDate"] = pd.to_datetime(master_df["Donem"] + "-01")
    return master_df.sort_values("SortDate"), None

# =============================================================================
# 4. METİN ANALİZİ (JS REFERANSLI FLESCH ALGORİTMASI)
# =============================================================================

NOUNS = {
    "cost","costs","expenditures","consumption","growth","output","demand","activity",
    "production","investment","productivity","labor","labour","job","jobs","participation",
    "wage","wages","recovery","slowdown","contraction","expansion","cycle","conditions",
    "credit","lending","borrowing","liquidity","stability","markets","volatility",
    "uncertainty","risks","easing","rates","policy","stance","outlook","pressures",
    "inflation","price","prices","gold",
    "oil price","oil prices","cyclical position","development","employment","unemployment"
}

HAWKISH_ADJECTIVES = {
    "high","higher","strong","stronger","increasing","increased","fast","faster","elevated","rising",
    "accelerating","robust","persistent","mounting","excessive","solid","resilient","vigorous",
    "overheating","tightening","restrictive","constrained","limited","upside","significant","notable"
}

DOVISH_ADJECTIVES = {
    "low","lower","weak","weaker","decreasing","decreased","slow","slower","falling","declining",
    "subdued","soft","softer","easing","moderate","moderating","cooling","softening","downside","adverse"
}

HAWKISH_SINGLE = {"tight","tightening","restrictive","elevated","high","overheating","pressures","pressure","risk","risks","upside","vigilant","decisive"}
DOVISH_SINGLE = {"disinflation","decline","declining","fall","falling","decrease","decreasing","lower","low","subdued","contained","anchored","cooling","slow","slower","improvement","better","easing","relief"}

def split_into_sentences(text):
    if not text: return []
    return re.split(r'[.!?]+', text)

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

def find_context_sentences(text, found_phrases):
    sentences = split_into_sentences(text)
    contexts = {}
    for phrase in found_phrases:
        matched_sentences = []
        for sent in sentences:
            if phrase in sent.lower():
                highlighted_sent = re.sub(f"({re.escape(phrase)})", r"**\1**", sent, flags=re.IGNORECASE)
                matched_sentences.append(highlighted_sent.strip())
        if matched_sentences:
            contexts[phrase] = matched_sentences
    return contexts

def make_ngrams(tokens, n):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def run_full_analysis(text):
    if not text: return 0, 0, 0, [], [], {}, {}, 0
    clean_text = text.lower()
    tokens = re.findall(r"[a-z']+", clean_text)
    token_counts = Counter(tokens)
    flesch_score = calculate_flesch_reading_ease(text)
    bigrams = make_ngrams(tokens, 2)
    trigrams = make_ngrams(tokens, 3)
    bigram_counts = Counter(bigrams)
    trigr_counts = Counter(trigrams)

    hawkish_phrases = {f"{adj} {noun}" for adj in HAWKISH_ADJECTIVES for noun in NOUNS}
    dovish_phrases  = {f"{adj} {noun}" for adj in DOVISH_ADJECTIVES  for noun in NOUNS}

    def phrase_count(phrase):
        n = len(phrase.split())
        if n == 2: return bigram_counts[phrase]
        elif n == 3: return trigr_counts[phrase]
        else: return 0

    used_hawkish_ngrams = {p: phrase_count(p) for p in hawkish_phrases if phrase_count(p) > 0}
    used_dovish_ngrams  = {p: phrase_count(p) for p in dovish_phrases  if phrase_count(p) > 0}
    hawk_ngram_count = sum(used_hawkish_ngrams.values())
    dove_ngram_count = sum(used_dovish_ngrams.values())
    used_hawkish_single = {w: token_counts[w] for w in HAWKISH_SINGLE if token_counts[w] > 0}
    used_dovish_single  = {w: token_counts[w] for w in DOVISH_SINGLE  if token_counts[w] > 0}
    hawk_single_count = sum(used_hawkish_single.values())
    dove_single_count = sum(used_dovish_single.values())
    hawk_total = hawk_ngram_count + hawk_single_count
    dove_total = dove_ngram_count + dove_single_count
    total_signal = hawk_total + dove_total
    if total_signal > 0:
        net_score = (float(hawk_total - dove_total) / float(total_signal)) * 100
    else:
        net_score = 0.0
    all_hawk_matches = {**used_hawkish_ngrams, **used_hawkish_single}
    all_dove_matches = {**used_dovish_ngrams, **used_dovish_single}
    hawk_list = [f"{k} ({v})" for k, v in sorted(all_hawk_matches.items(), key=lambda x: -x[1])]
    dove_list = [f"{k} ({v})" for k, v in sorted(all_dove_matches.items(), key=lambda x: -x[1])]
    hawk_contexts = find_context_sentences(text, all_hawk_matches.keys())
    dove_contexts = find_context_sentences(text, all_dove_matches.keys())
    return net_score, hawk_total, dove_total, hawk_list, dove_list, hawk_contexts, dove_contexts, flesch_score

# --- DERİN ANALİZ ARAÇLARI ---

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
    stops = set(["that", "with", "this", "from", "have", "which", "will", "been", "were", "market", "central", "bank", "committee", "monetary", "policy", "decision", "percent", "rates", "level"])
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
# 5. YENİ ML (RIDGE + LOGISTIC) ALGORİTMASI (GELİŞMİŞ TAHMİN)
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

# --- VERİ HAZIRLAMA & ANA MOTOR ---
def prepare_ml_dataset(df_logs, df_market):
    """DB verilerini ML motorunun beklediği formata sokar."""
    if df_logs.empty or df_market.empty: return pd.DataFrame()
    
    if 'Donem' not in df_logs.columns and 'period_date' in df_logs.columns:
        df_logs = df_logs.copy()
        df_logs['period_date'] = pd.to_datetime(df_logs['period_date'])
        df_logs['Donem'] = df_logs['period_date'].dt.strftime('%Y-%m')
    
    if 'Donem' not in df_market.columns:
        return pd.DataFrame()

    # 1. Merge
    df = pd.merge(df_logs, df_market, on="Donem", how="left")
    df = df.sort_values("period_date").reset_index(drop=True)
    
    # 2. Rate Change Calculation (Current - Previous)
    df['rate_change_bps'] = df['PPK Faizi'].diff().fillna(0.0) * 100
    
    # 3. Columns mapping
    ml_df = pd.DataFrame({
        "date": df['period_date'],
        "text": df['text_content'],
        "rate_change_bps": df['rate_change_bps']
    })
    
    # NaN temizliği
    ml_df = ml_df.dropna(subset=['text', 'rate_change_bps'])
    return ml_df

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
# 6. ABG (APEL, BLIX, GRIMALDI - 2019) ANALYZER
# =============================================================================

@dataclass(frozen=True)
class ModPattern:
    token_regexes: Tuple[re.Pattern, ...]

@dataclass(frozen=True)
class TermEntry:
    term_tokens: Tuple[str, ...]
    hawk_mods: Tuple[ModPattern, ...]
    dove_mods: Tuple[ModPattern, ...]

class ABG2019Analyzer:
    def __init__(self, window_size: int = 7):
        self.window_size = window_size
        self.entries: List[TermEntry] = self._build_dictionary_from_appendix()

    def split_sentences(self, text: str) -> List[str]:
        return re.split(r'[.!?]+', text)

    def tokenize(self, sentence: str) -> List[str]:
        s = sentence.lower()
        return re.findall(r"[a-z]+(?:-[a-z]+)*", s)

    def _compile_token_wildcard(self, token: str) -> re.Pattern:
        token = token.strip().lower()
        if "*" in token:
            base = re.escape(token.replace("*", ""))
            return re.compile(rf"^{base}[a-z-]*$")
        else:
            return re.compile(rf"^{re.escape(token)}$")

    def _compile_modifier(self, modifier: str) -> List[ModPattern]:
        modifier = modifier.strip().lower()
        if not modifier: return []
        alts = modifier.split("/")
        out: List[ModPattern] = []
        for alt in alts:
            alt = alt.strip()
            if not alt: continue
            parts = alt.split()
            token_regexes = tuple(self._compile_token_wildcard(p) for p in parts)
            out.append(ModPattern(token_regexes=token_regexes))
        return out

    def _entry(self, term: str, hawk_mods: List[str], dove_mods: List[str]) -> TermEntry:
        term_tokens = tuple(term.lower().split())
        h_patterns: List[ModPattern] = []
        d_patterns: List[ModPattern] = []
        for m in hawk_mods: h_patterns.extend(self._compile_modifier(m))
        for m in dove_mods: d_patterns.extend(self._compile_modifier(m))
        return TermEntry(term_tokens=term_tokens, hawk_mods=tuple(h_patterns), dove_mods=tuple(d_patterns))

    def _build_dictionary_from_appendix(self) -> List[TermEntry]:
        inflation_consumer_prices_hawk = ["accelerat*", "boost*", "elevat*", "escalat*", "high*", "increas*", "jump*", "pickup", "rise*", "rose", "rising", "run-up/runup", "strong*", "surg*", "up*"]
        inflation_consumer_prices_dove = ["decelerat*", "declin*", "decreas*", "down*", "drop*", "fall*", "fell", "low*", "muted", "reduc*", "slow*", "stable", "subdued", "weak*", "contained"]
        inflation_infl_pressure_hawk = ["accelerat*", "boost*", "build*", "elevat*", "emerg*", "great*", "height*", "high*", "increas*", "intensif*", "mount*", "pickup", "rise", "rose", "rising", "stok*", "strong*", "sustain*"]
        inflation_infl_pressure_dove = ["abat*", "contain*", "dampen*", "decelerat*", "declin*", "decreas*", "dimin*", "eas*", "fall*", "fell", "low*", "moderat*", "reced*", "reduc*", "subdued", "temper*"]
        econ_cons_spend_hawk = ["accelerat*", "edg* up", "expan*", "increas*", "pick* up", "pickup", "soft*", "strength*", "strong*", "weak*"]
        econ_cons_spend_dove = ["contract*", "decelerat*", "decreas*", "drop*", "retrench*", "slow*", "slugg*", "soft*", "subdued"]
        econ_activity_hawk = ["accelerat*"]; econ_activity_dove = ["contract*"]
        econ_growth_hawk = ["buoyant", "edg* up", "expan*", "increas*", "high*", "pick* up", "pickup", "rise*", "rose", "rising", "step* up", "strength*", "strong*", "upside"]
        econ_growth_dove = ["curtail*", "decelerat*", "declin*", "decreas*", "downside", "drop", "fall*", "fell", "low*", "moderat*", "slow*", "slugg*", "weak*"]
        resource_util_hawk = ["high*", "increas*", "rise", "rising", "rose", "tight*"]; resource_util_dove = ["declin*", "fall*", "fell", "loose*", "low*"]
        employment_hawk = ["expand*", "gain*", "improv*", "increas*", "pick* up", "pickup", "rais*", "rise*", "rising", "rose", "strength*", "turn* up"]
        employment_dove = ["slow*", "declin*", "reduc*", "weak*", "deteriorat*", "shrink*", "shrank", "fall*", "fell", "drop*", "contract*", "sluggish"]
        labor_market_hawk = ["strain*", "tight*"]; labor_market_dove = ["eased", "easing", "loos*", "soft*", "weak*"]
        unemployment_hawk = ["declin*", "fall*", "fell", "low*", "reduc*"]; unemployment_dove = ["elevat*", "high", "increas*", "ris*", "rose*"]

        return [
            self._entry("consumer prices", inflation_consumer_prices_hawk, inflation_consumer_prices_dove),
            self._entry("inflation",       inflation_consumer_prices_hawk, inflation_consumer_prices_dove),
            self._entry("inflation pressure", inflation_infl_pressure_hawk, inflation_infl_pressure_dove),
            self._entry("consumer spending", econ_cons_spend_hawk, econ_cons_spend_dove),
            self._entry("economic activity", econ_activity_hawk, econ_activity_dove),
            self._entry("economic growth",   econ_growth_hawk, econ_growth_dove),
            self._entry("resource utilization", resource_util_hawk, resource_util_dove),
            self._entry("employment", employment_hawk, employment_dove),
            self._entry("labor market", labor_market_hawk, labor_market_dove),
            self._entry("unemployment", unemployment_hawk, unemployment_dove),
        ]

    def _find_term_spans(self, tokens: List[str]) -> List[Tuple[int, int, TermEntry]]:
        raw: List[Tuple[int, int, TermEntry]] = []
        n = len(tokens)
        for entry in self.entries:
            t = entry.term_tokens
            L = len(t)
            if L == 0: continue
            for i in range(0, n - L + 1):
                if tuple(tokens[i:i+L]) == t: raw.append((i, i+L, entry))
        raw.sort(key=lambda x: (-(x[1]-x[0]), x[0]))
        chosen: List[Tuple[int, int, TermEntry]] = []
        occupied = [False] * n
        for s, e, entry in raw:
            if any(occupied[k] for k in range(s, e)): continue
            chosen.append((s, e, entry))
            for k in range(s, e): occupied[k] = True
        chosen.sort(key=lambda x: x[0])
        return chosen

    def _match_modifier_at(self, tokens: List[str], pos: int, pat: ModPattern) -> bool:
        L = len(pat.token_regexes)
        if pos + L > len(tokens): return False
        for j in range(L):
            if not pat.token_regexes[j].match(tokens[pos+j]): return False
        return True

    def analyze(self, text: str) -> Dict[str, Any]:
        sentences = self.split_sentences(text)
        hawk = 0; dove = 0; details: List[Dict[str, Any]] = []
        for sent in sentences:
            sent = sent.strip()
            if not sent: continue
            tokens = self.tokenize(sent)
            if not tokens: continue
            term_spans = self._find_term_spans(tokens)
            for (ts, te, entry) in term_spans:
                w_start = max(0, ts - self.window_size)
                w_end = min(len(tokens), te + self.window_size)
                for pat in entry.hawk_mods:
                    L = len(pat.token_regexes)
                    for p in range(w_start, w_end - L + 1):
                        if self._match_modifier_at(tokens, p, pat):
                            hawk += 1; mod_str = " ".join(tokens[p:p+L])
                            details.append({"type": "HAWK", "term": " ".join(entry.term_tokens), "modifier": mod_str, "sentence": sent})
                for pat in entry.dove_mods:
                    L = len(pat.token_regexes)
                    for p in range(w_start, w_end - L + 1):
                        if self._match_modifier_at(tokens, p, pat):
                            dove += 1; mod_str = " ".join(tokens[p:p+L])
                            details.append({"type": "DOVE", "term": " ".join(entry.term_tokens), "modifier": mod_str, "sentence": sent})
        total = hawk + dove
        net = 1.0 + ((hawk - dove) / total) if total > 0 else 1.0
        return {"net_hawkishness": net, "hawk_count": hawk, "dove_count": dove, "total_matches": total, "match_details": details}

class ABGAnalyzer:
    def __init__(self): self.engine = ABG2019Analyzer(window_size=7)
    def analyze(self, text): return self.engine.analyze(text)

def calculate_abg_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    analyzer = ABG2019Analyzer(window_size=7); rows = []
    for _, row in df.iterrows():
        res = analyzer.analyze(str(row.get("text_content", "")))
        donem_val = row.get("Donem")
        if (donem_val is None or donem_val == "") and ("period_date" in row):
            try: donem_val = pd.to_datetime(row["period_date"]).strftime("%Y-%m")
            except Exception: donem_val = ""
        rows.append({"period_date": row.get("period_date"), "Donem": donem_val, "abg_index": res["net_hawkishness"], "abg_hawk": res["hawk_count"], "abg_dove": res["dove_count"]})
    return pd.DataFrame(rows).sort_values("period_date", ascending=False)

# =============================================================================
# 7. VADER ANALİZİ
# =============================================================================

def calculate_vader_series(df: pd.DataFrame) -> pd.DataFrame:
    if not HAS_VADER or df is None or df.empty: return pd.DataFrame()
    analyzer = SentimentIntensityAnalyzer()
    results = []
    for _, row in df.iterrows():
        text = str(row.get("text_content", ""))
        scores = analyzer.polarity_scores(text)
        
        donem_val = row.get("Donem")
        if (donem_val is None or donem_val == "") and ("period_date" in row):
            try: donem_val = pd.to_datetime(row["period_date"]).strftime("%Y-%m")
            except Exception: donem_val = ""

        results.append({
            "period_date": row.get("period_date"),
            "Donem": donem_val,
            "vader_compound": scores["compound"],
            "vader_pos": scores["pos"],
            "vader_neg": scores["neg"],
            "vader_neu": scores["neu"]
        })
    return pd.DataFrame(results).sort_values("period_date", ascending=True)
