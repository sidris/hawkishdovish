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

# --- 1. EK KÜTÜPHANELER ---
try:
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.pyplot as plt
    HAS_ML_DEPS = True
except ImportError:
    HAS_ML_DEPS = False

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
    supabase = None

EVDS_BASE = "https://evds2.tcmb.gov.tr/service/evds"
EVDS_TUFE_SERIES = "TP.FG.J0"

# =============================================================================
# 3. VERİTABANI İŞLEMLERİ (MARKET LOGS & EVENTS)
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

# --- EVENTS (BU KISIM EKSİKTİ, EKLENDİ) ---
def fetch_events():
    if not supabase: return pd.DataFrame()
    try:
        res = supabase.table("event_logs").select("*").order("event_date", desc=True).execute()
        data = getattr(res, 'data', []) if res else []
        return pd.DataFrame(data)
    except Exception:
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

# --- MARKET DATA ---
@st.cache_data(ttl=600)
def fetch_market_data_adapter(start_date, end_date):
    if not EVDS_API_KEY: return pd.DataFrame(), "EVDS Anahtarı Eksik."
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
    except Exception as e: return pd.DataFrame(), f"TÜFE Hatası: {e}"

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
    except Exception as e: return pd.DataFrame(), f"BIS Hatası: {e}"

    master_df = pd.DataFrame()
    if not df_inf.empty and not df_pol.empty: master_df = pd.merge(df_inf, df_pol, on="Donem", how="outer")
    elif not df_inf.empty: master_df = df_inf
    elif not df_pol.empty: master_df = df_pol

    if master_df.empty: return pd.DataFrame(), "Veri bulunamadı."
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
        "trend", "period", "stance", "prices", "growth", "inflation"
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
           "hawk": [M("accelerat", True), M("boost", True), M("build", True), M("elevat", True), M("emerg", True), M("great", True), M("height", True), M("high", True), M("increas", True), M("intensif", True), M("mount", True), M("pickup"), M("rise"), M("rose"), M("rising"), M("stok", True), M("strong", True), M("sustain", True)],
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
# 6. ENTEGRASYON VE ML
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

def train_and_predict_rate(df_history, current_score):
    if not HAS_ML_DEPS: return None, None, "Kütüphane eksik"
    if df_history.empty or 'PPK Faizi' not in df_history.columns: return None, None, "Yetersiz Veri"
    
    df = df_history.sort_values('period_date').copy()
    df = df.drop_duplicates(subset=['Donem'], keep='last').copy()
    df['Next_Rate'] = df['PPK Faizi'].shift(-1)
    df['Rate_Change'] = df['Next_Rate'] - df['PPK Faizi']
    
    train_data = df.dropna(subset=['score_abg_scaled', 'Rate_Change'])
    if len(train_data) < 5: return None, None, "Model için en az 5 ay veri lazım."
    
    X = train_data[['score_abg_scaled']]
    y = train_data['Rate_Change']
    
    model = LinearRegression()
    model.fit(X, y)
    
    y_bps = (y * 100).round(0).astype(int)
    has_logit = False
    try:
        model_logit = LogisticRegression(max_iter=1000, random_state=42)
        model_logit.fit(X, y_bps)
        has_logit = True
    except: pass
    
    full_X = df.dropna(subset=['score_abg_scaled'])[['score_abg_scaled']]
    df.loc[full_X.index, 'Predicted_Change'] = model.predict(full_X)
    
    if has_logit:
        df.loc[full_X.index, 'Predicted_Change_Logit'] = model_logit.predict(full_X) / 100.0
    else:
        df.loc[full_X.index, 'Predicted_Change_Logit'] = np.nan
        
    prediction = model.predict([[current_score]])[0]
    prediction_logit = prediction
    if has_logit: prediction_logit = model_logit.predict([[current_score]])[0] / 100.0
    
    corr = np.corrcoef(train_data['score_abg_scaled'], train_data['Rate_Change'])[0,1]
    
    stats = {
        'prediction': prediction,
        'prediction_logit': prediction_logit,
        'correlation': corr,
        'sample_size': len(train_data),
        'coef': model.coef_[0]
    }
    
    return stats, df[['period_date', 'Donem', 'Rate_Change', 'Predicted_Change', 'Predicted_Change_Logit']].copy(), None
