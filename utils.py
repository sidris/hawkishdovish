import streamlit as st
from supabase import create_client, Client
import pandas as pd
import requests
import io
import datetime
import re
from collections import Counter, defaultdict

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

# =============================================================================
# 2. GELİŞMİŞ ALGORİTMA
# =============================================================================

WORD_RE = re.compile(r"[a-z']+")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# --- BAĞLAM SÖZLÜKLERİ ---
DOVE_CONTEXT_BIGRAMS = {("disinflationary", "levels"), ("disinflationary", "impact")}
DOVE_CONTEXT_UNIGRAMS = {"disinflationary", "weak", "weaken", "weakened", "slowdown", "slowed", "declined", "cooling", "moderation"}
DOVE_CONTEXT_TERMS = {"demand", "domestic", "conditions", "activity", "growth"}

HAWK_CONTEXT_UNIGRAMS = {"tight", "tightness", "tightening", "restrictive", "maintained", "maintain", "strengthen", "strengthening", "decisive", "prudently", "tools", "stability"}
HAWK_CONTEXT_TERMS = {"stance", "policy", "rate", "price", "stability", "inflation", "expectations", "lira"}

# --- GENEL SÖZLÜKLER ---
NOUNS = ["cost","costs","expenditures","consumption","growth","output","demand","activity","production","investment","productivity","labor","labour","job","jobs","participation","wage","wages","recovery","slowdown","contraction","expansion","cycle","conditions","credit","lending","borrowing","liquidity","stability","markets","volatility","uncertainty","risks","easing","rates","policy","stance","outlook","pressures","inflation","price", "prices","wage", "wages","oil price", "oil prices","cyclical position","growth","development","employment","unemployment","recovery","cost", "costs","gold"]

HAWKISH_ADJ = ["high", "higher","strong", "stronger","increasing", "increased","fast", "faster","elevated","rising","accelerating","robust","persistent","mounting","excessive","solid","resillent","vigorous","overheating","tightening","restrivtive","constrained","limited","upside","significant","notable"]
DOVISH_ADJ = ["low", "lower","weak", "weaker","decreasing", "decreased","slow", "slower","falling","declining","subdued","weak","weaker","soft","softer","easing","slow","slower","moderate","moderating","cooling","softening","downside","adverse"]

HAWKISH_SINGLE = {"tight","tightening","restrictive","elevated","high","overheating","pressures","pressure","risk","risks","upside","vigilant","decisive"}
DOVISH_SINGLE = {"disinflation","decline","declining","fall","falling","decrease","decreasing","lower","low","subdued","contained","anchored","cooling","slow","slower","improvement","improvement","better","easing","relief"}

HAWK_DICT = {}
DOVE_DICT = {}

for term in NOUNS:
    HAWK_DICT[term] = set(HAWKISH_ADJ)
    DOVE_DICT[term] = set(DOVISH_ADJ)

for w in HAWKISH_SINGLE: HAWK_DICT[w] = {w}
for w in DOVISH_SINGLE: DOVE_DICT[w] = {w}

HAWK_DICT["disinf_hawk"] = {"disinf_hawk"}
DOVE_DICT["disinf_dove"] = {"disinf_dove"}

# --- FONKSİYONLAR ---

def tokenize(text: str): return WORD_RE.findall(text.lower())

def build_positions(tokens):
    pos_uni = defaultdict(list)
    for i, w in enumerate(tokens): pos_uni[w].append(i)
    bigrams = [" ".join(tokens[i:i+2]) for i in range(len(tokens)-1)]
    pos_bi = defaultdict(list)
    for i, bg in enumerate(bigrams): pos_bi[bg].append(i)
    return pos_uni, pos_bi

def get_term_positions(term: str, pos_uni, pos_bi):
    return pos_bi.get(term, []) if " " in term else pos_uni.get(term, [])

def relabel_disinflation_in_sentence(tokens, window=6):
    out = tokens[:]
    bigrams = {(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)}
    for i, w in enumerate(tokens):
        if w == "disinflationary": out[i] = "disinf_dove"; continue
        if w != "disinflation": continue
        left = max(0, i - window); right = min(len(tokens), i + window + 1)
        ctx = tokens[left:right]; ctx_set = set(ctx)
        
        d_sc = 0; h_sc = 0
        if any(bg in bigrams for bg in DOVE_CONTEXT_BIGRAMS): d_sc += 2
        d_sc += sum(1 for t in ctx if t in DOVE_CONTEXT_UNIGRAMS)
        h_sc += sum(1 for t in ctx if t in HAWK_CONTEXT_UNIGRAMS)
        if ctx_set & DOVE_CONTEXT_TERMS: d_sc += 1
        if ctx_set & HAWK_CONTEXT_TERMS: h_sc += 1
        
        if d_sc > h_sc and d_sc >= 2: out[i] = "disinf_dove"
        elif h_sc > d_sc and h_sc >= 2: out[i] = "disinf_hawk"
        else: out[i] = "disinf_neutral"
    return out

def preprocess_disinflation(text: str):
    processed_sents = []
    for sent in SENT_SPLIT.split(text.strip()):
        toks = tokenize(sent)
        if not toks: continue
        processed_sents.append(" ".join(relabel_disinflation_in_sentence(toks, window=6)))
    return ". ".join(processed_sents)

def abg_2019_count(text: str, hawk_dict: dict, dove_dict: dict, window: int = 7):
    hawk_hits = Counter(); dove_hits = Counter()
    hawk_terms_sorted = sorted(hawk_dict.keys(), key=len, reverse=True)
    dove_terms_sorted = sorted(dove_dict.keys(), key=len, reverse=True)

    for sent in SENT_SPLIT.split(text.strip()):
        toks = tokenize(sent)
        if not toks: continue
        pos_uni, pos_bi = build_positions(toks)
        occupied_spans = []

        def overlaps(a, b): return not (a[1] <= b[0] or b[1] <= a[0])
        def is_occupied(start, length): return any(overlaps((start, start+length), sp) for sp in occupied_spans)
        def mark(start, length): occupied_spans.append((start, start+length))

        def count_side(term_list, term_to_mods, out_counter):
            for term in term_list:
                mods = term_to_mods[term]
                is_bigram = (" " in term); term_len = 2 if is_bigram else 1
                for term_i in get_term_positions(term, pos_uni, pos_bi):
                    if is_occupied(term_i, term_len): continue
                    left = max(0, term_i - window); right = min(len(toks), term_i + term_len + window + 1)
                    window_tokens = toks[left:right]
                    for m in mods:
                        c = window_tokens.count(m)
                        if c:
                            disp = term
                            if term == "disinf_hawk": disp = "disinflation (bağlam: şahin)"
                            if term == "disinf_dove": disp = "disinflation (bağlam: güvercin)"
                            key = f"{m} -> {disp}" if m != term else disp
                            out_counter[key] += c
                            mark(term_i, term_len)
                            break
        
        count_side(hawk_terms_sorted, hawk_dict, hawk_hits)
        count_side(dove_terms_sorted, dove_dict, dove_hits)

    H = sum(hawk_hits.values())
    D = sum(dove_hits.values())
    net_score = (H - D) / (H + D) if (H + D) > 0 else 0
    return net_score, hawk_hits, dove_hits

def run_full_analysis(text):
    """
    ÖNEMLİ GÜNCELLEME: Artık toplam sayıları da döndürüyor.
    Return: (score, hawk_count, dove_count, hawk_list, dove_list)
    """
    pp_text = preprocess_disinflation(text)
    score, h_hits, d_hits = abg_2019_count(pp_text, HAWK_DICT, DOVE_DICT, window=7)
    
    h_total = sum(h_hits.values())
    d_total = sum(d_hits.values())
    
    hawk_list = [f"{k} ({v})" for k, v in h_hits.most_common()]
    dove_list = [f"{k} ({v})" for k, v in d_hits.most_common()]
    
    return score, h_total, d_total, hawk_list, dove_list

# =============================================================================
# 3. VERİ ÇEKME & DB
# =============================================================================

@st.cache_data(ttl=600)
def fetch_market_data_adapter(start_date, end_date):
    if not EVDS_API_KEY: return pd.DataFrame(), "EVDS Anahtarı Eksik."
    # Enflasyon
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

    # Faiz
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

def fetch_all_data():
    if not supabase: return pd.DataFrame()
    res = supabase.table("market_logs").select("*").order("period_date", desc=True).execute()
    return pd.DataFrame(res.data)

def insert_entry(date, text, source, s_dict, s_abg, s_fb, l_fb):
    if not supabase: return
    data = {"period_date": str(date), "text_content": text, "source": source,
        "score_dict": s_dict, "score_abg": s_abg, "score_finbert": s_fb, "finbert_label": l_fb}
    supabase.table("market_logs").insert(data).execute()

def update_entry(rid, date, text, source, s_dict, s_abg, s_fb, l_fb):
    if not supabase: return
    data = {"period_date": str(date), "text_content": text, "source": source,
        "score_dict": s_dict, "score_abg": s_abg, "score_finbert": s_fb, "finbert_label": l_fb}
    supabase.table("market_logs").update(data).eq("id", rid).execute()

def delete_entry(rid):
    if supabase: supabase.table("market_logs").delete().eq("id", rid).execute()
