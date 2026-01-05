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
# 4. METİN ANALİZİ (ESKİ SİSTEM - Dashboard ve İlk Hesaplamalar İçin)
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

def count_syllables(word):
    word = word.lower().strip(".:;?!")
    if not word: return 0
    if len(word) <= 3: return 1
    word = re.sub(r'(?:[^laeiouy]es|ed|[^laeiouy]e)$', '', word)
    word = re.sub(r'^y', '', word)
    syllables = re.findall(r'[aeiouy]{1,2}', word)
    return len(syllables) if syllables else 1

def calculate_flesch_reading_ease(text):
    if not text: return 0
    sentences = [s for s in split_into_sentences(text) if len(s.strip()) > 0]
    words = re.findall(r"[a-z']+", text.lower())
    num_sentences = max(1, len(sentences))
    num_words = max(1, len(words))
    total_syllables = sum(count_syllables(w) for w in words)
    score = 206.835 - (1.015 * (num_words / num_sentences)) - (84.6 * (total_syllables / num_words))
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

# --- DERİN ANALİZ FONKSİYONLARI ---

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

def train_and_predict_rate(df_history, current_score):
    if not HAS_ML_DEPS: return None, None, "Kütüphane eksik"
    if df_history.empty or 'PPK Faizi' not in df_history.columns: return None, None, "Yetersiz Veri"
    df = df_history.sort_values('period_date').copy()
    df = df.drop_duplicates(subset=['Donem'], keep='last').copy()
    df['Next_Rate'] = df['PPK Faizi'].shift(-1)
    df['Rate_Change'] = df['Next_Rate'] - df['PPK Faizi']
    train_data = df.dropna(subset=['score_abg_scaled', 'Rate_Change'])
    if len(train_data) < 5: return None, None, "Model için en az 5 ay veri lazım."
    X = train_data[['score_abg_scaled']]; y = train_data['Rate_Change']
    model = LinearRegression(); model.fit(X, y)
    y_bps = (y * 100).round(0).astype(int)
    try:
        model_logit = LogisticRegression(max_iter=1000, random_state=42)
        model_logit.fit(X, y_bps)
        has_logit = True
    except: has_logit = False
    full_X = df.dropna(subset=['score_abg_scaled'])[['score_abg_scaled']]
    df.loc[full_X.index, 'Predicted_Change'] = model.predict(full_X)
    if has_logit: df.loc[full_X.index, 'Predicted_Change_Logit'] = model_logit.predict(full_X) / 100.0
    else: df.loc[full_X.index, 'Predicted_Change_Logit'] = np.nan
    prediction = model.predict([[current_score]])[0]
    prediction_logit = prediction
    if has_logit: prediction_logit = model_logit.predict([[current_score]])[0] / 100.0
    corr = np.corrcoef(train_data['score_abg_scaled'], train_data['Rate_Change'])[0,1]
    stats = {'prediction': prediction, 'prediction_logit': prediction_logit, 'correlation': corr, 'sample_size': len(train_data), 'coef': model.coef_[0]}
    return stats, df[['period_date', 'Donem', 'Rate_Change', 'Predicted_Change', 'Predicted_Change_Logit']].copy(), None

# =============================================================================
# 5. ABG (APEL, BLIX, GRIMALDI - 2019) ANALYZER [GÜNCELLENMİŞ VERSİYON]
# =============================================================================

@dataclass(frozen=True)
class ModPattern:
    # modifier örn: "edg* up" => [r"^edg[a-z-]*$", r"^up$"]
    token_regexes: Tuple[re.Pattern, ...]

@dataclass(frozen=True)
class TermEntry:
    # term örn: "inflation pressure" => ("inflation","pressure")
    term_tokens: Tuple[str, ...]
    hawk_mods: Tuple[ModPattern, ...]
    dove_mods: Tuple[ModPattern, ...]

class ABG2019Analyzer:
    """
    ABG (Apel, Blix Grimaldi, Hull 2019) Appendix Table 4-6 sözlüğünü bire bir uygular.
    - Term/modifier aynı cümlede, term etrafında +/-7 kelime penceresinde sayılır.
    - Term substring çakışmaları tek instance sayılır (uzun terim öncelikli).
    - '*' stem wildcard: aynı kökten başlayan kelimeler.
    """

    def __init__(self, window_size: int = 7):
        self.window_size = window_size
        self.entries: List[TermEntry] = self._build_dictionary_from_appendix()

    # ----------------------------
    # Tokenization / sentence split
    # ----------------------------
    def split_sentences(self, text: str) -> List[str]:
        # Basit cümle bölme (ABG cümle içi koşul dediği için yeterli pratik yaklaşım)
        return re.split(r'[.!?]+', text)

    def tokenize(self, sentence: str) -> List[str]:
        # Hyphen'ı koru: "run-up" gibi modifierlar için
        s = sentence.lower()
        return re.findall(r"[a-z]+(?:-[a-z]+)*", s)

    # ----------------------------
    # Dictionary (Appendix Table 4-6)
    # ----------------------------
    def _compile_token_wildcard(self, token: str) -> re.Pattern:
        # token örn "high*" => ^high[a-z-]*$
        # token örn "fell"  => ^fell$
        # Not: ABG '*' "same stem" diyor.
        token = token.strip().lower()
        if "*" in token:
            base = re.escape(token.replace("*", ""))
            return re.compile(rf"^{base}[a-z-]*$")
        else:
            return re.compile(rf"^{re.escape(token)}$")

    def _compile_modifier(self, modifier: str) -> List[ModPattern]:
        """
        modifier string’i ModPattern listesine çevirir.
        - "run-up/runup" => iki alternatif pattern
        - "edg* up" => iki tokenlı pattern
        - "drop" vs "drop*" vs "down*" vs "high*" vs ...
        """
        modifier = modifier.strip().lower()
        if not modifier:
            return []

        # run-up/runup gibi alternatifler
        alts = modifier.split("/")
        out: List[ModPattern] = []
        for alt in alts:
            alt = alt.strip()
            if not alt:
                continue
            parts = alt.split()  # "edg* up" => ["edg*","up"]
            token_regexes = tuple(self._compile_token_wildcard(p) for p in parts)
            out.append(ModPattern(token_regexes=token_regexes))
        return out

    def _entry(self, term: str, hawk_mods: List[str], dove_mods: List[str]) -> TermEntry:
        term_tokens = tuple(term.lower().split())
        h_patterns: List[ModPattern] = []
        d_patterns: List[ModPattern] = []
        for m in hawk_mods:
            h_patterns.extend(self._compile_modifier(m))
        for m in dove_mods:
            d_patterns.extend(self._compile_modifier(m))
        return TermEntry(
            term_tokens=term_tokens,
            hawk_mods=tuple(h_patterns),
            dove_mods=tuple(d_patterns),
        )

    def _build_dictionary_from_appendix(self) -> List[TermEntry]:
        # Appendix Table 4 (Inflation)
        inflation_consumer_prices_hawk = [
            "accelerat*", "boost*", "elevat*", "escalat*", "high*", "increas*", "jump*",
            "pickup", "rise*", "rose", "rising", "run-up/runup", "strong*", "surg*", "up*"
        ]
        inflation_consumer_prices_dove = [
            "decelerat*", "declin*", "decreas*", "down*", "drop*", "fall*", "fell", "low*",
            "muted", "reduc*", "slow*", "stable", "subdued", "weak*", "contained"
        ]

        inflation_infl_pressure_hawk = [
            "accelerat*", "boost*", "build*", "elevat*", "emerg*", "great*", "height*",
            "high*", "increas*", "intensif*", "mount*", "pickup", "rise", "rose", "rising",
            "stok*", "strong*", "sustain*"
        ]
        inflation_infl_pressure_dove = [
            "abat*", "contain*", "dampen*", "decelerat*", "declin*", "decreas*", "dimin*",
            "eas*", "fall*", "fell", "low*", "moderat*", "reced*", "reduc*", "subdued",
            "temper*"
        ]

        # Appendix Table 5 (Economic Activity)
        econ_cons_spend_hawk = [
            "accelerat*", "edg* up", "expan*", "increas*", "pick* up", "pickup",
            "soft*", "strength*", "strong*", "weak*"
        ]
        econ_cons_spend_dove = [
            "contract*", "decelerat*", "decreas*", "drop*", "retrench*", "slow*",
            "slugg*", "soft*", "subdued"
        ]

        econ_activity_hawk = ["accelerat*"]
        econ_activity_dove = ["contract*"]

        econ_growth_hawk = [
            "buoyant", "edg* up", "expan*", "increas*", "high*", "pick* up", "pickup",
            "rise*", "rose", "rising", "step* up", "strength*", "strong*", "upside"
        ]
        econ_growth_dove = [
            "curtail*", "decelerat*", "declin*", "decreas*", "downside", "drop", "fall*",
            "fell", "low*", "moderat*", "slow*", "slugg*", "weak*"
        ]

        resource_util_hawk = ["high*", "increas*", "rise", "rising", "rose", "tight*"]
        resource_util_dove = ["declin*", "fall*", "fell", "loose*", "low*"]

        # Appendix Table 6 (Employment + Labor Market + Unemployment)
        employment_hawk = [
            "expand*", "gain*", "improv*", "increas*", "pick* up", "pickup", "rais*",
            "rise*", "rising", "rose", "strength*", "turn* up"
        ]
        employment_dove = [
            "slow*", "declin*", "reduc*", "weak*", "deteriorat*", "shrink*", "shrank",
            "fall*", "fell", "drop*", "contract*", "sluggish"
        ]

        labor_market_hawk = ["strain*", "tight*"]
        labor_market_dove = ["eased", "easing", "loos*", "soft*", "weak*"]

        unemployment_hawk = ["declin*", "fall*", "fell", "low*", "reduc*"]
        unemployment_dove = ["elevat*", "high", "increas*", "ris*", "rose*"]

        return [
            # Inflation
            self._entry("consumer prices", inflation_consumer_prices_hawk, inflation_consumer_prices_dove),
            self._entry("inflation",       inflation_consumer_prices_hawk, inflation_consumer_prices_dove),
            self._entry("inflation pressure", inflation_infl_pressure_hawk, inflation_infl_pressure_dove),

            # Economic activity
            self._entry("consumer spending", econ_cons_spend_hawk, econ_cons_spend_dove),
            self._entry("economic activity", econ_activity_hawk, econ_activity_dove),
            self._entry("economic growth",   econ_growth_hawk, econ_growth_dove),
            self._entry("resource utilization", resource_util_hawk, resource_util_dove),

            # Employment
            self._entry("employment", employment_hawk, employment_dove),
            self._entry("labor market", labor_market_hawk, labor_market_dove),
            self._entry("unemployment", unemployment_hawk, unemployment_dove),
        ]

    # ----------------------------
    # Matching logic (ABG rules)
    # ----------------------------
    def _find_term_spans(self, tokens: List[str]) -> List[Tuple[int, int, TermEntry]]:
        """
        Tüm term eşleşmelerini bulur (span: [start, end) ).
        Sonra substring çakışmasını önlemek için: en uzun termleri önce seçip overlap etmeyenleri bırakır.
        """
        raw: List[Tuple[int, int, TermEntry]] = []
        n = len(tokens)

        for entry in self.entries:
            t = entry.term_tokens
            L = len(t)
            if L == 0:
                continue
            for i in range(0, n - L + 1):
                if tuple(tokens[i:i+L]) == t:
                    raw.append((i, i+L, entry))

        # Uzun terimler önce (substring çakışmalarını tek saymak için)
        raw.sort(key=lambda x: (-(x[1]-x[0]), x[0]))

        chosen: List[Tuple[int, int, TermEntry]] = []
        occupied = [False] * n
        for s, e, entry in raw:
            if any(occupied[k] for k in range(s, e)):
                continue
            chosen.append((s, e, entry))
            for k in range(s, e):
                occupied[k] = True

        # start sırasına koy
        chosen.sort(key=lambda x: x[0])
        return chosen

    def _match_modifier_at(self, tokens: List[str], pos: int, pat: ModPattern) -> bool:
        L = len(pat.token_regexes)
        if pos + L > len(tokens):
            return False
        for j in range(L):
            if not pat.token_regexes[j].match(tokens[pos+j]):
                return False
        return True

    def analyze(self, text: str) -> Dict[str, Any]:
        sentences = self.split_sentences(text)
        hawk = 0
        dove = 0
        details: List[Dict[str, Any]] = []

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            tokens = self.tokenize(sent)
            if not tokens:
                continue

            # Term spans (substring çakışmaları tek say)
            term_spans = self._find_term_spans(tokens)

            for (ts, te, entry) in term_spans:
                # ±7-word window around the term span
                w_start = max(0, ts - self.window_size)
                w_end = min(len(tokens), te + self.window_size)

                # Hawk modifiers
                for pat in entry.hawk_mods:
                    L = len(pat.token_regexes)
                    for p in range(w_start, w_end - L + 1):
                        if self._match_modifier_at(tokens, p, pat):
                            hawk += 1
                            mod_str = " ".join(tokens[p:p+L])
                            details.append({
                                "type": "HAWK",
                                "term": " ".join(entry.term_tokens),
                                "modifier": mod_str,
                                "sentence": sent
                            })

                # Dove modifiers
                for pat in entry.dove_mods:
                    L = len(pat.token_regexes)
                    for p in range(w_start, w_end - L + 1):
                        if self._match_modifier_at(tokens, p, pat):
                            dove += 1
                            mod_str = " ".join(tokens[p:p+L])
                            details.append({
                                "type": "DOVE",
                                "term": " ".join(entry.term_tokens),
                                "modifier": mod_str,
                                "sentence": sent
                            })

        total = hawk + dove
        net = 1.0 + ((hawk - dove) / total) if total > 0 else 1.0  # ABG eq (1)

        return {
            "net_hawkishness": net,
            "hawk_count": hawk,
            "dove_count": dove,
            "total_matches": total,
            "match_details": details
        }

# ABG Wrapper Fonksiyonu (Eski API'ye uyumluluk için)
class ABGAnalyzer:
    def __init__(self):
        self.engine = ABG2019Analyzer(window_size=7)
    
    def analyze(self, text):
        return self.engine.analyze(text)

def calculate_abg_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    analyzer = ABG2019Analyzer(window_size=7)
    rows = []

    for _, row in df.iterrows():
        res = analyzer.analyze(str(row.get("text_content", "")))

        donem_val = row.get("Donem")
        if (donem_val is None or donem_val == "") and ("period_date" in row):
            try:
                donem_val = pd.to_datetime(row["period_date"]).strftime("%Y-%m")
            except Exception:
                donem_val = ""

        rows.append({
            "period_date": row.get("period_date"),
            "Donem": donem_val,
            "abg_index": res["net_hawkishness"],
            "abg_hawk": res["hawk_count"],
            "abg_dove": res["dove_count"],
        })

    return pd.DataFrame(rows).sort_values("period_date", ascending=False)
