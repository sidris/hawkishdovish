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
# 4. METİN ANALİZİ (ESKİ SİSTEM)
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
# 5. ABG ANALYZER SINIFI (APEL, BLIX, GRIMALDI - 2019)
# =============================================================================

class ABGAnalyzer:
    def __init__(self):
        # NOT: Kelimelere '*' (wildcard) eklendi!
        self.inflation_dict = {
            "terms": ["consumer prices", "consumer price", "cpi", "inflation", "inflation pressure", "inflationary pressure", "price", "prices"],
            "hawkish_modifiers": ["accelerat*", "boost*", "elevat*", "escalat*", "high*", "increas*", "jump*", "pickup", "ris*", "ros*", "run-up", "runup", "strong*", "surg*", "up", "mount*", "intensif*", "stok*", "sustain*"],
            "dovish_modifiers": ["decelerat*", "declin*", "decreas*", "down", "drop*", "fall*", "fell", "low*", "muted", "reduc*", "slow*", "stable", "subdued", "weak*", "contained", "abat*", "dampen*", "dimin*", "eas*", "moderat*", "reced*", "temper*"]
        }
        self.growth_dict = {
            "terms": ["consumer spending", "economic activity", "economic growth", "resource utilization", "gdp", "output", "demand", "production"],
            "hawkish_modifiers": ["accelerat*", "edg* up", "expan*", "increas*", "pick* up", "pickup", "soft*", "strength*", "strong*", "buoyant", "high*", "ris*", "ros*", "step* up", "tight*", "upside"],
            "dovish_modifiers": ["contract*", "decelerat*", "decreas*", "drop*", "retrench*", "slow*", "slugg*", "soft*", "subdued", "weak*", "curtail*", "declin*", "downside", "fall*", "fell", "low*", "loose"]
        }
        self.employment_dict = {
            "terms": ["employment", "job", "jobs", "labor market", "labour market"],
            "hawkish_modifiers": ["expand*", "gain*", "improv*", "increas*", "pick* up", "pickup", "rais*", "ris*", "ros*", "strength*", "turn* up", "strain*", "tight*"],
            "dovish_modifiers": ["slow*", "declin*", "reduc*", "weak*", "deteriorat*", "shrink*", "shrank", "fall*", "fell", "drop*", "contract*", "sluggish", "eased", "easing", "loos*", "soft*"]
        }
        self.unemployment_dict = {
            "terms": ["unemployment"],
            "hawkish_modifiers": ["declin*", "fall*", "fell", "low*", "reduc*"],
            "dovish_modifiers": ["sluggish", "eas*", "loos*", "elevat*", "high*", "increas*", "ris*", "ros*"]
        }
        self.dictionaries = [self.inflation_dict, self.growth_dict, self.employment_dict, self.unemployment_dict]

    def split_sentences(self, text):
        return re.split(r'[.!?]+', text) # Basit ve etkili cümle bölme

    def analyze(self, text):
        raw_sentences = self.split_sentences(text)
        hawk_count = 0; dove_count = 0; matches = [] 
        
        for original_sentence in raw_sentences:
            if not original_sentence.strip(): continue
            
            # Punctuation temizliği: Sadece kelimeler ve boşluklar kalsın
            clean_sent = re.sub(r'[^\w\s]', '', original_sentence).lower()
            tokens = clean_sent.split()
            found_in_sentence = False
            
            for vocab in self.dictionaries:
                terms = vocab["terms"]
                h_mods = vocab["hawkish_modifiers"]
                d_mods = vocab["dovish_modifiers"]

                for i, word in enumerate(tokens):
                    matched_term = None; term_index = -1
                    
                    # Term Match Check
                    for term in terms:
                        term_parts = term.split()
                        if len(term_parts) == 1 and word == term_parts[0]:
                            matched_term = term; term_index = i
                        elif len(term_parts) > 1:
                            if tokens[i:i+len(term_parts)] == term_parts:
                                matched_term = term; term_index = i
                    
                    if matched_term:
                        start = max(0, term_index - 7)
                        end = min(len(tokens), term_index + 7 + 1)
                        window = tokens[start:end]

                        for mod in h_mods:
                            # "low*" -> "\blow\w*\b"
                            pattern = r"\b" + mod.replace("*", "\w*") + r"\b"
                            for w in window:
                                if re.match(pattern, w):
                                    hawk_count += 1
                                    matches.append({"type": "HAWK", "term": f"{matched_term} + {w}", "sentence": original_sentence.strip()})
                                    found_in_sentence = True
                                    break 
                            if found_in_sentence: break
                        
                        if found_in_sentence: break 

                        for mod in d_mods:
                            pattern = r"\b" + mod.replace("*", "\w*") + r"\b"
                            for w in window:
                                if re.match(pattern, w):
                                    dove_count += 1
                                    matches.append({"type": "DOVE", "term": f"{matched_term} + {w}", "sentence": original_sentence.strip()})
                                    found_in_sentence = True
                                    break
                            if found_in_sentence: break
                    
                    if found_in_sentence: break
                if found_in_sentence: break

        total = hawk_count + dove_count
        net_hawkishness = ((hawk_count - dove_count) / total) + 1 if total > 0 else 1.0

        return {
            "net_hawkishness": net_hawkishness,
            "hawk_count": hawk_count,
            "dove_count": dove_count,
            "total_matches": total,
            "match_details": matches
        }

def calculate_abg_scores(df):
    if df.empty: return pd.DataFrame()
    analyzer = ABGAnalyzer(); results = []
    for _, row in df.iterrows():
        res = analyzer.analyze(str(row['text_content']))
        donem_val = row.get('Donem')
        if not donem_val and 'period_date' in row:
            try: donem_val = pd.to_datetime(row['period_date']).strftime('%Y-%m')
            except: donem_val = ''
            
        results.append({
            'period_date': row['period_date'],
            'Donem': donem_val,
            'abg_index': res['net_hawkishness'],
            'abg_hawk': res['hawk_count'],
            'abg_dove': res['dove_count']
        })
    # DÜZELTME: En yeniden eskiye (Descending)
    return pd.DataFrame(results).sort_values('period_date', ascending=False)
