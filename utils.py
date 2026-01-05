import re
from collections import Counter

class ABGAnalyzer:
    def __init__(self):
        # TABLO 4: ENFLASYON (INFLATION) [cite: 762]
        # Terimler ve onlara eşlik eden niteleyiciler
        self.inflation_dict = {
            "terms": [
                "consumer prices", "consumer price", "cpi", "inflation", 
                "inflation pressure", "inflationary pressure", "price", "prices"
            ],
            "hawkish_modifiers": [
                "accelerat", "boost", "elevat", "escalat", "high", "increas", 
                "jump", "pickup", "ris", "rose", "run-up", "runup", 
                "strong", "surg", "up", "mount", "intensif", "stok", "sustain"
            ],
            "dovish_modifiers": [
                "decelerat", "declin", "decreas", "down", "drop", "fall", 
                "fell", "low", "muted", "reduc", "slow", "stable", 
                "subdued", "weak", "contained", "abat", "dampen", "dimin", 
                "eas", "moderat", "reced", "temper"
            ]
        }

        # TABLO 5: EKONOMİK AKTİVİTE (ECONOMIC ACTIVITY) [cite: 770]
        self.growth_dict = {
            "terms": [
                "consumer spending", "economic activity", "economic growth", 
                "resource utilization", "gdp", "output", "demand", "production"
            ],
            "hawkish_modifiers": [
                "accelerat", "edg* up", "expan", "increas", "pick* up", 
                "pickup", "soft", "strength", "strong", "buoyant", "high", 
                "ris", "rose", "step* up", "tight"
            ],
            "dovish_modifiers": [
                "contract", "decelerat", "decreas", "drop", "retrench", 
                "slow", "slugg", "soft", "subdued", "weak", "curtail", 
                "declin", "downside", "fall", "fell", "low", "loose"
            ]
        }

        # TABLO 6: İSTİHDAM (EMPLOYMENT) [cite: 777]
        # Dikkat: "Unemployment" için şahin/güvercin mantığı terstir.
        # Makale bunu iki ayrı grup olarak ele alır.
        
        # Grup A: İstihdam (Employment) -> Artış = Şahin (Güçlü Ekonomi)
        self.employment_dict = {
            "terms": ["employment", "job", "jobs", "labor market", "labour market"],
            "hawkish_modifiers": [
                "expand", "gain", "improv", "increas", "pick up", "pickup", 
                "rais", "ris", "rose", "strength", "turn* up", "strain", "tight"
            ],
            "dovish_modifiers": [
                "slow", "declin", "reduc", "weak", "deteriorat", "shrink", 
                "shrank", "fall", "fell", "drop", "contract", "soft"
            ]
        }

        # Grup B: İşsizlik (Unemployment) -> Düşüş = Şahin (Güçlü Ekonomi)
        self.unemployment_dict = {
            "terms": ["unemployment"],
            "hawkish_modifiers": [
                "declin", "fall", "fell", "low", "reduc"
            ],
            "dovish_modifiers": [
                "sluggish", "eas", "loos", "elevat", "high", "increas", 
                "ris", "rose"
            ]
        }
        
        # Tüm sözlüklerin listesi
        self.dictionaries = [
            self.inflation_dict, 
            self.growth_dict, 
            self.employment_dict, 
            self.unemployment_dict
        ]

    def clean_text(self, text):
        # Metni cümlelere ayır (Basit kural)
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        cleaned_sentences = []
        for s in sentences:
            # Noktalama işaretlerini kaldır ve küçük harfe çevir
            s = re.sub(r'[^\w\s]', '', s).lower()
            cleaned_sentences.append(s.split())
        return cleaned_sentences

    def analyze(self, text):
        sentences = self.clean_text(text)
        hawk_count = 0
        dove_count = 0
        
        matches = [] # Analiz kontrolü için eşleşmeleri saklayalım

        for sentence_tokens in sentences:
            for vocab in self.dictionaries:
                terms = vocab["terms"]
                h_mods = vocab["hawkish_modifiers"]
                d_mods = vocab["dovish_modifiers"]

                # Cümledeki her kelime için kontrol
                for i, word in enumerate(sentence_tokens):
                    
                    # 1. Terim Kontrolü (Tek veya çok kelimeli)
                    # Basitlik adına burada tek kelimelik token taraması yapılıyor. 
                    # "Labor market" gibi bigramlar için metin ön işlemede birleştirme yapılabilir.
                    # Ancak bu kodda pencere mantığı ile yakalayacağız.
                    
                    matched_term = None
                    term_index = -1
                    
                    # Cümlede terim var mı?
                    for term in terms:
                        term_parts = term.split()
                        # Eğer terim tek kelimeyse
                        if len(term_parts) == 1 and word == term_parts[0]:
                            matched_term = term
                            term_index = i
                        # Eğer terim çok kelimeyse (örn: labor market)
                        elif len(term_parts) > 1:
                            if sentence_tokens[i:i+len(term_parts)] == term_parts:
                                matched_term = term
                                term_index = i
                    
                    if matched_term:
                        # 2. Pencere Kontrolü (Window = 7) 
                        start = max(0, term_index - 7)
                        end = min(len(sentence_tokens), term_index + 7 + 1)
                        window_tokens = sentence_tokens[start:end]

                        # 3. Niteleyici (Modifier) Kontrolü
                        found_modifier = False
                        
                        # Şahin Niteleyiciler
                        for mod in h_mods:
                            # Regex ile stem eşleşmesi (örn: accelerat*)
                            pattern = r"\b" + mod.replace("*", "\w*") + r"\b"
                            for w in window_tokens:
                                if re.match(pattern, w):
                                    hawk_count += 1
                                    matches.append(f"HAWK: {matched_term} + {w}")
                                    found_modifier = True
                                    break 
                            if found_modifier: break
                        
                        if found_modifier: continue # Bir terim bir kere sayılır

                        # Güvercin Niteleyiciler
                        for mod in d_mods:
                            pattern = r"\b" + mod.replace("*", "\w*") + r"\b"
                            for w in window_tokens:
                                if re.match(pattern, w):
                                    dove_count += 1
                                    matches.append(f"DOVE: {matched_term} + {w}")
                                    found_modifier = True
                                    break
                            if found_modifier: break

        # 4. Endeks Hesaplama 
        total = hawk_count + dove_count
        if total > 0:
            net_hawkishness = ((hawk_count - dove_count) / total) + 1
        else:
            net_hawkishness = 1.0 # Nötr durum (Formülde +1 baz etkisi var)

        return {
            "net_hawkishness": net_hawkishness,
            "hawk_count": hawk_count,
            "dove_count": dove_count,
            "total_matches": total,
            "match_details": matches
        }

# ÖRNEK KULLANIM
text_example = """
Inflation expectations have accelerated recently due to high oil prices.
However, the unemployment rate rose unexpectedly, signaling weak demand.
Economic activity is showing strong growth in the industrial sector.
"""

analyzer = ABGAnalyzer()
result = analyzer.analyze(text_example)

print(f"Net Hawkishness Index: {result['net_hawkishness']:.2f}")
print(f"Details: {result}")
