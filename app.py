import streamlit as st
import pandas as pd
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
import utils
import uuid
import numpy as np
import re

st.set_page_config(page_title="Piyasa Analiz", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 5rem; }
    h1 { font-size: 1.8rem !important; }
    .stDataFrame { font-size: 0.8rem; }
    .stButton button { border-radius: 20px; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# --- GÜVENLİK ---
APP_PWD = "SahinGuvercin34"
ADMIN_PWD = "SahinGuvercin06"

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        st.markdown("<br><h3 style='text-align: center;'>🔐 Güvenli Giriş</h3>", unsafe_allow_html=True)
        pwd_input = st.text_input("Uygulama Şifresi", type="password")
        if st.button("Giriş Yap", type="primary"):
            if pwd_input == APP_PWD:
                st.session_state["logged_in"] = True
                st.success("Başarılı!")
                st.rerun()
            else:
                st.error("Hatalı!")
    st.stop()

# --- SESSION & STATE ---
if "form_data" not in st.session_state:
    st.session_state["form_data"] = {
        "id": None,
        "date": datetime.date.today().replace(day=1),
        "source": "TCMB",
        "text": ""
    }
if "table_key" not in st.session_state:
    st.session_state["table_key"] = str(uuid.uuid4())
if "collision_state" not in st.session_state:
    st.session_state["collision_state"] = {"active": False, "target_id": None, "pending_text": None, "target_date": None}
if "update_state" not in st.session_state:
    st.session_state["update_state"] = {"active": False, "pending_text": None}

if "stop_words_deep" not in st.session_state:
    st.session_state["stop_words_deep"] = []
if "stop_words_cloud" not in st.session_state:
    st.session_state["stop_words_cloud"] = []

# ✅ RoBERTa trend cache state (doğru yerde)
if "ai_trend_df" not in st.session_state:
    st.session_state["ai_trend_df"] = None

def add_deep_stop():
    word = st.session_state.get("deep_stop_in", "").strip()
    if word and word not in st.session_state["stop_words_deep"]:
        st.session_state["stop_words_deep"].append(word)
    st.session_state["deep_stop_in"] = ""

def add_cloud_stop():
    word = st.session_state.get("cloud_stop_in", "").strip()
    if word and word not in st.session_state["stop_words_cloud"]:
        st.session_state["stop_words_cloud"].append(word)
    st.session_state["cloud_stop_in"] = ""

def reset_form():
    st.session_state["form_data"] = {"id": None, "date": datetime.date.today(), "source": "TCMB", "text": ""}
    st.session_state["collision_state"] = {"active": False, "target_id": None, "pending_text": None, "target_date": None}
    st.session_state["update_state"] = {"active": False, "pending_text": None}
    st.session_state["table_key"] = str(uuid.uuid4())

# -------------------------
# ✅ Fallback: Cümle bazlı RoBERTa (utils boş dönerse)
# -------------------------
def fallback_sentence_roberta(text: str) -> pd.DataFrame:
    """
    utils.analyze_sentences_with_roberta boş dönerse:
    - basit sentence split
    - her cümle için utils.analyze_with_roberta çalıştır
    - df döndür
    """
    if not text or not isinstance(text, str):
        return pd.DataFrame()

    # kaba ama sağlam split (EN/TR karışık metinlerde iş görür)
    parts = re.split(r"(?<=[\.\!\?\;:])\s+", text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    # aşırı kısa cümleleri ele
    parts = [p for p in parts if len(p) >= 30 and len(p.split()) >= 4]
    if not parts:
        return pd.DataFrame()

    rows = []
    for sent in parts[:120]:  # güvenlik
        res = utils.analyze_with_roberta(sent) if hasattr(utils, "analyze_with_roberta") else None
        if not isinstance(res, dict):
            continue
        scores = res.get("scores_map", {}) or {}
        h = float(scores.get("HAWK", 0.0))
        d = float(scores.get("DOVE", 0.0))
        n = float(scores.get("NEUT", 0.0))
        diff = float(res.get("diff", h - d))
        stance = str(res.get("stance", ""))

        rows.append({
            "Cümle": sent,
            "Duruş": stance,
            "Diff (H-D)": diff,
            "HAWK": h,
            "DOVE": d,
            "NEUT": n
        })

    df = pd.DataFrame(rows)
    if not df.empty and "Diff (H-D)" in df.columns:
        df = df.sort_values("Diff (H-D)", ascending=False).reset_index(drop=True)
    return df

c_head1, c_head2 = st.columns([6, 1])
with c_head1:
    st.title("🦅 Şahin/Güvercin Paneli")
with c_head2:
    if st.button("Çıkış"):
        st.session_state["logged_in"] = False
        st.rerun()

tab1, tab2, tab3, tab4, tab_textdata, tab6, tab7, tab_roberta, tab_imp = st.tabs([
    "📈 Dashboard",
    "📝 Veri Girişi",
    "📊 Veriler",
    "🔍 Frekans",
    "📚 Text as Data (TF-IDF)",
    "☁️ WordCloud",
    "📜 ABF (2019)",
    "🧠 CB-RoBERTa",
    "📅 Haberler"
])

# ==============================================================================
# TAB 1: DASHBOARD
# ==============================================================================
with tab1:
    with st.spinner("Veriler Yükleniyor..."):
        df_logs = utils.fetch_all_data()
        df_events = utils.fetch_events()

    if df_logs is None or df_logs.empty:
        st.info("Kayıt yok.")
        st.stop()

    df_logs = df_logs.copy()
    df_logs["period_date"] = pd.to_datetime(df_logs["period_date"], errors="coerce")
    df_logs = df_logs.dropna(subset=["period_date"]).sort_values("period_date")
    df_logs["Donem"] = df_logs["period_date"].dt.strftime("%Y-%m")

    df_logs["word_count"] = df_logs["text_content"].apply(lambda x: len(str(x).split()) if x else 0)
    df_logs["flesch_score"] = df_logs["text_content"].apply(lambda x: utils.calculate_flesch_reading_ease(str(x)))
    if "score_abg" in df_logs.columns:
        df_logs["score_abg_scaled"] = df_logs["score_abg"].apply(lambda x: x * 100 if abs(x) <= 1 else x)
    else:
        df_logs["score_abg_scaled"] = np.nan

    abg_df = utils.calculate_abg_scores(df_logs)
    abg_df["abg_dashboard_val"] = (abg_df["abg_index"] - 1.0) * 100

    min_d = df_logs["period_date"].min().date()
    max_d = datetime.date.today()
    df_market, err = utils.fetch_market_data_adapter(min_d, max_d)
    if err:
        st.warning(f"Market veri uyarısı: {err}")

    merged = pd.merge(df_logs, df_market, on="Donem", how="left")
    merged = pd.merge(merged, abg_df[["period_date", "abg_dashboard_val"]], on="period_date", how="left")
    merged = merged.sort_values("period_date")

    # ✅ AI (mrince) merge (TEK kez)
    ai_df = st.session_state.get("ai_trend_df")
    if ai_df is not None and not ai_df.empty:
        ai_tmp = ai_df.copy()
        ai_tmp["AI_EMA"] = pd.to_numeric(ai_tmp.get("AI Score (EMA)", np.nan), errors="coerce")
        ai_tmp["AI_CALIB"] = pd.to_numeric(ai_tmp.get("AI Score (Calib)", np.nan), errors="coerce")

        if "Dönem" in ai_tmp.columns:
            ai_tmp["Donem"] = ai_tmp["Dönem"].astype(str)
        elif "period_date" in ai_tmp.columns:
            ai_tmp["Donem"] = pd.to_datetime(ai_tmp["period_date"], errors="coerce").dt.strftime("%Y-%m")
        else:
            ai_tmp["Donem"] = None

        ai_tmp = ai_tmp.dropna(subset=["Donem"])
        ai_tmp = ai_tmp[["Donem", "AI_EMA", "AI_CALIB"]].drop_duplicates(subset=["Donem"])
        merged = pd.merge(merged, ai_tmp, on="Donem", how="left")
    else:
        merged["AI_EMA"] = np.nan
        merged["AI_CALIB"] = np.nan

    if "Yıllık TÜFE" in merged.columns:
        merged["Yıllık TÜFE"] = pd.to_numeric(merged["Yıllık TÜFE"], errors="coerce")
    if "PPK Faizi" in merged.columns:
        merged["PPK Faizi"] = pd.to_numeric(merged["PPK Faizi"], errors="coerce")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=merged["period_date"], y=merged["word_count"], name="Metin Uzunluğu",
        marker=dict(color="gray"), opacity=0.10, yaxis="y3", hoverinfo="x+y+name"
    ))

    fig.add_trace(go.Scatter(
        x=merged["period_date"], y=merged["score_abg_scaled"],
        name="Şahin/Güvercin-Hibrit", line=dict(color="black", width=2, dash="dot"),
        marker=dict(size=6, color="black"), yaxis="y"
    ))
    fig.add_trace(go.Scatter(
        x=merged["period_date"], y=merged["abg_dashboard_val"],
        name="Şahin/Güvercin ABG 2019", line=dict(color="navy", width=4), yaxis="y"
    ))

    if merged["AI_EMA"].notna().any():
        fig.add_trace(go.Scatter(
            x=merged["period_date"], y=merged["AI_EMA"],
            name="AI Score (mrince, EMA)", line=dict(color="green", width=3), yaxis="y"
        ))
    if merged["AI_CALIB"].notna().any():
        fig.add_trace(go.Scatter(
            x=merged["period_date"], y=merged["AI_CALIB"],
            name="AI Score (mrince, Calib)", line=dict(color="green", width=2, dash="dot"), yaxis="y"
        ))

    if "Yıllık TÜFE" in merged.columns:
        fig.add_trace(go.Scatter(
            x=merged["period_date"], y=merged["Yıllık TÜFE"],
            name="Yıllık TÜFE (%)", line=dict(color="red", dash="dot"), yaxis="y"
        ))
    if "PPK Faizi" in merged.columns:
        fig.add_trace(go.Scatter(
            x=merged["period_date"], y=merged["PPK Faizi"],
            name="Faiz (%)", line=dict(color="orange", dash="dot"), yaxis="y"
        ))

    fig.add_trace(go.Scatter(
        x=merged["period_date"], y=merged["flesch_score"],
        name="Okunabilirlik (Flesch)", mode="markers",
        marker=dict(color="teal", size=8, opacity=0.8), yaxis="y"
    ))

    layout_shapes = [
        dict(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=0, y1=150,
             fillcolor="rgba(255, 0, 0, 0.08)", line_width=0, layer="below"),
        dict(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=-150, y1=0,
             fillcolor="rgba(0, 0, 255, 0.08)", line_width=0, layer="below"),
        dict(type="line", xref="paper", yref="y", x0=0, x1=1, y0=0, y1=0,
             line=dict(color="black", width=3), layer="below"),
    ]
    layout_annotations = [
        dict(x=0.02, y=130, xref="paper", yref="y", text="🦅 ŞAHİN", showarrow=False,
             font=dict(size=14, color="darkred", weight="bold"), xanchor="left"),
        dict(x=0.02, y=-130, xref="paper", yref="y", text="🕊️ GÜVERCİN", showarrow=False,
             font=dict(size=14, color="darkblue", weight="bold"), xanchor="left")
    ]

    governors = [("2020-11-01", "Naci Ağbal"), ("2021-04-01", "Şahap Kavcıoğlu"),
                 ("2023-06-01", "Hafize Gaye Erkan"), ("2024-02-01", "Fatih Karahan")]
    for start_date, name in governors:
        layout_shapes.append(dict(
            type="line", xref="x", yref="paper", x0=start_date, x1=start_date, y0=0, y1=1,
            line=dict(color="gray", width=1, dash="longdash"), layer="below"
        ))
        layout_annotations.append(dict(
            x=start_date, y=1.02, xref="x", yref="paper",
            text=f" <b>{name.split()[0][0]}.{name.split()[-1]}</b>",
            showarrow=False, xanchor="left", font=dict(size=9, color="#555")
        ))

    event_links_display = []
    if df_events is not None and not df_events.empty:
        for _, ev in df_events.iterrows():
            ev_date = pd.to_datetime(ev["event_date"]).strftime("%Y-%m-%d")

            layout_shapes.append(dict(
                type="line", xref="x", yref="paper",
                x0=ev_date, x1=ev_date, y0=0, y1=0.2,
                line=dict(color="purple", width=2, dash="dot")
            ))

            first_link = ev["links"].split("\n")[0] if ev.get("links") else ""
            layout_annotations.append(dict(
                x=ev_date, y=0.05, xref="x", yref="paper",
                text=f"ℹ️ <a href='{first_link}' target='_blank'>Haber</a>",
                showarrow=False, xanchor="left",
                font=dict(size=10, color="purple"),
                bgcolor="rgba(255,255,255,0.7)"
            ))

            if ev.get("links"):
                links_list = [l.strip() for l in ev["links"].split("\n") if l.strip()]
                event_links_display.append({"Tarih": ev_date, "Linkler": links_list})

    fig.update_layout(
        title="Merkez Bankası Analiz Paneli",
        hovermode="x unified",
        height=600,
        shapes=layout_shapes,
        annotations=layout_annotations,
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        yaxis=dict(title="Skor & Oranlar", range=[-150, 150], zeroline=False),
        yaxis2=dict(visible=False, overlaying="y", side="right"),
        yaxis3=dict(title="Kelime", overlaying="y", side="right",
                    showgrid=False, visible=False,
                    range=[0, max(1, merged["word_count"].max()) * 2])
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🤖 Yapay Zeka (RoBERTa) Trendi")

    if st.session_state.get("ai_trend_df") is not None and not st.session_state["ai_trend_df"].empty:
        fig_ai = utils.create_ai_trend_chart(st.session_state["ai_trend_df"]) if hasattr(utils, "create_ai_trend_chart") else None
        if fig_ai:
            st.plotly_chart(fig_ai, use_container_width=True, key="ai_chart_dashboard")
        else:
            st.warning("Grafik oluşturulamadı (Veri seti boş olabilir).")
    else:
        st.info("Yapay zeka analizi hesaplama gücü gerektirir. Görüntülemek için aşağıdaki butonu kullanın.")
        if st.button("🚀 AI Analizini Başlat (Dashboard)", key="btn_ai_dash"):
            if not utils.HAS_TRANSFORMERS:
                st.error("Kütüphaneler eksik.")
            else:
                with st.spinner("AI Modeli tüm geçmişi tarıyor..."):
                    df_all_data = utils.fetch_all_data()
                    res_df = utils.calculate_ai_trend_series(df_all_data)
                if res_df is None or res_df.empty:
                    st.error("Analiz sonucunda hiç veri dönmedi!")
                else:
                    st.session_state["ai_trend_df"] = res_df
                    st.rerun()

    if event_links_display:
        with st.expander("📅 Grafikteki Önemli Tarihler ve Haber Linkleri", expanded=False):
            for item in event_links_display:
                st.markdown(f"**{item['Tarih']}**")
                for link in item["Linkler"]:
                    st.markdown(f"- [Haber Linki]({link})")

    if st.button("🔄 Yenile"):
        st.cache_data.clear()
        st.rerun()

# ==============================================================================
# TAB 2: VERİ GİRİŞİ
# ==============================================================================
with tab2:
    st.subheader("Veri İşlemleri")
    st.info("ℹ️ **BİLGİ:** Aşağıdaki geçmiş kayıtlar listesinden istediğiniz dönemi seçerek, hangi cümlelerin hesaplamaya alındığını görebilirsiniz.")

    with st.container():
        df_all = utils.fetch_all_data()
        if df_all is not None and not df_all.empty:
            df_all = df_all.copy()
            df_all["period_date"] = pd.to_datetime(df_all["period_date"], errors="coerce")
            df_all = df_all.dropna(subset=["period_date"]).sort_values("period_date", ascending=False)
            df_all["date_only"] = df_all["period_date"].dt.date

            current_id = st.session_state["form_data"]["id"]
            with st.container(border=True):
                if st.button("➕ YENİ VERİ GİRİŞİ (Temizle)", type="secondary"):
                    reset_form()
                    st.rerun()

                st.markdown("---")
                c1, c2 = st.columns([1, 2])
                with c1:
                    val_date = st.session_state["form_data"]["date"]
                    selected_date = st.date_input("Tarih", value=val_date)
                    val_source = st.session_state["form_data"]["source"]
                    source = st.text_input("Kaynak", value=val_source)
                    st.caption(f"Dönem: **{selected_date.strftime('%Y-%m')}**")

                with c2:
                    val_text = st.session_state["form_data"]["text"]
                    txt = st.text_area("Metin", value=val_text, height=200, placeholder="Metni buraya yapıştırın...")

                st.markdown("---")

                if st.session_state["collision_state"]["active"]:
                    st.error("⚠️ Kayıt Çakışması")
                    admin_pass = st.text_input("Admin Şifresi", type="password", key="overwrite_pass")
                    if st.button("🚨 Üzerine Yaz", type="primary"):
                        if admin_pass == ADMIN_PWD:
                            p_txt = st.session_state["collision_state"]["pending_text"]
                            t_id = st.session_state["collision_state"]["target_id"]
                            s_abg, h_cnt, d_cnt, hawks, doves, h_ctx, d_ctx, flesch = utils.run_full_analysis(p_txt)
                            utils.update_entry(t_id, selected_date, p_txt, source, s_abg, s_abg)
                            st.success("Başarılı!")
                            reset_form()
                            st.rerun()
                        else:
                            st.error("Hatalı!")
                    if st.button("❌ İptal"):
                        st.session_state["collision_state"]["active"] = False
                        st.rerun()

                elif st.session_state["update_state"]["active"]:
                    st.warning("Güncelleme Onayı")
                    update_pass = st.text_input("Admin Şifresi", type="password", key="update_pass")
                    if st.button("💾 Güncelle", type="primary"):
                        if update_pass == ADMIN_PWD:
                            p_txt = st.session_state["update_state"]["pending_text"]
                            s_abg, h_cnt, d_cnt, hawks, doves, h_ctx, d_ctx, flesch = utils.run_full_analysis(p_txt)
                            utils.update_entry(current_id, selected_date, p_txt, source, s_abg, s_abg)
                            st.success("Güncellendi!")
                            reset_form()
                            st.rerun()
                        else:
                            st.error("Hatalı!")
                    if st.button("❌ İptal"):
                        st.session_state["update_state"]["active"] = False
                        st.rerun()

                else:
                    btn_label = "💾 Güncelle" if current_id else "💾 Kaydet"
                    if st.button(btn_label, type="primary"):
                        if txt:
                            collision_record = None
                            mask = df_all["date_only"] == selected_date
                            if mask.any():
                                collision_record = df_all[mask].iloc[0]

                            is_self_update = current_id and (
                                (collision_record is None) or
                                (collision_record is not None and int(collision_record["id"]) == current_id)
                            )

                            if is_self_update:
                                st.session_state["update_state"] = {"active": True, "pending_text": txt}
                                st.rerun()
                            elif collision_record is not None:
                                st.session_state["collision_state"] = {
                                    "active": True,
                                    "target_id": int(collision_record["id"]),
                                    "target_date": selected_date,
                                    "pending_text": txt
                                }
                                st.rerun()
                            else:
                                s_abg, h_cnt, d_cnt, hawks, doves, h_ctx, d_ctx, flesch = utils.run_full_analysis(txt)
                                utils.insert_entry(selected_date, txt, source, s_abg, s_abg)
                                st.success("Eklendi!")
                                reset_form()
                                st.rerun()
                        else:
                            st.error("Metin boş.")

                    if current_id:
                        with st.popover("🗑️ Sil"):
                            del_pass = st.text_input("Şifre", type="password", key="del_pass")
                            if st.button("🔥 Sil"):
                                if del_pass == ADMIN_PWD:
                                    utils.delete_entry(current_id)
                                    st.success("Silindi!")
                                    reset_form()
                                    st.rerun()
                                else:
                                    st.error("Hatalı!")

                if txt:
                    s_live, h_cnt, d_cnt, h_list, d_list, h_ctx, d_ctx, flesch_live = utils.run_full_analysis(txt)
                    st.markdown("---")
                    st.subheader("🔍 Analiz Detayları")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Şahin", f"{h_cnt}")
                    with c2:
                        st.metric("Güvercin", f"{d_cnt}")
                    with c3:
                        st.metric("Flesch", f"{flesch_live:.1f}")
                    st.caption(f"**Net Skor:** {s_live:.2f}")

                    with st.expander("📄 Tespit Edilen Cümleler", expanded=True):
                        k1, k2 = st.columns(2)
                        with k1:
                            st.markdown("#### 🦅 Şahin")
                            if h_list:
                                for item in h_list:
                                    t = item.split(" (")[0]
                                    st.markdown(f"**{item}**")
                                    if t in h_ctx:
                                        for s in h_ctx[t]:
                                            st.caption(f"📝 {s}")
                            else:
                                st.write("- Yok")
                        with k2:
                            st.markdown("#### 🕊️ Güvercin")
                            if d_list:
                                for item in d_list:
                                    t = item.split(" (")[0]
                                    st.markdown(f"**{item}**")
                                    if t in d_ctx:
                                        for s in d_ctx[t]:
                                            st.caption(f"📝 {s}")
                            else:
                                st.write("- Yok")

            st.markdown("### 📋 Kayıtlar")
            df_show = df_all.copy()
            df_show["Dönem"] = df_show["period_date"].dt.strftime("%Y-%m")
            if "score_abg" in df_show.columns:
                df_show["Görsel Skor"] = df_show["score_abg"].apply(lambda x: x * 100 if abs(x) <= 1 else x)
            else:
                df_show["Görsel Skor"] = np.nan

            event = st.dataframe(
                df_show[["id", "Dönem", "Görsel Skor"]],
                on_select="rerun",
                selection_mode="single-row",
                use_container_width=True,
                hide_index=True,
                key=st.session_state["table_key"]
            )

            if len(event.selection.rows) > 0:
                sel_id = df_show.iloc[event.selection.rows[0]]["id"]
                if st.session_state["collision_state"]["active"] or st.session_state["update_state"]["active"]:
                    st.session_state["collision_state"]["active"] = False
                    st.session_state["update_state"]["active"] = False
                if st.session_state["form_data"]["id"] != sel_id:
                    orig = df_all[df_all["id"] == sel_id].iloc[0]
                    st.session_state["form_data"] = {
                        "id": int(orig["id"]),
                        "date": pd.to_datetime(orig["period_date"]).date(),
                        "source": orig["source"],
                        "text": orig["text_content"]
                    }
                    st.rerun()
        else:
            st.info("Veri yok.")

# ==============================================================================
# TAB IMP: HABERLER
# ==============================================================================
with tab_imp:
    st.header("📅 Haberler (Event Logs)")
    st.caption("Seçtiğin tarihe haber linkleri ekle. Dashboard grafiğinde mor çizgiler olarak görünür.")

    try:
        c1, c2 = st.columns([1, 3])
        with c1:
            ev_date = st.date_input("Tarih", value=datetime.date.today(), key="ev_date_in")
        with c2:
            ev_links = st.text_area(
                "Haber linkleri (her satıra 1 link)",
                height=140,
                placeholder="https://...\nhttps://...\n...",
                key="ev_links_in"
            )

        col_add, col_refresh = st.columns([1, 1])
        with col_add:
            if st.button("➕ Kaydet", type="primary", key="btn_add_event"):
                links_clean = "\n".join([l.strip() for l in (ev_links or "").splitlines() if l.strip()])
                if not links_clean:
                    st.warning("En az 1 link gir.")
                else:
                    utils.add_event(ev_date, links_clean)
                    st.success("Eklendi!")
                    st.rerun()

        with col_refresh:
            if st.button("🔄 Yenile", key="btn_refresh_events"):
                st.rerun()

        st.divider()

        df_events = utils.fetch_events()

        if df_events is None or df_events.empty:
            st.info("Henüz kayıtlı haber yok.")
        else:
            df_events = df_events.copy()
            if "event_date" in df_events.columns:
                df_events["event_date"] = pd.to_datetime(df_events["event_date"], errors="coerce")
                df_events = df_events.dropna(subset=["event_date"]).sort_values("event_date", ascending=False)

            st.subheader("📌 Kayıtlı Haberler")

            for _, row in df_events.iterrows():
                rid = row.get("id", None)
                d = row.get("event_date", None)
                d_str = pd.to_datetime(d).strftime("%Y-%m-%d") if pd.notna(d) else "—"

                links_raw = row.get("links", "") or ""
                links_list = [l.strip() for l in links_raw.splitlines() if l.strip()]

                with st.container(border=True):
                    top1, top2 = st.columns([5, 1])
                    with top1:
                        st.markdown(f"### {d_str}")
                        if links_list:
                            for l in links_list:
                                st.markdown(f"- {l}")
                        else:
                            st.caption("Link yok")

                    with top2:
                        if rid is not None:
                            if st.button("🗑️ Sil", key=f"del_event_{rid}"):
                                utils.delete_event(int(rid))
                                st.success("Silindi!")
                                st.rerun()

            with st.expander("📋 Ham Tablo", expanded=False):
                st.dataframe(df_events, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error("Haberler sekmesi hata aldı.")
        st.exception(e)

# ==============================================================================
# TAB 3: PİYASA VERİLERİ
# ==============================================================================
with tab3:
    st.header("Piyasa Verileri")
    d1 = st.date_input("Başlangıç", datetime.date(2023, 1, 1))
    d2 = st.date_input("Bitiş", datetime.date.today())
    if st.button("Getir", key="get_market"):
        df, err = utils.fetch_market_data_adapter(d1, d2)
        if df is not None and not df.empty:
            fig_m = go.Figure()
            if "Yıllık TÜFE" in df.columns:
                fig_m.add_trace(go.Scatter(x=df["Donem"], y=df["Yıllık TÜFE"], name="Yıllık TÜFE", line=dict(color="red")))
            if "Aylık TÜFE" in df.columns:
                fig_m.add_trace(go.Scatter(x=df["Donem"], y=df["Aylık TÜFE"], name="Aylık TÜFE", line=dict(color="blue", dash="dot")))
            if "PPK Faizi" in df.columns:
                fig_m.add_trace(go.Scatter(x=df["Donem"], y=df["PPK Faizi"], name="Faiz", line=dict(color="orange")))
            st.plotly_chart(fig_m, use_container_width=True)
            st.dataframe(df, use_container_width=True)
        else:
            st.error(f"Hata: {err}")

# ==============================================================================
# TAB 4: FREKANS (İZLENEN TERİMLER)
# ==============================================================================
with tab4:
    st.header("🔍 Frekans (İzlenen Terimler)")

    DEFAULT_WATCH_TERMS = [
        "inflation", "disinflation", "stability", "growth", "gdp",
        "interest rate", "policy rate", "lowered", "macroprudential",
        "target", "monetary policy", "tightened", "risks", "exchange rate",
        "prudently", "global", "recession", "food"
    ]

    if "watch_terms" not in st.session_state:
        st.session_state["watch_terms"] = DEFAULT_WATCH_TERMS.copy()

    def add_watch_term():
        t = (st.session_state.get("watch_term_in", "") or "").strip().lower()
        if not t:
            return
        if t not in st.session_state["watch_terms"]:
            st.session_state["watch_terms"].append(t)

    def reset_watch_terms():
        st.session_state["watch_terms"] = DEFAULT_WATCH_TERMS.copy()

    def _fallback_build_watch_terms_timeseries(df_in: pd.DataFrame, terms: list) -> pd.DataFrame:
        rows = []
        for _, r in df_in.iterrows():
            txt = str(r.get("text_content", "") or "").lower()
            rec = {"period_date": r["period_date"], "Donem": r["Donem"]}
            for term in terms:
                rec[term] = txt.count(term.lower())
            rows.append(rec)
        return pd.DataFrame(rows).sort_values("period_date").reset_index(drop=True)

    df_all = utils.fetch_all_data()
    if df_all is None or df_all.empty:
        st.info("Yeterli veri yok.")
        st.stop()

    df_all = df_all.copy()
    df_all["period_date"] = pd.to_datetime(df_all.get("period_date", None), errors="coerce")
    df_all = df_all.dropna(subset=["period_date"]).sort_values("period_date", ascending=False).reset_index(drop=True)
    df_all["Donem"] = df_all["period_date"].dt.strftime("%Y-%m")

    st.caption(
        "Bu bölüm sadece izlediğin kelimeleri gösterir. "
        "Yeni kelime eklersen ve metinlerde geçiyorsa otomatik seriye girer."
    )

    c1, c2 = st.columns([3, 1])
    with c1:
        st.text_input(
            "➕ Kelime veya phrase ekle (Enter)",
            key="watch_term_in",
            on_change=add_watch_term,
            placeholder="ör: liquidity, demand, wage, credit growth"
        )
    with c2:
        if st.button("↩️ Reset", type="secondary"):
            reset_watch_terms()
            st.rerun()

    terms = st.session_state.get("watch_terms", [])
    if not terms:
        st.warning("İzlenen kelime yok. Yukarıdan ekleyebilirsin.")
        st.stop()

    st.write("Aktif izlenen terimler:")
    cols = st.columns(6)
    for i, term in enumerate(list(terms)):
        if cols[i % 6].button(f"{term} ✖", key=f"watch_del_{term}"):
            st.session_state["watch_terms"].remove(term)
            st.rerun()

    st.divider()

    if hasattr(utils, "build_watch_terms_timeseries"):
        freq_df = utils.build_watch_terms_timeseries(df_all, terms)
    else:
        freq_df = _fallback_build_watch_terms_timeseries(df_all, terms)

    if freq_df is None or freq_df.empty:
        st.info("Zaman serisi üretilemedi.")
    else:
        usable_terms = [
            t for t in terms
            if t in freq_df.columns and pd.to_numeric(freq_df[t], errors="coerce").fillna(0).sum() > 0
        ]
        if not usable_terms:
            st.info("Bu kelimeler metinlerde hiç geçmiyor.")
        else:
            fig = go.Figure()
            for t in usable_terms:
                fig.add_trace(go.Scatter(
                    x=freq_df["period_date"],
                    y=freq_df[t],
                    name=t,
                    mode="lines+markers"
                ))
            fig.update_layout(
                title="İzlenen Ekonomi Terimleri — Zaman Serisi",
                hovermode="x unified",
                height=420,
                legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📋 Ham tablo", expanded=False):
                show_cols = ["Donem"] + usable_terms
                st.dataframe(freq_df[show_cols], use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🔄 Metin Farkı (Diff) Analizi")

    c_diff1, c_diff2 = st.columns(2)
    with c_diff1:
        sel_date1 = st.selectbox("Eski Metin:", df_all["Donem"].tolist(), index=min(1, len(df_all) - 1), key="diff_old")
    with c_diff2:
        sel_date2 = st.selectbox("Yeni Metin:", df_all["Donem"].tolist(), index=0, key="diff_new")

    if st.button("Farkları Göster", type="primary", key="btn_diff"):
        t1 = str(df_all[df_all["Donem"] == sel_date1].iloc[0].get("text_content", "") or "")
        t2 = str(df_all[df_all["Donem"] == sel_date2].iloc[0].get("text_content", "") or "")
        diff_html = utils.generate_diff_html(t1, t2)
        st.markdown(f"**Kırmızı:** {sel_date1}'den silinenler | **Yeşil:** {sel_date2}'ye eklenenler")
        with st.container(border=True, height=400):
            st.markdown(diff_html, unsafe_allow_html=True)

# ==============================================================================
# TAB: TEXT AS DATA (TF-IDF) — HYBRID + CPI delta_bp tahmini
# ==============================================================================
with tab_textdata:
    st.header("📚 Text as Data (TF-IDF) — HYBRID + CPI PPK Kararı (delta_bp) Tahmini")

    if not utils.HAS_ML_DEPS:
        st.error("ML kütüphaneleri eksik (sklearn).")


    df_logs = utils.fetch_all_data()
    if df_logs is None or df_logs.empty:
        st.info("Veri yok.")


    df_logs = df_logs.copy()
    df_logs["period_date"] = pd.to_datetime(df_logs["period_date"], errors="coerce")
    df_logs = df_logs.dropna(subset=["period_date"]).sort_values("period_date")

    for c in ["policy_rate", "delta_bp"]:
        if c in df_logs.columns:
            df_logs[c] = pd.to_numeric(df_logs[c], errors="coerce")

    min_d = df_logs["period_date"].min().date()
    max_d = datetime.date.today()
    df_market, err = utils.fetch_market_data_adapter(min_d, max_d)
    if err:
        st.warning(f"Market veri uyarısı: {err}")

    df_td = utils.textasdata_prepare_df_hybrid_cpi(
        df_logs, df_market,
        text_col="text_content",
        date_col="period_date",
        y_col="delta_bp",
        rate_col="policy_rate"
    )

    if df_td is None or df_td.empty or df_td["delta_bp"].notna().sum() < 10:
        st.warning("HYBRID+CPI eğitim için yeterli gözlem yok. (En az ~10 kayıt önerilir)")


    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.info(
            "Bu sekme **English TF-IDF (word+char)** + **faiz geçmişi** + **TÜFE (lagged)** ile "
            "**delta_bp (bps)** tahmin eder. Walk-forward backtest gösterir."
        )
    with c2:
        min_df = st.number_input("min_df", min_value=1, max_value=10, value=2, step=1, key="td_min_df")
    with c3:
        alpha = st.number_input("Ridge alpha", min_value=0.1, max_value=80.0, value=10.0, step=1.0, key="td_alpha")

    if "textasdata_model" not in st.session_state:
        st.session_state["textasdata_model"] = None

    if st.button("🚀 Modeli Eğit / Yenile (HYBRID + CPI)", type="primary", key="btn_td_train"):
        with st.spinner("Eğitiliyor + walk-forward backtest..."):
            out = utils.train_textasdata_hybrid_cpi_ridge(
                df_td,
                min_df=int(min_df),
                alpha=float(alpha),
                n_splits=6,
                word_ngram=(1, 2),
                char_ngram=(3, 5),
                max_features_word=12000,
                max_features_char=20000
            )
        st.session_state["textasdata_model"] = out
        st.success("Hazır!")

    model_pack = st.session_state.get("textasdata_model")
    if not model_pack:
        st.info("Başlamak için yukarıdaki butona bas.")


    metrics = model_pack.get("metrics", {}) or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE (bps)", f"{metrics.get('mae', np.nan):.1f}")
    c2.metric("RMSE (bps)", f"{metrics.get('rmse', np.nan):.1f}")
    c3.metric("R²", f"{metrics.get('r2', np.nan):.2f}")
    c4.metric("Gözlem", f"{metrics.get('n', 0)}")

    df_pred = model_pack.get("pred_df")
    if df_pred is not None and not df_pred.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_pred["period_date"], y=df_pred["delta_bp"],
            name="Gerçek delta_bp", opacity=0.45
        ))
        fig.add_trace(go.Scatter(
            x=df_pred["period_date"], y=df_pred["pred_delta_bp"],
            name="Walk-forward Tahmin", mode="lines+markers"
        ))
        fig.add_hline(y=0, line_color="black", opacity=0.25)
        fig.update_layout(
            title="Text-as-Data HYBRID + CPI Backtest — delta_bp (bps)",
            hovermode="x unified",
            height=420,
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)

    coef_df = model_pack.get("coef_df")
    if coef_df is not None and not coef_df.empty:
        st.subheader("🧠 Which words push hike/cut? (word TF-IDF coefficients)")
        k = st.slider("Show top K", 10, 60, 25, step=5, key="td_topk")
        cpos, cneg = st.columns(2)
        with cpos:
            st.markdown("### 🔺 Hike-leaning (positive)")
            st.dataframe(coef_df.sort_values("coef", ascending=False).head(int(k)),
                         use_container_width=True, hide_index=True)
        with cneg:
            st.markdown("### 🔻 Cut-leaning (negative)")
            st.dataframe(coef_df.sort_values("coef", ascending=True).head(int(k)),
                         use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🔮 Single-text Prediction (HYBRID + CPI)")

    last_rate = float(df_td["policy_rate"].dropna().iloc[-1]) if df_td["policy_rate"].notna().any() else np.nan
    st.caption(f"Last known policy_rate: {last_rate if np.isfinite(last_rate) else '—'}")

    txt = st.text_area("Paste the statement text", height=220, placeholder="Paste PPK statement...", key="td_input")

    if st.button("🧾 Predict (HYBRID + CPI)", type="secondary", key="btn_td_pred"):
        if not txt or len(txt.strip()) < 30:
            st.warning("Text too short.")
        else:
            pred = utils.predict_textasdata_hybrid_cpi(model_pack, df_td, txt)
            pred_bp = float((pred or {}).get("pred_delta_bp", 0.0))
            implied = (last_rate + pred_bp / 100.0) if np.isfinite(last_rate) else np.nan
            c1, c2 = st.columns(2)
            c1.metric("Predicted delta_bp", f"{pred_bp:.0f} bps")
            c2.metric("Implied policy_rate", f"{implied:.2f}" if np.isfinite(implied) else "—")

# ==============================================================================
# TAB6: WordCloud
# ==============================================================================
with tab6:
    st.header("☁️ Kelime Bulutu (WordCloud)")

    df_all = utils.fetch_all_data()
    if df_all is None or df_all.empty:
        st.info("Veri yok.")
        st.stop()

    df_all = df_all.copy()
    df_all["period_date"] = pd.to_datetime(df_all["period_date"], errors="coerce")
    df_all = df_all.dropna(subset=["period_date"]).sort_values("period_date", ascending=False)
    df_all["Donem"] = df_all["period_date"].dt.strftime("%Y-%m")

    st.text_input("🚫 Buluttan Çıkarılacak Kelimeler (Enter)", key="cloud_stop_in", on_change=add_cloud_stop)

    if st.session_state.get("stop_words_cloud"):
        st.write("Filtreler:")
        cols = st.columns(8)
        for i, word in enumerate(st.session_state["stop_words_cloud"]):
            if cols[i % 8].button(f"{word} ✖", key=f"del_cloud_{word}"):
                st.session_state["stop_words_cloud"].remove(word)
                st.rerun()

    st.divider()

    dates = df_all["Donem"].tolist()
    sel_cloud_date = st.selectbox("Dönem Seçin:", ["Tüm Zamanlar"] + dates, key="cloud_sel")

    if st.button("Bulutu Oluştur", type="primary", key="btn_cloud"):
        if sel_cloud_date == "Tüm Zamanlar":
            text_cloud = " ".join(df_all["text_content"].astype(str).tolist())
        else:
            text_cloud = df_all[df_all["Donem"] == sel_cloud_date].iloc[0]["text_content"]

        fig_wc = utils.generate_wordcloud_img(text_cloud, st.session_state.get("stop_words_cloud", []))
        if fig_wc:
            st.pyplot(fig_wc)
        else:
            st.error("Kütüphane eksik veya metin boş.")

# ==============================================================================
# TAB7: ABF (2019)
# ==============================================================================
with tab7:
    st.header("📜 Apel, Blix ve Grimaldi (2019) Analizi")
    st.info("Bu yöntem, kelimeleri kategorilere ayırır ve sıfat bağlamına göre 'Şahin/Güvercin' puanlar.")

    df_abg_source = utils.fetch_all_data()
    if df_abg_source is None or df_abg_source.empty:
        st.info("Analiz için veri yok.")
        st.stop()

    df_abg_source = df_abg_source.copy()
    df_abg_source["period_date"] = pd.to_datetime(df_abg_source["period_date"], errors="coerce")
    df_abg_source = df_abg_source.dropna(subset=["period_date"]).sort_values("period_date")
    df_abg_source["Donem"] = df_abg_source["period_date"].dt.strftime("%Y-%m")

    abg_df = utils.calculate_abg_scores(df_abg_source)

    fig_abg = go.Figure()
    fig_abg.add_trace(go.Scatter(
        x=abg_df["period_date"], y=abg_df["abg_index"],
        name="ABF Net Hawkishness", line=dict(color="purple", width=3),
        marker=dict(size=8)
    ))
    fig_abg.add_shape(
        type="line",
        x0=abg_df["period_date"].min(), x1=abg_df["period_date"].max(),
        y0=1, y1=1,
        line=dict(color="gray", dash="dash")
    )
    fig_abg.update_layout(
        title="ABF (2019) Endeksi Zaman Serisi (Nötr=1.0)",
        yaxis_title="Hawkishness Index (0 - 2)",
        hovermode="x unified"
    )
    st.plotly_chart(fig_abg, use_container_width=True)

    st.divider()
    st.subheader("🔍 Dönem Bazlı Detaylar")

    sel_abg_period = st.selectbox("İncelenecek Dönem:", abg_df["Donem"].tolist(), key="abg_sel")
    subset = df_abg_source[df_abg_source["Donem"] == sel_abg_period]

    if subset.empty:
        st.error("Seçilen dönem için metin bulunamadı.")
        st.stop()

    text_abg = subset.iloc[0]["text_content"]
    res = utils.analyze_hawk_dove(
        text_abg,
        DICT=utils.DICT,
        window_words=10,
        dedupe_within_term_window=True,
        nearest_only=True
    )

    net_h = res.get("net_hawkishness", 0)
    h_cnt = res.get("hawk_count", 0)
    d_cnt = res.get("dove_count", 0)
    details = res.get("match_details", [])

    c1, c2, c3 = st.columns(3)
    c1.metric("Net Endeks", f"{net_h:.4f}")
    c2.metric("🦅 Şahin Eşleşme", h_cnt)
    c3.metric("🕊️ Güvercin Eşleşme", d_cnt)

    if "topic_counts" in res:
        with st.expander("Detaylı Kırılım (Topic Counts)"):
            st.json(res["topic_counts"])

    with st.expander("📝 Detaylı Eşleşme Tablosu (Cümle Bağlamı)", expanded=True):
        if details:
            detail_data = []
            for m in details:
                detail_data.append({
                    "Tip": "🦅 ŞAHİN" if m["type"] == "HAWK" else "🕊️ GÜVERCİN",
                    "Eşleşen Terim": m["term"],
                    "Cümle": m["sentence"]
                })
            st.dataframe(pd.DataFrame(detail_data), use_container_width=True, hide_index=True)
        else:
            st.info("Bu metinde herhangi bir ABF sözlük eşleşmesi bulunamadı.")

    with st.expander("Metin Önizleme"):
        st.write(text_abg)

# ==============================================================================
# TAB ROBERTA: CB-RoBERTa
# ==============================================================================
with tab_roberta:
    st.header("🧠 CentralBankRoBERTa (Yapay Zeka Analizi)")

    if not utils.HAS_TRANSFORMERS:
        st.error("Kütüphaneler eksik. (transformers/torch)")
        st.stop()

    st.subheader("📈 Tarihsel Trend (Calib + EMA + Hysteresis)")

    if st.session_state.get("ai_trend_df") is not None and not st.session_state["ai_trend_df"].empty:
        df_tr = st.session_state["ai_trend_df"]
        fig_trend = utils.create_ai_trend_chart(df_tr) if hasattr(utils, "create_ai_trend_chart") else None
        if fig_trend:
            st.plotly_chart(fig_trend, use_container_width=True, key="ai_chart_roberta")
        else:
            st.warning("Grafik oluşturulamadı.")

        cbtn1, _ = st.columns([1, 3])
        with cbtn1:
            if st.button("🔄 Tekrar Hesapla", key="btn_ai_recalc"):
                st.session_state["ai_trend_df"] = None
                st.rerun()
    else:
        st.info("Tarihsel trend için tüm metinler taranır. (Biraz zaman alabilir)")
        if st.button("🚀 Tüm Geçmişi Analiz Et", type="primary", key="btn_ai_run_all"):
            with st.spinner("Model tüm geçmişi tarıyor..."):
                df_all_rob = utils.fetch_all_data()
                res_df = utils.calculate_ai_trend_series(df_all_rob)

            if res_df is None or res_df.empty:
                st.error("Analiz sonucu boş geldi. (DB boş olabilir veya model hata vermiş olabilir)")
            else:
                st.session_state["ai_trend_df"] = res_df
                st.rerun()

    with st.expander("ℹ️ Bu grafik nasıl hesaplanıyor?", expanded=False):
        st.markdown("""
Bu grafik, modelin verdiği **3 sınıf olasılığından** (Şahin / Güvercin / Nötr) türetilmiş bir **endeks**tir.

1) Her metin için `P(HAWK)`, `P(DOVE)`, `P(NEUT)` alınır.  
2) Ham fark: **diff = P(HAWK) − P(DOVE)**  
3) Serinin kendi dağılımına göre **robust kalibrasyon** yapılır (median + MAD → robust z-score)  
4) `tanh` ile skor **−100..+100** bandına sıkıştırılır  
5) **EMA (span=7)** ile yumuşatılır  
6) Rejim etiketinde hızlı flip olmasın diye **histerezis** uygulanır (±25 eşikleri)
        """)

    st.divider()
    st.subheader("🔍 Tekil Dönem Detay Analizi")

    df_all_rob = utils.fetch_all_data()
    if df_all_rob is None or df_all_rob.empty:
        st.info("Tekil analiz için veritabanında kayıt yok.")
        st.stop()

    df_all_rob = df_all_rob.copy()
    df_all_rob["period_date"] = pd.to_datetime(df_all_rob["period_date"], errors="coerce")
    df_all_rob = df_all_rob.dropna(subset=["period_date"]).sort_values("period_date", ascending=False)
    df_all_rob["Donem"] = df_all_rob["period_date"].dt.strftime("%Y-%m")

    sel_rob_period = st.selectbox(
        "İncelenecek Dönem:",
        df_all_rob["Donem"].tolist(),
        index=0,
        key="rob_single_sel"
    )

    row_rob = df_all_rob[df_all_rob["Donem"] == sel_rob_period].iloc[0]
    txt_input = str(row_rob.get("text_content", "") or "")

    with st.expander("Metni Gör", expanded=False):
        st.write(txt_input)

    if st.button("🧪 Bu Metni Analiz Et", type="secondary", key="btn_ai_single"):
        with st.spinner("Analiz ediliyor..."):
            roberta_res = utils.analyze_with_roberta(txt_input)

        if not isinstance(roberta_res, dict):
            st.error(f"Model hata döndürdü: {roberta_res}")
            st.stop()

        scores = roberta_res.get("scores_map", {}) or {}
        h = float(scores.get("HAWK", 0.0))
        d = float(scores.get("DOVE", 0.0))
        n = float(scores.get("NEUT", 0.0))
        diff = float(roberta_res.get("diff", h - d))
        stance = str(roberta_res.get("stance", ""))

        ema_score = None
        if st.session_state.get("ai_trend_df") is not None and not st.session_state["ai_trend_df"].empty:
            tmp = st.session_state["ai_trend_df"]
            hit = tmp[tmp["Dönem"] == sel_rob_period]
            if not hit.empty and "AI Score (EMA)" in hit.columns:
                ema_score = float(hit.iloc[0]["AI Score (EMA)"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Duruş", stance)
        c2.metric("Diff (H-D)", f"{diff:.3f}")
        c3.metric("AI Score (EMA)", f"{ema_score:.1f}" if ema_score is not None else "—")

        st.write("Sınıf Skorları:")
        st.json({"HAWK": h, "DOVE": d, "NEUT": n})

        with st.expander("DEBUG (ham çıktı)", expanded=False):
            st.json(roberta_res)

        st.markdown("---")
        st.subheader("🧩 Cümle Bazlı Ayrıştırma (RoBERTa)")
        
        # 1) Policy Action (metin üstünden)
        act = utils.detect_policy_action(txt_input) if hasattr(utils, "detect_policy_action") else {"action":"UNKNOWN","bp":None,"weight_0_1":None}
        act_label = str(act.get("action", "UNKNOWN"))
        act_bp = act.get("bp", None)
        act_w = act.get("weight_0_1", None)
        
        # UI: Action + ağırlık
        cA, cB, cC, cD = st.columns(4)
        
        if act_bp is None:
            cA.metric("Policy Action", act_label)
        else:
            cA.metric("Policy Action", f"{act_label} ({act_bp:+d} bp)")
        
        # 500bp = 1.00 ölçeği (sunum için)
        if act_w is None:
            cB.metric("Action Weight (0-1)", "—")
        else:
            cB.metric("Action Weight (0-1)", f"{act_w:.2f}")
        
        # 2) Cümle bazlı RoBERTa
        if hasattr(utils, "analyze_sentences_with_roberta"):
            df_sent = utils.analyze_sentences_with_roberta(txt_input)
        
            # Eğer df_sent boşsa bunu saklama: kullanıcıya hata nedenini göster
            if df_sent is None or df_sent.empty:
                cC.metric("🦅 Şahin cümle", 0)
                cD.metric("🕊️ Güvercin cümle", 0)
                st.warning("Cümle bazlı analiz boş döndü. (split_sentences_nlp cümle üretemiyor olabilir veya model yüklenemiyor olabilir.)")
                st.caption("İpucu: utils.split_sentences_nlp fonksiyonunun gerçekten cümle listesi döndürdüğünü kontrol et.")
            else:
                # Özet
                summary = utils.summarize_sentence_roberta(df_sent) if hasattr(utils, "summarize_sentence_roberta") else {}
                cC.metric("🦅 Şahin cümle", int(summary.get("hawk_n", 0)))
                cD.metric("🕊️ Güvercin cümle", int(summary.get("dove_n", 0)))
        
                c1, c2, c3 = st.columns(3)
                n = int(summary.get("n", 0))
                c1.metric("Diff ortalama", f"{float(summary.get('diff_mean', np.nan)):.3f}" if n else "—")
                c2.metric("Pozitif toplam (hawk itişi)", f"{float(summary.get('pos_sum', np.nan)):.2f}" if n else "—")
                c3.metric("Negatif toplam (dove itişi)", f"{float(summary.get('neg_sum', np.nan)):.2f}" if n else "—")
        
                # Sunum için: Action büyüklüğünü "etki" diye de yaz
                st.caption(
                    "Not: Net duruş **cümle sayısından değil Diff (H−D) ağırlıklarından** gelir. "
                    "Ayrıca Action Weight, karar büyüklüğünü 0..1 ölçeğine indirger (500bp=1.0)."
                )
        
                # En güçlü 5 cümleyi ayrıca göster (sunumda çok iş görüyor)
                with st.expander("🎯 En güçlü cümleler (Top 5 hawk / Top 5 dove)", expanded=False):
                    top_h = df_sent.sort_values("Diff (H-D)", ascending=False).head(5)
                    top_d = df_sent.sort_values("Diff (H-D)", ascending=True).head(5)
                    st.markdown("**🦅 Hawk-leaning (Top 5)**")
                    st.dataframe(top_h[["Cümle", "Duruş", "Diff (H-D)", "HAWK", "DOVE", "NEUT"]], use_container_width=True, hide_index=True)
                    st.markdown("**🕊️ Dove-leaning (Top 5)**")
                    st.dataframe(top_d[["Cümle", "Duruş", "Diff (H-D)", "HAWK", "DOVE", "NEUT"]], use_container_width=True, hide_index=True)
        
                st.dataframe(df_sent, use_container_width=True)
        
        else:
            st.error("utils.analyze_sentences_with_roberta bulunamadı.")
