"""
Bus Service Reliability — Streamlit dashboard
=============================================
Interactive view of the Spark-processed incident data plus the trained-model
performance summary, for a Transport-Authority stakeholder.

The dashboard reads the *Spark-processed* Parquet with pandas (a justified
memory-scale read for the presentation layer — the heavy lifting already
happened in Spark) and the model-metric JSON artefacts.

Run:  .venv/bin/streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
PARQUET = ROOT / "data" / "processed" / "incidents_clean.parquet"
OUTPUTS = ROOT / "outputs"

st.set_page_config(page_title="Bus Service Reliability", layout="wide",
                   page_icon="🚌")


@st.cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET)
    df["occurred_ts"] = pd.to_datetime(df["occurred_ts"], errors="coerce")
    return df


@st.cache_data
def load_metrics():
    cls = json.loads((OUTPUTS / "classification_metrics.json").read_text()) \
        if (OUTPUTS / "classification_metrics.json").exists() else None
    reg = json.loads((OUTPUTS / "regression_metrics.json").read_text()) \
        if (OUTPUTS / "regression_metrics.json").exists() else None
    return cls, reg


df = load_data()
cls_metrics, reg_metrics = load_metrics()

# ---- header -----------------------------------------------------------------
st.title("🚌 Bus Service Reliability — Predictive Analytics")
st.caption("NYC Bus Breakdown & Delay incidents · PySpark pipeline · "
           f"{len(df):,} cleaned records (2015–2026)")

# ---- sidebar filters --------------------------------------------------------
st.sidebar.header("Filters")
years = sorted(int(y) for y in df["year"].dropna().unique())
yr = st.sidebar.select_slider("Year range", options=years,
                              value=(years[0], years[-1]))
boros = ["(All)"] + sorted(df["boro"].dropna().unique().tolist())
boro_sel = st.sidebar.selectbox("Borough / Area", boros)
band_sel = st.sidebar.multiselect(
    "Time band", ["AM_Peak", "Midday", "PM_Peak", "OffPeak"],
    default=["AM_Peak", "Midday", "PM_Peak", "OffPeak"])

mask = df["year"].between(yr[0], yr[1]) & df["time_band"].isin(band_sel)
if boro_sel != "(All)":
    mask &= df["boro"] == boro_sel
d = df[mask]

# ---- KPI row ----------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Incidents", f"{len(d):,}")
c2.metric("Breakdown rate", f"{d['is_breakdown'].mean()*100:.1f}%")
c3.metric("Avg delay (min)", f"{d['delay_minutes'].mean():.1f}")
c4.metric("On-time (≤15 min)", f"{d['on_time'].mean()*100:.1f}%")

st.divider()

tab1, tab2, tab3 = st.tabs(["📊 Operations", "🏢 Operators", "🤖 Models & Risk"])

with tab1:
    a, b = st.columns(2)
    by_boro = d.groupby("boro").size().reset_index(name="incidents") \
        .sort_values("incidents", ascending=True)
    a.plotly_chart(px.bar(by_boro, x="incidents", y="boro", orientation="h",
                          title="Incidents by Borough / Area"),
                   use_container_width=True)
    by_reason = d.groupby("reason").size().reset_index(name="incidents") \
        .sort_values("incidents", ascending=False).head(10)
    b.plotly_chart(px.bar(by_reason, x="reason", y="incidents",
                          title="Top Reasons"), use_container_width=True)

    by_hour = d.groupby("hour").size().reset_index(name="incidents")
    st.plotly_chart(px.line(by_hour, x="hour", y="incidents", markers=True,
                            title="Incidents by Hour of Day (peak demand)"),
                    use_container_width=True)

with tab2:
    league = (d.groupby("operator")
                .agg(incidents=("is_breakdown", "size"),
                     breakdown_rate=("is_breakdown", "mean"),
                     avg_delay=("delay_minutes", "mean"))
                .reset_index())
    league = league[league["incidents"] >= 500] \
        .sort_values("breakdown_rate", ascending=False).head(15)
    league["breakdown_rate"] = (league["breakdown_rate"] * 100).round(1)
    league["avg_delay"] = league["avg_delay"].round(1)
    st.subheader("Operator reliability league table (≥500 incidents)")
    st.dataframe(league, use_container_width=True, hide_index=True)
    st.plotly_chart(
        px.bar(league.sort_values("breakdown_rate"), x="breakdown_rate",
               y="operator", orientation="h",
               title="Breakdown rate (%) by operator"),
        use_container_width=True)

with tab3:
    st.subheader("Model performance (held-out test set)")
    if cls_metrics:
        st.markdown("**Classification — predict Breakdown vs Running Late**")
        cm = pd.DataFrame(cls_metrics["results"])[
            ["model", "accuracy", "precision_breakdown", "recall_breakdown",
             "f1_breakdown", "roc_auc", "pr_auc"]].round(3)
        st.dataframe(cm, use_container_width=True, hide_index=True)
        st.caption(f"Best model: **{cls_metrics['best_model']}** · "
                   f"CV ROC-AUC {cls_metrics['cross_validation']['best_cv_roc_auc']:.3f}")
    if reg_metrics:
        st.markdown("**Regression — predict delay minutes**")
        rm = pd.DataFrame(reg_metrics["results"])[
            ["model", "rmse", "mae", "r2", "train_seconds"]].round(3)
        st.dataframe(rm, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Empirical breakdown-risk lookup")
    colx, coly = st.columns(2)
    op = colx.selectbox("Operator", sorted(df["operator"].dropna().unique()))
    tb = coly.selectbox("Time band", ["AM_Peak", "Midday", "PM_Peak", "OffPeak"])
    sub = df[(df["operator"] == op) & (df["time_band"] == tb)]
    if len(sub):
        risk = sub["is_breakdown"].mean()
        st.metric(f"Historical breakdown risk — {op} / {tb}",
                  f"{risk*100:.1f}%", help=f"based on {len(sub):,} past incidents")
        st.progress(min(risk * 3, 1.0))
    else:
        st.info("No historical incidents for that combination.")

st.caption("Data: NYC Open Data 'Bus Breakdown and Delays' (CC0). "
           "Pipeline: PySpark 3.5 · MLlib · SQLite.")
