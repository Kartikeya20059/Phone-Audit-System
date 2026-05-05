# app_cached.py (modified: caching, performance improvements, minor fixes)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
import pickle
import plotly.express as px
import plotly.graph_objects as go

# helper imports
from helper import adb_devices, fetch_call_log_raw, parse_content_query, fetch_device_props, fetch_battery, fetch_meminfo, fetch_uptime

# preprocessor imports
from preprocessor import (
    normalize_dataframe,
    most_frequent,
    calls_by_hour,
    calls_per_day,
    calls_heatmap,
    caller_features,
    cluster_callers,
    heuristic_suspicious,
    isolation_anomalies,
    compute_advanced_anomalies,
    forecast_calls_per_day,
    generate_nl_most_frequent,
    generate_nl_hourly,
    generate_nl_forecast,
    generate_nl_heuristic,
    generate_nl_ai,
    generate_nl_clustering,
    normalize_device_props,
    parse_battery,
    embed_features,
    rolling_metrics,
    correlation_matrix,
    caller_time_series,
    predict_calls_for_number,
    train_suspicion_predictor,
    predict_suspicion,
    top_callers_forecast,
    compute_dendrogram,
)

# PDF utilities
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table as RLTable, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# config
APP_VERSION = "v0.9.5-pro-cached"
AUTHOR = "Aditya Mohan Srivastava"
st.set_page_config(page_title="Call Log Analyzer — Pro", layout="wide")
st.title("Call Log Analyzer — Pro (Advanced Dashboard & Audit) — Cached")
st.caption(f"Version {APP_VERSION} — {AUTHOR}")
st.markdown("---")

# -------------------- Utility: serialization for caching --------------------
def _df_serialize(df: pd.DataFrame) -> str:
    """Serialize dataframe to a compact CSV string for stable hashing in cache keys."""
    if df is None or df.empty:
        return ""
    # Keep only columns we rely upon for heavy ops to reduce serial size
    use_cols = [c for c in ["number", "duration_s", "datetime", "hour", "date", "call_type"] if c in df.columns]
    return df[use_cols].to_csv(index=False)

# -------------------- Cached wrappers --------------------
@st.cache_data(ttl=3600)
def cached_embed(df_csv: str, method: str, n_components: int, random_state: int):
    if not df_csv:
        return pd.DataFrame(columns=["emb_x", "emb_y"])
    df_local = pd.read_csv(io.StringIO(df_csv))
    # rehydrate datetime column if present
    if "datetime" in df_local.columns:
        try:
            df_local["datetime"] = pd.to_datetime(df_local["datetime"], errors="coerce")
        except Exception:
            pass
    return embed_features(df_local, method=method, n_components=n_components, random_state=random_state)

@st.cache_data(ttl=1800)
def cached_compute_anomalies(df_csv: str, contamination: float, use_pyod: bool, use_autoencoder: bool, pca_components: int, random_state: int):
    if not df_csv:
        return pd.DataFrame()
    df_local = pd.read_csv(io.StringIO(df_csv))
    if "datetime" in df_local.columns:
        df_local["datetime"] = pd.to_datetime(df_local["datetime"], errors="coerce")
    # Wrap the call to preprocessor's compute_advanced_anomalies
    try:
        return compute_advanced_anomalies(df_local, contamination=contamination, use_pyod=use_pyod, use_autoencoder=use_autoencoder, pca_components=pca_components, random_state=random_state)
    except Exception:
        # fallback
        return isolation_anomalies(df_local, contamination=contamination)

@st.cache_data(ttl=1800)
def cached_top_callers_forecast(df_csv: str, top_n: int, days_ahead: int):
    if not df_csv:
        return pd.DataFrame()
    df_local = pd.read_csv(io.StringIO(df_csv))
    if "datetime" in df_local.columns:
        df_local["datetime"] = pd.to_datetime(df_local["datetime"], errors="coerce")
    return top_callers_forecast(df_local, top_n=top_n, days_ahead=days_ahead)

@st.cache_data(ttl=1800)
def cached_predict_calls_for_number(df_csv: str, number: str, days_ahead: int):
    if not df_csv:
        return pd.DataFrame()
    df_local = pd.read_csv(io.StringIO(df_csv))
    if "datetime" in df_local.columns:
        df_local["datetime"] = pd.to_datetime(df_local["datetime"], errors="coerce")
    ts = caller_time_series(df_local, number)
    if ts.empty:
        return pd.DataFrame()
    return predict_calls_for_number(ts, days_ahead=days_ahead)

@st.cache_data(ttl=3600)
def cached_train_suspicion_predictor(df_csv: str, random_state: int):
    """Train and cache the suspicion predictor. Returns the pickled model dict to keep cache serializable."""
    if not df_csv:
        return None
    df_local = pd.read_csv(io.StringIO(df_csv))
    if "datetime" in df_local.columns:
        df_local["datetime"] = pd.to_datetime(df_local["datetime"], errors="coerce")
    md = train_suspicion_predictor(df_local, random_state=random_state)
    # Keep cacheable by pickling the lightweight dict; scikit objects may still be included.
    return md

# -------------------- Sidebar controls --------------------
st.sidebar.header("Controls & Device Info")
if st.sidebar.button("🔍 Check Device"):
    try:
        out = adb_devices()
        st.sidebar.code(out)
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        detected = any("\tdevice" in l or l.endswith("device") for l in lines[1:]) if len(lines) > 1 else False
        if detected:
            st.sidebar.success("Device detected and authorized.")
        else:
            st.sidebar.warning("No authorized device found. Enable USB debugging and accept on phone.")
    except Exception as e:
        st.sidebar.error(str(e))

if st.sidebar.button("⚙️ Fetch device info"):
    try:
        st.session_state["props_raw"] = fetch_device_props()
        st.session_state["battery_raw"] = fetch_battery()
        st.session_state["meminfo_raw"] = fetch_meminfo()
        st.session_state["uptime_raw"] = fetch_uptime()
        st.sidebar.success("Device info fetched.")
    except Exception as e:
        st.sidebar.error(str(e))

st.sidebar.markdown("---")
st.sidebar.header("Extraction & Analyze")
if st.sidebar.button("⏺️ Extract call logs (full)"):
    try:
        raw = fetch_call_log_raw()
        st.session_state["raw_output"] = raw
        st.sidebar.success("Raw output fetched and cached.")
    except Exception as e:
        st.sidebar.error(f"Extraction failed: {e}")

if st.sidebar.button("🔁 Parse & Analyze"):
    try:
        raw = st.session_state.get("raw_output")
        if raw is None:
            raw = fetch_call_log_raw()
            st.session_state["raw_output"] = raw
        rows = parse_content_query(raw)
        df = normalize_dataframe(rows)
        st.session_state["calls_df"] = df
        st.sidebar.success(f"Parsed {len(df)} records.")
    except Exception as e:
        st.sidebar.error(str(e))

st.sidebar.markdown("---")
st.sidebar.header("Advanced AI options")
cluster_k = st.sidebar.slider("Caller clusters (K)", 1, 12, 4)
forecast_days = st.sidebar.slider("Forecast horizon (days)", 3, 30, 7)
contamination = st.sidebar.slider("Anomaly contamination", 0.005, 0.2, 0.05, step=0.005)
embedding_method = st.sidebar.selectbox("Embedding", options=["pca", "umap"], index=0)
use_autoencoder = st.sidebar.checkbox("Enable Autoencoder detector (TensorFlow)", value=True)
use_pyod = st.sidebar.checkbox("Enable PyOD detectors (HBOS, PyOD IForest)", value=True)
ai_random_state = st.sidebar.number_input("Random seed", value=42, min_value=0, max_value=99999, step=1)
pca_components = st.sidebar.slider("Embedding PCA components (when used)", 0, 4, 2)

# optional-libs info
def _check_optional_libs():
    missing = []
    try:
        import tensorflow  # noqa
    except Exception:
        missing.append("tensorflow")
    try:
        import pyod  # noqa
    except Exception:
        missing.append("pyod")
    try:
        import umap  # noqa
    except Exception:
        missing.append("umap-learn")
    try:
        import scipy  # noqa
    except Exception:
        missing.append("scipy")
    return missing

missing = _check_optional_libs()
if missing:
    st.sidebar.info("Optional libs missing: " + ", ".join(missing) + ". Some advanced features disabled.")

st.sidebar.markdown("---")
st.sidebar.header("Quick exports")
if "calls_df" in st.session_state:
    st.sidebar.download_button("⬇️ Export current filtered CSV", data=st.session_state["calls_df"].to_csv(index=False).encode("utf-8"), file_name="call_logs_filtered.csv", mime="text/csv")

# require parsed data
if "calls_df" not in st.session_state:
    st.info("No parsed logs in session. Use sidebar: Extract -> Parse & Analyze.")
    st.stop()

# work on a copy
df_all = st.session_state["calls_df"].copy()

# Date & type filters
st.sidebar.markdown("### Date & Type filters")
min_date = df_all["date"].min() if not df_all["date"].isna().all() else None
max_date = df_all["date"].max() if not df_all["date"].isna().all() else None
date_range_value = (min_date, max_date) if min_date is not None and max_date is not None else None
date_range = st.sidebar.date_input("Date range", value=date_range_value)
call_types = st.sidebar.multiselect("Call types", options=df_all["call_type"].unique().tolist(), default=df_all["call_type"].unique().tolist())

# apply filters
df = df_all.copy()
if isinstance(date_range, tuple) and len(date_range) == 2 and all(date_range):
    start, end = date_range
    df = df[(df["date"] >= start) & (df["date"] <= end)]
elif isinstance(date_range, (list, tuple)) and len(date_range) == 1:
    start = date_range[0]
    df = df[df["date"] == start]
elif hasattr(date_range, "day"):
    df = df[df["date"] == date_range]
if call_types:
    df = df[df["call_type"].isin(call_types)]

# Tabs
tab_overview, tab_visuals, tab_anom, tab_models, tab_audit = st.tabs(["Overview", "Visuals & Clustering", "Anomalies & Explainability", "Models & Predictors", "Audit Report"])

# ------------- Overview -------------
with tab_overview:
    st.subheader("Executive summary")
    st.markdown("This dashboard provides interactive exploration, anomaly detection (ensemble), clustering, per-caller forecasts and a downloadable audit report in an industry-friendly format. Use the side controls to tune detectors/embeddings/forecasts.")
    st.markdown("### Key metrics")
    c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])
    with c1:
        st.metric("Total calls (filtered)", len(df))
    with c2:
        st.metric("Unique numbers", df["number"].nunique())
    with c3:
        st.metric("Total duration (s)", int(df["duration_s"].sum() if not df.empty else 0))
    with c4:
        st.metric("Calls today", int(df[df["date"] == datetime.now().date()].shape[0]))
    st.markdown("### Top callers")
    st.dataframe(most_frequent(df, top_n=50), height=300)

# ------------- Visuals & Clustering -------------
with tab_visuals:
    st.subheader("Embeddings, clusters & correlation")

    # Embedding
    st.markdown("#### 2D embedding (interactive)")
    df_csv = _df_serialize(df)
    emb = cached_embed(df_csv, method=embedding_method, n_components=2, random_state=int(ai_random_state))
    if emb.empty:
        st.info("Not enough data for embedding.")
    else:
        # safe concat by position
        df2 = df.reset_index(drop=True)
        emb2 = emb.reset_index(drop=True)
        if len(df2) == len(emb2):
            plot_df = pd.concat([df2, emb2], axis=1)
            fig = px.scatter(plot_df, x="emb_x", y="emb_y", color="call_type", hover_data=["number", "duration", "datetime"], title=f"{embedding_method.upper()} embedding")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Embedding length mismatch — skipping embedding plot.")

        st.markdown("Select range (lasso/box) on the plot to zoom into a subset, then use 'Analyze selection' to inspect.")
        st.info("Note: Streamlit's direct Plotly selection support is limited. Consider using the streamlit-plotly-events package for interactive selections.")

    # Clustering on caller aggregates
    st.markdown("#### Caller clustering (aggregate features)")
    agg = caller_features(df)
    if agg.empty:
        st.info("Not enough data to aggregate callers.")
    else:
        clustered = cluster_callers(agg, k=cluster_k)
        st.dataframe(clustered.sort_values("total_calls", ascending=False).head(200), height=300)
        figc = px.histogram(clustered, x="cluster", title="Callers per cluster")
        st.plotly_chart(figc, use_container_width=True)

        st.markdown("Parallel coordinates (top 50 contacts)")
        top50 = clustered.sort_values("total_calls", ascending=False).head(50)
        if not top50.empty:
            # guard: parallel_coordinates wants numeric columns present
            dims = [c for c in ["total_calls", "avg_duration", "pct_missed"] if c in top50.columns]
            if dims:
                figp = px.parallel_coordinates(top50, dimensions=dims, color="cluster")
                st.plotly_chart(figp, use_container_width=True)

    # dendrogram (if available)
    st.markdown("#### Caller dendrogram (hierarchical)")
    try:
        dend = compute_dendrogram(df)
        if dend is None:
            st.info("Not enough data or scipy not available for dendrogram.")
        else:
            Z = dend["linkage"]
            labels = dend["labels"]
            import scipy.cluster.hierarchy as sch
            dn = sch.dendrogram(Z, labels=labels, no_plot=True)
            icoord = dn["icoord"]
            dcoord = dn["dcoord"]
            fig_tree = go.Figure()
            for xs, ys in zip(icoord, dcoord):
                fig_tree.add_trace(go.Scatter(x=xs, y=ys, mode='lines', line=dict(color='gray'), hoverinfo='none', showlegend=False))
            fig_tree.update_layout(title="Dendrogram (hierarchical clustering)", xaxis=dict(showticklabels=False), yaxis_title="Distance", height=500)
            st.plotly_chart(fig_tree, use_container_width=True)
    except Exception as e:
        st.info(f"Dendrogram unavailable: {e}")

    # correlation matrix
    st.markdown("#### Correlation matrix (daily features)")
    corr = correlation_matrix(df)
    if corr.empty:
        st.info("Not enough daily data to compute correlation matrix.")
    else:
        fig_corr = px.imshow(corr, text_auto=True, title="Correlation matrix")
        st.plotly_chart(fig_corr, use_container_width=True)

# ------------- Anomalies & Explainability -------------
with tab_anom:
    st.subheader("Anomalies (advanced ensemble) & explainability")
    with st.spinner("Running advanced anomaly ensemble..."):
        try:
            df_csv_local = _df_serialize(df)
            anom_df = cached_compute_anomalies(df_csv_local, contamination=float(contamination), use_pyod=bool(use_pyod), use_autoencoder=bool(use_autoencoder), pca_components=int(pca_components), random_state=int(ai_random_state))
        except Exception as e:
            st.error(f"Advanced anomaly detector failed: {e}")
            anom_df = isolation_anomalies(df, contamination=float(contamination))

    if anom_df is None or anom_df.empty:
        st.info("No anomalies flagged or insufficient data for advanced detectors.")
    else:
        st.markdown("### Top anomalies (table)")
        st.dataframe(anom_df.head(200), height=300)
        st.markdown("### Anomaly score distribution")
        fig_hist = px.histogram(anom_df, x="anomaly_score", nbins=30, title="Anomaly scores")
        st.plotly_chart(fig_hist, use_container_width=True)

        try:
            surrogate_text = anom_df["explain"].iloc[0] if "explain" in anom_df.columns else ""
            st.markdown("### Surrogate explanation (summary)")
            st.write(surrogate_text)
        except Exception:
            pass

    # combined risk scoring shown as interactive table
    st.markdown("### Combined risk scoring (calls)")
    merged = df.copy()
    # build anomaly_map using robust ISO ms strings
    anomaly_map = {}
    if anom_df is not None and not anom_df.empty:
        tmp = anom_df.copy()
        if "datetime" in tmp.columns:
            tmp["dt_key"] = tmp["datetime"].astype('datetime64[ms]').astype(str)
        else:
            tmp["dt_key"] = tmp.index.astype(str)
        if "anomaly_score" in tmp.columns:
            anomaly_map = tmp.set_index(["number", "dt_key"]).to_dict().get("anomaly_score", {})
    if "datetime" in merged.columns:
        merged["dt_key"] = merged["datetime"].astype('datetime64[ms]').astype(str)
    else:
        merged["dt_key"] = merged.index.astype(str)
    merged["anomaly_score"] = merged.apply(lambda r: float(anomaly_map.get((r["number"], r["dt_key"]), 0)), axis=1)

    agg = caller_features(df)
    cached_clustered = cluster_callers(agg, k=cluster_k) if not agg.empty else pd.DataFrame()
    clustered_map = {}
    if not cached_clustered.empty:
        clustered_map = cached_clustered.set_index("number").to_dict().get("cluster_dist", {})
    merged["cluster_dist"] = merged["number"].map(clustered_map).fillna(0)
    merged["heuristic"] = merged.apply(lambda r: (1.0 if r.get("duration_s", 0) >= 3600 else 0) + (0.5 if (pd.notna(r.get("hour")) and isinstance(r.get("hour"), (int, float)) and (0 <= int(r.get("hour")) <= 5) and r.get("duration_s", 0) > 0) else 0), axis=1)
    merged["risk_score"] = merged["anomaly_score"] * 0.6 + merged["cluster_dist"] * 0.2 + merged["heuristic"] * 0.2

    top_risk = merged.sort_values("risk_score", ascending=False).head(200)
    if not top_risk.empty:
        top_risk_display = top_risk.copy()
        top_risk_display["datetime"] = top_risk_display["datetime"].astype(str)
        st.dataframe(top_risk_display[["number", "datetime", "call_type", "duration", "anomaly_score", "cluster_dist", "heuristic", "risk_score"]].head(200), height=300)

# ------------- Models & Predictors -------------
with tab_models:
    st.subheader("Models, Predictors & Forecasts")

    st.markdown("### Train suspicion predictor (weak supervision from heuristics)")
    if st.button("Train suspicion classifier"):
        with st.spinner("Training..."):
            df_csv_local = _df_serialize(df)
            mdict = cached_train_suspicion_predictor(df_csv_local, random_state=int(ai_random_state))
            if mdict is None:
                st.error("Could not train — no positive heuristic labels or insufficient data.")
            else:
                st.session_state["suspicion_model"] = mdict
                st.success("Trained suspicion model.")
                if mdict.get("auc") is not None:
                    st.write("AUC:", mdict.get("auc"))

    if "suspicion_model" in st.session_state and st.session_state["suspicion_model"] is not None:
        md = st.session_state["suspicion_model"]
        st.markdown("Feature coefficients (logistic):")
        fi = pd.DataFrame({"feature": md["feature_names"], "coef": md["coef"]}).sort_values("coef", ascending=False)
        st.bar_chart(fi.set_index("feature")["coef"])
        pred_df = predict_suspicion(md, df)
        pred_df["datetime"] = pred_df["datetime"].astype(str)
        st.markdown("Top predicted suspicious calls")
        st.dataframe(pred_df.sort_values("suspicion_prob", ascending=False).head(200), height=300)
        st.download_button("Download suspicion model (pickle)", data=pickle.dumps(md), file_name="suspicion_model.pkl", mime="application/octet-stream")

    # Top-callers forecast
    st.markdown("### Top-callers forecast")
    top_n = st.number_input("Top N", value=10, min_value=1, max_value=100, step=1)
    days_a = st.number_input("Horizon (days)", value=7, min_value=1, max_value=30, step=1)
    if st.button("Compute top callers forecast"):
        with st.spinner("Computing..."):
            df_csv_local = _df_serialize(df)
            tcf = cached_top_callers_forecast(df_csv_local, top_n=top_n, days_ahead=int(days_a))
            st.dataframe(tcf, height=400)
            st.session_state["last_top_fc"] = tcf

    if "last_top_fc" in st.session_state:
        st.download_button("Download top-callers forecast CSV", data=st.session_state["last_top_fc"].to_csv(index=False).encode("utf-8"), file_name="top_callers_forecast.csv", mime="text/csv")

    # per-number drill-down forecast plot
    st.markdown("### Drill-down: per-number trend & forecast")
    numbers = most_frequent(df, top_n=200)["number"].tolist()
    sel = st.selectbox("Choose number to inspect", options=numbers)
    if sel:
        df_csv_local = _df_serialize(df)
        fc = cached_predict_calls_for_number(df_csv_local, number=sel, days_ahead=int(days_a))
        if fc is None or fc.empty:
            st.info("No time-series for this number.")
        else:
            fig_fc = px.line(fc, x="date", y="count", title=f"Calls for {sel} (historical + forecast)")
            st.plotly_chart(fig_fc, use_container_width=True)
            st.dataframe(fc.tail(int(days_a) + 10))

# ------------- Audit Report -------------
with tab_audit:
    st.subheader("Industry-style Audit Report (text + tables)")

    st.markdown("Choose report components and generate an industry-standard PDF (Executive Summary, Methodology, Findings, Recommendations, Annex: tables).")
    include_charts = st.checkbox("Include charts as images in PDF (optional)", value=True)
    include_tables = st.checkbox("Include data tables (Top callers, Heuristic alerts, Anomalies, Top risk)", value=True)

    if st.button("Generate audit report (PDF)"):
        with st.spinner("Generating PDF..."):
            device_props = normalize_device_props(st.session_state.get("props_raw", "")) if st.session_state.get("props_raw") else {}
            battery_info = parse_battery(st.session_state.get("battery_raw", "")) if st.session_state.get("battery_raw") else {}
            top_callers_tbl = most_frequent(df, top_n=50)
            sus_tbl = heuristic_suspicious(df)
            try:
                df_csv_local = _df_serialize(df)
                advanced_anoms = cached_compute_anomalies(df_csv_local, contamination=float(contamination), use_pyod=bool(use_pyod), use_autoencoder=bool(use_autoencoder), pca_components=int(pca_components), random_state=int(ai_random_state))
            except Exception:
                advanced_anoms = isolation_anomalies(df, contamination=float(contamination))

            # top risk (reuse earlier)
            merged_local = df.copy()
            anomaly_map_local = {}
            if advanced_anoms is not None and not advanced_anoms.empty:
                tmp = advanced_anoms.copy()
                if "datetime" in tmp.columns:
                    tmp["dt_key"] = tmp["datetime"].astype('datetime64[ms]').astype(str)
                else:
                    tmp["dt_key"] = tmp.index.astype(str)
                if "anomaly_score" in tmp.columns:
                    anomaly_map_local = tmp.set_index(["number", "dt_key"]).to_dict().get("anomaly_score", {})
            if "datetime" in merged_local.columns:
                merged_local["dt_key"] = merged_local["datetime"].astype('datetime64[ms]').astype(str)
            else:
                merged_local["dt_key"] = merged_local.index.astype(str)
            merged_local["anomaly_score"] = merged_local.apply(lambda r: float(anomaly_map_local.get((r["number"], r["dt_key"]), 0)), axis=1)

            agg_local = caller_features(df)
            clustered_local = cluster_callers(agg_local, k=cluster_k) if not agg_local.empty else pd.DataFrame()
            clustered_map_local = {}
            if not clustered_local.empty:
                clustered_map_local = clustered_local.set_index("number").to_dict().get("cluster_dist", {})
            merged_local["cluster_dist"] = merged_local["number"].map(clustered_map_local).fillna(0)
            merged_local["heuristic"] = merged_local.apply(lambda r: (1.0 if r.get("duration_s", 0) >= 3600 else 0) + (0.5 if (pd.notna(r.get("hour")) and isinstance(r.get("hour"), (int, float)) and (0 <= int(r.get("hour")) <= 5) and r.get("duration_s", 0) > 0) else 0), axis=1)
            merged_local["risk_score"] = merged_local["anomaly_score"] * 0.6 + merged_local["cluster_dist"] * 0.2 + merged_local["heuristic"] * 0.2
            top_risk_tbl = merged_local.sort_values("risk_score", ascending=False).head(200)

            # build PDF
            def build_pdf_bytes():
                buf = io.BytesIO()
                doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
                styles = getSampleStyleSheet()
                elems = []
                elems.append(Paragraph("Call Log Analyzer — Audit Report", styles["Title"]))
                elems.append(Spacer(1, 6))
                elems.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
                elems.append(Paragraph(f"Report version: {APP_VERSION} — Author: {AUTHOR}", styles["Normal"]))
                elems.append(Spacer(1, 12))
                elems.append(Paragraph("Executive Summary", styles["Heading2"]))
                es = f"Dataset contains {len(df)} calls (filtered). Top findings: {len(sus_tbl)} heuristic alerts and {len(advanced_anoms) if advanced_anoms is not None else 0} AI-detected anomalies. See Findings & Recommendations below."
                elems.append(Paragraph(es, styles["Normal"]))
                elems.append(Spacer(1, 12))
                elems.append(Paragraph("Methodology", styles["Heading2"]))
                meth = "We extract call logs using ADB and normalize into per-call records. Analysis includes rule-based heuristics, an ensemble of anomaly detectors (IsolationForest, LOF, HBOS / PyOD when available, and an Autoencoder where TensorFlow is installed), clustering of callers, and per-caller forecasting (ARIMA fallback to linear). Explainability is provided by a surrogate decision tree and feature-level summaries."
                elems.append(Paragraph(meth, styles["Normal"]))
                elems.append(Spacer(1, 12))
                elems.append(Paragraph("Findings", styles["Heading2"]))
                findings = f"- Total calls: {len(df)}  \n- Heuristic alerts: {len(sus_tbl)}  \n- AI anomalies: {len(advanced_anoms) if advanced_anoms is not None else 0}  \n- Top risk contacts (top 10): {', '.join(top_risk_tbl['number'].head(10).astype(str).tolist())}"
                elems.append(Paragraph(findings, styles["Normal"]))
                elems.append(Spacer(1, 12))
                elems.append(Paragraph("Recommendations", styles["Heading2"]))
                recs = "1. Investigate top risk contacts listed in Annex.  2. Where possible, collect ground-truth labels for anomalies to benchmark models.  3. Use the exported model artifacts for reproducible re-analysis.  4. Consider longer-term monitoring and threshold tuning for operational alerting."
                elems.append(Paragraph(recs, styles["Normal"]))
                elems.append(Spacer(1, 12))
                if include_tables:
                    elems.append(Paragraph("Annex: Top callers (table)", styles["Heading3"]))
                    rows = [["Number", "Count"]]
                    for _, r in top_callers_tbl.iterrows():
                        rows.append([str(r["number"]), str(int(r["count"]))])
                    t = RLTable(rows, colWidths=[300, 200])
                    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.2, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))
                    elems.append(t)
                    elems.append(Spacer(1, 12))

                    elems.append(Paragraph("Annex: Heuristic alerts (top 50)", styles["Heading3"]))
                    if sus_tbl.empty:
                        elems.append(Paragraph("No heuristic alerts.", styles["Normal"]))
                    else:
                        rows = [["Number", "Datetime", "Reason", "Score"]]
                        for _, r in sus_tbl.head(50).iterrows():
                            rows.append([str(r["number"]), str(r["datetime"]), str(r.get("reason", "")), f"{r.get('score',0):.2f}"])
                        t = RLTable(rows, colWidths=[120, 120, 180, 80])
                        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.2, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("FONTSIZE", (0,0), (-1,-1), 8)]))
                        elems.append(t)
                        elems.append(Spacer(1, 12))

                    elems.append(Paragraph("Annex: AI anomalies (top 50)", styles["Heading3"]))
                    if advanced_anoms is None or advanced_anoms.empty:
                        elems.append(Paragraph("No AI anomalies.", styles["Normal"]))
                    else:
                        rows = [["Number", "Datetime", "Duration_s", "Score", "Explain"]]
                        for _, r in advanced_anoms.head(50).iterrows():
                            rows.append([str(r["number"]), str(r["datetime"]), str(int(r.get("duration_s",0))), f"{float(r.get('anomaly_score',0)):.3f}", str(r.get("explain",""))])
                        t = RLTable(rows, colWidths=[100, 110, 60, 60, 200])
                        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.2, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("FONTSIZE", (0,0), (-1,-1), 8)]))
                        elems.append(t)
                        elems.append(Spacer(1, 12))

                    elems.append(Paragraph("Annex: Top risk contacts (top 50)", styles["Heading3"]))
                    rows = [["Number", "Datetime", "Type", "Duration", "Anomaly", "ClusterDist", "Heuristic", "Risk"]]
                    for _, r in top_risk_tbl.head(50).iterrows():
                        rows.append([str(r["number"]), str(r["datetime"]), str(r.get("call_type","")), str(r.get("duration","")), f"{float(r.get('anomaly_score',0)):.3f}", f"{float(r.get('cluster_dist',0)):.3f}", f"{float(r.get('heuristic',0)):.2f}", f"{float(r.get('risk_score',0)):.3f}"])
                    t = RLTable(rows, colWidths=[80, 90, 50, 50, 50, 60, 50, 50])
                    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.2, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("FONTSIZE", (0,0), (-1,-1), 8)]))
                    elems.append(t)
                    elems.append(Spacer(1, 12))

                elems.append(Paragraph("End of report", styles["Italic"]))
                doc.build(elems)
                buf.seek(0)
                return buf.read()

            try:
                pdf_bytes = build_pdf_bytes()
                st.session_state["last_audit_pdf"] = pdf_bytes
                st.success("Audit PDF generated.")
            except Exception as e:
                st.error(f"Failed to build PDF: {e}")

    if "last_audit_pdf" in st.session_state:
        st.download_button("⬇️ Download Audit PDF", data=st.session_state["last_audit_pdf"], file_name=f"calllog_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")

# footer
st.markdown("---")
st.caption("Advanced dashboard. For research/forensic/audit use: only run on devices you own or have permission to inspect.")
