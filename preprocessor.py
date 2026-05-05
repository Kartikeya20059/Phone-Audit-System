# preprocessor.py
"""
Robust preprocessor for Call Log Analyzer.

- Lazy-imports optional packages inside functions to avoid import-time failures.
- Exposes functions required by app.py, including compute_advanced_anomalies and NL helpers.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")

# mapping typical Android type codes
TYPE_MAP = {
    1: "INCOMING",
    2: "OUTGOING",
    3: "MISSED",
    5: "REJECTED",
    0: "UNKNOWN",
    "1": "INCOMING",
    "2": "OUTGOING",
    "3": "MISSED",
    "5": "REJECTED",
    "0": "UNKNOWN",
}

# ---------- basic helpers ----------
def ms_to_dt(ms):
    try:
        if ms is None or ms == "" or pd.isna(ms):
            return pd.NaT
        s = int(ms) / 1000.0
        return datetime.fromtimestamp(s)
    except Exception:
        return pd.NaT

def sec_to_hms(s):
    try:
        s = int(s)
    except Exception:
        return "00:00:00"
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"

# ---------- normalization ----------
def normalize_dataframe(rows):
    if not rows:
        return pd.DataFrame(columns=["number", "call_type", "duration_s", "duration", "datetime", "hour", "date", "weekday"])
    df = pd.DataFrame(rows)
    for alt in ("number", "normalized_number", "formatted_number", "matched_number", "data1"):
        if alt in df.columns:
            df = df.rename(columns={alt: "number"})
            break
    if "number" not in df.columns:
        df["number"] = ""
    if "duration" in df.columns:
        df["duration_s"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0).astype(int)
    else:
        df["duration_s"] = 0
    if "date" in df.columns:
        df["timestamp_ms"] = pd.to_numeric(df["date"], errors="coerce")
    elif "last_modified" in df.columns:
        df["timestamp_ms"] = pd.to_numeric(df["last_modified"], errors="coerce")
    else:
        df["timestamp_ms"] = pd.NA
    df["datetime"] = df["timestamp_ms"].apply(ms_to_dt)
    df["duration"] = df["duration_s"].apply(sec_to_hms)
    if "type" in df.columns:
        df["call_type"] = df["type"].astype(str).map(TYPE_MAP).fillna("UNKNOWN")
    elif "phone_call_type" in df.columns:
        df["call_type"] = df["phone_call_type"].astype(str).map(TYPE_MAP).fillna("UNKNOWN")
    else:
        df["call_type"] = "UNKNOWN"
    df["hour"] = df["datetime"].apply(lambda d: d.hour if not pd.isna(d) else pd.NA)
    df["date"] = df["datetime"].apply(lambda d: d.date() if not pd.isna(d) else pd.NaT)
    df["weekday"] = df["datetime"].apply(lambda d: d.weekday() if not pd.isna(d) else pd.NA)
    df["number"] = df["number"].astype(str).str.strip()
    df["number_display"] = df["number"].str.lstrip("+")
    out = df[["number_display", "call_type", "duration_s", "duration", "datetime", "hour", "date", "weekday"]].copy()
    out = out.rename(columns={"number_display": "number"})
    out = out.sort_values("datetime", ascending=False).reset_index(drop=True)
    return out

# ---------- simple analytics ----------
def most_frequent(df, top_n=10):
    if df.empty:
        return pd.DataFrame(columns=["number", "count"])
    freq = df["number"].value_counts().reset_index()
    freq.columns = ["number", "count"]
    return freq.head(top_n)

def calls_by_hour(df):
    if df.empty:
        return pd.DataFrame()
    pivot = df.groupby(["hour", "call_type"]).size().unstack(fill_value=0)
    pivot = pivot.reset_index().sort_values("hour")
    return pivot

def calls_per_day(df):
    if df.empty:
        return pd.DataFrame()
    s = df.groupby("date").size().reset_index(name="count").sort_values("date")
    return s

def calls_heatmap(df):
    if df.empty:
        return pd.DataFrame(0, index=range(24), columns=range(7))
    tmp = df.copy().dropna(subset=["hour", "weekday"])
    if tmp.empty:
        return pd.DataFrame(0, index=range(24), columns=range(7))
    tmp["hour"] = tmp["hour"].astype(int)
    tmp["weekday"] = tmp["weekday"].astype(int)
    pivot = tmp.groupby(["hour", "weekday"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(index=range(24), columns=range(7), fill_value=0)
    return pivot

# ---------- caller aggregates & clustering ----------
def caller_features(df):
    if df.empty:
        return pd.DataFrame()
    agg = df.groupby("number").agg(
        total_calls=("number", "size"),
        avg_duration=("duration_s", "mean"),
        total_duration=("duration_s", "sum"),
        missed_calls=("call_type", lambda s: (s == "MISSED").sum()),
    ).reset_index()
    agg["pct_missed"] = agg["missed_calls"] / agg["total_calls"]
    return agg

def cluster_callers(agg_df, k=3):
    if agg_df.empty:
        return agg_df
    X = agg_df[["total_calls", "avg_duration", "pct_missed"]].fillna(0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    k = max(1, min(k, Xs.shape[0]))
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(Xs)
    agg = agg_df.copy()
    agg["cluster"] = labels
    centers = kmeans.cluster_centers_
    if centers.shape[0] == 1:
        dists = np.linalg.norm(Xs - centers[0], axis=1)
    else:
        dists = np.linalg.norm(Xs - centers[labels], axis=1)
    agg["cluster_dist"] = dists
    return agg

# ---------- heuristic suspicious ----------
def heuristic_suspicious(
    df,
    night_start=0,
    night_end=5,
    long_call_threshold_s=3600,
    frequent_window_minutes=30,
    frequent_calls_threshold=5,
    short_call_seconds=10,
):
    suspicious = []
    if df.empty:
        return pd.DataFrame(columns=["number", "datetime", "call_type", "duration_s", "reason", "score"])
    for idx, row in df.iterrows():
        score = 0.0
        reasons = []
        try:
            dur = int(row.get("duration_s", 0))
        except Exception:
            dur = 0
        if dur >= long_call_threshold_s:
            reasons.append("Long call")
            score += 0.6
        if row["hour"] is not pd.NA and isinstance(row["hour"], (int, float)):
            h = int(row["hour"])
            if (night_start <= h <= night_end) and dur > 0:
                reasons.append("Odd-night call")
                score += 0.4
        if reasons:
            suspicious.append(
                {
                    "number": row["number"],
                    "datetime": row["datetime"],
                    "call_type": row["call_type"],
                    "duration_s": dur,
                    "reason": "; ".join(reasons),
                    "score": min(score, 1.0),
                }
            )
    df_ts = df.dropna(subset=["datetime"]).sort_values("datetime")
    if not df_ts.empty:
        for number, group in df_ts.groupby("number"):
            times = list(group["datetime"])
            durations = list(group["duration_s"])
            i = 0
            n = len(times)
            while i < n:
                j = i
                window_calls = 1
                window_durations = durations[i]
                while j + 1 < n and (times[j + 1] - times[i]).total_seconds() <= frequent_window_minutes * 60:
                    j += 1
                    window_calls += 1
                    window_durations += durations[j]
                if window_calls >= frequent_calls_threshold and (window_durations / window_calls) <= short_call_seconds:
                    for k in range(i, j + 1):
                        suspicious.append(
                            {
                                "number": number,
                                "datetime": times[k],
                                "call_type": df_ts.iloc[k]["call_type"],
                                "duration_s": df_ts.iloc[k]["duration_s"],
                                "reason": f"Frequent short calls ({window_calls} in {frequent_window_minutes}m)",
                                "score": 0.7,
                            }
                        )
                i += 1
    sus_df = pd.DataFrame(suspicious)
    if sus_df.empty:
        return sus_df
    sus_df = sus_df.drop_duplicates(subset=["number", "datetime"]).sort_values("score", ascending=False).reset_index(drop=True)
    return sus_df

# ---------- classic anomaly detectors ----------
def isolation_anomalies(df, contamination=0.05):
    if df.empty:
        return pd.DataFrame()
    tmp = df.copy()
    tmp["dayofweek"] = tmp["datetime"].apply(lambda d: d.weekday() if pd.notna(d) else -1)
    freq = tmp["number"].value_counts().to_dict()
    tmp["count_by_number"] = tmp["number"].map(freq).fillna(0)
    X = tmp[["duration_s", "hour", "dayofweek", "count_by_number"]].fillna(-1).astype(float).values
    if X.shape[0] < 5:
        return pd.DataFrame()
    iso = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    preds = iso.fit_predict(X)
    raw_scores = iso.decision_function(X)
    smax = float(np.max(raw_scores))
    smin = float(np.min(raw_scores))
    denom = smax - smin if (smax - smin) != 0 else 1.0
    anomaly_score = (smax - raw_scores) / denom
    tmp["anomaly_score"] = anomaly_score
    tmp["anomaly_flag"] = preds == -1
    res = tmp[tmp["anomaly_flag"]].copy()
    if res.empty:
        return res
    res = res[["number", "datetime", "call_type", "duration_s", "anomaly_score"]].sort_values("anomaly_score", ascending=False)
    return res

# ---------- compute_advanced_anomalies (definitive) ----------
def compute_advanced_anomalies(
    df,
    contamination=0.05,
    use_pyod=True,
    use_autoencoder=True,
    pca_components=2,
    random_state=42,
):
    """
    Combined anomaly detection wrapper. Uses sklearn detectors by default and attempts optional detectors
    (pyod HBOS/IF, tensorflow autoencoder) via lazy imports. Returns anomalies with anomaly_score and explain.
    """
    if df.empty:
        return pd.DataFrame()
    tmp = df.copy()
    tmp["dayofweek"] = tmp["datetime"].apply(lambda d: d.weekday() if pd.notna(d) else -1)
    freq = tmp["number"].value_counts().to_dict()
    tmp["count_by_number"] = tmp["number"].map(freq).fillna(0)
    X_df = tmp[["duration_s", "hour", "dayofweek", "count_by_number"]].fillna(-1)
    X = X_df.astype(float).values

    # preprocess
    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)
    if pca_components and pca_components > 0:
        pca = PCA(n_components=min(pca_components, Xs.shape[1]), random_state=random_state)
        Xp = pca.fit_transform(Xs)
    else:
        Xp = Xs

    scores = []
    detectors = {}

    # IsolationForest
    try:
        iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=random_state)
        iso.fit(Xp)
        iso_raw = iso.decision_function(Xp)  # higher=less anomalous
        iso_score = iso_raw.max() - iso_raw
        iso_score = (iso_score - iso_score.min()) / (iso_score.max() - iso_score.min() + 1e-12)
        scores.append(iso_score)
        detectors["isolation"] = iso_score
    except Exception:
        pass

    # LOF
    try:
        lof = LocalOutlierFactor(n_neighbors=min(20, max(5, len(Xp)-1)), contamination=contamination, novelty=False)
        lof.fit(Xp)
        lof_raw = -lof.negative_outlier_factor_
        lof_score = (lof_raw - lof_raw.min()) / (lof_raw.max() - lof_raw.min() + 1e-12)
        scores.append(lof_score)
        detectors["lof"] = lof_score
    except Exception:
        pass

    # Lazy PyOD HBOS
    if use_pyod:
        try:
            from pyod.models.hbos import HBOS
            h = HBOS(contamination=contamination)
            h.fit(Xp)
            h_score = h.decision_scores_
            h_score = (h_score - h_score.min()) / (h_score.max() - h_score.min() + 1e-12)
            scores.append(h_score)
            detectors["hbos"] = h_score
        except Exception:
            pass

    # Lazy PyOD IForest
    if use_pyod:
        try:
            from pyod.models.iforest import IForest as PYOD_IF
            pif = PYOD_IF(contamination=contamination)
            pif.fit(Xp)
            pif_score = pif.decision_scores_
            pif_score = (pif_score - pif_score.min()) / (pif_score.max() - pif_score.min() + 1e-12)
            scores.append(pif_score)
            detectors["pyod_if"] = pif_score
        except Exception:
            pass

    # Optional autoencoder (lazy TF import)
    if use_autoencoder:
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models, optimizers
            # small autoencoder
            input_dim = Xs.shape[1]
            latent = max(1, min(8, input_dim // 2))
            inputs = layers.Input(shape=(input_dim,))
            x = layers.Dense(max(32, input_dim * 2), activation="relu")(inputs)
            x = layers.Dense(max(16, input_dim), activation="relu")(x)
            bottleneck = layers.Dense(latent, activation="relu")(x)
            x = layers.Dense(max(16, input_dim), activation="relu")(bottleneck)
            x = layers.Dense(max(32, input_dim * 2), activation="relu")(x)
            outputs = layers.Dense(input_dim, activation="linear")(x)
            ae = models.Model(inputs, outputs)
            ae.compile(optimizer=optimizers.Adam(1e-3), loss="mse")
            ae.fit(Xs, Xs, epochs=8, batch_size=32, verbose=0)
            Xp_pred = ae.predict(Xs)
            mse = np.mean((Xs - Xp_pred) ** 2, axis=1)
            ae_score = (mse - mse.min()) / (mse.max() - mse.min() + 1e-12)
            scores.append(ae_score)
            detectors["autoencoder"] = ae_score
        except Exception:
            pass

    if not scores:
        return pd.DataFrame()

    # combine scores
    stacked = np.vstack(scores)
    combo = np.mean(stacked, axis=0)
    combo = (combo - combo.min()) / (combo.max() - combo.min() + 1e-12)
    tmp["anomaly_score"] = combo
    cutoff = np.quantile(combo, 1.0 - contamination) if contamination > 0 else combo.max() + 1
    tmp["anomaly_flag"] = tmp["anomaly_score"] >= cutoff

    # surrogate decision tree (explainability)
    try:
        clf = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=random_state)
        clf.fit(Xp, tmp["anomaly_flag"].astype(int).values)
        rules = export_text(clf, feature_names=list(X_df.columns))
        tmp["surrogate_rule_text"] = rules
        fi = clf.feature_importances_
        tmp["surrogate_feature_importance"] = [dict(zip(list(X_df.columns), fi)) for _ in range(len(tmp))]
    except Exception:
        tmp["surrogate_rule_text"] = ""
        tmp["surrogate_feature_importance"] = [{} for _ in range(len(tmp))]

    # attach detector-specific scores
    for name, arr in detectors.items():
        tmp[f"score_{name}"] = arr

    res = tmp[tmp["anomaly_flag"]].copy()
    if res.empty:
        return res

    def explain_row(i):
        dets = sorted(detectors.keys(), key=lambda k: float(detectors[k][i] if i < len(detectors[k]) else 0), reverse=True)[:3]
        fi_map = tmp.iloc[i].get("surrogate_feature_importance", {})
        top_feats = []
        for k, v in sorted(fi_map.items(), key=lambda kv: kv[1], reverse=True)[:2]:
            top_feats.append(f"{k}:{v:.2f}")
        parts = []
        if dets:
            parts.append("Detectors:" + ",".join(dets))
        if top_feats:
            parts.append("Features:" + ",".join(top_feats))
        return " | ".join(parts)

    res["explain"] = [explain_row(i) for i in res.index]
    return res[["number", "datetime", "call_type", "duration_s", "anomaly_score", "explain"]].sort_values("anomaly_score", ascending=False)

# ---------- forecasting ----------
def forecast_calls_per_day(series_df, days_ahead=7):
    import pandas as pd
    if series_df.empty:
        return pd.DataFrame()
    ts = series_df.copy()
    ts["date"] = pd.to_datetime(ts["date"])
    ts = ts.set_index("date").asfreq("D").fillna(0)
    y = ts["count"].astype(float)
    try:
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(y, order=(1, 1, 0))
        fit = model.fit()
        pred = fit.get_forecast(steps=days_ahead)
        mean = pred.predicted_mean
        ci = pred.conf_int(alpha=0.05)
        future_idx = pd.date_range(start=y.index[-1] + pd.Timedelta(days=1), periods=days_ahead, freq="D")
        fc = pd.DataFrame({"date": future_idx, "count": mean.values, "lower": ci.iloc[:, 0].values, "upper": ci.iloc[:, 1].values}).set_index("date")
        existing = pd.DataFrame({"date": y.index, "count": y.values}).set_index("date")
        out = pd.concat([existing, fc], axis=0).reset_index()
        out["date"] = out["date"].dt.date
        out["lower"] = out.get("lower", out["count"])
        out["upper"] = out.get("upper", out["count"])
        return out.reset_index(drop=True)
    except Exception:
        idx = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression()
        model.fit(idx, y.values)
        future_idx = np.arange(len(y), len(y) + days_ahead).reshape(-1, 1)
        preds = model.predict(future_idx)
        future_dates = pd.date_range(start=y.index[-1] + pd.Timedelta(days=1), periods=days_ahead, freq="D")
        out_existing = pd.DataFrame({"date": y.index.date, "count": y.values})
        out_future = pd.DataFrame({"date": future_dates.date, "count": preds, "lower": preds * 0.9, "upper": preds * 1.1})
        out = pd.concat([out_existing, out_future], axis=0).reset_index(drop=True)
        return out

def caller_time_series(df, number):
    if df.empty:
        return pd.DataFrame(columns=["date", "count"])
    sub = df[df["number"] == number].copy()
    if sub.empty:
        return pd.DataFrame(columns=["date", "count"])
    s = sub.groupby("date").size().reset_index(name="count").sort_values("date")
    return s

def predict_calls_for_number(series_df, days_ahead=7):
    return forecast_calls_per_day(series_df, days_ahead=days_ahead)

# ---------- rolling & correlation ----------
def rolling_metrics(df, window_days=7):
    s = calls_per_day(df)
    if s.empty:
        return pd.DataFrame(columns=["date", "count", "rolling_mean", "rolling_std"]).set_index("date")
    ts = s.set_index("date")["count"].astype(float)
    rm = ts.rolling(window=window_days, min_periods=1).mean()
    rs = ts.rolling(window=window_days, min_periods=1).std().fillna(0)
    out = pd.DataFrame({"count": ts, "rolling_mean": rm, "rolling_std": rs})
    return out.reset_index().set_index("date")

def correlation_matrix(df):
    if df.empty:
        return pd.DataFrame()
    daily = df.groupby("date").agg(count=("number", "size"), mean_duration=("duration_s", "mean"), median_duration=("duration_s", "median"))
    return daily.corr()

# ---------- embedding (lazy import of umap) ----------
def embed_features(df, method="pca", n_components=2, random_state=42):
    if df.empty:
        return pd.DataFrame(columns=["emb_x", "emb_y"])
    tmp = df.copy()
    tmp["dayofweek"] = tmp["datetime"].apply(lambda d: d.weekday() if pd.notna(d) else -1)
    X = tmp[["duration_s", "hour", "dayofweek"]].fillna(-1).astype(float).values
    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)
    if method == "umap":
        try:
            import umap as umaplib
            reducer = umaplib.UMAP(n_components=n_components, random_state=random_state)
            emb = reducer.fit_transform(Xs)
        except Exception:
            pca = PCA(n_components=min(n_components, Xs.shape[1]), random_state=random_state)
            emb = pca.fit_transform(Xs)
    else:
        pca = PCA(n_components=min(n_components, Xs.shape[1]), random_state=random_state)
        emb = pca.fit_transform(Xs)
    df_emb = pd.DataFrame(emb[:, :2], columns=["emb_x", "emb_y"], index=tmp.index)
    return df_emb

# ---------- supervised weak-supervision predictor ----------
def make_feature_table_for_learning(df):
    if df.empty:
        return pd.DataFrame(), np.array([])
    tmp = df.copy()
    tmp["dayofweek"] = tmp["datetime"].apply(lambda d: d.weekday() if pd.notna(d) else -1)
    freq = tmp["number"].value_counts().to_dict()
    tmp["count_by_number"] = tmp["number"].map(freq).fillna(0)
    features = tmp[["duration_s", "hour", "dayofweek", "count_by_number"]].fillna(-1).astype(float)
    sus = heuristic_suspicious(df)
    if sus.empty:
        return features, np.array([])
    sus_keys = set((r["number"], r["datetime"]) for _, r in sus.iterrows())
    y = np.array([1 if (row["number"], row["datetime"]) in sus_keys else 0 for _, row in tmp.iterrows()], dtype=int)
    return features, y

def train_suspicion_predictor(df, test_size=0.2, random_state=42):
    X, y = make_feature_table_for_learning(df)
    if X.empty or y.size == 0 or int(y.sum()) == 0:
        return None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y if y.sum() > 1 else None, random_state=random_state)
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = LogisticRegression(class_weight="balanced", random_state=random_state, max_iter=200)
    model.fit(X_train_s, y_train)
    y_pred_prob = model.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob) if len(np.unique(y_test)) > 1 else None
    feature_names = list(X.columns)
    coef = model.coef_.flatten() if hasattr(model, "coef_") else np.zeros(len(feature_names))
    return {"model": model, "scaler": scaler, "feature_names": feature_names, "coef": coef, "auc": auc, "trained_on": {"n_train": len(X_train), "n_test": len(X_test), "pos_train": int(y_train.sum()), "pos_test": int(y_test.sum())}}

def predict_suspicion(model_dict, df):
    if model_dict is None:
        return df.copy()
    model = model_dict["model"]
    scaler = model_dict["scaler"]
    feature_names = model_dict["feature_names"]
    tmp = df.copy()
    tmp["dayofweek"] = tmp["datetime"].apply(lambda d: d.weekday() if pd.notna(d) else -1)
    freq = tmp["number"].value_counts().to_dict()
    tmp["count_by_number"] = tmp["number"].map(freq).fillna(0)
    X = tmp[feature_names].fillna(-1).astype(float)
    Xs = scaler.transform(X)
    probs = model.predict_proba(Xs)[:, 1]
    tmp["suspicion_prob"] = probs
    tmp["suspicion_label"] = (tmp["suspicion_prob"] >= 0.5).astype(int)
    return tmp

# ---------- top callers forecast ----------
def top_callers_forecast(df, top_n=10, days_ahead=7):
    if df.empty:
        return pd.DataFrame()
    top = most_frequent(df, top_n=top_n)
    rows = []
    for _, r in top.iterrows():
        number = r["number"]
        ts = caller_time_series(df, number)
        if ts.empty:
            preds = [0] * days_ahead
        else:
            fc = predict_calls_for_number(ts, days_ahead=days_ahead)
            preds = list(fc.tail(days_ahead)["count"].astype(float))
        rows.append({"number": number, **{f"next_{i+1}": preds[i] for i in range(days_ahead)}, "mean_next": float(np.mean(preds) if len(preds) else 0)})
    cols = ["number"] + [f"next_{i+1}" for i in range(days_ahead)] + ["mean_next"]
    return pd.DataFrame(rows)[cols]

# ---------- dendrogram helper (lazy scipy) ----------
def compute_dendrogram(df, metric="euclidean", method="ward"):
    try:
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import pdist
    except Exception:
        return None
    if df.empty:
        return None
    agg = caller_features(df)
    X = agg[["total_calls", "avg_duration", "pct_missed"]].fillna(0).values
    if X.shape[0] < 2:
        return None
    dist = pdist(X, metric=metric)
    Z = linkage(dist, method=method)
    return {"linkage": Z, "labels": agg["number"].tolist()}

# ---------- device / battery parsers ----------
def normalize_device_props(raw):
    out = {}
    if not raw:
        return out
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("[") and "]: [" in line:
            try:
                k, v = line.split("]: [", 1)
                k = k.lstrip("[").strip()
                v = v.rstrip("]").strip()
                out[k] = v
            except Exception:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    out[parts[0].strip().strip("[]")] = parts[1].strip().strip("[]")
        else:
            if ":" in line:
                k, v = line.split(":", 1)
                out[k.strip().strip("[]")] = v.strip().strip("[]")
    return out

def parse_battery(raw):
    data = {}
    if not raw:
        return data
    for line in raw.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            data[k.strip()] = v.strip()
    return data

# ---------- Natural-language summary helpers ----------
def generate_nl_most_frequent(freq_df):
    if freq_df is None or freq_df.empty:
        return "No callers found."
    top = freq_df.iloc[0]
    number = top["number"]
    count = int(top["count"])
    others = int(freq_df["count"].iloc[1:4].sum()) if len(freq_df) > 1 else 0
    return f"Most frequent caller: {number} called {count} times. Next top callers total {others} calls."

def generate_nl_hourly(df):
    if df is None or df.empty:
        return "No hourly data."
    pivot = df.groupby("hour").size()
    if pivot.empty:
        return "No hourly calls."
    peak = int(pivot.idxmax())
    peak_count = int(pivot.max())
    return f"Peak calling hour: {peak:02d}:00 with {peak_count} calls."

def generate_nl_forecast(forecast_df, days=7):
    if forecast_df is None or forecast_df.empty:
        return "No forecast available."
    tail = forecast_df.tail(days)
    mean = float(tail["count"].mean())
    std = float(tail["count"].std(ddof=0)) if len(tail) > 1 else 0.0
    return f"Next {days}-day forecast (mean±std): {mean:.1f} ± {std:.1f} calls/day."

def generate_nl_heuristic(sus_df):
    if sus_df is None or sus_df.empty:
        return "No heuristic suspicious activities detected."
    top = sus_df.iloc[0]
    num = top["number"]
    reason = top.get("reason", "")
    return f"Top heuristic alert: {num} — {reason}."

def generate_nl_ai(anom_df):
    if anom_df is None or anom_df.empty:
        return "No AI anomalies detected."
    top = anom_df.iloc[0]
    num = top["number"]
    score = float(top.get("anomaly_score", 0))
    return f"Top AI anomaly: {num} (score {score:.2f})."

def generate_nl_clustering(clustered_df):
    if clustered_df is None or clustered_df.empty:
        return "Not enough data for clustering."
    grp = clustered_df.groupby("cluster")["total_calls"].mean().sort_values(ascending=False)
    top_cluster = int(grp.index[0])
    return f"Cluster {top_cluster} has highest average calls per contact."
