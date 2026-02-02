# app.py — F1 Win Probability Predictor
# (Leakage-Safe + Season/Race/Driver selectors + Compare + Compact Metrics + ROC + Streaks + Insights)

import os
import numpy as np
import pandas as pd
import streamlit as st

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score, brier_score_loss,
    precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
)

import matplotlib.pyplot as plt
import seaborn as sns

# ---- Compact plot theme (keeps everything small and readable)
plt.rcParams.update({
    "figure.dpi": 110,          # moderate DPI
    "figure.figsize": (4.8, 3), # small default figure
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

# --------------------------- CONFIG ---------------------------------
CSV_PATH = r"C:\Users\aishw\OneDrive\Desktop\F1predictoncap\Data\f1_data.csv"
APP_TITLE = "Formula 1 — Win Probability Predictor (Leakage-Safe)"
st.set_page_config(page_title="F1 Win Predictor", layout="wide")

# Columns we absolutely forbid (post-race leakage)
LEAKY_COLS = {"points", "position", "positionorder"}

# Columns we require (pre-race model inputs)
REQUIRED_FEATURES = [
    "grid",
    "qual_position",
    "driver_points_to_date",
    "constructor_points_to_date",
    "year",
    "round",
]

# Optional: map alternate names from your CSV to the required names above
RENAME_MAP = {
    # "qualifying_position": "qual_position",
    # "driver_pts_to_date": "driver_points_to_date",
    # "constructor_pts_to_date": "constructor_points_to_date",
}

# --------------------------- HELPERS --------------------------------
def assert_no_leakage(columns):
    lower = {c.lower() for c in columns}
    bad = lower & LEAKY_COLS
    if bad:
        raise ValueError(
            f"❌ Leaky columns detected in features: {bad}. Drop them from the dataset or feature set."
        )

def _driver_name_series(df: pd.DataFrame) -> pd.Series:
    """Return a nice driver display name from columns present in df."""
    cols = set(df.columns)
    if {"forename", "surname"}.issubset(cols):
        return (df["forename"].astype(str) + " " + df["surname"].astype(str)).fillna("Unknown")
    if "driver_name" in cols:
        return df["driver_name"].astype(str).fillna("Unknown")
    if "driverId" in cols:
        return df["driverId"].astype(str).fillna("Unknown")
    return pd.Series(["Unknown"] * len(df))

@st.cache_data
def load_data(csv_path: str, _cache_bust: float):
    """Load CSV, lock features to REQUIRED_FEATURES (order preserved), return masks for last-season test."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    if RENAME_MAP:
        df = df.rename(columns=RENAME_MAP)

    if {"year", "round"}.issubset(df.columns):
        df = df.sort_values(["year", "round"]).reset_index(drop=True)
    else:
        raise KeyError("'year' and 'round' columns are required in the CSV.")

    if "win" not in df.columns:
        raise KeyError("Target column 'win' not found in the CSV.")

    numeric = df.select_dtypes(include=["number"])
    assert_no_leakage(numeric.columns)

    missing = [c for c in REQUIRED_FEATURES if c not in numeric.columns]
    if missing:
        raise KeyError(f"Your CSV is missing required numeric columns: {missing}")

    X = numeric[REQUIRED_FEATURES].copy()
    y = df["win"].astype(int)

    last_season = int(df["year"].max())
    train_mask = df["year"] < last_season
    test_mask  = df["year"] == last_season

    meta = {
        "last_season": last_season,
        "feature_candidates": list(X.columns),
        "n_rows": len(df),
    }
    return df, X, y, train_mask, test_mask, meta

@st.cache_resource
def train_and_eval(X, y, train_mask, test_mask):
    """Train on < last season; evaluate on last season only."""
    model = XGBClassifier(
        random_state=42,
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=-1,
    )
    model.fit(X[train_mask], y[train_mask])

    prob = model.predict_proba(X[test_mask])[:, 1]
    pred = (prob >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y[test_mask], pred),
        "roc_auc": roc_auc_score(y[test_mask], prob),
        "pr_auc":  average_precision_score(y[test_mask], prob),
        "brier":   brier_score_loss(y[test_mask], prob),
        "precision": precision_score(y[test_mask], pred, zero_division=0),
        "recall":    recall_score(y[test_mask], pred, zero_division=0),
        "f1":        f1_score(y[test_mask], pred, zero_division=0),
        "cm":        confusion_matrix(y[test_mask], pred),
        "feature_order": list(X.columns),        # equals REQUIRED_FEATURES
        "test_probabilities": prob,
        "y_test_true": y[test_mask].values,
        "test_mask": test_mask,  # for diagnostics use
    }
    return model, metrics

# ---------- Dropdown helpers (Season → Race → Driver) ----------
def build_lookup_frames(df):
    if not {"year", "round"}.issubset(df.columns):
        raise KeyError("Dataset must contain 'year' and 'round'.")
    return df

def options_for_season(df):
    return sorted(df["year"].unique().tolist())

def options_for_race(df, season):
    sub = df[df["year"] == season]
    if "name" in sub.columns:
        return sub[["round", "name"]].drop_duplicates().sort_values("round")
    else:
        return sub[["round"]].drop_duplicates().sort_values("round").assign(
            name=lambda x: "Round " + x["round"].astype(str)
        )

def options_for_drivers(df, season, round_):
    sub = df[(df["year"] == season) & (df["round"] == round_)].copy()
    sub["driver_name"] = _driver_name_series(sub)
    cols = ["driver_name", "driver_points_to_date", "constructor_points_to_date", "grid", "qual_position"]
    keep = [c for c in cols if c in sub.columns]
    return sub[keep].dropna(subset=["driver_name"]).sort_values("driver_name")

def plot_hist(prob_array):
    fig, ax = plt.subplots(figsize=(4.8, 3.0))  # smaller
    ax.hist(prob_array, bins=20, edgecolor="black")
    ax.set_title("Predicted Win Probabilities (Held-Out Season)")
    ax.set_xlabel("Win Probability")
    ax.set_ylabel("Number of Drivers")
    fig.tight_layout()
    return fig

# --------------------------- APP ------------------------------------
def main():
    st.title(APP_TITLE)

    # --- Personal header / style ---
    st.markdown("""
<style>
  /* Reduce overlap at top */
  .block-container {
      padding-top: 2.5rem !important;  /* adds nice breathing space under Streamlit header */
  }

  .small-muted{font-size:0.92rem;color:#666;}
  .stProgress > div > div > div { background: linear-gradient(90deg,#0ea5e9,#6366f1) !important; }

  /* Metric tile tweaks */
  [data-testid="stMetricValue"] { 
      font-weight: 700; 
      font-size:1rem !important; 
  }
  [data-testid="stMetricLabel"]{ 
      font-size:0.85rem !important; 
      color:#555; 
  }

  /* Optional: Slightly reduce tab spacing */
  [data-baseweb="tab"] button p {
      font-size: 0.92rem;
      font-weight: 500;
  }
</style>
""", unsafe_allow_html=True)

    st.write(
        "Built by **Aishwarya Gurrala Muni** · MS-BANA Capstone — "
        "a race-day tool that turns **grid & form** into clean win probabilities."
    )
    st.caption("Because F1 isn’t just fast cars—it’s data, strategy, and tiny edges.")

    # Sidebar quick actions / about
    if st.sidebar.button("↻ Refresh data & model"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.experimental_rerun()

    with st.sidebar.expander("👋 About this app", expanded=True):
        st.write("Predicts F1 win odds using **pre-race** signals only (no leakage).")
        st.write("**Stack:** XGBoost · scikit-learn · Streamlit")
        st.write("**Validation:** the **entire last season** is held out.")

    # Load data/model with cache-bust via file mtime
    mtime = os.path.getmtime(CSV_PATH)
    df, X, y, train_mask, test_mask, meta = load_data(CSV_PATH, mtime)
    model, metrics = train_and_eval(X, y, train_mask, test_mask)
    df = build_lookup_frames(df)

    # ---- Global selectors (used across tabs)
    st.sidebar.header("Select Real Race Context")
    seasons = options_for_season(df)
    season = st.sidebar.selectbox("Season", seasons, index=len(seasons)-1)

    race_opts = options_for_race(df, season)
    race_labels = race_opts["name"].tolist()
    race_rounds = race_opts["round"].tolist()
    race_label = st.sidebar.selectbox("Grand Prix", race_labels, index=0)
    round_selected = race_rounds[race_labels.index(race_label)]

    driver_table = options_for_drivers(df, season, round_selected)
    driver_choices = driver_table["driver_name"].tolist()
    chosen_driver = st.sidebar.selectbox("Driver", driver_choices, index=0)

    # Prefill from chosen driver
    row_prefill = driver_table[driver_table["driver_name"] == chosen_driver].iloc[0].to_dict()
    pref_grid = int(row_prefill.get("grid", 1) if pd.notna(row_prefill.get("grid", np.nan)) else 1)
    pref_qual = int(row_prefill.get("qual_position", pref_grid))
    pref_dpts = float(row_prefill.get("driver_points_to_date", 0.0) or 0.0)
    pref_cpts = float(row_prefill.get("constructor_points_to_date", 0.0) or 0.0)

    st.sidebar.header("Input Race Conditions (pre-race)")
    grid = st.sidebar.number_input("Grid (starting position)", min_value=1, max_value=30, value=pref_grid, step=1, key="grid")
    qual_position = st.sidebar.number_input("Qualifying position", min_value=1, max_value=30, value=pref_qual, step=1, key="qual")
    driver_points_to_date = st.sidebar.number_input("Driver points to date", min_value=0.0, value=round(pref_dpts, 2), step=0.5, key="dpts")
    constructor_points_to_date = st.sidebar.number_input("Constructor points to date", min_value=0.0, value=round(pref_cpts, 2), step=0.5, key="cpts")
    st.sidebar.caption("Rule of thumb: **lower grid & quali** + **higher points** → higher win odds.")

    # Tabs (Insights replaces Tableau)
    tab_pred, tab_compare, tab_diag, tab_streaks, tab_insight = st.tabs(
        ["Predictor", "Compare", "Diagnostics", "Streaks", "Insights"]
    )

    # ========== Predictor ==========
    with tab_pred:
        st.subheader(f"Validation on Held-Out Season {meta['last_season']}")

        # Compact metrics row
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        with c1: st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        with c2: st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
        with c3: st.metric("PR-AUC",  f"{metrics['pr_auc']:.3f}")
        with c4: st.metric("Brier ↓", f"{metrics['brier']:.3f}")
        st.caption("Tip: Lower grid (P1–P3), higher driver/constructor points, and better qualifying typically increase win chance.")

        # Build one-row input EXACTLY in training feature order
        feature_order = metrics["feature_order"]  # equals REQUIRED_FEATURES
        user_row = pd.DataFrame([{
            "grid": grid,
            "qual_position": qual_position,
            "driver_points_to_date": driver_points_to_date,
            "constructor_points_to_date": constructor_points_to_date,
            "year": season,
            "round": round_selected,
        }])[feature_order]

        prob = float(model.predict_proba(user_row)[:, 1][0])

        left, right = st.columns([1, 2])
        with left:
            st.subheader("Predicted Win Probability")
            st.progress(prob)
            st.write(f"**{prob*100:.2f}%** chance to win, given these inputs.")

            # Scenario naming + note
            scen_name = st.text_input("Scenario name", value=f"{season} {race_label} — {chosen_driver}")
            scen_note = st.text_area("Optional note", placeholder="Why I’m testing this setup…")

            if st.button("Save this scenario to CSV"):
                out = user_row.copy()
                out["predicted_win_probability"] = prob
                out["driver"] = chosen_driver
                out["season"] = season
                out["round"] = round_selected
                out["race"] = race_label
                out["scenario_name"] = scen_name
                out["note"] = scen_note
                out_path = os.path.join(os.path.dirname(CSV_PATH), "saved_scenarios.csv")
                if os.path.exists(out_path):
                    out.to_csv(out_path, mode="a", header=False, index=False)
                else:
                    out.to_csv(out_path, index=False)
                st.success(f"Saved “{scen_name}” → {out_path}")

        with right:
            st.subheader("Classification Metrics (Held-Out Season)")
            m1, m2, m3 = st.columns([1, 1, 1])
            with m1: st.metric("Precision", f"{metrics['precision']:.3f}")
            with m2: st.metric("Recall",    f"{metrics['recall']:.3f}")
            with m3: st.metric("F1-Score",  f"{metrics['f1']:.3f}")

            with st.expander("Confusion Matrix"):
                fig, ax = plt.subplots(figsize=(4.8, 3.6))
                sns.heatmap(
                    metrics["cm"], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred 0 (No-Win)", "Pred 1 (Win)"],
                    yticklabels=["Actual 0 (No-Win)", "Actual 1 (Win)"]
                )
                ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
                ax.set_title("Confusion Matrix — Held-Out Season")
                st.pyplot(fig)

            with st.expander("Debug: model inputs"):
                st.write("Feature order:", feature_order)
                st.dataframe(user_row, use_container_width=True)

    # ========== Compare ==========
    with tab_compare:
        st.subheader("Compare Multiple Drivers (same race)")
        st.write("Pick a few drivers from **this race** and I’ll rank their win chances.")
        multi_sel = st.multiselect("Pick drivers to compare", driver_choices[:], default=driver_choices[:3])

        if multi_sel:
            rows = []
            for name in multi_sel:
                r = driver_table[driver_table["driver_name"] == name].iloc[0].to_dict()
                row = pd.DataFrame([{
                    "grid": r.get("grid", 99) or 99,
                    "qual_position": r.get("qual_position", r.get("grid", 99)) or r.get("grid", 99),
                    "driver_points_to_date": r.get("driver_points_to_date", 0.0) or 0.0,
                    "constructor_points_to_date": r.get("constructor_points_to_date", 0.0) or 0.0,
                    "year": season, "round": round_selected,
                }])[feature_order]
                p = float(model.predict_proba(row)[:, 1][0])
                rows.append({"Driver": name, "Win Probability (%)": round(p*100, 2)})

            comp_df = pd.DataFrame(rows).sort_values("Win Probability (%)", ascending=False)
            st.dataframe(comp_df, use_container_width=True)
            st.bar_chart(comp_df.set_index("Driver"))

            if not comp_df.empty:
                top_row = comp_df.iloc[0]
                st.caption(f"📌 **Takeaway:** **{top_row['Driver']}** leads with **{top_row['Win Probability (%)']:.1f}%**.")

    # ========== Diagnostics ==========
    with tab_diag:
        st.subheader("Model Diagnostics")
        st.info("Design: I validate on the **entire last season** only. Keeps evaluation honest and future-looking.")

        st.write("Distribution of predicted win probabilities (held-out season).")
        st.pyplot(plot_hist(metrics["test_probabilities"]))

        # ROC Curve (smaller)
        fpr, tpr, _ = roc_curve(metrics["y_test_true"], metrics["test_probabilities"])
        roc_auc = auc(fpr, tpr)
        fig2, ax2 = plt.subplots(figsize=(4.8, 3.2))
        ax2.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
        ax2.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
        ax2.set_xlabel("False Positive Rate (1 - Specificity)")
        ax2.set_ylabel("True Positive Rate (Recall)")
        ax2.set_title("ROC Curve — Held-Out Season")
        ax2.legend(loc="lower right")
        st.pyplot(fig2)

        st.write("**Features used (in training order):**")
        st.code("\n".join(metrics["feature_order"]))

        with st.expander("Dataset info"):
            st.write(f"Rows: **{meta['n_rows']}**")
            st.code(", ".join(meta["feature_candidates"]))

    # ========== Streaks (simple momentum view) ==========
    with tab_streaks:
        st.subheader("Momentum / Streaks (within season)")
        sdf = df[df["year"] == season].copy()
        sdf["driver_name"] = _driver_name_series(sdf)

        # Rolling 3-round win count up to current round
        sdf = sdf.sort_values(["driver_name", "round"])
        sdf["rolling_wins_3"] = sdf.groupby("driver_name")["win"].rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
        cur = sdf[sdf["round"] == round_selected][["driver_name", "rolling_wins_3"]].sort_values("rolling_wins_3", ascending=False).head(10)
        st.caption(f"Top rolling **3-round** winners up to Round {round_selected}")
        st.dataframe(cur.rename(columns={"driver_name": "Driver", "rolling_wins_3": "Wins (last 3 rounds)"}), use_container_width=True)
        if not cur.empty:
            fig, ax = plt.subplots(figsize=(5.2, 3.0))
            ax.barh(cur["driver_name"][::-1], cur["rolling_wins_3"][::-1])
            ax.set_xlabel("Wins (last 3 rounds)")
            ax.set_ylabel("")
            st.pyplot(fig)

    # ========== Insights ==========
    with tab_insight:
        st.subheader("Insights & Highlights")
        st.write("Let’s interpret what your model just told us 👇")

        # Build race subset + names
        race_df = df[(df["year"] == season) & (df["round"] == round_selected)].copy()
        race_df["driver_name"] = _driver_name_series(race_df)

        # Ensure feature columns present; predict per-driver win prob for this race
        feature_order = metrics["feature_order"]
        safe_cols = [c for c in feature_order if c in race_df.columns]
        # if any are missing, fill in from current UI (season/round already match)
        for col, val in {
            "grid": grid,
            "qual_position": qual_position,
            "driver_points_to_date": driver_points_to_date,
            "constructor_points_to_date": constructor_points_to_date,
            "year": season,
            "round": round_selected,
        }.items():
            if col not in race_df.columns:
                race_df[col] = val

        race_df["predicted_win_prob"] = model.predict_proba(
            race_df[feature_order].fillna(0)
        )[:, 1]

        # Field averages & top3
        avg_prob = race_df["predicted_win_prob"].mean()
        top3 = race_df.sort_values("predicted_win_prob", ascending=False).head(3)

        st.markdown(f"### 🏎️ Race Context — {race_label} ({season})")
        st.metric("Field Average Win Probability", f"{avg_prob*100:.2f}%")

        driver_prob = float(prob)
        diff = (driver_prob - avg_prob) * 100
        if diff > 5:
            comment = f"🔥 **{chosen_driver}** has a **stronger-than-average** win chance this round! Momentum and starting spot look good."
        elif diff < -5:
            comment = f"⚠️ **{chosen_driver}** is **below the field average** — likely facing grid or pace disadvantages."
        else:
            comment = f"⚖️ **{chosen_driver}** is roughly **on par** with the grid — it could swing either way."

        st.markdown(f"**Insight:** {comment}")

        # Visual: small bar comparison
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.barh(["Field Avg", chosen_driver], [avg_prob * 100, driver_prob * 100],
                color=["#94a3b8", "#0ea5e9"])
        ax.set_xlim(0, 100)
        ax.set_xlabel("Predicted Win Probability (%)")
        for i, v in enumerate([avg_prob * 100, driver_prob * 100]):
            ax.text(min(v + 1, 99.5), i, f"{v:.1f}%", va="center", fontsize=10)
        st.pyplot(fig)

        # Top 3 table
        st.markdown("### 🥇 Top 3 Predicted Drivers for this Race")
        top3_display = top3[["driver_name", "grid", "predicted_win_prob"]].copy()
        top3_display["predicted_win_prob"] = (top3_display["predicted_win_prob"] * 100).round(2)
        st.dataframe(
            top3_display.rename(columns={
                "driver_name": "Driver", "grid": "Grid Position", "predicted_win_prob": "Win Probability (%)"
            }),
            use_container_width=True
        )

        # Optional: full odds board (mini leaderboard)
        with st.expander("Show full odds board (all drivers in this race)"):
            board = race_df[["driver_name", "grid", "predicted_win_prob"]].copy()
            board["Win Probability (%)"] = (board["predicted_win_prob"] * 100).round(2)
            board = board.sort_values("predicted_win_prob", ascending=False)
            st.dataframe(board.drop(columns=["predicted_win_prob"]).rename(columns={"driver_name": "Driver", "grid": "Grid"}), use_container_width=True)

        st.info(
            "💬 **Race Takeaway:** Win probability reflects not just driver skill — but how grid position, "
            "constructor form, and qualifying align. Surprises may still happen — even data can’t predict rain or Lap-1 chaos!"
        )

    # Footer
    st.markdown(
        """
        <div style="margin-top:2rem;color:#6b7280;font-size:0.9rem;text-align:center;">
          © Aishwarya Gurrala Muni — Formula 1 Win Probability · MS-BANA Capstone
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
