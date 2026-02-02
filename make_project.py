import os, json

DATA_DIR = r"C:\Users\aishw\OneDrive\Desktop\F1predictoncap\Data"

folders = [".streamlit", "src", "models", "reports", os.path.join("data","processed")]
for d in folders:
    os.makedirs(d, exist_ok=True)

open(".streamlit/config.toml","w", encoding="utf-8").write('[theme]\nbase="light"\nprimaryColor="#1f6feb"\nbackgroundColor="#ffffff"\nsecondaryBackgroundColor="#f6f8fa"\ntextColor="#0b1526"\nfont="sans serif"\n')

open("requirements.txt","w", encoding="utf-8").write("pandas\nnumpy\nscikit-learn\njoblib\nopenpyxl\nstreamlit>=1.32\n")

open("src/data_prep.py","w", encoding="utf-8").write(f'''
import os
import pandas as pd

DATA_DIR = os.environ.get("F1_DATA_DIR", r"{DATA_DIR}")

def _read(name: str):
    for ext in [".xlsx", ".csv"]:
        p = os.path.join(DATA_DIR, name + ext)
        if os.path.exists(p):
            return pd.read_excel(p) if ext == ".xlsx" else pd.read_csv(p)
    raise FileNotFoundError(f"Missing file for {{name}} in {{DATA_DIR}}")

def main():
    results = _read("results")
    needed = {{'raceId','driverId','constructorId','grid','positionOrder','points'}}
    if not needed.issubset(results.columns):
        raise ValueError(f"`results` missing columns. Needs: {{needed}}")

    df = results[list(needed)].copy()

    # Optional joins
    try:
        q = _read("qualifying")[["raceId","driverId","position"]].rename(columns={{"position":"qual_position"}})
        df = df.merge(q, on=["raceId","driverId"], how="left")
    except Exception:
        pass

    try:
        ds = _read("driver_standings")[["raceId","driverId","points"]].rename(columns={{"points":"driver_points_to_date"}})
        df = df.merge(ds, on=["raceId","driverId"], how="left")
    except Exception:
        pass

    try:
        cs = _read("constructor_standings")[["raceId","constructorId","points"]].rename(columns={{"points":"constructor_points_to_date"}})
        df = df.merge(cs, on=["raceId","constructorId"], how="left")
    except Exception:
        pass

    df["win"] = (df["positionOrder"] == 1).astype(int)

    for c in ["qual_position","driver_points_to_date","constructor_points_to_date","points","grid"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/training_data.csv", index=False)
    print("Saved data/processed/training_data.csv", df.shape)

if __name__ == "__main__":
    main()
''')

open("src/train.py","w", encoding="utf-8").write('''
import os, json
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.ensemble import GradientBoostingClassifier

DATA_PATH = "data/processed/training_data.csv"

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Run `python -m src.data_prep` first.")

    df = pd.read_csv(DATA_PATH)
    features = [c for c in ["grid","qual_position","driver_points_to_date","constructor_points_to_date","points"] if c in df.columns]
    X, y = df[features], df["win"]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = GradientBoostingClassifier(random_state=42).fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:,1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "features_used": features,
        "accuracy": float(accuracy_score(yte, pred)),
        "roc_auc": float(roc_auc_score(yte, proba)),
        "brier": float(brier_score_loss(yte, proba))
    }

    os.makedirs("models", exist_ok=True)
    dump(model, "models/race_predictor.pkl")

    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics.json","w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Model saved to models/race_predictor.pkl")
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()
''')

open("app.py","w", encoding="utf-8").write('''
import os, json
import pandas as pd
import streamlit as st
from joblib import load

st.set_page_config(page_title="F1 Predictor", page_icon=":checkered_flag:", layout="wide")
st.title("Formula 1 — Win Probability")

try:
    model = load("models/race_predictor.pkl")
except Exception:
    st.error("Model not found. Run: `python -m src.data_prep` then `python -m src.train`.")
    st.stop()

metrics = None
if os.path.exists("reports/metrics.json"):
    with open("reports/metrics.json","r", encoding="utf-8") as f:
        metrics = json.load(f)

with st.sidebar:
    st.header("Input Race Conditions")
    grid = st.number_input("Grid (starting position)", 1, 30, 5)
    qual = st.number_input("Qualifying position", 1, 30, 5)
    drv_pts = st.number_input("Driver points to date", 0.0, 500.0, 75.0)
    cons_pts = st.number_input("Constructor points to date", 0.0, 1000.0, 150.0)
    recent = st.number_input("Recent points (last race/avg)", 0.0, 50.0, 8.0)

row = {
    "grid": grid,
    "qual_position": qual,
    "driver_points_to_date": drv_pts,
    "constructor_points_to_date": cons_pts,
    "points": recent
}
X = pd.DataFrame([row])
need = getattr(model, "feature_names_in_", None)
if need is not None:
    X = X[[c for c in need if c in X.columns]]

proba = float(model.predict_proba(X)[0,1])

col1, col2 = st.columns([1,2])
with col1:
    st.subheader("Predicted Win Probability")
    st.metric("Win Probability", f"{proba:.2%}")
    st.progress(min(max(proba,0.0),1.0))

if metrics:
    with col2:
        st.subheader("Validation Metrics")
        st.write(metrics)
''')

print("Files created. Next steps:")
print("1) python -m venv venv")
print("2) venv\\Scripts\\activate.bat")
print("3) pip install -r requirements.txt")
print("4) python -m src.data_prep")
print("5) python -m src.train")
print("6) streamlit run app.py")
