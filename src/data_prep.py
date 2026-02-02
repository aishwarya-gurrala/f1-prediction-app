import os
import pandas as pd

DATA_DIR = os.environ.get("F1_DATA_DIR", r"C:\Users\aishw\OneDrive\Desktop\F1predictoncap\Data")

def _read_file(name):
    for ext in [".xlsx", ".csv"]:
        path = os.path.join(DATA_DIR, name + ext)
        if os.path.exists(path):
            if ext == ".xlsx":
                return pd.read_excel(path)
            else:
                return pd.read_csv(path)
    raise FileNotFoundError(f"{name} not found in {DATA_DIR}")

def build_dataset():
    results = _read_file("results")
    base_cols = ['raceId','driverId','constructorId','grid','positionOrder','points']
    df = results[base_cols].copy()

    # optional joins
    try:
        q = _read_file("qualifying")[["raceId","driverId","position"]].rename(columns={"position":"qual_position"})
        df = df.merge(q, on=["raceId","driverId"], how="left")
    except Exception:
        pass

    try:
        ds = _read_file("driver_standings")[["raceId","driverId","points"]].rename(columns={"points":"driver_points_to_date"})
        df = df.merge(ds, on=["raceId","driverId"], how="left")
    except Exception:
        pass

    try:
        cs = _read_file("constructor_standings")[["raceId","constructorId","points"]].rename(columns={"points":"constructor_points_to_date"})
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
    build_dataset()
