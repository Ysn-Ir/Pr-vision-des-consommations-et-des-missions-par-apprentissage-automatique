"""Entraînement et évaluation des modèles de prévision pour Marrakech et Laâyoune.

Reprend exactement le protocole rigoureux de Tétouan :
- Pas horaire, prévision à J+1 (24 heures)
- Variables calendaires, météo (ERA5 Open-Meteo), et thermiques (HDD/CDD)
- Variables de retards strictement décalées d'au moins 24 h (anti-fuite)
- Découpage chronologique strict : entraînement -> validation -> essai
- Comparaison : Référence naïve, Régression Linéaire, Forêt Aléatoire, XGBoost, XGBoost Résiduel
"""
from pathlib import Path
import json, time, warnings
import numpy as np
import pandas as pd
import httpx
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from forecast_utils import ResidualModel

warnings.filterwarnings("ignore")
SEED = 42

DATA_DIR = Path("data") / "morocco_cities"
ART = Path("artifacts")
ART.mkdir(exist_ok=True)

# Facteurs d'émission du réseau national marocain (kgCO2/kWh)
FACTORS = {"bas": 0.596, "central": 0.658, "haut": 0.720}

# Jours fériés au Maroc en 2023
HOLIDAYS_2023 = pd.to_datetime([
    "2023-01-01", "2023-01-11", "2023-05-01", "2023-07-30", "2023-08-14",
    "2023-08-20", "2023-08-21", "2023-11-06", "2023-11-18",
    "2023-04-21", "2023-04-22", "2023-06-28", "2023-06-29", "2023-07-19", "2023-09-27"
])
# Période de Ramadan 2023
RAMADAN_START_2023 = pd.Timestamp("2023-03-23")
RAMADAN_END_2023 = pd.Timestamp("2023-04-21")

FEATURES = [
    "hour", "dow", "is_weekend", "is_holiday", "is_ramadan",
    "temp", "humidity", "wind", "hdd", "cdd",
    "lag_24", "lag_48", "lag_168", "roll_24", "roll_168"
]


def metrics(y_true, y_pred):
    e = np.asarray(y_pred) - np.asarray(y_true)
    y_true_arr = np.asarray(y_true)
    # Éviter division par zéro pour MAPE
    nonzero = y_true_arr > 1e-3
    mape = float(np.mean(np.abs(e[nonzero]) / y_true_arr[nonzero])) * 100.0
    return {
        "MAE": float(np.mean(np.abs(e))),
        "RMSE": float(np.sqrt(np.mean(e**2))),
        "MAPE_%": float(mape),
        "biais": float(np.mean(e))
    }


def fetch_or_load_weather(city_name, lat, lon, start_date, end_date):
    cache = DATA_DIR / f"weather_{city_name}_{start_date}_{end_date}.csv"
    if cache.exists():
        print(f"Chargement météo locale depuis le cache : {cache}")
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    
    print(f"Téléchargement météo ERA5 pour {city_name}...")
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    resp = httpx.get(url, timeout=40)
    data = resp.json()["hourly"]
    df = pd.DataFrame({
        "DateTime": pd.to_datetime(data["time"]),
        "temp": data["temperature_2m"],
        "humidity": data["relative_humidity_2m"],
        "wind": data["wind_speed_10m"]
    }).set_index("DateTime")
    df.to_csv(cache)
    return df


def build_features(df_load, df_weather, target_col="load_kw"):
    df = df_load.join(df_weather, how="inner")
    df["temp"] = df["temp"].interpolate(method="time")
    df["humidity"] = df["humidity"].interpolate(method="time")
    df["wind"] = df["wind"].interpolate(method="time")
    df[target_col] = df[target_col].interpolate(method="time")
    
    # Variables calendaires
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    df["is_holiday"] = df.index.normalize().isin(HOLIDAYS_2023).astype(int)
    df["is_ramadan"] = ((df.index >= RAMADAN_START_2023) & (df.index <= RAMADAN_END_2023)).astype(int)
    
    # Variables thermiques
    df["hdd"] = np.maximum(0, 15.0 - df["temp"])
    df["cdd"] = np.maximum(0, df["temp"] - 18.0)
    
    # Variables décalées (Lags) strictement >= 24 h
    s = df[target_col]
    df["lag_24"] = s.shift(24)
    df["lag_48"] = s.shift(48)
    df["lag_168"] = s.shift(168)
    df["roll_24"] = s.shift(24).rolling(24).mean()
    df["roll_168"] = s.shift(24).rolling(168).mean()
    
    return df.dropna()


def train_and_evaluate_city(city_key, df, target_col="load_kw", train_end="2023-08-31 23:00", val_end="2023-10-31 23:00"):
    print(f"\n========================================================")
    print(f"  ENTRAÎNEMENT & ÉVALUATION : {city_key.upper()}")
    print(f"========================================================")
    print(f"Total observations après retards : {len(df)}")
    
    tr = df.loc[:train_end]
    va = df.loc[pd.Timestamp(train_end) + pd.Timedelta(hours=1) : val_end]
    te = df.loc[pd.Timestamp(val_end) + pd.Timedelta(hours=1) :]
    trva = df.loc[:val_end]
    
    print(f"Découpage : Entraînement {len(tr)} h | Validation {len(va)} h | Essai {len(te)} h")
    
    # 1. Référence Naïve 24 h
    ref_pred_val = va["lag_24"]
    ref_pred_te = te["lag_24"]
    ref_metrics = metrics(te[target_col], ref_pred_te)
    ref_metrics.update({"entraînement_s": 0.0, "prédiction_s": 0.0, "gain_RMSE_%": 0.0, "gain_MAE_%": 0.0})
    print(f"Référence (Naïve 24h) Essai -> RMSE: {ref_metrics['RMSE']:.1f}, MAE: {ref_metrics['MAE']:.1f}, MAPE: {ref_metrics['MAPE_%']:.2f}%")
    
    # 2. Modèles
    pre = ColumnTransformer([
        ("oh", OneHotEncoder(handle_unknown="ignore"), ["hour", "dow"]),
        ("num", StandardScaler(), [c for c in FEATURES if c not in ("hour", "dow")])
    ])
    
    makers = {
        "Régression linéaire": lambda: Pipeline([("pre", pre), ("lr", LinearRegression())]),
        "Forêt aléatoire": lambda: RandomForestRegressor(n_estimators=150, max_depth=12, random_state=SEED, n_jobs=-1),
        "XGBoost": lambda: XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.04, random_state=SEED, n_jobs=-1, tree_method="hist"),
        "XGBoost sur écart à la veille": lambda: ResidualModel(
            XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.04, random_state=SEED, n_jobs=-1, tree_method="hist"),
            base_col="lag_24"
        )
    }
    
    # Sélection sur validation
    val_scores = {}
    for name, mk in makers.items():
        m = mk().fit(tr[FEATURES], tr[target_col])
        p = m.predict(va[FEATURES])
        val_scores[name] = metrics(va[target_col], p)
        print(f"Validation [{name:30s}] -> RMSE: {val_scores[name]['RMSE']:.1f} | MAE: {val_scores[name]['MAE']:.1f}")
        
    selected_name = min(val_scores, key=lambda k: val_scores[k]["RMSE"])
    print(f"--> Modèle retenu sur la validation : {selected_name}")
    
    # Ré-entraînement sur Train+Val et Évaluation sur l'Essai
    test_results = {"Référence (Naïve 24 h)": ref_metrics}
    fitted_models = {}
    
    for name, mk in makers.items():
        t0 = time.time()
        m = mk().fit(trva[FEATURES], trva[target_col])
        t_fit = time.time() - t0
        
        t0 = time.time()
        pred = m.predict(te[FEATURES])
        t_pred = time.time() - t0
        
        score = metrics(te[target_col], pred)
        gain_rmse = (1.0 - score["RMSE"] / ref_metrics["RMSE"]) * 100.0
        gain_mae = (1.0 - score["MAE"] / ref_metrics["MAE"]) * 100.0
        score.update({
            "entraînement_s": round(t_fit, 3),
            "prédiction_s": round(t_pred, 3),
            "gain_RMSE_%": round(gain_rmse, 2),
            "gain_MAE_%": round(gain_mae, 2)
        })
        test_results[name] = score
        fitted_models[name] = m
        print(f"Essai [{name:30s}] -> RMSE: {score['RMSE']:6.1f} ({gain_rmse:+5.1f} %) | MAE: {score['MAE']:6.1f} | MAPE: {score['MAPE_%']:.2f}%")
        
    # Sauvegarde des artefacts
    df.to_csv(ART / f"features_{city_key}_hourly.csv")
    joblib.dump({
        "model": fitted_models[selected_name],
        "name": selected_name,
        "features": FEATURES,
        "factors": FACTORS,
        "city": city_key,
        "target_col": target_col
    }, ART / f"model_{city_key}.joblib")
    
    with open(ART / f"results_{city_key}.json", "w", encoding="utf-8") as f:
        json.dump({
            "city": city_key,
            "target": target_col,
            "selected_on_validation": selected_name,
            "val_scores": val_scores,
            "test_metrics": test_results
        }, f, indent=2, ensure_ascii=False)
        
    print(f"Artefacts sauvegardés pour {city_key} dans artifacts/")
    return test_results


def run():
    # -------------------------------------------------------------
    # 1. MARRAKECH
    # -------------------------------------------------------------
    m_raw = pd.read_csv(DATA_DIR / "Marrakech_30T.csv", sep=";")
    m_raw["DateTime"] = pd.to_datetime(m_raw["DateTime"])
    m_df = m_raw.set_index("DateTime").resample("1h").mean()
    m_df["load_kw"] = m_df["zone1"] + m_df["zone2"]
    
    w_marrakech = fetch_or_load_weather("marrakech", 31.6295, -7.9811, "2023-01-09", "2024-01-08")
    marrakech_features = build_features(m_df, w_marrakech, target_col="load_kw")
    train_and_evaluate_city(
        "marrakech",
        marrakech_features,
        target_col="load_kw",
        train_end="2023-08-31 23:00",
        val_end="2023-10-31 23:00"
    )
    
    # -------------------------------------------------------------
    # 2. LAAYOUNE (Courant en Ampères sur départs 22 kV)
    # -------------------------------------------------------------
    l_raw = pd.read_csv(DATA_DIR / "Laayoune_10T.csv", sep=";")
    l_raw["DateTime"] = pd.to_datetime(l_raw["DateTime"], format="%d/%m/%Y %H:%M")
    for c in ["zone1", "zone2", "zone3", "zone4", "zone5"]:
        if l_raw[c].dtype == object:
            l_raw[c] = l_raw[c].astype(str).str.replace(",", ".").str.strip().astype(float)
    l_df = l_raw.set_index("DateTime").resample("1h").mean()
    l_df["current_A"] = l_df[["zone1", "zone2", "zone3", "zone4", "zone5"]].sum(axis=1)
    
    w_laayoune = fetch_or_load_weather("laayoune", 27.1536, -13.2033, "2022-09-14", "2023-09-14")
    laayoune_features = build_features(l_df, w_laayoune, target_col="current_A")
    train_and_evaluate_city(
        "laayoune",
        laayoune_features,
        target_col="current_A",
        train_end="2023-04-30 23:00",
        val_end="2023-06-30 23:00"
    )


if __name__ == "__main__":
    run()
