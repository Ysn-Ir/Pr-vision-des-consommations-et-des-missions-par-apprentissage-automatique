"""Utilitaires pour l'exploration multi-villes au Maroc (Marrakech, Laâyoune, Fès-Meknès, Tétouan).

Données issues de :
- Tétouan : Amendis SCADA / UCI Repository (id=849)
- Marrakech & Laâyoune : Zenodo (DOI: 10.5281/zenodo.16423697) / Mendeley Data (DOI: 10.17632/6s9mfp3bd8.1)
- Fès-Meknès : Open-Meteo Archive API (ERA5 Reanalysis)
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import httpx

DATA_DIR = Path(__file__).parent / "data"
CITIES_DIR = DATA_DIR / "morocco_cities"


def load_marrakech_hourly() -> pd.DataFrame:
    """Charge et agrège les données de compteurs intelligents de Marrakech au pas horaire (kW)."""
    csv_file = CITIES_DIR / "Marrakech_30T.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"Fichier introuvable : {csv_file}")
    df = pd.read_csv(csv_file, sep=";")
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df.set_index("DateTime").sort_index()
    # Résumé horaire (moyenne des deux pas de 30 min)
    hourly = df.resample("1h").mean().dropna()
    hourly["total_kw"] = hourly["zone1"] + hourly["zone2"]
    hourly["hour"] = hourly.index.hour
    hourly["dow"] = hourly.index.dayofweek
    hourly["is_weekend"] = hourly["dow"].isin([5, 6]).astype(int)
    return hourly


def load_laayoune_hourly() -> pd.DataFrame:
    """Charge et agrège les données de compteurs intelligents de Laâyoune au pas horaire (Courant en A)."""
    csv_file = CITIES_DIR / "Laayoune_10T.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"Fichier introuvable : {csv_file}")
    df = pd.read_csv(csv_file, sep=";")
    df["DateTime"] = pd.to_datetime(df["DateTime"], format="%d/%m/%Y %H:%M")
    # Conversion des virgules décimales en flottants
    for col in ["zone1", "zone2", "zone3", "zone4", "zone5"]:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(",", ".").str.strip().astype(float)
    df = df.set_index("DateTime").sort_index()
    hourly = df.resample("1h").mean().dropna()
    hourly["total_current_A"] = hourly[["zone1", "zone2", "zone3", "zone4", "zone5"]].sum(axis=1)
    hourly["hour"] = hourly.index.hour
    hourly["dow"] = hourly.index.dayofweek
    hourly["is_weekend"] = hourly["dow"].isin([5, 6]).astype(int)
    return hourly


def fetch_fes_weather(start_date: str = "2017-01-01", end_date: str = "2017-01-31") -> pd.DataFrame:
    """Récupère les données météo horaires réanalyse ERA5 pour Fès via l'API Open-Meteo."""
    cache_file = CITIES_DIR / f"weather_fes_{start_date}_{end_date}.csv"
    if cache_file.exists():
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)
    
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude=34.0333&longitude=-5.0000&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    try:
        resp = httpx.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()["hourly"]
            df = pd.DataFrame({
                "DateTime": pd.to_datetime(data["time"]),
                "temp": data["temperature_2m"],
                "humidity": data["relative_humidity_2m"],
                "wind": data["wind_speed_10m"]
            }).set_index("DateTime")
            df.to_csv(cache_file)
            return df
    except Exception as e:
        print(f"Avertissement : échec Open-Meteo ({e})")
    
    # Fallback synthétique si réseau indisponible
    dr = pd.date_range(start_date, end_date, freq="1h")
    return pd.DataFrame({"temp": 15.0, "humidity": 60.0, "wind": 2.0}, index=dr)


def get_moroccan_cities_overview() -> dict:
    """Synthèse comparative multi-territoires au Maroc pour le rapport et le tableau de bord."""
    meta_path = CITIES_DIR / "morocco_cities_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    
    # Analyse Marrakech
    m_info = {"statut": "Données intégrées"}
    try:
        m_df = load_marrakech_hourly()
        m_info.update({
            "periode": f"{m_df.index.min().date()} au {m_df.index.max().date()}",
            "heures": len(m_df),
            "charge_moyenne_kw": round(float(m_df["total_kw"].mean()), 1),
            "charge_pointe_kw": round(float(m_df["total_kw"].max()), 1),
            "heure_pointe_typique": int(m_df.groupby("hour")["total_kw"].mean().idxmax()),
            "ratio_weekend_semaine": round(float(m_df[m_df.is_weekend == 1]["total_kw"].mean() / m_df[m_df.is_weekend == 0]["total_kw"].mean()), 3)
        })
    except Exception as e:
        m_info["erreur"] = str(e)

    # Analyse Laayoune
    l_info = {"statut": "Données intégrées"}
    try:
        l_df = load_laayoune_hourly()
        l_info.update({
            "periode": f"{l_df.index.min().date()} au {l_df.index.max().date()}",
            "heures": len(l_df),
            "courant_moyen_A": round(float(l_df["total_current_A"].mean()), 1),
            "courant_pointe_A": round(float(l_df["total_current_A"].max()), 1),
            "heure_pointe_typique": int(l_df.groupby("hour")["total_current_A"].mean().idxmax()),
            "zones_comptabilisees": 5
        })
    except Exception as e:
        l_info["erreur"] = str(e)

    return {
        "metadata": meta,
        "marrakech": m_info,
        "laayoune": l_info,
        "transposition_fes_meknes": {
            "statut": "Méthodologie transposable",
            "climat": "Continental méditerranéen (étés caniculaires, hivers froids)",
            "impact_prevision": "Forte sensibilité de la charge au chauffage hivernal et à la climatisation estivale (amplitude thermique plus élevée qu'à Tétouan)"
        }
    }
