"""Application Web et API Multi-Villes de prévision de charge et d'émissions de CO₂.

Supporte l'inférence et la visualisation pour 3 territoires marocains réels :
1. Tétouan (SCADA Amendis, 2017 - Unité : kW)
2. Marrakech (Compteurs communicants 22 kV, 2023 - Unité : kW)
3. Laâyoune (Compteurs 22 kV sur 5 départs, 2022-2023 - Unité : Ampères)
Et la transposition climatique vers Fès-Meknès (ERA5).

Lancement :
    uvicorn api:app --reload
Accès :
    http://127.0.0.1:8000
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

import forecast_utils
from forecast_utils import (
    ResidualModel,  # noqa: F401
    ZONE_SHARES,
    FACTORS_DEFAULT,
    compute_prediction_intervals,
    decompose_zones,
    simulate_weather_anomaly,
    calculate_emissions_summary
)
import multi_city_utils

ROOT = Path(__file__).parent
ART = ROOT / "artifacts"

# ------------------------------------------------------------------------------
# Chargement des modèles et données multi-villes
# ------------------------------------------------------------------------------
CITY_CONFIG = {
    "tetouan": {
        "name": "Tétouan",
        "region": "Tanger-Tétouan-Al Hoceïma",
        "unit": "kW",
        "energy_unit": "MWh",
        "target_col": "load_kw",
        "default_date": "2017-11-15",
        "model_file": ART / "model.joblib",
        "data_file": ART / "features_hourly.csv",
        "results_file": ART / "results.json"
    },
    "marrakech": {
        "name": "Marrakech",
        "region": "Marrakech-Safi",
        "unit": "kW",
        "energy_unit": "MWh",
        "target_col": "load_kw",
        "default_date": "2023-11-15",
        "model_file": ART / "model_marrakech.joblib",
        "data_file": ART / "features_marrakech_hourly.csv",
        "results_file": ART / "results_marrakech.json"
    },
    "laayoune": {
        "name": "Laâyoune",
        "region": "Laâyoune-Sakia El Hamra",
        "unit": "A",
        "energy_unit": "kAh",
        "target_col": "current_A",
        "default_date": "2023-07-15",
        "model_file": ART / "model_laayoune.joblib",
        "data_file": ART / "features_laayoune_hourly.csv",
        "results_file": ART / "results_laayoune.json"
    },
    "fes": {
        "name": "Fès-Meknès",
        "region": "Fès-Meknès (Transfer Learning ENSA)",
        "unit": "kW",
        "energy_unit": "MWh",
        "target_col": "load_kw",
        "default_date": "2023-11-15",
        "model_file": ART / "model_fes.joblib",
        "data_file": ART / "features_fes_hourly.csv",
        "results_file": ART / "results_fes.json"
    }
}

MODELS = {}
DATASETS = {}
RESULTS = {}

for c_key, cfg in CITY_CONFIG.items():
    if cfg["model_file"].exists() and cfg["data_file"].exists():
        bundle = joblib.load(cfg["model_file"])
        MODELS[c_key] = bundle
        DATASETS[c_key] = pd.read_csv(cfg["data_file"], index_col=0, parse_dates=True)
        if cfg["results_file"].exists():
            RESULTS[c_key] = json.loads(cfg["results_file"].read_text(encoding="utf-8"))

app = FastAPI(
    title="Plateforme Multi-Villes de Prévision Énergétique — Maroc",
    description="Prévision de la charge horaire par apprentissage automatique pour Tétouan, Marrakech et Laâyoune.",
    version="2.0.0"
)


# ==============================================================================
# Endpoints API REST
# ==============================================================================

@app.get("/health")
def health():
    """État du service et disponibilité des modèles par ville."""
    available_cities = {}
    for c_key, cfg in CITY_CONFIG.items():
        if c_key in DATASETS:
            df = DATASETS[c_key]
            available_cities[c_key] = {
                "nom": cfg["name"],
                "modele": MODELS[c_key]["name"],
                "unite": cfg["unit"],
                "plage_dates": [str(df.index.min().date()), str(df.index.max().date())]
            }
    return {
        "status": "ok",
        "territoires_actifs": available_cities,
        "transposition": "Fès-Meknès (méthodologie transposable, ERA5)"
    }


@app.get("/forecast")
def forecast(
    city: str = Query("tetouan", description="Ville cible : tetouan, marrakech, laayoune"),
    date: str = Query(..., description="Date au format AAAA-MM-JJ"),
    delta_temp: float = Query(0.0, description="Anomalie thermique simulée en °C (-5 à +5)"),
    emission_factor: float = Query(None, description="Facteur d'émission personnalisé en kgCO2/kWh")
):
    """Prévision horaire (24h) multi-villes avec incertitude, impact météo 'what-if' et émissions."""
    city_key = city.lower()
    if city_key not in DATASETS:
        raise HTTPException(400, f"Ville non reconnue : {city}. Villes disponibles : {list(DATASETS.keys())}")
    
    cfg = CITY_CONFIG[city_key]
    model_bundle = MODELS[city_key]
    model = model_bundle["model"]
    features = model_bundle["features"]
    data = DATASETS[city_key]
    target_col = cfg["target_col"]
    
    try:
        day = pd.Timestamp(date).normalize()
    except Exception:
        raise HTTPException(422, "Format de date invalide. Format attendu : AAAA-MM-JJ")
    
    rows = data.loc[day : day + pd.Timedelta(hours=23)]
    if len(rows) == 0:
        raise HTTPException(
            404,
            f"Date {date} absente pour {cfg['name']}. Plage disponible : {data.index.min().date()} à {data.index.max().date()}."
        )
    
    # 1. Prévision
    base_pred = model.predict(rows[features])
    
    # 2. Simulation météo "what-if"
    simulated_pred = simulate_weather_anomaly(rows, base_pred, delta_temp) if city_key != "laayoune" else base_pred
    
    # 3. Intervalles d'incertitude (90%)
    rmse_map = {"tetouan": 2037.4, "marrakech": 142.2, "laayoune": 26.9, "fes": 1944.0}
    rmse_val = rmse_map.get(city_key, 2000.0)
    lower_90, upper_90 = compute_prediction_intervals(simulated_pred, rmse=rmse_val, confidence=0.90)
    
    # 4. Total journalier
    total_day = float(simulated_pred.sum())
    base_total = float(base_pred.sum())
    
    # Observé
    has_obs = target_col in rows.columns and not rows[target_col].isna().all()
    obs_vals = [round(float(v), 1) for v in rows[target_col]] if has_obs else []
    obs_total = float(rows[target_col].sum()) if has_obs else None
    
    # Émissions (si en kW)
    emissions = None
    if cfg["unit"] == "kW":
        emissions = calculate_emissions_summary(total_day, emission_factor)
        
    temperatures = [round(float(t), 1) for t in rows["temp"]] if "temp" in rows.columns else []
    
    return {
        "ville": cfg["name"],
        "region": cfg["region"],
        "date": str(day.date()),
        "unite": cfg["unit"],
        "energy_unit": cfg["energy_unit"],
        "modele": model_bundle["name"],
        "heures": len(rows),
        "delta_temp_simulee_degC": delta_temp,
        "prevision": [round(float(p), 1) for p in simulated_pred],
        "prevision_base": [round(float(p), 1) for p in base_pred],
        "prevision_lower_90": [round(float(p), 1) for p in lower_90],
        "prevision_upper_90": [round(float(p), 1) for p in upper_90],
        "observe": obs_vals,
        "temperature_degC": temperatures,
        "total_prevu": round(total_day, 1),
        "total_observe": round(obs_total, 1) if obs_total is not None else None,
        "emissions_estimees_tCO2": emissions,
        "zones": decompose_zones(simulated_pred) if city_key == "tetouan" else None
    }


@app.get("/api/overview")
def overview(city: str = "tetouan"):
    """Métriques comparatives des modèles pour la ville sélectionnée."""
    city_key = city.lower()
    return RESULTS.get(city_key, RESULTS.get("tetouan", {}))


@app.get("/api/morocco-cities")
def morocco_cities():
    """Synthèse comparative multi-territoires au Maroc."""
    return multi_city_utils.get_moroccan_cities_overview()


@app.get("/api/presets")
def presets(city: str = "tetouan"):
    """Scénarios types selon la ville."""
    if city == "marrakech":
        return [
            {"label": "Automne (Essai)", "date": "2023-11-15", "desc": "Semaine typique de test 2023"},
            {"label": "Pointe Canicule (Août)", "date": "2023-08-11", "desc": "Pointe diurne extrême à 15h00 (>44°C)"},
            {"label": "Hiver froid", "date": "2023-01-25", "desc": "Basses températures nocturnes"},
            {"label": "Ramadan 2023", "date": "2023-04-05", "desc": "Activité nocturne décalée"}
        ]
    elif city == "laayoune":
        return [
            {"label": "Été Saharien (Essai)", "date": "2023-07-15", "desc": "Période d'évaluation d'été"},
            {"label": "Pointe Soirée (21h)", "date": "2023-06-20", "desc": "Pic d'éclairage et ventilation"},
            {"label": "Hiver Océanique", "date": "2023-01-10", "desc": "Profil thermique modéré"}
        ]
    elif city == "fes":
        return [
            {"label": "15 Nov (Essai Hiver)", "date": "2023-11-15", "desc": "Semaine typique d'évaluation 2023"},
            {"label": "12 Août (Canicule 46°C)", "date": "2023-08-12", "desc": "Pointe de climatisation extrême à Fès"},
            {"label": "25 Jan (Gel nocturne -1°C)", "date": "2023-01-25", "desc": "Forte demande de chauffage"},
            {"label": "05 Avr (Ramadan 2023)", "date": "2023-04-05", "desc": "Activité nocturne et rupture du jeûne"}
        ]
    else:
        return [
            {"label": "15 Nov (Essai typique)", "date": "2017-11-15", "desc": "Période d'évaluation hivernale"},
            {"label": "04 Août (Canicule)", "date": "2017-08-04", "desc": "Pic de climatisation estivale"},
            {"label": "18 Jan (Hiver froid)", "date": "2017-01-18", "desc": "Températures < 4°C et chauffage"},
            {"label": "05 Juin (Ramadan)", "date": "2017-06-05", "desc": "Rupture du jeûne (iftar)"},
            {"label": "18 Nov (Fête Nationale)", "date": "2017-11-18", "desc": "Jour férié national"}
        ]


# ==============================================================================
# Interface Web Interactive (Dashboard HTML)
# ==============================================================================

@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """Application web interactive de pilotage énergétique multi-villes."""
    html_content = """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Plateforme de Prévision Énergétique Multi-Villes — Maroc</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Chart.js CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <!-- FontAwesome Icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        body { font-family: 'Inter', sans-serif; background-color: #0f172a; color: #f8fafc; }
        .card { background-color: #1e293b; border: 1px solid #334155; }
        .badge { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.3); }
        .tab-btn.active { background-color: #3b82f6; color: white; }
    </style>
</head>
<body class="min-h-screen pb-12">

    <!-- Header Navigation -->
    <header class="border-b border-slate-700 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex flex-wrap items-center justify-between gap-4">
            <div class="flex items-center space-x-3">
                <div class="p-2.5 bg-emerald-500/20 border border-emerald-500/40 rounded-xl">
                    <i class="fa-solid fa-bolt text-emerald-400 text-xl"></i>
                </div>
                <div>
                    <h1 class="text-lg font-bold tracking-tight text-white flex items-center gap-2">
                        Prévision Énergétique Multi-Territoires &bull; Maroc
                        <span class="text-xs px-2 py-0.5 rounded-full badge font-mono">ENSA Fès</span>
                    </h1>
                    <p class="text-xs text-slate-400">Modélisation Machine Learning : Tétouan &middot; Marrakech &middot; Laâyoune &middot; Transposition Fès</p>
                </div>
            </div>

            <!-- City Selector Pills -->
            <div class="flex items-center gap-2 bg-slate-800 p-1.5 rounded-xl border border-slate-700">
                <button onclick="selectCity('tetouan')" id="pillTetouan" class="px-3 py-1 rounded-lg text-xs font-semibold bg-blue-600 text-white transition flex items-center gap-1.5">
                    <i class="fa-solid fa-water text-cyan-300"></i> Tétouan (2017)
                </button>
                <button onclick="selectCity('marrakech')" id="pillMarrakech" class="px-3 py-1 rounded-lg text-xs font-semibold text-slate-400 hover:text-white transition flex items-center gap-1.5">
                    <i class="fa-solid fa-sun text-amber-400"></i> Marrakech (2023)
                </button>
                <button onclick="selectCity('laayoune')" id="pillLaayoune" class="px-3 py-1 rounded-lg text-xs font-semibold text-slate-400 hover:text-white transition flex items-center gap-1.5">
                    <i class="fa-solid fa-wind text-purple-400"></i> Laâyoune (2022-24)
                </button>
                <button onclick="selectCity('fes')" id="pillFes" class="px-3 py-1 rounded-lg text-xs font-semibold text-slate-400 hover:text-white transition flex items-center gap-1.5">
                    <i class="fa-solid fa-graduation-cap text-emerald-400"></i> Fès–Meknès (ENSA)
                </button>
            </div>

            <div class="flex items-center gap-3">
                <a href="/docs" target="_blank" class="text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 font-medium flex items-center gap-1.5">
                    <i class="fa-solid fa-code"></i> API Swagger
                </a>
            </div>
        </div>
    </header>

    <!-- Main Container -->
    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-6 space-y-6">

        <!-- Quick Presets Bar -->
        <div class="flex flex-wrap items-center justify-between gap-3 p-3.5 rounded-xl card">
            <div class="flex items-center gap-2 text-xs text-slate-400">
                <i class="fa-solid fa-calendar-day text-blue-400"></i>
                <span class="font-medium" id="presetsTitle">Scénarios représentatifs pour Tétouan :</span>
            </div>
            <div class="flex flex-wrap gap-2" id="presetsContainer">
                <!-- Injected via JS -->
            </div>
        </div>

        <!-- Metric KPI Cards -->
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div class="card p-4 rounded-xl">
                <div class="flex justify-between items-start">
                    <div>
                        <p class="text-xs font-medium text-slate-400 uppercase tracking-wider" id="kpiEnergyLabel">Charge Quotidienne Prévue</p>
                        <h3 class="text-2xl font-bold text-white mt-1" id="kpiEnergy">-</h3>
                    </div>
                    <div class="p-2.5 bg-blue-500/10 rounded-lg text-blue-400">
                        <i class="fa-solid fa-battery-three-quarters text-lg"></i>
                    </div>
                </div>
                <div class="mt-3 flex items-center text-xs text-slate-400">
                    <span>Observé : <strong class="text-slate-200" id="kpiObsEnergy">-</strong></span>
                </div>
            </div>

            <div class="card p-4 rounded-xl">
                <div class="flex justify-between items-start">
                    <div>
                        <p class="text-xs font-medium text-slate-400 uppercase tracking-wider">Pointe Journalière</p>
                        <h3 class="text-2xl font-bold text-amber-400 mt-1" id="kpiPeakLoad">-</h3>
                    </div>
                    <div class="p-2.5 bg-amber-500/10 rounded-lg text-amber-400">
                        <i class="fa-solid fa-chart-line text-lg"></i>
                    </div>
                </div>
                <div class="mt-3 flex items-center text-xs text-slate-400">
                    <span>Heure de pointe : <strong class="text-slate-200" id="kpiPeakHour">- h</strong></span>
                </div>
            </div>

            <div class="card p-4 rounded-xl">
                <div class="flex justify-between items-start">
                    <div>
                        <p class="text-xs font-medium text-slate-400 uppercase tracking-wider">Émissions Estimées</p>
                        <h3 class="text-2xl font-bold text-emerald-400 mt-1" id="kpiEmissions">-</h3>
                    </div>
                    <div class="p-2.5 bg-emerald-500/10 rounded-lg text-emerald-400">
                        <i class="fa-solid fa-smog text-lg"></i>
                    </div>
                </div>
                <div class="mt-3 flex items-center text-xs text-slate-400">
                    <span id="kpiEmissionsSub">Intensité carbone : 0.658 kg/kWh</span>
                </div>
            </div>

            <div class="card p-4 rounded-xl">
                <div class="flex justify-between items-start">
                    <div>
                        <p class="text-xs font-medium text-slate-400 uppercase tracking-wider">Modèle Certifié</p>
                        <h3 class="text-sm font-bold text-purple-400 mt-1" id="kpiModel">-</h3>
                    </div>
                    <div class="p-2.5 bg-purple-500/10 rounded-lg text-purple-400">
                        <i class="fa-solid fa-bullseye text-lg"></i>
                    </div>
                </div>
                <div class="mt-3 flex items-center text-xs text-emerald-400" id="kpiGain">
                    <i class="fa-solid fa-arrow-down mr-1"></i> Gain vs Persistance
                </div>
            </div>
        </div>

        <!-- Main Chart & Controls Section -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            
            <!-- Left Chart Area (2 Cols) -->
            <div class="lg:col-span-2 card p-5 rounded-xl space-y-4">
                <div class="flex flex-wrap items-center justify-between gap-3">
                    <div>
                        <h2 class="text-base font-semibold text-white flex items-center gap-2">
                            <span id="chartTitle">Courbe de Charge Horaire</span>
                            <span class="text-xs px-2 py-0.5 rounded bg-slate-700 text-slate-300 font-mono" id="currentDateDisplay">-</span>
                        </h2>
                        <p class="text-xs text-slate-400" id="chartSubtitle">Prévision J+1, observé réseau et intervalle à 90%</p>
                    </div>

                    <!-- Zone Switcher (Only for Tetouan) -->
                    <div id="zoneSwitchContainer" class="flex bg-slate-800 p-1 rounded-lg border border-slate-700 text-xs">
                        <button onclick="setZone('total')" id="btnZoneTotal" class="px-2.5 py-1 rounded font-medium bg-blue-600 text-white transition">Total</button>
                        <button onclick="setZone('zone1')" id="btnZone1" class="px-2.5 py-1 rounded font-medium text-slate-400 hover:text-white transition">Z1</button>
                        <button onclick="setZone('zone2')" id="btnZone2" class="px-2.5 py-1 rounded font-medium text-slate-400 hover:text-white transition">Z2</button>
                        <button onclick="setZone('zone3')" id="btnZone3" class="px-2.5 py-1 rounded font-medium text-slate-400 hover:text-white transition">Z3</button>
                    </div>
                </div>

                <!-- Canvas -->
                <div class="h-80 w-full relative">
                    <canvas id="forecastChart"></canvas>
                </div>

                <!-- Chart Legend Footer -->
                <div class="pt-2 border-t border-slate-800 flex flex-wrap items-center justify-between text-xs text-slate-400">
                    <div class="flex items-center gap-4">
                        <span class="flex items-center gap-1.5"><span class="w-3 h-3 rounded bg-blue-500 inline-block"></span> Prévision J+1</span>
                        <span class="flex items-center gap-1.5"><span class="w-3 h-3 rounded bg-emerald-400 inline-block"></span> Observé Réseau</span>
                        <span class="flex items-center gap-1.5"><span class="w-3 h-3 rounded bg-amber-500/40 inline-block"></span> Intervalle 90%</span>
                        <span id="whatifLegend" class="hidden flex items-center gap-1.5"><span class="w-3 h-3 rounded bg-purple-400 inline-block"></span> Simulation Météo</span>
                    </div>
                    <span class="italic text-[11px]" id="unitFootnote">Unité : Kilowatts (kW)</span>
                </div>
            </div>

            <!-- Right Interactive Simulator Controls (1 Col) -->
            <div class="card p-5 rounded-xl space-y-5">
                <h3 class="text-sm font-semibold text-white uppercase tracking-wider border-b border-slate-800 pb-2 flex items-center gap-2">
                    <i class="fa-solid fa-sliders text-blue-400"></i> Paramètres & Inférence
                </h3>

                <!-- Date Selector Form -->
                <div class="space-y-1.5">
                    <label class="text-xs font-medium text-slate-300">Date d'évaluation :</label>
                    <div class="flex gap-2">
                        <input type="date" id="inputDate"
                               class="bg-slate-900 border border-slate-700 text-white rounded-lg px-3 py-1.5 text-xs w-full focus:outline-none focus:border-blue-500">
                        <button onclick="updateForecast()" class="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 rounded-lg text-xs font-semibold text-white transition flex items-center gap-1">
                            <i class="fa-solid fa-arrows-rotate"></i>
                        </button>
                    </div>
                </div>

                <!-- Climate Sensitivity Slider (What-If) -->
                <div class="space-y-2 pt-2 border-t border-slate-800" id="climateSliderSection">
                    <div class="flex justify-between items-center text-xs">
                        <label class="font-medium text-slate-300 flex items-center gap-1.5">
                            <i class="fa-solid fa-temperature-half text-amber-400"></i> Simulation Thermique (What-If)
                        </label>
                        <span class="font-mono font-bold text-amber-400" id="deltaTempDisplay">0.0 °C</span>
                    </div>
                    <input type="range" id="sliderDeltaTemp" min="-5" max="5" step="0.5" value="0"
                           oninput="onTempSliderChange(this.value)"
                           class="w-full accent-amber-500 bg-slate-900 cursor-pointer">
                    <div class="flex justify-between text-[10px] text-slate-500">
                        <span>-5°C (Grand froid)</span>
                        <span>0°C (Observé)</span>
                        <span>+5°C (Canicule)</span>
                    </div>
                </div>

                <!-- Carbon Factor Slider -->
                <div class="space-y-2 pt-2 border-t border-slate-800" id="carbonSliderSection">
                    <div class="flex justify-between items-center text-xs">
                        <label class="font-medium text-slate-300 flex items-center gap-1.5">
                            <i class="fa-solid fa-leaf text-emerald-400"></i> Intensité Carbone du Réseau
                        </label>
                        <span class="font-mono font-bold text-emerald-400" id="factorDisplay">0.658 kg/kWh</span>
                    </div>
                    <input type="range" id="sliderFactor" min="0.400" max="0.800" step="0.010" value="0.658"
                           oninput="onFactorSliderChange(this.value)"
                           class="w-full accent-emerald-500 bg-slate-900 cursor-pointer">
                    <div class="flex justify-between text-[10px] text-slate-500">
                        <span>0.400 (Solaire/Éolien)</span>
                        <span>0.658 (Central)</span>
                        <span>0.720 (Haut)</span>
                    </div>
                </div>

                <!-- Transition Impact Summary Box -->
                <div class="bg-slate-900/80 p-3 rounded-lg border border-slate-800 text-xs space-y-1.5" id="transitionBox">
                    <p class="font-semibold text-slate-300 flex items-center gap-1.5">
                        <i class="fa-solid fa-solar-panel text-amber-400"></i> Écrêtement Solaire Diurne (15%)
                    </p>
                    <p class="text-emerald-400 font-mono font-semibold" id="solarSavingsText">
                        Calcul en cours...
                    </p>
                </div>

            </div>
        </div>

        <!-- Comparative Multi-City Cards -->
        <div class="card p-5 rounded-xl space-y-4">
            <h3 class="text-sm font-semibold text-white uppercase tracking-wider flex items-center gap-2">
                <i class="fa-solid fa-city text-blue-400"></i> Synthèse Comparative des Villes Marocaines
            </h3>
            <div class="grid grid-cols-1 md:grid-cols-4 gap-4 text-xs">
                
                <!-- Tetouan -->
                <div class="bg-slate-900/60 p-3.5 rounded-lg border border-slate-700/60 space-y-1.5">
                    <div class="flex justify-between items-center">
                        <h4 class="font-bold text-white">Tétouan</h4>
                        <span class="badge text-[10px] px-2 py-0.5 rounded">SCADA</span>
                    </div>
                    <p class="text-slate-400">Tanger-Tétouan &middot; Méditerranéen</p>
                    <div class="font-mono text-slate-300 pt-1 border-t border-slate-800 space-y-0.5">
                        <div>Charge moy : <strong>64.3 MW</strong></div>
                        <div>Pointe : <strong>132.8 MW (20h)</strong></div>
                        <div>Modèle : <strong>XGBoost écart (+36%)</strong></div>
                    </div>
                </div>

                <!-- Marrakech -->
                <div class="bg-slate-900/60 p-3.5 rounded-lg border border-slate-700/60 space-y-1.5">
                    <div class="flex justify-between items-center">
                        <h4 class="font-bold text-white">Marrakech</h4>
                        <span class="badge text-[10px] px-2 py-0.5 rounded">2023</span>
                    </div>
                    <p class="text-slate-400">Marrakech-Safi &middot; Semi-aride</p>
                    <div class="font-mono text-slate-300 pt-1 border-t border-slate-800 space-y-0.5">
                        <div>Charge moy : <strong>1.32 MW</strong></div>
                        <div>Pointe : <strong>2.51 MW (15h)</strong></div>
                        <div>Modèle : <strong>Linéaire (+20%)</strong></div>
                    </div>
                </div>

                <!-- Laayoune -->
                <div class="bg-slate-900/60 p-3.5 rounded-lg border border-slate-700/60 space-y-1.5">
                    <div class="flex justify-between items-center">
                        <h4 class="font-bold text-white">Laâyoune</h4>
                        <span class="badge text-[10px] px-2 py-0.5 rounded">22 kV</span>
                    </div>
                    <p class="text-slate-400">Sahara &middot; Océanique</p>
                    <div class="font-mono text-slate-300 pt-1 border-t border-slate-800 space-y-0.5">
                        <div>Courant moy : <strong>561 A</strong></div>
                        <div>Pointe : <strong>1 284 A (21h)</strong></div>
                        <div>Modèle : <strong>Linéaire (+11%)</strong></div>
                    </div>
                </div>

                <!-- Fes -->
                <div class="bg-slate-900/60 p-3.5 rounded-lg border border-blue-500/40 space-y-1.5">
                    <div class="flex justify-between items-center">
                        <h4 class="font-bold text-white">Fès–Meknès</h4>
                        <span class="badge text-[10px] px-2 py-0.5 rounded bg-blue-500/20 text-blue-300 border-blue-500/40">ENSA Fès</span>
                    </div>
                    <p class="text-slate-400">Continental méditerranéen</p>
                    <div class="font-mono text-slate-300 pt-1 border-t border-slate-800 space-y-0.5">
                        <div>Transposition méthodologique</div>
                        <div>Sensibilité thermique extrême</div>
                        <div>Chauffage hiver & Clim été</div>
                    </div>
                </div>

            </div>
        </div>

    </main>

    <!-- Embedded Client JS -->
    <script>
        let currentCity = 'tetouan';
        let currentChart = null;
        let currentData = null;
        let currentZone = 'total';
        let currentDeltaTemp = 0.0;
        let currentFactor = 0.658;

        const cityDates = {
            tetouan: { default: "2017-11-15", min: "2017-01-08", max: "2017-12-30" },
            marrakech: { default: "2023-11-15", min: "2023-01-16", max: "2024-01-08" },
            laayoune: { default: "2023-07-15", min: "2022-09-22", max: "2023-09-14" },
            fes: { default: "2023-11-15", min: "2023-01-16", max: "2023-12-31" }
        };

        function selectCity(city) {
            currentCity = city;
            ['pillTetouan', 'pillMarrakech', 'pillLaayoune', 'pillFes'].forEach(id => {
                const el = document.getElementById(id);
                el.className = 'px-3 py-1 rounded-lg text-xs font-semibold text-slate-400 hover:text-white transition flex items-center gap-1.5';
            });
            if (city === 'tetouan') document.getElementById('pillTetouan').className = 'px-3 py-1 rounded-lg text-xs font-semibold bg-blue-600 text-white transition flex items-center gap-1.5';
            if (city === 'marrakech') document.getElementById('pillMarrakech').className = 'px-3 py-1 rounded-lg text-xs font-semibold bg-blue-600 text-white transition flex items-center gap-1.5';
            if (city === 'laayoune') document.getElementById('pillLaayoune').className = 'px-3 py-1 rounded-lg text-xs font-semibold bg-blue-600 text-white transition flex items-center gap-1.5';
            if (city === 'fes') document.getElementById('pillFes').className = 'px-3 py-1 rounded-lg text-xs font-semibold bg-blue-600 text-white transition flex items-center gap-1.5';

            // Show/hide zone switcher
            const zSwitch = document.getElementById('zoneSwitchContainer');
            if (city === 'tetouan') zSwitch.classList.remove('hidden'); else zSwitch.classList.add('hidden');

            // Configure date input
            const dCfg = cityDates[city];
            const inp = document.getElementById('inputDate');
            inp.min = dCfg.min;
            inp.max = dCfg.max;
            inp.value = dCfg.default;

            loadPresets();
            updateForecast();
        }

        async function loadPresets() {
            const res = await fetch(`/api/presets?city=${currentCity}`);
            const presets = await res.json();
            const container = document.getElementById('presetsContainer');
            container.innerHTML = '';
            document.getElementById('presetsTitle').innerText = `Scénarios représentatifs pour ${currentCity.toUpperCase()} :`;

            presets.forEach(p => {
                const btn = document.createElement('button');
                btn.className = 'text-xs px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 transition border border-slate-700';
                btn.innerText = p.label;
                btn.onclick = () => {
                    document.getElementById('inputDate').value = p.date;
                    updateForecast();
                };
                container.appendChild(btn);
            });
        }

        async function updateForecast() {
            const date = document.getElementById('inputDate').value;
            document.getElementById('currentDateDisplay').innerText = `${currentCity.toUpperCase()} &bull; ${date}`;
            
            try {
                const url = `/forecast?city=${currentCity}&date=${date}&delta_temp=${currentDeltaTemp}&emission_factor=${currentFactor}`;
                const res = await fetch(url);
                if (!res.ok) {
                    const err = await res.json();
                    alert("Erreur : " + err.detail);
                    return;
                }
                currentData = await res.json();
                renderKPIs(currentData);
                renderChart(currentData);
            } catch (e) {
                console.error("Erreur de chargement :", e);
            }
        }

        function renderKPIs(d) {
            const unit = d.unite;
            const energyUnit = d.energy_unit;
            
            const totalVal = unit === 'kW' ? (d.total_prevu / 1000).toFixed(1) : d.total_prevu.toFixed(1);
            document.getElementById('kpiEnergy').innerText = `${Number(totalVal).toLocaleString('fr-FR')} ${energyUnit}`;
            
            if (d.total_observe) {
                const obsVal = unit === 'kW' ? (d.total_observe / 1000).toFixed(1) : d.total_observe.toFixed(1);
                document.getElementById('kpiObsEnergy').innerText = `${Number(obsVal).toLocaleString('fr-FR')} ${energyUnit}`;
            } else {
                document.getElementById('kpiObsEnergy').innerText = "Non disponible";
            }

            // Peak
            let maxVal = -1, peakHour = 0;
            d.prevision.forEach((val, h) => {
                if (val > maxVal) { maxVal = val; peakHour = h; }
            });
            document.getElementById('kpiPeakLoad').innerText = `${Math.round(maxVal).toLocaleString('fr-FR')} ${unit}`;
            document.getElementById('kpiPeakHour').innerText = `${peakHour}h00`;

            // Model KPI
            document.getElementById('kpiModel').innerText = d.modele;
            if (currentCity === 'tetouan') {
                document.getElementById('kpiGain').innerHTML = '<i class="fa-solid fa-arrow-down mr-1"></i> -36.1% vs persistance (RMSE)';
            } else if (currentCity === 'marrakech') {
                document.getElementById('kpiGain').innerHTML = '<i class="fa-solid fa-arrow-down mr-1"></i> -20.0% vs persistance (RMSE)';
            } else if (currentCity === 'laayoune') {
                document.getElementById('kpiGain').innerHTML = '<i class="fa-solid fa-arrow-down mr-1"></i> -11.4% vs persistance (RMSE)';
            } else {
                document.getElementById('kpiGain').innerHTML = '<i class="fa-solid fa-arrow-down mr-1"></i> -70.6% vs persistance (RMSE)';
            }

            // Emissions
            if (d.emissions_estimees_tCO2) {
                document.getElementById('carbonSliderSection').classList.remove('hidden');
                document.getElementById('transitionBox').classList.remove('hidden');
                const emCentral = d.emissions_estimees_tCO2.central || d.emissions_estimees_tCO2.personnalise;
                document.getElementById('kpiEmissions').innerText = `${Number(emCentral).toLocaleString('fr-FR')} tCO₂`;
                document.getElementById('kpiEmissionsSub').innerText = `Scénarios : ${d.emissions_estimees_tCO2.bas} à ${d.emissions_estimees_tCO2.haut} t`;
                
                const solarMwh = (totalVal * 0.15).toFixed(1);
                const solarCo2 = (solarMwh * currentFactor).toFixed(1);
                document.getElementById('solarSavingsText').innerHTML = `&approx; -${solarMwh} ${energyUnit} &bull; -${solarCo2} tCO₂ / jour`;
            } else {
                document.getElementById('carbonSliderSection').classList.add('hidden');
                document.getElementById('transitionBox').classList.add('hidden');
                document.getElementById('kpiEmissions').innerText = "Non applicable";
                document.getElementById('kpiEmissionsSub').innerText = "Mesures en Ampères (non kWh)";
            }
        }

        function renderChart(d) {
            const ctx = document.getElementById('forecastChart').getContext('2d');
            const hours = Array.from({length: 24}, (_, i) => `${i}h`);
            const unit = d.unite;
            document.getElementById('unitFootnote').innerText = `Unité : ${unit === 'kW' ? 'Kilowatts (kW)' : 'Ampères (A)'}`;

            let seriesPred = d.prevision;
            let seriesLower = d.prevision_lower_90;
            let seriesUpper = d.prevision_upper_90;
            let seriesObs = d.observe;

            if (currentCity === 'tetouan' && d.zones) {
                if (currentZone === 'zone1') {
                    seriesPred = d.zones.zone1_kw;
                    seriesLower = seriesPred.map(v => v * 0.95);
                    seriesUpper = seriesPred.map(v => v * 1.05);
                    seriesObs = d.observe.map(v => v * 0.48);
                } else if (currentZone === 'zone2') {
                    seriesPred = d.zones.zone2_kw;
                    seriesLower = seriesPred.map(v => v * 0.95);
                    seriesUpper = seriesPred.map(v => v * 1.05);
                    seriesObs = d.observe.map(v => v * 0.313);
                } else if (currentZone === 'zone3') {
                    seriesPred = d.zones.zone3_kw;
                    seriesLower = seriesPred.map(v => v * 0.95);
                    seriesUpper = seriesPred.map(v => v * 1.05);
                    seriesObs = d.observe.map(v => v * 0.207);
                }
            }

            if (currentChart) currentChart.destroy();

            const datasets = [
                {
                    label: `Prévision (${unit})`,
                    data: seriesPred,
                    borderColor: '#3b82f6',
                    backgroundColor: '#3b82f6',
                    borderWidth: 2.5,
                    tension: 0.3,
                    pointRadius: 2,
                    zIndex: 10
                },
                {
                    label: `Observé (${unit})`,
                    data: seriesObs,
                    borderColor: '#10b981',
                    backgroundColor: '#10b981',
                    borderWidth: 2,
                    borderDash: [4, 4],
                    tension: 0.3,
                    pointRadius: 2,
                    zIndex: 9
                },
                {
                    label: 'Borne haute 90%',
                    data: seriesUpper,
                    borderColor: 'rgba(245, 158, 11, 0.1)',
                    backgroundColor: 'rgba(245, 158, 11, 0.15)',
                    borderWidth: 0,
                    fill: '+1',
                    pointRadius: 0
                },
                {
                    label: 'Borne basse 90%',
                    data: seriesLower,
                    borderColor: 'rgba(245, 158, 11, 0.1)',
                    backgroundColor: 'transparent',
                    borderWidth: 0,
                    pointRadius: 0
                }
            ];

            if (Math.abs(currentDeltaTemp) > 1e-3 && currentCity !== 'laayoune') {
                document.getElementById('whatifLegend').classList.remove('hidden');
                datasets.push({
                    label: 'Sans anomalie thermique',
                    data: d.prevision_base,
                    borderColor: '#c084fc',
                    borderWidth: 1.5,
                    borderDash: [2, 2],
                    pointRadius: 0
                });
            } else {
                document.getElementById('whatifLegend').classList.add('hidden');
            }

            currentChart = new Chart(ctx, {
                type: 'line',
                data: { labels: hours, datasets: datasets },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: { mode: 'index', intersect: false },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: '#0f172a',
                            titleColor: '#f8fafc',
                            borderColor: '#334155',
                            borderWidth: 1,
                            padding: 10
                        }
                    },
                    scales: {
                        x: {
                            grid: { color: 'rgba(255, 255, 255, 0.05)' },
                            ticks: { color: '#94a3b8', font: { size: 10 } }
                        },
                        y: {
                            grid: { color: 'rgba(255, 255, 255, 0.05)' },
                            ticks: {
                                color: '#94a3b8',
                                font: { size: 10 },
                                callback: val => (unit === 'kW' && val >= 1000) ? (val / 1000).toFixed(0) + ' MW' : val + ' ' + unit
                            }
                        }
                    }
                }
            });
        }

        function setZone(z) {
            currentZone = z;
            ['btnZoneTotal', 'btnZone1', 'btnZone2', 'btnZone3'].forEach(id => {
                document.getElementById(id).className = 'px-2.5 py-1 rounded font-medium text-slate-400 hover:text-white transition';
            });
            if (z === 'total') document.getElementById('btnZoneTotal').className = 'px-2.5 py-1 rounded font-medium bg-blue-600 text-white transition';
            if (z === 'zone1') document.getElementById('btnZone1').className = 'px-2.5 py-1 rounded font-medium bg-blue-600 text-white transition';
            if (z === 'zone2') document.getElementById('btnZone2').className = 'px-2.5 py-1 rounded font-medium bg-blue-600 text-white transition';
            if (z === 'zone3') document.getElementById('btnZone3').className = 'px-2.5 py-1 rounded font-medium bg-blue-600 text-white transition';
            if (currentData) renderChart(currentData);
        }

        function onTempSliderChange(val) {
            currentDeltaTemp = parseFloat(val);
            document.getElementById('deltaTempDisplay').innerText = (currentDeltaTemp > 0 ? '+' : '') + currentDeltaTemp.toFixed(1) + ' °C';
            updateForecast();
        }

        function onFactorSliderChange(val) {
            currentFactor = parseFloat(val);
            document.getElementById('factorDisplay').innerText = currentFactor.toFixed(3) + ' kg/kWh';
            if (currentData && currentData.unite === 'kW') {
                const kwh = currentData.total_prevu;
                const co2 = (kwh * currentFactor / 1000).toFixed(1);
                document.getElementById('kpiEmissions').innerText = `${Number(co2).toLocaleString('fr-FR')} tCO₂`;
            }
        }

        window.addEventListener('DOMContentLoaded', () => {
            selectCity('tetouan');
        });
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content, status_code=200)
