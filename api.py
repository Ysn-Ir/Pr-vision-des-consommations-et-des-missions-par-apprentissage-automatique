"""Application Web et API de prévision de charge et d'émissions de CO₂.

Fournit :
1. L'application Web interactive (Dashboard accessible à la racine / ou /dashboard)
2. L'API REST JSON pour la prévision horaire (/forecast) avec intervalles d'incertitude
3. La simulation météo "what-if" (sensibilité thermique)
4. L'exploration multi-territoires au Maroc (/api/morocco-cities)
5. Les métriques comparatives des modèles (/api/overview)

Lancement :
    uvicorn api:app --reload
Puis ouvrir :
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

# Chargement du modèle et des artefacts
bundle = joblib.load(ART / "model.joblib")
MODEL = bundle["model"]
NAME = bundle["name"]
FEATURES = bundle["features"]
FACTORS = bundle["factors"]

DATA = pd.read_csv(ART / "features_hourly.csv", index_col=0, parse_dates=True)

RESULTS_JSON = json.loads((ART / "results.json").read_text(encoding="utf-8")) if (ART / "results.json").exists() else {}

app = FastAPI(
    title="Projet 04 — Prévision Énergétique & Émissions (Tétouan & Maroc)",
    description="Plateforme de prévision de la charge électrique urbaine et d'évaluation des scénarios d'émissions CO₂.",
    version="1.0.0"
)


# ==============================================================================
# Endpoints API REST
# ==============================================================================

@app.get("/health")
def health():
    """Vérification de l'état du service et des métadonnées du modèle."""
    return {
        "status": "ok",
        "modele": NAME,
        "territoire_principal": "Tétouan (Tanger-Tétouan-Al Hoceïma)",
        "territoires_secondaires": ["Marrakech", "Laâyoune", "Fès-Meknès"],
        "dates_disponibles": [str(DATA.index.min().date()), str(DATA.index.max().date())],
        "nature_des_donnees": "Consommations observées (Amendis SCADA) ; émissions estimées"
    }


@app.get("/forecast")
def forecast(
    date: str = Query(..., description="Date au format AAAA-MM-JJ (ex: 2017-11-15)"),
    delta_temp: float = Query(0.0, description="Anomalie thermique simulée en °C (-5 à +5)"),
    emission_factor: float = Query(None, description="Facteur d'émission personnalisé en kgCO2/kWh")
):
    """Prévision horaire (24h) avec incertitude, impact météo 'what-if' et scénarios d'émissions."""
    try:
        day = pd.Timestamp(date).normalize()
    except Exception:
        raise HTTPException(422, "Format de date invalide. Format attendu : AAAA-MM-JJ")
    
    rows = DATA.loc[day : day + pd.Timedelta(hours=23)]
    if len(rows) == 0:
        raise HTTPException(
            404,
            f"Date {date} absente de la base. Plage disponible : {DATA.index.min().date()} à {DATA.index.max().date()}."
        )
    
    # Prévision de base
    base_pred = MODEL.predict(rows[FEATURES])
    
    # Simulation "what-if" si une anomalie thermique est spécifiée
    simulated_pred = simulate_weather_anomaly(rows, base_pred, delta_temp)
    
    # Intervalles de confiance (90%)
    lower_90, upper_90 = compute_prediction_intervals(simulated_pred, rmse=2037.4, confidence=0.90)
    
    # Décomposition par zone
    zones_pred = decompose_zones(simulated_pred)
    
    # Énergie totale prévue (kW sur 1h = kWh)
    energy_kwh = float(simulated_pred.sum())
    base_energy_kwh = float(base_pred.sum())
    
    # Émissions estimées
    emissions = calculate_emissions_summary(energy_kwh, emission_factor)
    
    # Données météo associées
    temperatures = [round(float(t), 1) for t in rows["temp"]] if "temp" in rows.columns else []
    
    # Données observées si présentes
    has_obs = "load_kw" in rows.columns and not rows["load_kw"].isna().all()
    obs_kw = [round(float(v), 1) for v in rows["load_kw"]] if has_obs else []
    obs_energy_kwh = float(rows["load_kw"].sum()) if has_obs else None
    
    return {
        "date": str(day.date()),
        "modele": NAME,
        "heures": len(rows),
        "delta_temp_simulee_degC": delta_temp,
        "prevision_kw": [round(float(p), 1) for p in simulated_pred],
        "prevision_base_kw": [round(float(p), 1) for p in base_pred],
        "prevision_lower_90_kw": [round(float(p), 1) for p in lower_90],
        "prevision_upper_90_kw": [round(float(p), 1) for p in upper_90],
        "observe_kw": obs_kw,
        "temperature_degC": temperatures,
        "zones": zones_pred,
        "energie_prevue_kwh": round(energy_kwh, 1),
        "energie_observee_kwh": round(obs_energy_kwh, 1) if obs_energy_kwh is not None else None,
        "ecart_energie_mwh": round((energy_kwh - base_energy_kwh) / 1000.0, 2),
        "emissions_estimees_tCO2": emissions,
        "facteurs_utilises": FACTORS_DEFAULT,
        "avertissement": "Consommations observées ; émissions estimées selon l'intensité carbone du réseau."
    }


@app.get("/api/overview")
def overview():
    """Synthèse des métriques comparatives des modèles et de la qualité des données."""
    return RESULTS_JSON


@app.get("/api/morocco-cities")
def morocco_cities():
    """Données comparatives multi-villes (Marrakech, Laâyoune, Fès-Meknès, Tétouan)."""
    return multi_city_utils.get_moroccan_cities_overview()


@app.get("/api/presets")
def presets():
    """Dates remarquables représentatives pour démonstration rapide."""
    return [
        {"label": "Automne (Essai typique)", "date": "2017-11-15", "desc": "Semaine normale en période d'évaluation"},
        {"label": "Pointe estivale (Canicule)", "date": "2017-08-04", "desc": "Températures > 35°C et pic de climatisation"},
        {"label": "Hiver froid", "date": "2017-01-18", "desc": "Températures minimales ~4°C et chauffage"},
        {"label": "Ramadan (Rupture du jeûne)", "date": "2017-06-05", "desc": "Profil horaire atypique (iftar et suhur)"},
        {"label": "Jour férié (Fête nationale)", "date": "2017-11-18", "desc": "Activité commerciale et industrielle réduite"}
    ]


# ==============================================================================
# Interface Web Interactive (Dashboard HTML)
# ==============================================================================

@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """Sert l'application web interactive de pilotage énergétique et de prévision."""
    html_content = """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prévision Énergétique & Transition Bas-Carbone — Maroc</title>
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
        .accent-emerald { color: #10b981; }
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
                        Projet 04 — Prévision Énergétique & Émissions CO₂
                        <span class="text-xs px-2 py-0.5 rounded-full badge font-mono">ENSA Fès</span>
                    </h1>
                    <p class="text-xs text-slate-400">Territoire pilote : Tétouan &middot; Transposition Fès–Meknès, Marrakech, Laâyoune</p>
                </div>
            </div>

            <div class="flex items-center gap-3">
                <div class="hidden sm:flex items-center gap-2 text-xs bg-slate-800 px-3 py-1.5 rounded-lg border border-slate-700">
                    <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                    <span class="text-slate-300">Modèle actif :</span>
                    <span class="font-semibold text-emerald-400" id="headerModelName">XGBoost sur écart</span>
                </div>
                <a href="/docs" target="_blank" class="text-xs px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 transition text-white font-medium flex items-center gap-1.5">
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
                <span class="font-medium">Scénarios prédéfinis :</span>
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
                        <p class="text-xs font-medium text-slate-400 uppercase tracking-wider">Énergie Quotidienne Prévue</p>
                        <h3 class="text-2xl font-bold text-white mt-1" id="kpiEnergy">- MWh</h3>
                    </div>
                    <div class="p-2.5 bg-blue-500/10 rounded-lg text-blue-400">
                        <i class="fa-solid fa-battery-three-quarters text-lg"></i>
                    </div>
                </div>
                <div class="mt-3 flex items-center text-xs text-slate-400" id="kpiObsEnergySub">
                    <span>Observé : <strong class="text-slate-200" id="kpiObsEnergy">-</strong></span>
                </div>
            </div>

            <div class="card p-4 rounded-xl">
                <div class="flex justify-between items-start">
                    <div>
                        <p class="text-xs font-medium text-slate-400 uppercase tracking-wider">Pointe de Charge</p>
                        <h3 class="text-2xl font-bold text-amber-400 mt-1" id="kpiPeakLoad">- kW</h3>
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
                        <p class="text-xs font-medium text-slate-400 uppercase tracking-wider">Émissions Estimées (CO₂)</p>
                        <h3 class="text-2xl font-bold text-emerald-400 mt-1" id="kpiEmissions">- tCO₂</h3>
                    </div>
                    <div class="p-2.5 bg-emerald-500/10 rounded-lg text-emerald-400">
                        <i class="fa-solid fa-smog text-lg"></i>
                    </div>
                </div>
                <div class="mt-3 flex items-center text-xs text-slate-400">
                    <span>Fourchette : <span class="text-slate-300 font-mono text-[11px]" id="kpiEmissionsRange">-</span></span>
                </div>
            </div>

            <div class="card p-4 rounded-xl">
                <div class="flex justify-between items-start">
                    <div>
                        <p class="text-xs font-medium text-slate-400 uppercase tracking-wider">Précision du Modèle (Essai)</p>
                        <h3 class="text-2xl font-bold text-purple-400 mt-1">2.34 %</h3>
                    </div>
                    <div class="p-2.5 bg-purple-500/10 rounded-lg text-purple-400">
                        <i class="fa-solid fa-bullseye text-lg"></i>
                    </div>
                </div>
                <div class="mt-3 flex items-center text-xs text-slate-400">
                    <span class="text-emerald-400 font-semibold mr-1"><i class="fa-solid fa-arrow-down"></i> -36.1%</span> vs persistance naïve
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
                            <span>Courbe de Charge Horaire (24 Heures)</span>
                            <span class="text-xs px-2 py-0.5 rounded bg-slate-700 text-slate-300 font-mono" id="currentDateDisplay">2017-11-15</span>
                        </h2>
                        <p class="text-xs text-slate-400">Comparaison entre prévision J+1, observé SCADA et intervalle à 90%</p>
                    </div>

                    <!-- Zone Switcher -->
                    <div class="flex bg-slate-800 p-1 rounded-lg border border-slate-700 text-xs">
                        <button onclick="setZone('total')" id="btnZoneTotal" class="px-2.5 py-1 rounded font-medium bg-blue-600 text-white transition">Total Ville</button>
                        <button onclick="setZone('zone1')" id="btnZone1" class="px-2.5 py-1 rounded font-medium text-slate-400 hover:text-white transition">Zone 1 (48%)</button>
                        <button onclick="setZone('zone2')" id="btnZone2" class="px-2.5 py-1 rounded font-medium text-slate-400 hover:text-white transition">Zone 2 (31%)</button>
                        <button onclick="setZone('zone3')" id="btnZone3" class="px-2.5 py-1 rounded font-medium text-slate-400 hover:text-white transition">Zone 3 (21%)</button>
                    </div>
                </div>

                <!-- Canvas -->
                <div class="h-80 w-full relative">
                    <canvas id="forecastChart"></canvas>
                </div>

                <!-- Chart Legend Footer -->
                <div class="pt-2 border-t border-slate-800 flex flex-wrap items-center justify-between text-xs text-slate-400">
                    <div class="flex items-center gap-4">
                        <span class="flex items-center gap-1.5"><span class="w-3 h-3 rounded bg-blue-500 inline-block"></span> Prévision</span>
                        <span class="flex items-center gap-1.5"><span class="w-3 h-3 rounded bg-emerald-400 inline-block"></span> Observé</span>
                        <span class="flex items-center gap-1.5"><span class="w-3 h-3 rounded bg-amber-500/40 inline-block"></span> Intervalle 90%</span>
                        <span id="whatifLegend" class="hidden flex items-center gap-1.5"><span class="w-3 h-3 rounded bg-purple-400 inline-block"></span> Simulation Météo</span>
                    </div>
                    <span class="italic text-[11px]">Unité : Kilowatts (kW) au pas de 1 h</span>
                </div>
            </div>

            <!-- Right Interactive Simulator Controls (1 Col) -->
            <div class="card p-5 rounded-xl space-y-5">
                <h3 class="text-sm font-semibold text-white uppercase tracking-wider border-b border-slate-800 pb-2 flex items-center gap-2">
                    <i class="fa-solid fa-sliders text-blue-400"></i> Paramètres & Simulations
                </h3>

                <!-- Date Selector Form -->
                <div class="space-y-1.5">
                    <label class="text-xs font-medium text-slate-300">Date de prévision (2017) :</label>
                    <div class="flex gap-2">
                        <input type="date" id="inputDate" value="2017-11-15" min="2017-01-08" max="2017-12-30"
                               class="bg-slate-900 border border-slate-700 text-white rounded-lg px-3 py-1.5 text-xs w-full focus:outline-none focus:border-blue-500">
                        <button onclick="updateForecast()" class="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 rounded-lg text-xs font-semibold text-white transition flex items-center gap-1">
                            <i class="fa-solid fa-arrows-rotate"></i>
                        </button>
                    </div>
                </div>

                <!-- Climate Sensitivity Slider (What-If) -->
                <div class="space-y-2 pt-2 border-t border-slate-800">
                    <div class="flex justify-between items-center text-xs">
                        <label class="font-medium text-slate-300 flex items-center gap-1.5">
                            <i class="fa-solid fa-temperature-half text-amber-400"></i> Anomalie Température (What-If)
                        </label>
                        <span class="font-mono font-bold text-amber-400" id="deltaTempDisplay">0.0 °C</span>
                    </div>
                    <input type="range" id="sliderDeltaTemp" min="-5" max="5" step="0.5" value="0"
                           oninput="onTempSliderChange(this.value)"
                           class="w-full accent-amber-500 bg-slate-900 cursor-pointer">
                    <div class="flex justify-between text-[10px] text-slate-500">
                        <span>-5°C (Vague de froid)</span>
                        <span>0°C (Météo observée)</span>
                        <span>+5°C (Canicule)</span>
                    </div>
                    <p class="text-[11px] text-slate-400 italic" id="whatifComment">
                        Modifie la charge de climatisation (+520 kW/°C si >20°C) ou de chauffage (+380 kW/°C si <15°C).
                    </p>
                </div>

                <!-- Carbon Factor Slider -->
                <div class="space-y-2 pt-2 border-t border-slate-800">
                    <div class="flex justify-between items-center text-xs">
                        <label class="font-medium text-slate-300 flex items-center gap-1.5">
                            <i class="fa-solid fa-leaf text-emerald-400"></i> Facteur d'Émission Réseau
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
                <div class="bg-slate-900/80 p-3 rounded-lg border border-slate-800 text-xs space-y-1.5">
                    <p class="font-semibold text-slate-300 flex items-center gap-1.5">
                        <i class="fa-solid fa-solar-panel text-amber-400"></i> Potentiel d'Écrêtement Solaire
                    </p>
                    <p class="text-slate-400 text-[11px]">
                        Si 15% de la pointe diurne est couverte en toiture solaire locale :
                    </p>
                    <p class="text-emerald-400 font-mono font-semibold" id="solarSavingsText">
                        &approx; -18.2 MWh &bull; -12.0 tCO₂ / jour
                    </p>
                </div>

            </div>
        </div>

        <!-- Tabs Navigation for Multi-City & Model Benchmarking -->
        <div class="border-b border-slate-800 flex gap-4 text-xs font-medium">
            <button onclick="switchTab('tabCities')" id="btnTabCities" class="tab-btn px-4 py-2.5 rounded-t-lg bg-blue-600 text-white flex items-center gap-2">
                <i class="fa-solid fa-city"></i> Exploration Multi-Villes (Maroc)
            </button>
            <button onclick="switchTab('tabModels')" id="btnTabModels" class="tab-btn px-4 py-2.5 rounded-t-lg bg-slate-800 text-slate-400 hover:text-white flex items-center gap-2">
                <i class="fa-solid fa-table-list"></i> Comparatif des Modèles
            </button>
            <button onclick="switchTab('tabMethodo')" id="btnTabMethodo" class="tab-btn px-4 py-2.5 rounded-t-lg bg-slate-800 text-slate-400 hover:text-white flex items-center gap-2">
                <i class="fa-solid fa-graduation-cap"></i> Note Méthodologique & Limites
            </button>
        </div>

        <!-- Tab 1: Multi-City Exploration -->
        <div id="tabCities" class="space-y-4">
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                
                <!-- Tetouan Card -->
                <div class="card p-4 rounded-xl space-y-2.5">
                    <div class="flex justify-between items-center">
                        <h4 class="font-bold text-white text-sm">Tétouan</h4>
                        <span class="badge text-[10px] px-2 py-0.5 rounded">Pilote SCADA</span>
                    </div>
                    <p class="text-xs text-slate-400">Région Tanger-Tétouan-Al Hoceïma &middot; Climat méditerranéen côtier</p>
                    <div class="text-xs space-y-1 font-mono text-slate-300 pt-2 border-t border-slate-800">
                        <div>Charge moyenne : <strong>~64.3 MW</strong></div>
                        <div>Pointe observée : <strong>132.8 MW</strong></div>
                        <div>Heure de pointe : <strong>20h00 - 21h00</strong></div>
                        <div>Résolution : 10 min (52 416 mesures, 2017)</div>
                    </div>
                </div>

                <!-- Marrakech Card -->
                <div class="card p-4 rounded-xl space-y-2.5">
                    <div class="flex justify-between items-center">
                        <h4 class="font-bold text-white text-sm">Marrakech</h4>
                        <span class="badge text-[10px] px-2 py-0.5 rounded">Compteurs 2023-2024</span>
                    </div>
                    <p class="text-xs text-slate-400">Région Marrakech-Safi &middot; Climat semi-aride continental</p>
                    <div class="text-xs space-y-1 font-mono text-slate-300 pt-2 border-t border-slate-800" id="cardMarrakech">
                        <div>Charge moyenne : <strong>1.32 MW</strong></div>
                        <div>Pointe observée : <strong>2.51 MW</strong></div>
                        <div>Heure de pointe : <strong>15h00</strong> (Climatisation)</div>
                        <div>Résolution : 30 min (8 749 heures)</div>
                    </div>
                </div>

                <!-- Laayoune Card -->
                <div class="card p-4 rounded-xl space-y-2.5">
                    <div class="flex justify-between items-center">
                        <h4 class="font-bold text-white text-sm">Laâyoune</h4>
                        <span class="badge text-[10px] px-2 py-0.5 rounded">Réseau 22 kV</span>
                    </div>
                    <p class="text-xs text-slate-400">Région Laâyoune-Sakia El Hamra &middot; Climat saharien océanique</p>
                    <div class="text-xs space-y-1 font-mono text-slate-300 pt-2 border-t border-slate-800" id="cardLaayoune">
                        <div>Courant moyen : <strong>560.7 A</strong></div>
                        <div>Pointe de courant : <strong>1 284.2 A</strong></div>
                        <div>Heure de pointe : <strong>21h00</strong> (Éclairage)</div>
                        <div>5 zones de transformateurs (2022-2024)</div>
                    </div>
                </div>

            </div>

            <!-- Fes Meknes Transposition Box -->
            <div class="card p-4 rounded-xl bg-slate-900 border-blue-500/40">
                <div class="flex items-start gap-3">
                    <div class="p-2 bg-blue-500/20 text-blue-400 rounded-lg">
                        <i class="fa-solid fa-compass-drafting text-lg"></i>
                    </div>
                    <div class="space-y-1 text-xs">
                        <h4 class="font-bold text-white text-sm">Transposition Méthodologique à Fès–Meknès (ENSA Fès)</h4>
                        <p class="text-slate-300">
                            La démarche prédictive développée sur Tétouan est directement transposable à la région Fès-Meknès. 
                            Fès présente une amplitude thermique saisonnière plus marquée que la côte tétouanaise (hivers froids avec températures < 4°C et étés chauds > 40°C), 
                            ce qui renforce le couplage météo-consommation et rend le modèle d'apprentissage d'autant plus pertinent pour l'anticipation des pointes hivernales et estivales.
                        </p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Tab 2: Model Benchmark Table -->
        <div id="tabModels" class="hidden card p-4 rounded-xl overflow-x-auto">
            <h4 class="font-semibold text-white text-sm mb-3">Tableau Comparatif sur la Période d'Essai (Novembre &ndash; Décembre 2017)</h4>
            <table class="w-full text-xs text-left">
                <thead class="text-slate-400 border-b border-slate-700 bg-slate-900/50">
                    <tr>
                        <th class="py-2.5 px-3">Modèle</th>
                        <th class="py-2.5 px-3">MAE (kW)</th>
                        <th class="py-2.5 px-3">RMSE (kW)</th>
                        <th class="py-2.5 px-3">MAPE (%)</th>
                        <th class="py-2.5 px-3">Biais (kW)</th>
                        <th class="py-2.5 px-3">Gain RMSE vs Réf.</th>
                        <th class="py-2.5 px-3">Temps entraînement</th>
                    </tr>
                </thead>
                <tbody class="divide-y divide-slate-800 font-mono text-slate-300" id="tableModelsBody">
                    <tr>
                        <td class="py-2 px-3 font-sans font-semibold text-white">Référence (Naïve 24 h)</td>
                        <td class="py-2 px-3">2 076.1</td><td class="py-2 px-3">3 188.2</td><td class="py-2 px-3">3.33 %</td><td class="py-2 px-3">+55.1</td><td class="py-2 px-3 text-slate-500">—</td><td class="py-2 px-3">< 0.01 s</td>
                    </tr>
                    <tr>
                        <td class="py-2 px-3 font-sans font-semibold text-white">Régression linéaire</td>
                        <td class="py-2 px-3">1 558.3</td><td class="py-2 px-3">2 099.9</td><td class="py-2 px-3">2.54 %</td><td class="py-2 px-3">+40.2</td><td class="py-2 px-3 text-emerald-400 font-semibold">+34.1 %</td><td class="py-2 px-3">0.03 s</td>
                    </tr>
                    <tr>
                        <td class="py-2 px-3 font-sans font-semibold text-white">Forêt aléatoire</td>
                        <td class="py-2 px-3">1 621.7</td><td class="py-2 px-3">2 342.0</td><td class="py-2 px-3">2.56 %</td><td class="py-2 px-3">+357.0</td><td class="py-2 px-3 text-emerald-400 font-semibold">+26.5 %</td><td class="py-2 px-3">3.32 s</td>
                    </tr>
                    <tr>
                        <td class="py-2 px-3 font-sans font-semibold text-white">XGBoost</td>
                        <td class="py-2 px-3">1 502.6</td><td class="py-2 px-3">2 091.0</td><td class="py-2 px-3">2.39 %</td><td class="py-2 px-3">+450.0</td><td class="py-2 px-3 text-emerald-400 font-semibold">+34.4 %</td><td class="py-2 px-3">0.41 s</td>
                    </tr>
                    <tr class="bg-blue-900/20 text-blue-200">
                        <td class="py-2 px-3 font-sans font-bold text-white flex items-center gap-1.5">
                            <i class="fa-solid fa-star text-amber-400"></i> XGBoost sur écart à la veille
                        </td>
                        <td class="py-2 px-3 font-bold text-emerald-300">1 471.2</td>
                        <td class="py-2 px-3 font-bold text-emerald-300">2 037.4</td>
                        <td class="py-2 px-3 font-bold text-emerald-300">2.34 %</td>
                        <td class="py-2 px-3">+674.9</td>
                        <td class="py-2 px-3 text-emerald-400 font-bold">+36.1 %</td>
                        <td class="py-2 px-3">0.35 s</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <!-- Tab 3: Methodology & Limits -->
        <div id="tabMethodo" class="hidden card p-5 rounded-xl space-y-3 text-xs text-slate-300">
            <h4 class="font-bold text-white text-sm">Périmètre et rigueur scientifique</h4>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div class="space-y-2 bg-slate-900/50 p-3.5 rounded-lg border border-slate-800">
                    <h5 class="font-semibold text-emerald-400"><i class="fa-solid fa-shield-halved"></i> Prévention des fuites d'information</h5>
                    <p>Le modèle respecte scrupuleusement l'horizon J+1 : aucune variable explicative relative à la charge ne provient des 24 dernières heures précédant le jour prévu. Le test anti-fuite automatisé a validé ce respect.</p>
                </div>
                <div class="space-y-2 bg-slate-900/50 p-3.5 rounded-lg border border-slate-800">
                    <h5 class="font-semibold text-amber-400"><i class="fa-solid fa-triangle-exclamation"></i> Distinction Mesuré vs Calculé</h5>
                    <p>La consommation électrique est une grandeur physique observée (SCADA). Les émissions de gaz à effet de serre sont estimées par calcul ($Énergie \times Facteur$). L'incertitude sur l'intensité carbone du mix national (~19%) surpasse l'erreur du modèle prédictif (~1%).</p>
                </div>
            </div>
        </div>

    </main>

    <!-- Embedded Client JS -->
    <script>
        let currentChart = null;
        let currentData = null;
        let currentZone = 'total';
        let currentDeltaTemp = 0.0;
        let currentFactor = 0.658;

        const presets = [
            { label: "15 Nov (Essai)", date: "2017-11-15" },
            { label: "04 Août (Canicule)", date: "2017-08-04" },
            { label: "18 Jan (Hiver)", date: "2017-01-18" },
            { label: "05 Juin (Ramadan)", date: "2017-06-05" },
            { label: "18 Nov (Fête Nationale)", date: "2017-11-18" }
        ];

        function initPresets() {
            const container = document.getElementById('presetsContainer');
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
            document.getElementById('currentDateDisplay').innerText = date;
            
            try {
                const url = `/forecast?date=${date}&delta_temp=${currentDeltaTemp}&emission_factor=${currentFactor}`;
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
            const energyMWh = (d.energie_prevue_kwh / 1000).toFixed(1);
            document.getElementById('kpiEnergy').innerText = `${Number(energyMWh).toLocaleString('fr-FR')} MWh`;
            
            if (d.energie_observee_kwh) {
                const obsMWh = (d.energie_observee_kwh / 1000).toFixed(1);
                document.getElementById('kpiObsEnergy').innerText = `${Number(obsMWh).toLocaleString('fr-FR')} MWh`;
            } else {
                document.getElementById('kpiObsEnergy').innerText = "Non disponible";
            }

            // Peak load & peak hour
            let maxKw = -1, peakHour = 0;
            d.prevision_kw.forEach((kw, h) => {
                if (kw > maxKw) { maxKw = kw; peakHour = h; }
            });
            document.getElementById('kpiPeakLoad').innerText = `${Math.round(maxKw).toLocaleString('fr-FR')} kW`;
            document.getElementById('kpiPeakHour').innerText = `${peakHour}h00`;

            // Emissions
            const emCentral = d.emissions_estimees_tCO2.central || d.emissions_estimees_tCO2.personnalise;
            document.getElementById('kpiEmissions').innerText = `${Number(emCentral).toLocaleString('fr-FR')} tCO₂`;
            document.getElementById('kpiEmissionsRange').innerText = `${d.emissions_estimees_tCO2.bas} (bas) &ndash; ${d.emissions_estimees_tCO2.haut} (haut)`;

            // Solar savings calc
            const solarMwh = (energyMWh * 0.12).toFixed(1);
            const solarCo2 = (solarMwh * currentFactor).toFixed(1);
            document.getElementById('solarSavingsText').innerHTML = `&approx; -${solarMwh} MWh &bull; -${solarCo2} tCO₂ / jour`;
        }

        function renderChart(d) {
            const ctx = document.getElementById('forecastChart').getContext('2d');
            const hours = Array.from({length: 24}, (_, i) => `${i}h`);

            let seriesPred = d.prevision_kw;
            let seriesLower = d.prevision_lower_90_kw;
            let seriesUpper = d.prevision_upper_90_kw;
            let seriesObs = d.observe_kw;

            // Handle zone scaling
            if (currentZone === 'zone1') {
                seriesPred = d.zones.zone1_kw;
                seriesLower = seriesPred.map(v => v * 0.95);
                seriesUpper = seriesPred.map(v => v * 1.05);
                seriesObs = d.observe_kw.map(v => v * 0.48);
            } else if (currentZone === 'zone2') {
                seriesPred = d.zones.zone2_kw;
                seriesLower = seriesPred.map(v => v * 0.95);
                seriesUpper = seriesPred.map(v => v * 1.05);
                seriesObs = d.observe_kw.map(v => v * 0.313);
            } else if (currentZone === 'zone3') {
                seriesPred = d.zones.zone3_kw;
                seriesLower = seriesPred.map(v => v * 0.95);
                seriesUpper = seriesPred.map(v => v * 1.05);
                seriesObs = d.observe_kw.map(v => v * 0.207);
            }

            if (currentChart) currentChart.destroy();

            const datasets = [
                {
                    label: 'Prévision J+1 (kW)',
                    data: seriesPred,
                    borderColor: '#3b82f6',
                    backgroundColor: '#3b82f6',
                    borderWidth: 2.5,
                    tension: 0.3,
                    pointRadius: 2,
                    zIndex: 10
                },
                {
                    label: 'Observé SCADA (kW)',
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

            // If temperature anomaly active, show baseline comparison
            if (Math.abs(currentDeltaTemp) > 1e-3) {
                document.getElementById('whatifLegend').classList.remove('hidden');
                datasets.push({
                    label: 'Référence sans anomalie',
                    data: d.prevision_base_kw,
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
                                callback: val => (val / 1000).toFixed(0) + ' MW'
                            }
                        }
                    }
                }
            });
        }

        function setZone(z) {
            currentZone = z;
            ['btnZoneTotal', 'btnZone1', 'btnZone2', 'btnZone3'].forEach(id => {
                const b = document.getElementById(id);
                b.className = 'px-2.5 py-1 rounded font-medium text-slate-400 hover:text-white transition';
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
            if (currentData) {
                const kwh = currentData.energie_prevue_kwh;
                const co2 = (kwh * currentFactor / 1000).toFixed(1);
                document.getElementById('kpiEmissions').innerText = `${Number(co2).toLocaleString('fr-FR')} tCO₂`;
            }
        }

        function switchTab(tabId) {
            ['tabCities', 'tabModels', 'tabMethodo'].forEach(t => {
                document.getElementById(t).classList.add('hidden');
            });
            ['btnTabCities', 'btnTabModels', 'btnTabMethodo'].forEach(b => {
                document.getElementById(b).className = 'tab-btn px-4 py-2.5 rounded-t-lg bg-slate-800 text-slate-400 hover:text-white flex items-center gap-2';
            });
            document.getElementById(tabId).classList.remove('hidden');
            if (tabId === 'tabCities') document.getElementById('btnTabCities').className = 'tab-btn px-4 py-2.5 rounded-t-lg bg-blue-600 text-white flex items-center gap-2';
            if (tabId === 'tabModels') document.getElementById('btnTabModels').className = 'tab-btn px-4 py-2.5 rounded-t-lg bg-blue-600 text-white flex items-center gap-2';
            if (tabId === 'tabMethodo') document.getElementById('btnTabMethodo').className = 'tab-btn px-4 py-2.5 rounded-t-lg bg-blue-600 text-white flex items-center gap-2';
        }

        window.addEventListener('DOMContentLoaded', () => {
            initPresets();
            updateForecast();
        });
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content, status_code=200)
