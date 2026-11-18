"""Utilitaires partagés par le notebook, l'API et l'application web.

Contient la classe ResidualModel requise pour désérialiser model.joblib,
ainsi que les outils de calcul d'intervalles de confiance, de simulation météo
"what-if", et de ventilation par zones urbaines.
"""
import numpy as np
import pandas as pd


class ResidualModel:
    """Apprend l'écart entre la charge à prévoir et la charge de la veille (colonne base_col).

    Une forêt ou XGBoost ne peut pas prédire au-delà des valeurs vues à l'entraînement ;
    prédire un écart plutôt qu'un niveau évite ce plafonnement quand la saison change.
    """

    def __init__(self, model, base_col="lag_24"):
        self.model, self.base_col = model, base_col

    def fit(self, X, y):
        self.model.fit(X, np.asarray(y) - X[self.base_col].values)
        return self

    def predict(self, X):
        return self.model.predict(X) + X[self.base_col].values


# Ratio historique moyen des 3 zones de Tétouan (Amendis SCADA)
ZONE_SHARES = {
    "zone1": 0.480,  # ~48.0 % de la charge totale
    "zone2": 0.313,  # ~31.3 % de la charge totale
    "zone3": 0.207,  # ~20.7 % de la charge totale
}

# Facteurs d'émission de référence pour le réseau électrique marocain (kgCO2/kWh)
FACTORS_DEFAULT = {
    "bas": 0.596,     # Source Ember 2025
    "central": 0.658, # Moyenne / médiane des estimations nationales
    "haut": 0.720,    # Source Climate Analytics / IEA
}


def compute_prediction_intervals(pred: np.ndarray, rmse: float = 2037.4, confidence: float = 0.90) -> tuple[np.ndarray, np.ndarray]:
    """Calcule les bornes inférieure et supérieure de la prévision.

    Utilise l'écart-type résiduel empirique (RMSE de test = ~2037 kW).
    Pour un niveau de 90%, z ≈ 1.645 ; pour 95%, z ≈ 1.96.
    """
    z_map = {0.80: 1.282, 0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    z = z_map.get(confidence, 1.645)
    margin = z * rmse
    lower = np.maximum(0, pred - margin)
    upper = pred + margin
    return lower, upper


def decompose_zones(pred_total: np.ndarray) -> dict[str, list[float]]:
    """Décompose la prévision totale en sous-zones de distribution (Zone 1, 2 et 3)."""
    return {
        "zone1_kw": [round(float(p * ZONE_SHARES["zone1"]), 1) for p in pred_total],
        "zone2_kw": [round(float(p * ZONE_SHARES["zone2"]), 1) for p in pred_total],
        "zone3_kw": [round(float(p * ZONE_SHARES["zone3"]), 1) for p in pred_total],
    }


def simulate_weather_anomaly(rows: pd.DataFrame, base_pred: np.ndarray, delta_temp: float = 0.0) -> np.ndarray:
    """Simule l'impact d'une anomalie thermique (vague de chaleur ou grand froid) sur la prévision.

    Sensibilité thermique estimée sur Tétouan :
    - En été (T > 20°C) : climatisation ~ +520 kW par °C supplémentaire.
    - En hiver (T < 15°C) : chauffage d'appoint ~ +380 kW par °C de baisse.
    """
    if abs(delta_temp) < 1e-4:
        return base_pred.copy()
    
    adjusted_pred = np.array(base_pred, copy=True, dtype=float)
    temps = rows["temp"].values if "temp" in rows.columns else np.full(len(base_pred), 20.0)
    
    for i, t in enumerate(temps):
        new_t = t + delta_temp
        impact = 0.0
        # Sensibilité climatisation (au-dessus du confort ~20°C)
        if new_t > 20.0:
            cdd_diff = max(0.0, new_t - 20.0) - max(0.0, t - 20.0)
            impact += cdd_diff * 520.0
        # Sensibilité chauffage (en-dessous de 15°C)
        if new_t < 15.0:
            hdd_diff = max(0.0, 15.0 - new_t) - max(0.0, 15.0 - t)
            impact += hdd_diff * 380.0
        
        adjusted_pred[i] = max(1000.0, adjusted_pred[i] + impact)
    
    return adjusted_pred


def calculate_emissions_summary(energy_kwh: float, custom_factor: float = None) -> dict:
    """Calcule les émissions de CO2 estimées selon les scénarios officiels et optionnel personnalisé."""
    res = {k: round(energy_kwh * f / 1000.0, 2) for k, f in FACTORS_DEFAULT.items()}
    if custom_factor is not None and custom_factor > 0:
        res["personnalise"] = round(energy_kwh * custom_factor / 1000.0, 2)
    return res
