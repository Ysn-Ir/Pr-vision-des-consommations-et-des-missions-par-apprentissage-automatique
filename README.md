# Projet 04 — Prévision de la consommation électrique et des émissions de CO₂ au Maroc

**Management de projet et transition bas-carbone — ENSA Fès**  
Territoire pilote : **Tanger–Tétouan–Al Hoceïma** · Transposition : **Fès–Meknès, Marrakech-Safi, Laâyoune-Sakia El Hamra**.

---

## 1. Description du projet

Ce projet développe une chaîne complète d'apprentissage automatique pour prévoir la charge électrique urbaine à l'horizon J+1 (pas horaire, 24 heures) et en déduire une estimation rigoureuse des émissions de gaz à effet de serre ($\text{CO}_2$). 

Développé initialement sur les données SCADA de Tétouan (2017), le projet a été enrichi avec les jeux de données récents de compteurs intelligents de **Marrakech** et **Laâyoune** (2022–2024), ainsi que la réanalyse météorologique ERA5 pour cadrer la transposition méthodologique à **Fès–Meknès**.

### Distinction méthodologique essentielle
* **Consommation d'électricité (kW / kWh) :** Grandeur physique mesurée par les systèmes de télégestion réseau (SCADA Amendis, compteurs 22 kV).
* **Émissions de $\text{CO}_2$ (kg / t) :** Grandeur calculée ($Énergie \times Facteur\ d'intensité\ carbone$), évaluée selon 3 scénarios nationaux (bas : 0.596, central : 0.658, haut : 0.720 $\text{kgCO}_2/\text{kWh}$).

---

## 2. Structure du répertoire

```text
project4/
├── data/
│   ├── Tetuan City power consumption.csv       # Données SCADA Tétouan (10 min, 52 416 mesures, 2017)
│   └── morocco_cities/                         # Extension multi-territoires au Maroc
│       ├── Marrakech_30T.csv                   # Compteurs Marrakech (30 min, 17 501 mesures, 2023–2024)
│       ├── Laayoune_10T.csv                    # Compteurs Laâyoune 22 kV (10 min, 88 890 mesures, 2022–2024)
│       └── morocco_cities_metadata.json        # Fiche de traçabilité, licences et citations DOI
├── artifacts/
│   ├── features_hourly.csv                     # Jeu horaire Tétouan enrichi (calendrier + météo + retards)
│   ├── model.joblib                            # Modèle sélectionné sérialisé (XGBoost sur écart à la veille)
│   └── results.json                            # Métriques de test, qualité des données, facteurs
├── projet04_prevision_tetouan.ipynb            # Pipeline complet d'expérimentation, validation et analyse
├── forecast_utils.py                           # Classe ResidualModel, incertitude 90%, simulateur météo
├── multi_city_utils.py                         # Analyse comparative multi-villes et connecteur Open-Meteo
├── api.py                                      # Application Web interactive (Dashboard) et API FastAPI
├── requirements.txt                            # Dépendances Python
└── README.md                                   # Documentation du projet
```

---

## 3. Installation et exécution

### 3.1 Dépendances requises
Installez les dépendances avec pip :
```bash
pip install -r requirements.txt
```

### 3.2 Lancement de l'Application Web & de l'API
Démarrez le serveur FastAPI / Uvicorn :
```bash
uvicorn api:app --reload
```
* **Application Web Interactive (Dashboard) :** Ouvrez votre navigateur sur **`http://127.0.0.1:8000/`** (ou `/dashboard`).
* **Documentation interactive Swagger API :** Accessible sur **`http://127.0.0.1:8000/docs`**.

#### Fonctionnalités de l'Application Web :
- **Visualisation dynamique de la charge :** Comparaison de la courbe prévue à J+1, de la courbe observée SCADA, et du ruban d'incertitude à 90 % ($\pm 1.645 \times \text{RMSE}$).
- **Décomposition par zone :** Bascule instantanée entre la charge globale urbaine et les trois sous-zones de distribution (Zone 1 : 48 %, Zone 2 : 31 %, Zone 3 : 21 %).
- **Simulateur climatique "What-If" :** Curseur d'anomalie thermique ($\Delta T \in [-5^\circ\text{C}, +5^\circ\text{C}]$) simulant l'impact d'une vague de froid ou d'une canicule sur la climatisation et le chauffage.
- **Simulateur de transition bas-carbone :** Curseur dynamique du mix électrique national et calcul de l'évitement carbone par autoconsommation solaire diurne.
- **Explorateur multi-villes :** Comparaison des profils horaires et saisonniers de Tétouan, Marrakech, Laâyoune et Fès–Meknès.

#### Points d'accès de l'API REST :
- `GET /forecast?date=2017-11-15&delta_temp=0.0&emission_factor=0.658` : Prévision 24h, incertitudes, décomposition en zones et émissions.
- `GET /health` : Statut, métadonnées du modèle et territoires couverts.
- `GET /api/overview` : Métriques comparatives de tous les modèles et contrôle qualité des données.
- `GET /api/morocco-cities` : Synthèse des indicateurs énergétiques pour Marrakech, Laâyoune et Fès.
- `GET /api/presets` : Liste de journées représentatives (Canicule, Vague de froid, Ramadan, etc.).

### 3.3 Réexécution du notebook et des modèles multi-villes
* Pour réexécuter le notebook principal de Tétouan :
```bash
jupyter nbconvert --to notebook --execute --inplace projet04_prevision_tetouan.ipynb
```
* Pour réentraîner et évaluer les modèles sur Marrakech et Laâyoune :
```bash
python train_multi_city.py
```
* Pour entraîner le modèle de transfert et de transposition climatique sur Fès–Meknès :
```bash
python train_fes_transfer.py
```

---

## 4. Résultats certifiés sur la période d'essai

### 4.1 Tétouan (Novembre–Décembre 2017, 1 440 heures)

| Modèle | MAE (kW) | RMSE (kW) | MAPE (%) | Biais (kW) | Gain RMSE vs Réf. | Gain MAE vs Réf. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Référence (Naïve 24 h)** | 2 076.1 | 3 188.2 | 3.33 % | +55.1 | — | — |
| **Régression linéaire** | 1 558.3 | 2 099.9 | 2.54 % | +40.2 | +34.1 % | +24.9 % |
| **Forêt aléatoire** | 1 621.7 | 2 342.0 | 2.56 % | +357.0 | +26.5 % | +21.9 % |
| **XGBoost standard** | 1 502.6 | 2 091.0 | 2.39 % | +450.0 | +34.4 % | +27.6 % |
| **XGBoost sur écart à la veille** ⭐ | **1 471.2** | **2 037.4** | **2.34 %** | +674.9 | **+36.1 %** | **+29.1 %** |

⭐ *Modèle retenu sur la validation de Tétouan.*

### 4.2 Benchmark multi-territoires (Tétouan, Marrakech, Laâyoune, Fès-Meknès)

| Territoire | Grandeur & Période d'essai | Modèle Retenu | RMSE | MAE | MAPE (%) | Gain RMSE vs Réf. |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Tétouan** | Charge urbaine (kW) &middot; Nov–Déc 2017 | XGBoost sur écart | 2 037.4 kW | 1 471.2 kW | 2.34 % | **+36.1 %** |
| **Marrakech** | Puissance active 22 kV (kW) &middot; Nov–Jan 2023 | Régression Linéaire | 142.2 kW | 115.6 kW | 9.16 % | **+20.0 %** |
| **Laâyoune** | Courant total 22 kV (A) &middot; Juil–Août 2023 | Régression Linéaire | 26.9 A | 20.4 A | 3.97 % | **+11.4 %** |
| **Fès–Meknès** | Charge urbaine (kW, ERA5) &middot; Nov–Déc 2023 | Régression Linéaire | 1 944.0 kW | 1 496.0 kW | 2.05 % | **+70.6 %** |

---

## 5. Synthèse des analyses critiques

1. **Significativité statistique (Bootstrap 2 000 tirages) :** Tous les modèles surpassent très nettement la persistance naïve ($IC_{95\%} = [0.53\ ;\ 0.77]$ pour le ratio de RMSE). Cependant, le gain du modèle non linéaire par rapport à la simple régression linéaire n'est pas statistiquement significatif en hiver ($IC_{95\%} = [0.84\ ;\ 1.10]$).
2. **Décrochage estival des arbres de décision (Validation glissante) :** En juillet (pic de chaleur estival inédit pour des modèles entraînés sur l'hiver/printemps), les arbres de décision classiques plafonnent et sous-estiment lourdement la charge (biais de $-13.5\ \text{MW}$ pour XGBoost standard). Prédire l'écart à la veille (`ResidualModel`) absorbe cette dérive saisonnière et réduit l'erreur estivale de plus de 55 %.
3. **Apport des données multi-villes marocaines :**
   - **Marrakech :** Forte pointe en milieu d'après-midi (15h00) tirée par la climatisation extrême en climat semi-aride.
   - **Laâyoune :** Profil avec pic régulier en soirée (21h00) et faible amplitude saisonnière.
   - **Fès–Meknès :** Climat continental à forte amplitude thermique où le recours au modèle résiduel est indispensable.
4. **Impact de l'intensité carbone :** L'énergie totale consommée sur l'essai est de 92.8 GWh (erreur prédictive de +1.05 %). L'écart entre les scénarios d'émissions bas et haut ($11\ 625\ \text{tCO}_2$, soit $18.8\ \%$) dépasse de loin l'erreur du modèle d'apprentissage.
