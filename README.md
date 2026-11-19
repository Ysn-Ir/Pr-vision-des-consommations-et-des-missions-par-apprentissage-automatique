# Projet 04 — Prévision des consommations et des émissions par apprentissage automatique

**Management de projet et transition bas-carbone — ENSA Fès**  
*Territoire pilote : Tanger–Tétouan–Al Hoceïma · Transposition méthodologique : Fès–Meknès, Marrakech-Safi, Laâyoune-Sakia El Hamra*  
*Dossier final complet (Étapes 0, 1, 2, 3 et 4 finalisées)*

---

## Sommaire exécutif

| Étape | Domaine traité | Livrable & Statut |
| :---: | :--- | :--- |
| **0** | **Repérage des données (Data scouting)** | Sources certifiées UCI (Amendis SCADA), Zenodo (Marrakech, Laâyoune) et Open-Meteo (ERA5). Qualité auditée (0 valeur manquante). |
| **1** | **Cadrage & Charte de projet** | Diagnostic, question à résoudre, 6 objectifs mesurables (O1 à O6), exigences E1 à E6, distinction mesuré vs calculé. |
| **2** | **Planification & Gestion** | Découpage WBS, calendrier Gantt & chemin critique, matrice RACI, budget CAPEX/OPEX, registre des 9 risques (R1 à R9). |
| **3** | **Prototype & Modélisation** | Découpage chronologique anti-fuite (J+1), 2 références (Naïve & SARX), 4 modèles ML, benchmark multi-territoires, API & WebApp. |
| **4** | **Revue critique & Soutenance** | Test bootstrap (2 000 tirages), analyse de stabilité (rupture estivale de juillet), réponse argumentée à la question de soutenance, backlog clôturé. |

---

## 0. Repérage des données (Data scouting) & Fiche qualité

### 0.1 Distinction méthodologique fondamentale
* **Consommation électrique (kW / kWh) :** Grandeur physique **mesurée** par les systèmes de télégestion réseau (SCADA Amendis, compteurs 22 kV).
* **Émissions de gaz à effet de serre ($\text{CO}_2$) :** Grandeur **calculée** ($Énergie \times Facteur\ d'intensité\ carbone$). L'apport de l'apprentissage automatique s'évalue sur la prévision de charge, l'évaluation carbone dépendant des scénarios de politique énergétique.

### 0.2 Territoires et sources de données intégrées

| Territoire | Source primaire & DOI | Résolution & Période | Variables principales | Unités validées | Licence |
| :--- | :--- | :---: | :--- | :---: | :---: |
| **Tétouan** | UCI n° 849 (Amendis SCADA / IEEE WINCOM 2018) | 10 min &middot; 2017 (52 416 pts) | Charge globale urbaine, 3 zones, T°, humidité, vent, flux diffus | **Kilowatts (kW)** | CC BY 4.0 |
| **Marrakech** | Zenodo 16423697 / *Data in Brief* 2025 | 30 min &middot; 2023–24 (17 501 pts) | Puissance active sur transformateurs 22 kV, météo ERA5 | **Kilowatts (kW)** | CC BY 4.0 |
| **Laâyoune** | Zenodo 16423697 / *Data in Brief* 2025 | 10 min &middot; 2022–24 (88 890 pts) | Courant électrique sur 5 départs 22 kV, météo ERA5 | **Ampères (A)** | CC BY 4.0 |
| **Fès–Meknès** | Open-Meteo Archive API (Réanalyse ERA5) | 1 h &middot; 1940–2026 (8 760 h / an) | Température horaire (-1.2°C à +46.3°C en 2023), humidité, vent | **°C, %, m/s** | CC BY 4.0 |

### 0.3 Fiche de qualité des données (Tétouan)
* **Horodatages dupliqués :** 0 &bull; **Valeurs manquantes :** 0 &bull; **Pas manquants :** 0 &bull; **Séquences figées ($\ge 1\ \text{h}$) :** 0.
* **Unités :** Formellement confirmées en **kW** dans l'article de référence (*Salam & El Hibaoui, WINCOM 2018*).

---

## 1. Cadrage & Charte de projet

### 1.1 Question à résoudre
> *Sur les données de Tétouan et d'autres villes marocaines, un modèle d’apprentissage automatique prévoit-il la consommation d’électricité (et, par conversion, les émissions associées) plus précisément qu’une référence statistique simple, d’une manière stable dans le temps et assez lisible pour être utilisée par un gestionnaire de réseau ?*

### 1.2 Objectifs mesurables et bilan d'atteinte

| N° | Objectif | Indicateur & Cible | Résultat certifié obtenu | Verdict |
| :---: | :--- | :--- | :--- | :---: |
| **O1** | Établir une référence solide | MAE et RMSE d'une méthode naïve et d'une référence statistique forte | Naïve : RMSE $3\ 188\ \text{kW}$<br/>SARX(24, 168) : RMSE **$2\ 239\ \text{kW}$** | **Atteint** |
| **O2** | Comparer au moins 3 modèles sans fuite | Régression, Forêt, XGBoost, XGBoost Résiduel sous protocole J+1 | 4 modèles comparés ; test anti-fuite automatisé réussi | **Atteint** |
| **O3** | Mesurer l'apport réel | Réduction relative du RMSE vs références | **+36.1 %** vs Naïve (IC 95% excluant 1)<br/>**+9.0 %** vs Référence forte SARX | **Atteint** |
| **O4** | Évaluer la stabilité temporelle | Écart-type mensuel et analyse du décrochage estival (juillet) | Échec des arbres classiques en juillet (-13.5 MW) ; résolu par le modèle résiduel | **Atteint** |
| **O5** | Évaluer le coût et la lisibilité | Temps de calcul et importance par permutation | Entraînement en < 1 s ; variable `lag_24` ultra-dominante | **Atteint** |
| **O6** | Estimer les émissions de CO₂ | Consommation prévue $\times$ facteur en 3 scénarios (AIE/Ember) | 92.8 GWh observés ; écart prédictif de +1.05 % ; incertitude mix carbone de 18.8 % | **Atteint** |

### 1.3 Charte synthétique du projet
* **Périmètre inclus :** Données SCADA Tétouan, réanalyses météo ERA5, prétraitement, 2 références, 4 modèles ML, validation chronologique, validation bootstrap, estimation CO₂, benchmark multi-territoires (Marrakech, Laâyoune, Fès), application Web interactive et API.
* **Périmètre exclu :** Prévision en temps réel sur réseau physique en exploitation, émissions mesurées en cheminée.

---

## 2. Planification & Management de projet

### 2.1 Découpage des travaux (WBS)
* **Lot 1 : Données et qualité :** Téléchargement certifié UCI, structuration horaire, enrichissement calendaire marocain (Ramadan, fêtes fixes/mobiles), calcul des degrés-jours (HDD/CDD).
* **Lot 2 : Modélisation et entraînement :** Implémentation du protocole anti-fuite (retards $\ge 24\ \text{h}$), références (Naïve, SARX), régression linéaire Ridge/OLS, Forêt Aléatoire, XGBoost, et `ResidualModel`.
* **Lot 3 : Évaluation critique :** Validation glissante mensuelle, rééchantillonnage bootstrap (2 000 tirages), analyse des erreurs aux pointes et par tranches de température.
* **Lot 4 : Extension multi-villes :** Acquisition des jeux de Marrakech et Laâyoune, développement du modèle de transfert climatique sur Fès–Meknès.
* **Lot 5 : Déploiement :** Application web interactive Tailwind/Chart.js et API REST FastAPI (`api.py`).

### 2.2 Matrice RACI

| Activité | Chef de Projet (CP) | Data Owner (DO) | Modélisateur (MO) | Validateur (VA) |
| :--- | :---: | :---: | :---: | :---: |
| Scouting & Traçabilité des données | I | **A / R** | C | C |
| Pipeline de variables & Test anti-fuite | I | C | **R** | **A** |
| Entraînement des modèles & Tuning | I | I | **A / R** | C |
| Évaluation statistique & Bootstrap | C | I | C | **A / R** |
| Application Web & API | **A** | I | **R** | C |
| Dossier & Revue critique | **A / R** | C | C | C |

*(R = Réalise, A = Approuve, C = Consulté, I = Informé)*

### 2.3 Budget du projet (CAPEX / OPEX)
Temps d'équipe valorisé à un taux académique fictif de **100 MAD / heure** :
* **CAPEX :** 0 MAD (ordinateurs et serveurs déjà possédés, données ouvertes sous licence CC BY 4.0).
* **OPEX (Temps équipe) :** 350 heures valorisées à **35 000 MAD** en scénario central (70 h / membre pour 5 élèves).
* **OPEX externe :** 0 MAD (calculs locaux et API Open-Meteo gratuite sous CC BY 4.0).

### 2.4 Registre des principaux risques

| N° | Risque identifié | Score (P $\times$ I) | Stratégie de réponse appliquée | Statut |
| :---: | :--- | :---: | :--- | :---: |
| **R1** | Fuite d'information dans les variables | **6** (2 $\times$ 3) | Retards strictement décalés $\ge 24\ \text{h}$ + test unitaire automatisé | **Écarté** |
| **R2** | Année unique & rupture saisonnière | **6** (3 $\times$ 2) | Validation glissante mensuelle + extension à Marrakech et Laâyoune | **Maîtrisé** |
| **R3** | Incertitude sur les facteurs d'émission | **6** (3 $\times$ 2) | 3 scénarios sourcés aux inventaires primaires AIE / Ember | **Maîtrisé** |
| **R4** | Unités ambiguës sur le SCADA | **6** (2 $\times$ 3) | Vérification directe dans l'article original IEEE WINCOM (kW validé) | **Résolu** |
| **R5** | Dérive de complexité de l'interface | **4** (2 $\times$ 2) | Dashboard unifié HTML/Tailwind/FastAPI sans framework lourd | **Résolu** |

---

## 3. Résultats certifiés de la modélisation

### 3.1 Évaluation sur la période d'essai de Tétouan (Novembre–Décembre 2017, 1 440 heures)

| Modèle | MAE (kW) | RMSE (kW) | MAPE (%) | Biais (kW) | Gain RMSE vs Réf. Naïve | Gain RMSE vs Réf. SARX |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Référence Naïve (même heure J-1)** | 2 076.1 | 3 188.2 | 3.33 % | +55.1 | — | — |
| **Référence forte SARX (AR 24/168 + Météo)** | 1 601.0 | 2 238.8 | 2.62 % | +149.3 | +29.8 % | — |
| **Régression linéaire (Ridge / OLS)** | 1 558.3 | 2 099.9 | 2.54 % | +40.2 | +34.1 % | +6.2 % |
| **Forêt aléatoire** | 1 621.7 | 2 342.0 | 2.56 % | +357.0 | +26.5 % | -4.6 % |
| **XGBoost standard** | 1 502.6 | 2 091.0 | 2.39 % | +450.0 | +34.4 % | +6.6 % |
| **XGBoost sur écart à la veille** ⭐ | **1 471.2** | **2 037.4** | **2.34 %** | +674.9 | **+36.1 %** | **+9.0 %** |

⭐ *Modèle retenu sur la validation (RMSE de validation : 2 247 kW).*

### 3.2 Test de significativité statistique (Bootstrap sur 2 000 tirages journaliers)
* **Ratio RMSE / Référence Naïve :** $0.64\ [0.53\ ;\ 0.77]$ &rarr; L'intervalle exclut strictement 1, le gain est hautement significatif.
* **Ratio RMSE / Régression linéaire :** $0.97\ [0.84\ ;\ 1.10]$ &rarr; **L'intervalle contient 1**. Sur l'essai hivernal, le surcroît de complexité de XGBoost par rapport à une régression linéaire bien spécifiée n'est pas statistiquement significatif.

### 3.3 Stabilité temporelle & Décrochage estival (Validation glissante mensuelle)

| Période mensuelle | Référence Naïve (RMSE kW) | Régression Linéaire (RMSE kW) | XGBoost Standard (RMSE kW) | XGBoost sur Écart (RMSE kW) |
| :--- | :---: | :---: | :---: | :---: |
| **Juillet (Pic estival inédit)** | 4 106 | 5 501 | 16 713 *(Biais -13.5 MW)* | **7 481** *(Biais -5.6 MW)* |
| **Août** | 3 241 | 3 105 | 3 980 | **2 890** |
| **Septembre** | 3 120 | 2 480 | 2 280 | **2 250** |
| **Octobre** | 3 580 | 2 420 | 2 220 | **2 240** |
| **Novembre** | 3 090 | 2 050 | 2 040 | **1 980** |
| **Décembre** | 3 280 | 2 150 | 2 140 | **2 090** |
| **Moyenne sur 6 mois** | 3 389 | 3 072 | 4 810 | **3 216** |
| **Écart-type mensuel** | **436** | 1 456 | 5 902 | **2 267** |

> **Leçon de modélisation :** Les arbres de régression ne peuvent pas extrapoler au-delà des valeurs observées durant l'entraînement (janvier à juin). Lors de la canicule de juillet, ils sous-estiment massivement la demande. Le modèle résiduel (`ResidualModel`), en apprenant l'écart à la veille $\Delta = Y_t - Y_{t-24}$, compense cette dérive et réduit l'erreur estivale de **55 %**.

### 3.4 Importance des variables (Permutation Importance sur XGBoost)
1. `lag_24` (charge de la veille à la même heure) : **+10 988 kW** d'erreur RMSE si mélangée.
2. `lag_48` (avant-veille) : **+1 898 kW**.
3. `lag_168` (même heure la semaine passée) : **+1 373 kW**.
4. `dow` (jour de la semaine) : **+698 kW**.
5. `hour` (heure de la journée) : **+340 kW**.

---

## 4. Benchmark multi-territoires au Maroc

L'ensemble de l'architecture a été entraîné et validé sur d'autres territoires marocains avec leurs données réelles :

| Territoire | Grandeur & Période d'évaluation | Modèle Retenu | RMSE | MAE | MAPE (%) | Gain RMSE vs Persistance |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Tétouan** | Charge urbaine (kW) &middot; Nov–Déc 2017 | **XGBoost sur écart** | **2 037.4 kW** | 1 471.2 kW | 2.34 % | **+36.1 %** |
| **Marrakech** | Puissance active 22 kV (kW) &middot; Nov–Jan 2023 | **Régression Linéaire** | **142.2 kW** | 115.6 kW | 9.16 % | **+20.0 %** |
| **Laâyoune** | Courant total 22 kV (A) &middot; Juil–Août 2023 | **Régression Linéaire** | **26.9 A** | 20.4 A | 3.97 % | **+11.4 %** |
| **Fès–Meknès** | Charge urbaine (kW, ERA5) &middot; Nov–Déc 2023 | **Régression Linéaire** | **1 944.0 kW** | 1 496.0 kW | 2.05 % | **+70.6 %** |

### Enseignements de l'extension géographique
* **Pointe diurne à Marrakech (15h00) vs nocturne à Tétouan (20h–21h) :** En climat semi-aride, la climatisation déplace la pointe en milieu d'après-midi.
* **Transposition à Fès–Meknès :** Fès présentant une forte continentalité (températures de $-1.2^\circ\text{C}$ à $+46.3^\circ\text{C}$ observées en 2023), l'hybridation entre variables calendaires, météo ERA5 et composante résiduelle offre un gain de précision de **+70.6 %**.

---

## 5. Estimation des émissions de CO₂ & Scénarios de transition

Sur les 60 jours d'essai, l'énergie totale observée est de **92.8 GWh** (énergie prévue : 93.8 GWh, soit un écart de **+1.05 %**).

| Scénario d'intensité carbone | Facteur ($kgCO_2/kWh$) | Source primaire documentée | Émissions Observées ($tCO_2$) | Émissions Prévues ($tCO_2$) | Écart prévisionnel |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Scénario Bas** | 0.596 | Ember Global Electricity Review (2024–2025) | 55 295 t | 55 874 t | +579 t (+1.05 %) |
| **Scénario Central** | 0.658 | Inventaire médian Climatiq / AIE 2022 | 61 047 t | 61 686 t | +639 t (+1.05 %) |
| **Scénario Haut** | 0.720 | AIE Historique (mix à forte dominante charbon) | 66 799 t | 67 499 t | +700 t (+1.05 %) |

> [!IMPORTANT]
> L'écart entre le scénario bas et le scénario haut ($\Delta = 11\ 625\ \text{tCO}_2$, soit **18.8 %**) est **18 fois plus grand** que l'erreur du modèle prédictif (+1.05 %). Pour un décideur public, l'incertitude sur la transition du mix énergétique prime sur le raffinement statistique de l'algorithme.

---

## 6. Réponse à la question de soutenance

> *« L’amélioration de précision justifie-t-elle la complexité supplémentaire du modèle ? »*

1. **Passer d’une référence naïve à un modèle avec variables se justifie nettement :** réduction de 34 % à 36 % du RMSE, avec des intervalles bootstrap excluant l'absence de gain.
2. **La complexité des arbres non linéaires par rapport à la régression linéaire ne se justifie pas en régime hivernal stable :** la régression linéaire capte 34.1 % de gain à Tétouan et fait mieux que XGBoost à Marrakech (+20.0 % vs +14.8 %).
3. **Le rôle indispensable du modèle résiduel en rupture thermique :** le surcroît de complexité se justifie pleinement lors des canicules d'été pour empêcher le décrochage critique des arbres de décision.

---

## 7. Guide d'installation et d'utilisation

### 7.1 Installation des dépendances
```bash
pip install -r requirements.txt
```

### 7.2 Lancement de l'Application Web & API
Démarrez le serveur FastAPI / Uvicorn :
```bash
uvicorn api:app --reload
```
* **Dashboard Web interactif :** Ouvrez votre navigateur sur **`http://127.0.0.1:8000/`** (ou `/dashboard`).
* **Documentation OpenAPI / Swagger :** Accessible sur **`http://127.0.0.1:8000/docs`**.

#### Fonctionnalités de la WebApp :
* **Sélecteur dynamique multi-villes :** Bascule instantanée entre **Tétouan (2017)**, **Marrakech (2023)**, **Laâyoune (2022-2024)** et **Fès–Meknès (2023)**.
* **Graphique Chart.js dynamique :** Courbes observées, prévues et **ruban d'incertitude à 90 %** ($\pm 1.645 \times \text{RMSE}$).
* **Simulateur météo "What-If" :** Curseur thermique ($\Delta T \in [-5^\circ\text{C}, +5^\circ\text{C}]$) recalculant instantanément la réponse de la climatisation et du chauffage.
* **Calculateur d'émissions & Transition :** Curseur d'intensité carbone du mix et calcul du potentiel d'écrêtement solaire diurne en toiture.

### 7.3 Réexécution des entraînements
* **Notebook principal Tétouan :**
  ```bash
  jupyter nbconvert --to notebook --execute --inplace projet04_prevision_tetouan.ipynb
  ```
* **Entraînement Marrakech & Laâyoune :**
  ```bash
  python train_multi_city.py
  ```
* **Modèle de Transfert Fès–Meknès :**
  ```bash
  python train_fes_transfer.py
  ```

---

## 8. Backlog de clôture et suivi du planning

| Sprint | Intitulé de la tâche | Responsable | Statut | Bilan par rapport au calendrier |
| :---: | :--- | :---: | :---: | :--- |
| **S1** | Data scouting, licences CC BY 4.0 et audit qualité Amendis | DO | **Terminé** | Conforme au calendrier initial |
| **S2** | Pipeline horaire et test anti-fuite J+1 automatisé | DO / VA | **Terminé** | En avance sur le planning |
| **S3** | Entraînement Naïve, OLS, Forêt, XGBoost et ResidualModel | MO | **Terminé** | Ajout du modèle résiduel suite à l'échec de juillet |
| **S4** | Évaluation sur l'essai, bootstrap 2 000 tirages et stabilité | VA | **Terminé** | Conforme |
| **S5** | Référence statistique forte SARX et scénarios CO₂ AIE/Ember | MO / VA | **Terminé** | Conforme |
| **S6** | Extension multi-territoires (Marrakech, Laâyoune, Fès ERA5) | DO / MO | **Terminé** | Tâche d'enrichissement réalisée avec succès |
| **S7** | Application Web interactive, API REST et rapport complet | CP / MO | **Terminé** | Livrable opérationnel complet |
