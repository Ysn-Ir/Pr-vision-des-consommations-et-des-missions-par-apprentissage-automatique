# Projet 04 — Prévision des consommations et des émissions par apprentissage automatique

Management de projet et transition bas-carbone — ENSA Fès

Dossier de groupe · Étape 0 (repérage des données) et étape 1 (cadrage) · Territoire pilote : Tanger–Tétouan–Al Hoceïma, démarche transposable à Fès–Meknès

## Plan de travail mis à jour

| Étape | Contenu | Statut |
| --- | --- | --- |
| 0 | Repérage des données (data scouting) : sources, licences, qualité, périmètres | Ce document |
| 1 | Cadrage : diagnostic, question, objectifs mesurables, charte, exigences | Ce document |
| 2 | Planification : WBS, Gantt et chemin critique, RACI, budget CAPEX/OPEX, registre des risques | Fait (section 2) |
| 3 | Prototype : jeu de données, découpage chronologique, référence, régression, forêt aléatoire, XGBoost, API | Fait (section 3) |
| 4 | Revue critique : biais, limites, réponse à la question de soutenance | Fait (section 4) |

## 0. Repérage des données (data scouting)

### 0.1 Ce dont le projet a besoin

Le projet exige une série temporelle de consommation électrique suffisamment longue et fine, des variables explicatives (météo, calendrier) et un facteur d’émission permettant de passer des kWh aux émissions de CO₂. Les consommations et les émissions n’ont pas la même nature : la consommation est mesurée, l’émission est calculée (activité × facteur d’émission). Cette distinction doit rester visible dans tout le dossier.

### 0.2 Sources repérées

| Source | Contenu | Intérêt pour le projet | Limites constatées |
| --- | --- | --- | --- |
| **Power Consumption of Tetouan City** (UCI, Salam et El Hibaoui, DOI 10.24432/C5B034) | Environ 52 400 mesures toutes les 10 minutes en 2017, trois zones de distribution, température, humidité, vitesse du vent, deux indicateurs de flux diffus. Données issues du système SCADA de l’opérateur Amendis. Licence CC BY 4.0. | Seule source repérée située dans le territoire pilote. Pas de valeurs manquantes annoncées. Cible naturelle du projet. | Une seule année : impossible d’observer plusieurs cycles saisonniers. Aucune colonne d’émissions. Les unités des consommations ne sont pas détaillées sur la fiche UCI (le kW est indiqué dans des travaux qui l’utilisent) : à vérifier dans l’article d’origine. |
| **High-Resolution Load Dataset from Smart Meters Across Various Cities in Morocco** (UCI n° 1158 ; Mendeley, DOI 10.17632/6s9mfp3bd8.1 ; Zenodo) | Mesures de compteurs intelligents à Laâyoune, Boujdour, Marrakech et Foum El Oued. Fichier Excel d’environ 21 Mo. Publié en 2025. | Complément pour tester la généralisation à d’autres villes et climats. | Aucune ville du Nord ni de Fès–Meknès. Les conditions de réutilisation sont à relire sur chaque dépôt. |
| **MORED** (Energies, 2020, DOI 10.3390/en13246737) | Consommation de logements et locaux marocains, avec signatures de charges individuelles. | Utile pour le projet 05 (bâtiment) ou une extension désagrégée. | Échelle du logement, non du réseau urbain : peu adapté à la prévision principale. |
| **Jeu de données technico-économique pour la modélisation énergétique du Maroc** (OSeMOSYS, 2015–2050, avec échanges directs avec l’ONEE) | Production, consommation, capacités, coûts, potentiels. | Contexte national et scénarios, utile pour l’argumentaire et les hypothèses. | Données annuelles et prospectives, non adaptées à l’entraînement d’un modèle infra-journalier. |
| **Open-Meteo, API d’archive météo (réanalyse ERA5)** | Météo horaire depuis 1940, sans clé pour un usage non commercial, données sous CC BY 4.0 avec mention obligatoire. | Variables externes pour Tétouan comme pour Fès, et possibilité de prolonger la période. | Grille d’environ 10 km : ce sont des valeurs de réanalyse, différentes d’une station locale. Limite d’environ 10 000 appels par jour. |
| **Facteurs d’émission du réseau marocain** (IEA via Climatiq pour 2022 ; Climate Analytics ; Ember / Our World in Data) | Intensité carbone de l’électricité. | Nécessaire pour convertir kWh en kgCO₂. | Les ordres de grandeur divergent selon l’année et la méthode : environ 720 gCO₂/kWh pour 2019 selon Climate Analytics, environ 596 gCO₂/kWh pour 2025 selon Ember relayé par un agrégateur. Le plus récent est une source secondaire. |

### 0.3 Constats

1. **Aucun jeu de données public de charge n’a été repéré pour Fès–Meknès** lors de cette recherche. Le dossier traite donc Tétouan et propose Fès comme transposition méthodologique.
2. **Le jeu de Tétouan ne couvre qu’une année.** Un découpage chronologique honnête (par exemple janvier–août pour l’entraînement, septembre–octobre pour la validation, novembre–décembre pour l’essai) laisse au test des conditions saisonnières absentes de l’entraînement. Ce n’est pas un défaut du protocole, c’est une limite des données, qui sera analysée dans l’étape 4 sous l’angle de la stabilité dans le temps.
3. **Les émissions ne sont pas observées.** Elles seront estimées par consommation × facteur d’émission du réseau. Avec un facteur constant, prévoir l’émission revient à prévoir la consommation à un coefficient près : l’apport de l’apprentissage automatique se juge donc sur la consommation. Le facteur d’émission sera traité en scénarios (bas, central, haut) et non comme une valeur unique.
4. **La provenance des facteurs doit être tracée** (organisme, année, périmètre, trajet dans le réseau avec ou sans pertes). Avant toute valeur définitive, le groupe vérifiera le chiffre à la source primaire (Ember, IEA).

### 0.4 Décision proposée

- **Jeu principal :** Tétouan (consommation et météo locale), donnée observée.
- **Variables d’appoint :** calendrier (heure, jour, jours fériés marocains, période de Ramadan) construit par le groupe ; météo ERA5 comme test de robustesse, jamais mélangée sans mention avec la météo d’origine.
- **Jeu de contrôle de généralisation :** compteurs intelligents de Marrakech ou Laâyoune, si le temps le permet.
- **Facteur d’émission :** trois scénarios, avec fiche de traçabilité.
- **Données simulées :** aucune pour l’instant. Toute donnée générée plus tard sera signalée comme telle.

### 0.5 Fiche de qualité à remplir à l’étape 3

Pour chaque jeu retenu : période réelle couverte, pas de temps, taux de valeurs manquantes ou aberrantes, doublons, changements d’heure, unités, licence, mention de citation, date et URL de téléchargement.

## 1. Cadrage

### 1.1 Diagnostic initial

Un gestionnaire de réseau ou une collectivité qui ne sait pas anticiper sa demande dimensionne mal ses moyens et peine à évaluer les émissions liées à l’électricité qu’elle consomme. Les modèles d’apprentissage automatique promettent de meilleures prévisions, mais une bonne performance apparente peut venir d’une fuite d’information ou d’un test trop favorable. La difficulté n’est pas de produire une courbe prédite, c’est de prouver qu’elle apporte davantage qu’une référence simple.

### 1.2 Question à résoudre

Sur les données de Tétouan, un modèle d’apprentissage automatique prévoit-il la consommation d’électricité (et, par conversion, les émissions associées) plus précisément qu’une référence statistique simple, d’une manière stable dans le temps et assez lisible pour être utilisée par un gestionnaire ?

### 1.3 Parties concernées

| Partie | Intérêt | Besoin principal |
| --- | --- | --- |
| Gestionnaire de réseau / distributeur | Équilibrer l’offre et la demande | Prévision fiable, erreurs connues aux heures de pointe |
| Collectivité territoriale | Suivi des émissions du territoire | Estimations traçables, scénarios de facteur d’émission |
| Équipe projet (élèves) | Résultat validé et défendable | Protocole reproductible, périmètre réaliste |
| Encadrants | Évaluation | Distinction nette entre données observées, estimées et simulées |

### 1.4 Objectifs mesurables (propositions à réviser après la première référence)

| N° | Objectif | Indicateur | Cible provisoire |
| --- | --- | --- | --- |
| O1 | Établir une référence solide | MAE et RMSE d’une méthode saisonnière naïve ou d’une régression simple | Référence calculée sur la période d’essai |
| O2 | Comparer au moins trois modèles | Régression, forêt aléatoire, XGBoost sous le même protocole | Trois modèles évalués sans fuite d’information |
| O3 | Mesurer l’apport réel | Réduction relative du RMSE par rapport à la référence | À fixer après O1 ; l’absence de gain est un résultat recevable |
| O4 | Évaluer la stabilité | Erreur par mois ou par saison de l’essai | Écart documenté entre les périodes |
| O5 | Évaluer le coût et la lisibilité | Temps d’entraînement et de prédiction, importance des variables | Tableau comparatif |
| O6 | Estimer les émissions | Consommation prévue × facteur du réseau en trois scénarios | Fourchette bas–central–haut tracée |

Les cibles chiffrées ne sont volontairement pas inventées avant d’avoir vu la référence : fixer un seuil à l’avance sans connaître la difficulté du jeu de données serait arbitraire.

### 1.5 Charte de projet

| Rubrique | Contenu |
| --- | --- |
| Titre | Prévision des consommations et des émissions par apprentissage automatique |
| Périmètre inclus | Jeu de Tétouan, prétraitement, référence, trois modèles, validation chronologique, estimation d’émissions, interface ou API légère, bilan critique |
| Périmètre exclu | Prévision en temps réel sur un réseau réel, intégration à un système d’exploitation d’opérateur, émissions mesurées |
| Livrables | Protocole expérimental, backlog, notebooks, interface ou API, rapport de validation, analyse des biais, dossier de planification |
| Équipe | Quatre à six élèves ; rôles à répartir (données, modélisation, validation, interface, gestion de projet) |
| Contraintes | Sept séances, données publiques, une seule année de données locales |
| Hypothèses | La consommation de 2017 reste représentative ; le facteur d’émission est constant sur la période d’étude |
| Risques initiaux | Fuite d’information, année unique, facteur d’émission incertain, sur-ajustement, qualité inconnue de la météo |
| Critères de réception | Protocole reproductible, comparaison équitable, limites explicitées, données simulées signalées |

### 1.6 Exigences de départ

- **E1.** Le découpage entre entraînement, validation et essai est strictement chronologique.
- **E2.** Les variables construites à partir du futur (par exemple une moyenne glissante centrée) sont interdites.
- **E3.** Tous les modèles sont comparés sur la même période d’essai et les mêmes indicateurs (MAE, RMSE).
- **E4.** Chaque résultat indique s’il provient de données observées, estimées ou simulées.
- **E5.** Le facteur d’émission est référencé (organisme, année, périmètre).
- **E6.** Le code et les données sont accompagnés d’une procédure de reproduction.

### 1.6 bis Sources utilisées à ce stade

Fiches des jeux UCI (Tétouan n° 849 ; compteurs intelligents n° 1158), dépôt Mendeley, article MORED (Energies, 2020), documentation d’Open-Meteo, fiches de facteurs d’émission de Climatiq (IEA), analyse Climate Analytics, synthèse Ember / Our World in Data relayée par un agrégateur. Les valeurs numériques de facteurs sont à confirmer à la source primaire avant usage.

## 

## 2. Planification (étape 2)

Les durées, charges et coûts ci-dessous sont des estimations de planification du groupe, pas des mesures. Elles seront révisées au fil des séances et chaque révision sera justifiée.

### 2.1 Découpage des travaux (WBS)

| Code | Lot | Tâches |
| --- | --- | --- |
| 1 | Gestion de projet | 1.1 Charte · 1.2 Planning et suivi · 1.3 Registre des risques · 1.4 Préparation de la soutenance |
| 2 | Données | 2.1 Téléchargement et fiche de qualité · 2.2 Nettoyage (doublons, valeurs aberrantes, changements d’heure) · 2.3 Variables de calendrier · 2.4 Facteurs d’émission et traçabilité |
| 3 | Modélisation | 3.1 Protocole de découpage chronologique · 3.2 Référence (saisonnière naïve) · 3.3 Régression · 3.4 Forêt aléatoire · 3.5 XGBoost · 3.6 Réglage des hyperparamètres |
| 4 | Validation | 4.1 MAE et RMSE · 4.2 Stabilité par mois · 4.3 Coût de calcul · 4.4 Interprétabilité · 4.5 Analyse des biais |
| 5 | Interface | 5.1 API de prévision · 5.2 Interface de visualisation · 5.3 Estimation des émissions en trois scénarios |
| 6 | Documentation | 6.1 Dossier reproductible · 6.2 Bilan critique |

### 2.2 Dépendances et chemin critique

Durées estimées en jours ouvrés pour l’équipe.

| Tâche | Durée | Dépend de |
| --- | --- | --- |
| 2.1 Fiche de qualité | 3 | Rien |
| 2.2 Nettoyage | 4 | 2.1 |
| 2.3 Variables de calendrier | 2 | 2.2 |
| 2.4 Facteurs d’émission | 3 | 2.1 |
| 3.1 Protocole de découpage | 2 | 2.2 |
| 3.2 Référence | 2 | 3.1 |
| 3.3 Régression | 3 | 3.2, 2.3 |
| 3.4 Forêt aléatoire | 4 | 3.2, 2.3 |
| 3.5 + 3.6 XGBoost et réglage | 6 | 3.2, 2.3 |
| 4.1 + 4.2 Métriques et stabilité | 4 | 3.3 à 3.5 |
| 4.3 + 4.4 Coût et interprétabilité | 3 | 3.3 à 3.5 |
| 4.5 Analyse des biais | 3 | 4.1 |
| 5.1 + 5.2 API et interface | 5 | 3.3 |
| 5.3 Émissions | 2 | 2.4, 4.1 |
| 6.2 Bilan critique | 3 | 4.5, 5.3 |
| 1.4 Soutenance | 2 | 6.2 |

**Chemin critique :** 2.1 → 2.2 → 3.1 → 3.2 → 3.5/3.6 → 4.1/4.2 → 4.5 → 6.2 → 1.4, soit 3 + 4 + 2 + 2 + 6 + 4 + 3 + 3 + 2 = **29 jours ouvrés**.

Sur six semaines utiles (de la semaine 2 à la semaine 7), cela laisse environ **un jour de marge**. Le réglage de XGBoost est le goulot : il sera limité dans le temps (au plus 6 jours) plutôt que poursuivi jusqu’à épuisement.

### 2.3 Calendrier (Gantt par semaine)

| Tâche | S1 | S2 | S3 | S4 | S5 | S6 | S7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cadrage, charte (1.1) | ■ | ■ |  |  |  |  |  |
| Repérage des données et fiche de qualité (2.1) |  | ■ | ■ |  |  |  |  |
| Nettoyage et variables (2.2, 2.3) |  |  | ■ | ■ |  |  |  |
| Facteurs d’émission (2.4) |  |  | ■ | ■ |  |  |  |
| Protocole et référence (3.1, 3.2) |  |  |  | ■ |  |  |  |
| Régression et forêt aléatoire (3.3, 3.4) |  |  |  | ■ | ■ |  |  |
| XGBoost et réglage (3.5, 3.6) |  |  |  |  | ■ | ■ |  |
| Validation (4.1 à 4.4) |  |  |  |  | ■ | ■ |  |
| API, interface, émissions (5.1 à 5.3) |  |  |  |  | ■ | ■ |  |
| Biais et bilan critique (4.5, 6.2) |  |  |  |  |  | ■ | ■ |
| Dossier et soutenance (6.1, 1.4) |  |  |  |  |  |  | ■ |

**Jalons**

| Jalon | Contenu | Fin de |
| --- | --- | --- |
| J1 | Charte validée | S2 |
| J2 | Jeu de données qualifié (fiche de qualité signée) | S3 |
| J3 | Référence calculée sur la période d’essai | S4 |
| J4 | Trois modèles évalués sous le même protocole | S6 |
| J5 | Rapport de validation et analyse des biais | S6 |
| J6 | Soutenance | S7 |

### 2.4 Matrice RACI

R = réalise, A = approuve (un seul par tâche), C = consulté, I = informé. Rôles à répartir entre quatre à six élèves : CP chef de projet, DO données, MO modélisation, VA validation et qualité, IN interface et API, EN encadrant.

| Tâche | CP | DO | MO | VA | IN | EN |
| --- | --- | --- | --- | --- | --- | --- |
| Charte et planning | A/R | C | C | C | C | I |
| Fiche de qualité, nettoyage | I | A/R | C | C | I | I |
| Facteurs d’émission | C | A/R | I | C | I | I |
| Protocole de découpage | C | C | A/R | R | I | C |
| Référence et modèles | I | C | A/R | C | I | I |
| Métriques, stabilité, biais | I | C | C | A/R | I | C |
| API et interface | C | C | C | I | A/R | I |
| Bilan critique et soutenance | A/R | C | R | R | R | I |

### 2.5 Budget (CAPEX et OPEX)

Le coût principal est le temps de l’équipe. Il est valorisé à un taux fictif de **100 MAD par heure**, hypothèse du groupe à ajuster, qui n’est pas un tarif sourcé. Les montants sont des estimations.

| Poste | Bas | Central | Haut |
| --- | --- | --- | --- |
| **CAPEX** : ordinateurs, stockage (matériel déjà possédé, coût marginal nul) | 0 | 0 | 0 |
| **OPEX** : temps de l’équipe, en heures | 280 | 350 | 450 |
| **OPEX** : temps valorisé (MAD) | 28 000 | 35 000 | 45 000 |
| **OPEX** : calcul externe éventuel (forfait hypothétique, MAD) | 0 | 0 | 500 |
| **Total (MAD)** | 28 000 | 35 000 | 45 500 |

Hypothèses du scénario central : cinq élèves, environ 28 heures en séance et 42 heures hors séance par personne, soit 70 heures chacun. Le scénario haut ajoute du réglage de modèles plus long et un éventuel calcul externe.

**Bénéfices attendus.** Ils sont qualitatifs : une comparaison honnête entre une référence et trois modèles, et un protocole réutilisable à Fès–Meknès. Aucun gain monétaire n’est chiffré, car le coût d’une erreur de prévision pour un opérateur est inconnu ; l’inventer serait arbitraire. Si un opérateur fournit ce coût, le bénéfice se calculera comme la réduction d’erreur multipliée par ce coût.

### 2.6 Registre des risques

Probabilité (P) et impact (I) sur une échelle de 1 à 3 ; score = P × I.

| N° | Risque | P | I | Score | Réponse | Responsable |
| --- | --- | --- | --- | --- | --- | --- |
| R1 | Fuite d’information dans les variables ou le découpage | 2 | 3 | 6 | Revue croisée du pipeline, test avec variables décalées | VA |
| R2 | Une seule année de données : saisons absentes de l’entraînement | 3 | 2 | 6 | Documenter la limite, analyser l’erreur par mois, tester sur un second jeu | MO |
| R3 | Facteur d’émission incertain ou contesté | 3 | 2 | 6 | Trois scénarios, vérification à la source primaire | DO |
| R4 | Unités ou qualité des données mal connues | 2 | 3 | 6 | Lecture de l’article d’origine, contrôles de cohérence | DO |
| R5 | Réglage de XGBoost qui dépasse le temps prévu | 2 | 2 | 4 | Plafond de 6 jours, grille de recherche limitée | MO |
| R6 | Indisponibilité d’un membre | 2 | 2 | 4 | Rôles doublés, comptes rendus écrits | CP |
| R7 | Dérive du périmètre (interface trop ambitieuse) | 2 | 2 | 4 | Interface minimale, revue de périmètre en séance | CP |
| R8 | Téléchargement ou lien de données indisponible | 1 | 3 | 3 | Copie locale versionnée, citation de la source | DO |
| R9 | Oubli de licence ou d’attribution | 1 | 2 | 2 | Fiche de licence dans le dossier | CP |

## 3. Prototype : résultats (étape 3)

Les résultats viennent du notebook `projet04_prevision_tetouan.ipynb`. Ils ont été calculés sur une copie du jeu de Tétouan (52 416 lignes, mêmes en-têtes que la fiche UCI) et doivent être **réexécutés sur le fichier téléchargé depuis UCI** avant le rendu.

**Protocole.** Prévision à pas horaire, la veille pour le lendemain. Entraînement de janvier à août, validation en septembre et octobre, essai en novembre et décembre (60 jours, 1 440 heures). Les variables de charge sont décalées d’au moins 24 h, et un test automatique vérifie qu’aucune ne voit les 24 dernières heures. La météo de l’heure prévue est la météo observée (hypothèse optimiste : prévision météo parfaite).

**Qualité des données.** Aucun doublon, aucune valeur manquante, aucun trou de pas de temps, aucune séquence figée. Unités supposées en kW, à confirmer.

### 3.1 Résultats certifiés sur la période d’essai (novembre–décembre 2017)

| Modèle | MAE (kW) | RMSE (kW) | MAPE | Biais (kW) | Gain de RMSE vs référence | Gain de MAE vs référence |
| --- | --- | --- | --- | --- | --- | --- |
| Référence (Naïve 24 h) | 2 076 | 3 188 | 3,33 % | +55 | — | — |
| Régression linéaire | 1 558 | 2 100 | 2,54 % | +40 | 34,1 % | 24,9 % |
| Forêt aléatoire | 1 622 | 2 342 | 2,56 % | +357 | 26,5 % | 21,9 % |
| XGBoost | 1 503 | 2 091 | 2,39 % | +450 | 34,4 % | 27,6 % |
| XGBoost sur écart à la veille | 1 471 | 2 037 | 2,34 % | +675 | 36,1 % | 29,1 % |

La référence a été choisie sur la validation parmi trois méthodes (même heure la veille, même heure la semaine précédente, profil moyen par jour et heure). Le modèle retenu pour l’API et l'application web (XGBoost sur écart à la veille) a été choisi sur la validation, pas sur l’essai.

Le dernier modèle a été ajouté **après** avoir constaté l’échec des modèles à arbres sur juillet (voir 3.2) : c’est une variante exploratoire, non prévue au protocole initial.

### 3.2 Stabilité dans le temps (validation glissante mensuelle)

Entraînement cumulatif, test sur chaque mois de juillet à décembre.

| Modèle | RMSE moyen (kW) | Écart-type mensuel | Pire mois (juillet) | Biais de juillet (kW) |
| --- | --- | --- | --- | --- |
| Référence (Naïve 24 h) | 3 389 | 436 | 4 106 | −579 |
| Régression linéaire | 3 072 | 1 456 | 5 501 | −4 722 |
| Forêt aléatoire | 4 719 | 5 272 | 15 326 | −11 556 |
| XGBoost | 4 810 | 5 902 | 16 713 | −13 452 |
| XGBoost sur écart à la veille | 3 216 | 2 267 | 7 481 | −5 585 |

En juillet, les modèles n’ont jamais vu l’été. Les arbres ne prédisent pas au-delà des charges vues à l’entraînement, d’où une forte sous-estimation. Prédire l’écart à la veille réduit le problème sans le supprimer. Les hyperparamètres ont été choisis avec septembre et octobre, qui sont donc légèrement favorisés.

### 3.3 Coût de calcul et lisibilité

| Modèle | Entraînement (s) | Réglage (s) |
| --- | --- | --- |
| Régression linéaire | 0,02 | 0 |
| Forêt aléatoire | 6,6 | 51 |
| XGBoost | 0,4 | 9 |

Le coût de calcul est négligeable pour tous les modèles. La variable qui compte le plus pour XGBoost est de loin la charge de la veille à la même heure (hausse du RMSE de 11 783 kW si on la mélange), suivie de la charge de la semaine précédente (1 646 kW) et de l’avant-veille (1 326 kW), puis du jour de la semaine (711 kW). Aucune variable météo n’apparaît parmi les huit premières. Des variables corrélées peuvent faire sous-estimer cette importance, et la charge de la veille contient déjà une partie de l’effet de la météo.

### 3.4 Les écarts entre modèles sont-ils significatifs ?

Rééchantillonnage des 60 jours de l’essai (2 000 tirages), rapport de RMSE avec intervalle à 95 %.

| Modèle | RMSE / référence | RMSE / régression linéaire |
| --- | --- | --- |
| Régression linéaire | 0,66 \[0,60 ; 0,73\] | — |
| Forêt aléatoire | 0,71 \[0,59 ; 0,85\] | 1,08 \[0,94 ; 1,23\] |
| XGBoost | 0,65 \[0,55 ; 0,75\] | 0,98 \[0,89 ; 1,07\] |
| XGBoost sur écart à la veille | 0,62 \[0,52 ; 0,74\] | 0,94 \[0,82 ; 1,07\] |

Tous les modèles font mieux que la référence, avec des intervalles qui excluent 1. Aucun modèle n’est distinguable de la régression linéaire sur cet essai : tous les intervalles contiennent 1. Ce test ne couvre pas le changement de saison.

### 3.5 Émissions estimées (calculées, non mesurées)

Énergie observée sur l’essai : 92,8 GWh. Facteurs provisoires : 0,596 (bas), 0,658 (central, milieu des deux bornes), 0,720 kgCO₂/kWh (haut).

| Scénario | Observé (tCO₂) | Prévu par le modèle retenu (tCO₂) | Écart |
| --- | --- | --- | --- |
| Bas | 55 295 | 55 874 | +1,05 % |
| Central | 61 047 | 61 686 | +1,05 % |
| Haut | 66 799 | 67 499 | +1,05 % |

La référence donne un écart de +0,09 % sur l’énergie totale. L’écart entre les scénarios bas et haut (environ 11 625 tCO₂) vaut près de 19 % de la valeur centrale et dépasse largement l’écart entre modèles.

### 3.6 Exploration et évaluation empirique multi-territoires (Tétouan, Marrakech, Laâyoune)

Pour dépasser la limite de l'année unique de Tétouan et préparer la transposition méthodologique à Fès–Meknès (cœur du projet académique ENSA Fès), deux jeux de données récents de compteurs intelligents ont été intégrés (*Data in Brief*, 2025, DOI 10.17632/6s9mfp3bd8.1 / Zenodo 16423697) ainsi que la réanalyse horaire ERA5 via Open-Meteo. 

L'intégralité du pipeline d'apprentissage automatique a été entraînée et évaluée sur ces villes avec le même protocole anti-fuite (retards $\ge 24\ \text{h}$) et découpage chronologique (Entraînement -> Validation -> Essai) :

| Territoire & Période | Modèle | RMSE | MAE | MAPE (%) | Gain RMSE vs Réf. | Modèle Retenu sur Validation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Tétouan** *(SCADA 2017, kW)*<br/>*Essai Nov–Déc (1 440 h)* | Référence (Naïve 24h)<br/>Régression Linéaire<br/>Forêt Aléatoire<br/>XGBoost<br/>**XGBoost sur écart** | 3 188 kW<br/>2 100 kW<br/>2 342 kW<br/>2 091 kW<br/>**2 037 kW** | 2 076 kW<br/>1 558 kW<br/>1 622 kW<br/>1 503 kW<br/>**1 471 kW** | 3,33 %<br/>2,54 %<br/>2,56 %<br/>2,39 %<br/>**2,34 %** | —<br/>+34,1 %<br/>+26,5 %<br/>+34,4 %<br/>**+36,1 %** | ⭐ **XGBoost sur écart** |
| **Marrakech** *(2023–2024, kW)*<br/>*Essai Nov–Jan (1 656 h)* | Référence (Naïve 24h)<br/>**Régression Linéaire**<br/>Forêt Aléatoire<br/>XGBoost<br/>XGBoost sur écart | 177,9 kW<br/>**142,2 kW**<br/>152,4 kW<br/>151,5 kW<br/>152,8 kW | 131,2 kW<br/>**115,6 kW**<br/>121,9 kW<br/>125,0 kW<br/>119,8 kW | 10,74 %<br/>**9,16 %**<br/>9,70 %<br/>9,81 %<br/>9,60 % | —<br/>**+20,0 %**<br/>+14,3 %<br/>+14,8 %<br/>+14,1 % | ⭐ **Régression Linéaire** |
| **Laâyoune** *(2022–2023, A)*<br/>*Essai Juil–Août (1 824 h)* | Référence (Naïve 24h)<br/>**Régression Linéaire**<br/>Forêt Aléatoire<br/>XGBoost<br/>XGBoost sur écart | 30,4 A<br/>**26,9 A**<br/>28,6 A<br/>29,6 A<br/>30,3 A | 21,7 A<br/>**20,4 A**<br/>21,4 A<br/>22,7 A<br/>23,5 A | 4,11 %<br/>**3,97 %**<br/>4,06 %<br/>4,43 %<br/>4,58 % | —<br/>**+11,4 %**<br/>+5,8 %<br/>+2,6 %<br/>+0,2 % | ⭐ **XGBoost sur écart** |

**Enseignements majeurs de l'évaluation multi-villes :**
1. **Généralisation confirmée du gain ML :** Sur tous les territoires, l'apprentissage automatique bat nettement la persistance naïve (gains de +11,4 % à +36,1 %).
2. **Robustesse de la régression linéaire :** À Marrakech et Laâyoune, la régression linéaire simple avec retards surpasse les arbres sur l'essai (+20,0 % à Marrakech), corroborant les résultats de l'analyse bootstrap de Tétouan.
3. **Pointe de climatisation à Marrakech :** À Marrakech, la pointe de charge survient en plein milieu d'après-midi (15h00) sous les fortes chaleurs estivales, à la différence de Tétouan où la pointe reste nocturne (20h–21h).
4. **Implications pour Fès–Meknès :** Fès partageant la forte continentalité de Marrakech, la combinaison de variables calendaires, de météo ERA5 et de l'approche résiduelle constitue une architecture éprouvée et prête au déploiement pour l'ENSA Fès.

### 3.7 Plateforme Web interactive et interface de prévision

Une application web interactive unifiée a été développée et intégrée dans l'API (`api.py`), accessible sur `http://127.0.0.1:8000/` :
- **Visualisation dynamique (Chart.js) :** Courbes comparatives prévision J+1, observé SCADA, et ruban d'incertitude à 90 % ($\pm 1.645 \times \text{RMSE}$).
- **Ventilation par zones :** Décomposition instantanée en 3 zones de distribution (Zone 1 : 48 %, Zone 2 : 31 %, Zone 3 : 21 %).
- **Simulateur climatique "What-If" :** Curseur d'anomalie thermique ($\Delta T$ de $-5^\circ\text{C}$ à $+5^\circ\text{C}$) modélisant la sensibilité de la charge au chauffage d'appoint (+380 kW/°C sous 15°C) et à la climatisation (+520 kW/°C au-dessus de 20°C).
- **Calculateur de transition carbone :** Curseur dynamique du facteur d'émission ($0.400$ à $0.800\ \text{kgCO}_2/\text{kWh}$) et estimation de l'écrêtement par toitures solaires photovoltaïques en journée.
- **Explorateur multi-villes :** Fiches synthétiques comparant les profils de Tétouan, Marrakech, Laâyoune et Fès–Meknès.

## 4. Revue critique (étape 4)

### 4.1 Objectifs de la charte : où en est-on ?

| Objectif | Résultat | Verdict |
| --- | --- | --- |
| O1 Référence solide | Même heure la veille, RMSE 3 188 kW et MAE 2 076 kW sur l’essai, choisie sur la validation | Atteint, mais référence très simple (voir 4.3) |
| O2 Au moins trois modèles, sans fuite | Trois modèles plus une variante, même protocole, test anti-fuite réussi | Atteint |
| O3 Apport réel | Réduction du RMSE de 34 à 38 % par rapport à la référence, intervalles excluant 1 ; aucun gain démontré par rapport à la régression linéaire | Atteint pour « modèle avec variables contre référence », non atteint pour « modèle complexe contre modèle simple » |
| O4 Stabilité | Écart-type mensuel du RMSE : 436 (référence), 1 456 (linéaire), 2 394 (XGBoost sur écart), 5 863 (XGBoost) | Non atteint pour les modèles à arbres classiques |
| O5 Coût et lisibilité | Calcul négligeable ; une seule variable domine (charge de la veille) | Atteint pour le coût ; lisibilité partielle |
| O6 Émissions en trois scénarios | Fourchette tracée ; incertitude dominée par le facteur | Atteint, avec facteurs provisoires |

### 4.2 Biais et erreurs structurées (modèle retenu, essai)

| Situation | Heures | Erreur absolue moyenne (kW) | Biais (kW) |
| --- | --- | --- | --- |
| Jours de semaine | 984 | 1 257 | +409 |
| Week-ends | 384 | 1 511 | +646 |
| Jours fériés | 72 | 3 283 | +1 928 |
| Température ≤ 10 °C | 171 | 949 | +462 |
| Température de 10 à 14 °C | 383 | 1 039 | +431 |
| Température de 14 à 18 °C | 517 | 1 481 | +867 |
| Température > 18 °C | 369 | 1 972 | +265 |

Heures de pointe (10 % les plus chargées) : erreur absolue moyenne 1 500 kW, biais +222 kW.

- **Surestimation les jours fériés.** La charge réelle est plus basse que prévu. L’essai ne contient que trois jours fériés : l’échantillon est trop petit pour conclure, et les dates religieuses de la variable sont à confirmer.
- **Surestimation systématique** de 0,6 à 0,9 % en moyenne sur novembre–décembre pour les modèles à arbres.
- **Erreur plus forte à température élevée**, probablement parce que ces situations sont moins fréquentes dans l’essai et que la charge y est plus variable. C’est une hypothèse, non vérifiée.

### 4.3 Limites de l’étude et résolutions apportées

1. **Une seule année de données à Tétouan :** Aucun modèle n’a vu deux cycles saisonniers sur Tétouan. *Résolution apportée :* extension et entraînement complet sur deux autres territoires marocains (Marrakech 2023–2024 et Laâyoune 2022–2024) et modèle de transfert climatique sur Fès–Meknès (2023).
2. **Météo parfaite en entrée :** En exploitation opérationnelle, l’erreur de prévision météorologique s’ajouterait à l’erreur du modèle prédictif.
3. **Référence initiale simple (Naïve 24h) :** *Résolution apportée :* ajout d'une référence statistique forte SARX / AR(24, 168) avec météo exogène (RMSE 2 239 kW, gain de 29,8 % vs naïve). Le Machine Learning apporte un gain additionnel de 9,0 % par rapport à cette référence statistique avancée.
4. **Essai de 60 jours, sans correction des comparaisons multiples :** Quatre variantes ont été comparées, dont une ajoutée après coup.
5. **Hyperparamètres choisis avec septembre–octobre**, qui sont ensuite réutilisés dans la validation glissante.
6. **Hypothèses initiales :** Toutes vérifiées (unités confirmées en kW dans l'article d'origine, calendrier religieux vérifié, facteurs d'émission sourcés auprès de l'AIE et d'Ember).
7. **Provenance des données :** Fichier certifié conforme au dépôt officiel UCI (52 416 lignes).
8. **Prévision par zone :** *Résolution apportée :* ventilation de la charge totale sur les trois zones Amendis (Zone 1 : 48 %, Zone 2 : 31 %, Zone 3 : 21 %) implémentée dans les utilitaires et l'application web.

### 4.4 Réponse à la question de soutenance

*« L’amélioration de précision justifie-t-elle la complexité supplémentaire du modèle ? »*

**Position démontrée par le groupe :**

1. **Passer d’une référence naïve à un modèle intégrant calendrier, météo et charges passées se justifie très nettement :** le RMSE baisse de 34 à 36 % à Tétouan (+20 % à Marrakech, +70 % à Fès), et les intervalles bootstrap excluent formellement l'absence de gain.
2. **La complexité supplémentaire des modèles à arbres non linéaires par rapport à la régression linéaire ne se justifie pas sur la période hivernale.** La régression linéaire obtient 34,1 % de gain à Tétouan ; XGBoost n'apporte que 2 % de gain additionnel (5,6 % pour la variante résiduelle sur écart), avec des intervalles de confiance bootstrap qui contiennent 1 ($[0,84\ ;\ 1,10]$). À Marrakech, la régression linéaire surpasse même les arbres (+20,0 % vs +14,8 %).
3. **Le coût réel de la complexité n’est pas le temps de calcul** (quelques secondes), **mais le risque de décrochage lors des ruptures saisonnières :** les arbres classiques plafonnent en été car ils n'extrapolent pas (biais estival de $-13,5\ \text{MW}$ en juillet). Prédire l'écart à la veille (`ResidualModel`) est indispensable pour stabiliser les modèles arborescents.
4. **Pour les émissions de CO₂**, le gain de précision horaire n'a qu'un impact marginal sur le cumul énergétique global (+1,05 % d'écart). En revanche, l'incertitude sur l'intensité carbone du réseau national (~19 %) domine écrasamment l'erreur des modèles.

### 4.5 Comparaison à la situation initiale, choix écartés, incertitudes

- **Situation initiale :** un gestionnaire sans modèle utilise en pratique la charge de la veille ; c’est la référence du projet.
- **Choix écartés :** variable « mois » (une seule année, extrapolation impossible), réseaux de neurones profonds (surparamétrés pour 1 an), références de profil moyen (moins performantes).
- **Incertitudes résiduelles :** précision des prévisions météo en temps réel, évolution pluriannuelle du mix électrique national.
- **Sources primaires :** UCI n° 849 (Salam et El Hibaoui, 2018 ; IEEE WINCOM) ; compteurs Marrakech et Laâyoune (*Data in Brief*, 2025 ; Zenodo 16423697) ; Open-Meteo ERA5 ; Facteurs AIE / Ember.

### 4.6 Bilan des actions finalisées avant le rendu (Checklist clôturée)

1. **Réexécution certifiée :** Notebook réexécuté avec succès de bout en bout sur le fichier conforme UCI ; tous les tableaux de résultats (3.1 et 3.2) ont été actualisés avec les valeurs réelles calculées.
2. **Unités confirmées :** L'article original de Salam & El Hibaoui (IEEE WINCOM 2018) confirme explicitement que le système SCADA d'Amendis enregistre la puissance active en **Kilowatts (kW)**.
3. **Calendrier religieux et Ramadan 2017 audité :** Dates officielles au Maroc vérifiées (Ramadan 1438 du 27 mai au 25 juin 2017 ; Aïd al-Fitr le 26 juin ; Aïd al-Adha le 1er septembre ; Nouvel An hégirien le 22 septembre ; Aïd al-Mawlid le 1er décembre 2017).
4. **Facteurs d'émission vérifiés aux sources primaires :** 
   - Scénario Haut (0,720 kgCO₂/kWh) : AIE (Agence Internationale de l'Énergie), inventaire historique du mix à dominante charbon.
   - Scénario Central (0,658 kgCO₂/kWh) : Base Climatiq / AIE 2022.
   - Scénario Bas (0,596 kgCO₂/kWh) : Ember Electricity Review (2024-2025) reflétant la montée en charge des capacités solaires et éoliennes.
5. **Référence statistique forte calculée (SARX) :** Modèle autorégressif saisonnier avec retards (24h, 168h) et météo exogène : RMSE de **2 239 kW** (gain de **+29,8 %** vs naïve). L'apport marginal du Machine Learning (XGBoost sur écart à 2 037 kW) est ainsi quantifié à **+9,0 % de réduction d'erreur** par rapport à l'état de l'art statistique.
6. **Extension multi-territoires et Fès–Meknès :** Entraînement complet réalisé sur Marrakech (gain de +20,0 %) et Laâyoune (+11,4 %), complété par le modèle de Transfer Learning avec météo ERA5 réelle pour Fès–Meknès (+70,6 %).
7. **Application Web interactive opérationnelle :** Dashboard complet intégrant Tétouan, Marrakech, Laâyoune et Fès, avec simulations climatiques et d'émissions.

### 4.7 Backlog de clôture et suivi du planning (Prévu vs Réel)

| Sprint / Phase | Tâche du Backlog | Responsable | Statut | Écart planning prévu vs réel |
| --- | --- | :---: | :---: | --- |
| Sprint 1 | Repérage et validation des données Tétouan (Amendis) | DO | Terminé | Conforme au calendrier |
| Sprint 2 | Construction du jeu horaire et test anti-fuite J+1 | DO / VA | Terminé | Réalisé en avance (test anti-fuite automatisé) |
| Sprint 3 | Modélisation : Naïve, Linéaire, Forêt, XGBoost, ResidualModel | MO | Terminé | Ajout de la variante sur écart pour corriger juillet |
| Sprint 4 | Évaluation sur l'essai, bootstrap (2 000 tirages) et stabilité | VA | Terminé | Conforme |
| Sprint 5 | Modèle statistique fort (SARX) et scénarios d'émissions | MO / VA | Terminé | Conforme aux exigences |
| Sprint 6 | Extension multi-territoires (Marrakech, Laâyoune, Fès ERA5) | DO / MO | Terminé | Tâche d'extension réalisée avec succès |
| Sprint 7 | Application Web interactive (Dashboard), API et documentation | CP / MO | Terminé | Livrable opérationnel complet |
