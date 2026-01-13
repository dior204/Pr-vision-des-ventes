# 📊 Favorita Grocery Sales Forecasting  
### Projet Projet de Machine Learning de prévision des ventes – REG09

## 📝 Description
Ce projet vise à développer un modèle de prévision des ventes quotidiennes
pour la chaîne de supermarchés équatorienne Favorita, à partir de données
historiques multi-sources.

Le projet a été réalisé dans le cadre du module Machine Learning,
avec une approche complète allant de l’exploration des données jusqu’à la
visualisation des résultats via un dashboard.

---

## 🎯 Objectifs

### Objectif principal
Développer un modèle de Machine Learning capable de prédire de manière fiable
les ventes journalières (`unit_sales`) par produit et par magasin.

### Objectifs spécifiques
- Explorer et comprendre les données
- Nettoyer et préparer les données brutes
- Concevoir des variables explicatives pertinentes
- Comparer plusieurs modèles de Machine Learning
- Sélectionner un modèle performant et interprétable
- Visualiser et interpréter les résultats

---

## 🗂️ Données utilisées

Les données proviennent du jeu **Favorita Grocery Sales Forecasting** et
comprennent :

- `train.csv` : historique des ventes (variable cible `unit_sales`)
- `test.csv` : données sans variable cible
- `sample_submission.csv` : format attendu des prédictions
- `items.csv` : informations produits
- `stores.csv` : informations magasins
- `transactions.csv` : volumes de transactions
- `oil.csv` : prix journalier du pétrole
- `holidays_events.csv` : jours fériés et événements

Les données sont **temporelles**, **volumineuses** et **multi-sources**.

Veuillez trouver le lien vers la base [ici](https://drive.google.com/file/d/1iM4J3dU2LuY9FHlGI-I04fmYgzCuSjf1/view?usp=drive_link)
---

## 🏗️ Architecture du projet

Le projet est structuré autour d’un pipeline Machine Learning clair et
reproductible :

1. Exploration des données (EDA)
2. Prétraitement des données et Feature engineering
4. Modélisation et sélection du meilleur modèle
5. Dashboard
6.  Lien de la présentation

---

## 📘 Structure des notebooks

### 📙 Notebook 01 – EDA
- Analyse des distributions
- Étude des tendances temporelles
- Analyse de l’impact des promotions et des catégories
- Identification des valeurs manquantes et des valeurs extrêmes

Veuillez trouver le lien [ici](https://www.kaggle.com/code/jacquesilly/eda-pr-vision-des-ventes)
### 📙 Notebook 02 – Prétraitement & Feature Engineering & Modélisation (Pipeline)
- Traitement des valeurs manquantes
- Correction des valeurs négatives
- Harmonisation des formats et types
- Création de variables temporelles
- Création de lags et statistiques glissantes
- Encodage des variables catégorielles
- Fusion des sources de données
- Construction du DataFrame final
- Découpage temporel train / validation
- Implémentation de modèles de base (baseline)
- Entraînement de plusieurs modèles ML
- Évaluation à l’aide de MAE et RMSE
- Sélection du modèle final

### 📙 Dashboard
Veuillez trouver le lien vers le githup du Dashboard [ici](https://github.com/Khadijah19/Pr-vision-des-ventes)
---

## 📘 Lien vers la présentation [ici](https://www.canva.com/design/DAG-KZn1N-8/a8oNo0hmCXUdjgNboP18Ww/edit?utm_content=DAG-KZn1N-8&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

## 🤖 Modèles et métriques

### Modèles testés
Modèles évalués

Plusieurs modèles ont été testés afin de comparer leurs performances et d’identifier la meilleure approche pour la prévision des ventes :

🔹 Modèles linéaires régularisés

Ridge Regression (α = 1.0)

Lasso Regression (α = 0.01)

ElasticNet (α = 0.01, l1_ratio = 0.5)

Ces modèles servent de références interprétables, permettant d’évaluer la capacité des relations linéaires à expliquer les ventes.

🔹 Modèles avancés basés sur les arbres

LightGBM

CatBoost

Ces modèles sont capables de capturer des relations non linéaires complexes, des interactions entre variables ainsi que des effets saisonniers.

🔹 Modèles de référence (baselines)

Naive model

Seasonal Naive model

Ces modèles simples servent de points de comparaison, afin de vérifier que les modèles avancés apportent un réel gain de performance.
### Métriques d’évaluation
Les performances ont été évaluées à l’aide des métriques suivantes, calculées sur les ensembles d’entraînement et de test :

MSE (Mean Squared Error)

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

MAPE (Mean Absolute Percentage Error)

R² (coefficient de détermination)

Une séparation temporelle a été utilisée afin de respecter la nature chronologique des données.
---

## 📈 Résultats

Le modèle final sélectionné :
- offre les meilleures performances globales
- capture la saisonnalité et les effets de promotion
- met en évidence l’importance des ventes passées

Les résultats sont présentés à travers un dashboard facilitant
l’interprétation et la prise de décision.

---

## ⚠️ Limites

- Données volumineuses impliquant des temps de calcul élevés
- Modèle relativement lourd en ressources
- Absence de certaines variables clés (stocks, prix réels)
- Sensibilité aux évolutions futures du contexte économique

---

## 🚀 Perspectives

- Optimisation du pipeline de traitement
- Exploration de modèles plus légers ou plus avancés
- Intégration de données externes supplémentaires
- Entraînement distribué / cloud
- Déploiement en environnement de production

---

## 👥 Équipe

Projet réalisé par :
- Khadidiatou DIAKHATE
- Aissatou Sega DIALLO  
- Haba Fromo Francis
- Jacques ILLY
- Dior MBENGUE

---

## ✅ Conclusion

Ce projet illustre la mise en œuvre complète d’un pipeline de Machine Learning
appliqué à un problème réel de prévision des ventes, avec une approche
structurée, méthodologique et orientée décision.

