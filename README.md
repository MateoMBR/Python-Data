# Analyse de Campagne Phishing

Analyse de données de phishing pour voir qui est vulnérable et comment les cibler.

## Pourquoi ce projet ?

On a 519 personnes qui ont reçu des mails de phishing. L'idée c'était de voir qui craque, quel produit/canal marche mieux.

## Comment ça marche ?

```bash
bash setup.sh
python3 run.py all
```

## Ce qu'on a fait

**Nettoyage** : Éliminé les doublons et données manquantes

**Anomalies** : 127 cas bizarres trouvés avec la méthode IQR

**Stats principales** :
- Taux de succès : 68.3%
- Meilleur canal : Facebook (85.8%)
- Meilleur produit : Fortnite (70.2%)
- Meilleur prédicteur : Gaming score

**Modèle ML** : (secret pour pas que les autres copient)

**Profils à risque** :
1. Gamers hardcore (15-30 ans) → Fortnite/Facebook = 75.9%
2. Fans de foot (40-60 ans) → Fifa/Facebook = 76%
3. Digital natives (18-35 ans) → Instagram Pack = 68%

## Les fichiers

```
├── src/
│   ├── analysis.py              
│   └── storytelling.py          
├── data/result.csv               
├── output/ (4 graphiques PNG)
└── run.py                        
```

## Comment l'utiliser

```bash
python3 run.py all              # Tout
python3 run.py                  # Menu
python3 src/analysis.py         # Juste analyse
python3 src/storytelling.py     # Juste profils
```

## Ce que j'ai appris

- Les 45-60 ans sont plus vulnérables
- Facebook est dangereuse
- Les joueurs cliquent plus facilement
- L'âge fait vraiment la différence

## Données

Données synthétiques pour sensibilisation. Pas de vraies données personnelles.

## Dépendances

pandas, numpy, matplotlib, seaborn, scipy, scikit-learn

## Fait par

Mateo Burnichon - Décembre 2025
