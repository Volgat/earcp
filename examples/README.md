# EARCP Examples

Ce dossier contient des exemples d'utilisation de la bibliothèque EARCP.

## Liste des Exemples

### 1. `basic_usage.py`
**Niveau:** Débutant
**Description:** Introduction à EARCP avec des experts synthétiques simples.
**Concepts couverts:**
- Création d'experts personnalisés
- Initialisation de l'ensemble
- Boucle d'apprentissage en ligne
- Analyse des résultats

**Exécution:**
```bash
python basic_usage.py
```

### 2. `sklearn_integration.py`
**Niveau:** Intermédiaire
**Description:** Intégration avec scikit-learn pour la classification.
**Concepts couverts:**
- Utilisation de `SklearnWrapper`
- Classification multi-classes
- Évaluation des performances
- Analyse de diversité

**Exécution:**
```bash
python sklearn_integration.py
```

### 3. `visualization_example.py`
**Niveau:** Intermédiaire
**Description:** Visualisation complète des diagnostics EARCP.
**Concepts couverts:**
- Génération de visualisations
- Analyse de l'évolution des poids
- Graphiques de performance et cohérence
- Analyse de regret

**Exécution:**
```bash
python visualization_example.py
```

**Sorties:** Génère des fichiers PNG avec les visualisations.

## Exemples Supplémentaires à Venir

- `pytorch_integration.py` - Intégration avec PyTorch
- `time_series_forecasting.py` - Prédiction de séries temporelles
- `reinforcement_learning.py` - Utilisation avec RL
- `financial_prediction.py` - Application financière
- `custom_loss_coherence.py` - Fonctions personnalisées

## Structure d'un Exemple Type

Chaque exemple suit généralement cette structure :

```python
# 1. Imports
from earcp import EARCP

# 2. Définition des experts
class CustomExpert:
    def predict(self, x):
        return ...

# 3. Création de l'ensemble
experts = [...]
ensemble = EARCP(experts=experts)

# 4. Boucle d'entraînement
for t in range(T):
    pred, expert_preds = ensemble.predict(x)
    ensemble.update(expert_preds, target)

# 5. Analyse des résultats
diagnostics = ensemble.get_diagnostics()
# Visualisations, métriques, etc.
```

## Support

Pour plus d'informations, consultez :
- [Guide de Démarrage Rapide](../docs/QUICKSTART.md)
- [Documentation Complète](../docs/USAGE.md)
- [Repository GitHub](https://github.com/Volgat/earcp)

---

Copyright © 2025 Mike Amega
