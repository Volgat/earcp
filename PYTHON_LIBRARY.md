# EARCP - Bibliothèque Python

**Implémentation Python complète et professionnelle de l'architecture EARCP**

---

## 📦 Structure de la Bibliothèque

```
earcp/
├── earcp/                      # Package principal
│   ├── __init__.py            # Exports publics
│   ├── config.py              # Configuration et presets
│   ├── core/                  # Modules cœur
│   │   ├── __init__.py
│   │   ├── performance_tracker.py    # Suivi des performances
│   │   ├── coherence_metrics.py      # Métriques de cohérence
│   │   └── ensemble_weighting.py     # Calcul des poids
│   ├── models/                # Modèles
│   │   ├── __init__.py
│   │   └── earcp_model.py            # Classe EARCP principale
│   └── utils/                 # Utilitaires
│       ├── __init__.py
│       ├── visualization.py          # Visualisations
│       ├── metrics.py                # Métriques d'évaluation
│       └── wrappers.py               # Wrappers pour frameworks ML
├── examples/                   # Exemples d'utilisation
│   ├── README.md
│   ├── basic_usage.py
│   ├── sklearn_integration.py
│   └── visualization_example.py
├── docs/                       # Documentation
│   ├── USAGE.md               # Guide complet
│   └── QUICKSTART.md          # Démarrage rapide
├── tests/                      # Tests
│   └── test_basic.py
├── setup.py                    # Configuration d'installation
├── requirements.txt            # Dépendances
├── MANIFEST.in                # Fichiers à inclure dans la distribution
└── LICENSE                     # Licence MIT

```

## 🚀 Installation

### Depuis PyPI (quand publié)

```bash
pip install earcp
```

### Depuis le dépôt Git

```bash
git clone https://github.com/Volgat/earcp.git
cd earcp
pip install -e .
```

### Avec dépendances optionnelles

```bash
# Support complet
pip install earcp[full]

# Uniquement PyTorch
pip install earcp[torch]

# Uniquement scikit-learn
pip install earcp[sklearn]
```

## 📖 Utilisation

### Import Simple

```python
from earcp import EARCP

# Créer des experts
experts = [model1, model2, model3]

# Initialiser EARCP
ensemble = EARCP(experts=experts, beta=0.7, eta_s=5.0)

# Prédire et mettre à jour
for x, y in data:
    pred, expert_preds = ensemble.predict(x)
    ensemble.update(expert_preds, y)
```

### Avec Configuration Avancée

```python
from earcp import EARCP, EARCPConfig, get_preset_config

# Option 1: Configuration personnalisée
config = EARCPConfig(
    alpha_P=0.9,
    alpha_C=0.85,
    beta=0.7,
    eta_s=5.0,
    w_min=0.05,
    track_diagnostics=True
)

ensemble = EARCP(experts=experts, config=config)

# Option 2: Preset
config = get_preset_config('balanced')
ensemble = EARCP(experts=experts, config=config)
```

### Intégration avec scikit-learn

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from earcp import EARCP
from earcp.utils.wrappers import SklearnWrapper

# Entraîner des modèles sklearn
models = [Ridge(), RandomForestRegressor()]
for model in models:
    model.fit(X_train, y_train)

# Utiliser avec EARCP
experts = [SklearnWrapper(model) for model in models]
ensemble = EARCP(experts=experts)
```

## 📊 Modules Principaux

### 1. Core Modules

#### `PerformanceTracker`
Suit et lisse les scores de performance des experts.
```python
from earcp.core import PerformanceTracker

tracker = PerformanceTracker(n_experts=3, alpha=0.9)
scores = tracker.update(predictions, target)
```

#### `CoherenceMetrics`
Calcule la cohérence (accord) entre experts.
```python
from earcp.core import CoherenceMetrics

coherence = CoherenceMetrics(n_experts=3, alpha=0.85)
scores = coherence.update(predictions)
```

#### `EnsembleWeighting`
Combine performance et cohérence pour calculer les poids.
```python
from earcp.core import EnsembleWeighting

weighting = EnsembleWeighting(n_experts=3, beta=0.7, eta_s=5.0)
weights = weighting.update_weights(perf_scores, coh_scores)
```

### 2. Models

#### `EARCP`
Classe principale orchestrant tous les composants.
```python
from earcp import EARCP

ensemble = EARCP(experts=experts)
pred, expert_preds = ensemble.predict(x)
metrics = ensemble.update(expert_preds, target)
diagnostics = ensemble.get_diagnostics()
```

### 3. Utils

#### Visualisation
```python
from earcp.utils.visualization import plot_diagnostics, plot_weights

plot_diagnostics(diagnostics, save_path='results.png')
plot_weights(weights_history, expert_names=['E1', 'E2', 'E3'])
```

#### Métriques
```python
from earcp.utils.metrics import compute_regret, compute_diversity

regret = compute_regret(expert_losses, ensemble_loss)
diversity = compute_diversity(weights_history)
```

#### Wrappers
```python
from earcp.utils.wrappers import (
    SklearnWrapper,
    TorchWrapper,
    KerasWrapper,
    CallableWrapper
)

# Wrapper pour n'importe quel framework
expert = SklearnWrapper(sklearn_model)
expert = TorchWrapper(torch_model, device='cuda')
expert = KerasWrapper(keras_model)
expert = CallableWrapper(custom_function)
```

## 🧪 Tests

Exécuter les tests:

```bash
# Tests de base
python tests/test_basic.py

# Avec pytest (si installé)
pytest tests/

# Avec couverture
pytest --cov=earcp tests/
```

## 📝 Exemples

Tous les exemples sont dans le dossier `examples/`:

```bash
# Exemple basique
python examples/basic_usage.py

# Intégration sklearn
python examples/sklearn_integration.py

# Visualisations
python examples/visualization_example.py
```

## 🔧 Développement

### Installation en mode développement

```bash
git clone https://github.com/Volgat/earcp.git
cd earcp
pip install -e ".[dev]"
```

### Ajouter vos propres experts

```python
class MonExpert:
    """Votre expert personnalisé."""

    def predict(self, x):
        """
        Méthode obligatoire pour EARCP.

        Parameters
        ----------
        x : array-like
            Entrée

        Returns
        -------
        np.ndarray
            Prédiction
        """
        # Votre logique ici
        return prediction
```

### Fonctions de perte personnalisées

```python
def ma_perte(y_pred, y_true):
    """
    Fonction de perte personnalisée.

    Doit retourner une valeur dans [0, 1].
    """
    erreur = np.abs(y_pred - y_true)
    return np.tanh(erreur)  # Normaliser à [0, 1]

config = EARCPConfig(loss_fn=ma_perte)
ensemble = EARCP(experts=experts, config=config)
```

## 📚 Documentation Complète

- **Guide Complet**: [docs/USAGE.md](docs/USAGE.md)
- **Démarrage Rapide**: [docs/QUICKSTART.md](docs/QUICKSTART.md)
- **Whitepaper Technique**: [EARCP_Technical_Whitepaper.md](EARCP_Technical_Whitepaper.md)
- **Article Académique**: [EARCP_paper.tex](EARCP_paper.tex)

## 🎯 Cas d'Usage

La bibliothèque EARCP est adaptée pour:

- ✅ Prédiction de séries temporelles
- ✅ Classification / Régression
- ✅ Apprentissage par renforcement
- ✅ Prévisions financières
- ✅ Systèmes de recommandation
- ✅ Traitement du signal
- ✅ Tout problème de décision séquentielle

## 🤝 Contribution

Les contributions sont les bienvenues! Consultez [CONTRIBUTING.md](CONTRIBUTING.md) pour les directives.

### Domaines de contribution:

- 🔧 Nouvelles fonctionnalités
- 📖 Documentation et tutoriels
- 🧪 Tests supplémentaires
- 🎨 Visualisations
- 🔬 Benchmarks sur nouveaux domaines

## 📜 Licence

MIT License - Copyright (c) 2025 Mike Amega

Voir [LICENSE](LICENSE) pour plus de détails.

**Note:** Des termes supplémentaires s'appliquent pour l'usage commercial. Contactez info@amewebstudio.com.

## 📧 Contact

**Auteur:** Mike Amega
**Email:** info@amewebstudio.com
**GitHub:** https://github.com/Volgat/earcp
**LinkedIn:** https://www.linkedin.com/in/mike-amega-486329184/

---

## ⭐ Citation

Si vous utilisez EARCP dans vos travaux, merci de citer:

```bibtex
@software{amega2025earcp,
  title={EARCP: Ensemble Auto-Régulé par Cohérence et Performance},
  author={Amega, Mike},
  year={2025},
  url={https://github.com/Volgat/earcp},
  note={Python library for adaptive ensemble learning}
}
```

---

**Dernière mise à jour:** Novembre 13, 2025
**Version:** 1.0.0
**Date de publication du prior art:** Novembre 13, 2025
