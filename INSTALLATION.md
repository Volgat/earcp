# Installation et Utilisation de la Bibliothèque EARCP

## Installation Rapide

### Option 1: Installation locale (développement)

```bash
# Cloner le dépôt
git clone https://github.com/Volgat/earcp.git
cd earcp

# Installer en mode éditable
pip install -e .
```

### Option 2: Installation avec toutes les dépendances

```bash
# Installation complète (sklearn, visualisation, etc.)
pip install -e ".[full]"

# Ou juste les dépendances de base
pip install -e .
```

## Vérification de l'Installation

```bash
# Lancer les tests
python tests/test_basic.py

# Si tout fonctionne, vous devriez voir:
# ✓ ALL TESTS PASSED!
```

## Premier Exemple

Créez un fichier `test_earcp.py`:

```python
from earcp import EARCP
import numpy as np

# Définir des experts simples
class Expert:
    def __init__(self, factor):
        self.factor = factor

    def predict(self, x):
        return self.factor * x

# Créer l'ensemble
experts = [Expert(1.0), Expert(2.0), Expert(1.5)]
ensemble = EARCP(experts=experts)

# Test rapide
print("Poids initiaux:", ensemble.get_weights())

# Simuler quelques itérations
for t in range(50):
    x = np.array([t * 0.1])
    target = np.array([1.5 * t * 0.1])  # La cible favorise Expert 3

    pred, expert_preds = ensemble.predict(x)
    ensemble.update(expert_preds, target)

print("Poids finaux:", ensemble.get_weights())
print("L'expert avec coefficient 1.5 devrait avoir le poids le plus élevé!")
```

Exécutez:
```bash
python test_earcp.py
```

## Exemples Complets

```bash
# Exemple basique avec analyse détaillée
python examples/basic_usage.py

# Intégration avec scikit-learn (classification)
python examples/sklearn_integration.py

# Visualisations complètes
python examples/visualization_example.py
```

## Utilisation dans Vos Projets

Une fois installé, importez simplement EARCP:

```python
from earcp import EARCP

# Vos modèles existants
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

# Créer des experts
models = [Ridge(), RandomForestRegressor()]
for model in models:
    model.fit(X_train, y_train)

# Wrapper pour EARCP
from earcp.utils.wrappers import SklearnWrapper
experts = [SklearnWrapper(m) for m in models]

# Utiliser EARCP
ensemble = EARCP(experts=experts, beta=0.7)

# Apprentissage en ligne
for x, y in zip(X_test, y_test):
    pred, expert_preds = ensemble.predict(x.reshape(1, -1))
    ensemble.update(expert_preds, y.reshape(-1, 1))

# Résultats
print("Poids finaux:", ensemble.get_weights())
diagnostics = ensemble.get_diagnostics()
```

## Documentation

- **Démarrage rapide**: `docs/QUICKSTART.md`
- **Guide complet**: `docs/USAGE.md`
- **Référence de la bibliothèque**: `PYTHON_LIBRARY.md`

## Structure du Projet

```
earcp/
├── earcp/              # Code source de la bibliothèque
├── examples/           # Exemples d'utilisation
├── docs/              # Documentation complète
├── tests/             # Tests unitaires
├── setup.py           # Configuration d'installation
└── requirements.txt   # Dépendances
```

## Dépendances

**Requises:**
- numpy >= 1.20.0
- scipy >= 1.7.0
- matplotlib >= 3.3.0

**Optionnelles:**
- scikit-learn >= 0.24.0 (pour SklearnWrapper)
- torch >= 1.9.0 (pour TorchWrapper)
- tensorflow >= 2.0.0 (pour KerasWrapper)

## Résolution de Problèmes

### ImportError: cannot import name 'EARCP'

```bash
# Assurez-vous d'avoir installé le package
pip install -e .

# Vérifiez que vous êtes dans le bon répertoire
pwd  # Devrait afficher .../earcp
```

### ModuleNotFoundError: No module named 'numpy'

```bash
# Installez les dépendances
pip install numpy scipy matplotlib
```

### Tests échouent

```bash
# Réinstallez les dépendances
pip install -e ".[full]"

# Relancez les tests
python tests/test_basic.py
```

## Support

- **Issues**: https://github.com/Volgat/earcp/issues
- **Email**: info@amewebstudio.com
- **Documentation**: `docs/`

## Licence

MIT License - Copyright (c) 2025 Mike Amega

Voir `LICENSE` pour plus de détails.

---

**Prêt à utiliser EARCP dans vos projets!** 🚀
