# Installation Rapide - EARCP Python Library

## ⚠️ Important: Nom de Branche

La bibliothèque Python est sur la branche:
```
earcp-lib
```

## 🚀 Installation Simple (3 Étapes)

### Étape 1: Cloner
```bash
git clone https://github.com/Volgat/earcp.git
cd earcp
```

### Étape 2: Basculer sur la bonne branche
```bash
git checkout earcp-lib
```

### Étape 3: Installer
```bash
pip install -e .
```

## ✅ Vérification

```bash
python -c "from earcp import EARCP; print('✓ Installation réussie!')"
```

## 🧪 Test Rapide

```python
from earcp import EARCP
import numpy as np

class Expert:
    def __init__(self, factor):
        self.factor = factor
    def predict(self, x):
        return self.factor * x

# Créer l'ensemble
experts = [Expert(1.0), Expert(2.0), Expert(1.5)]
ensemble = EARCP(experts=experts)

# Test
for t in range(20):
    x = np.array([t * 0.1])
    target = np.array([1.5 * t * 0.1])
    pred, expert_preds = ensemble.predict(x)
    ensemble.update(expert_preds, target)

print("Poids finaux:", ensemble.get_weights())
# L'expert avec factor=1.5 devrait avoir le poids le plus élevé
```

## 📦 Installation Directe (Alternative)

```bash
pip install git+https://github.com/Volgat/earcp.git@earcp-lib
```

## 🔧 Résolution de Problèmes

### Erreur: "does not appear to be a Python project"
**Cause**: Vous êtes sur la mauvaise branche
**Solution**:
```bash
git checkout earcp-lib
```

### Erreur: "No module named 'numpy'"
**Solution**:
```bash
pip install numpy scipy matplotlib
```

## 📚 Documentation Complète

- **Installation détaillée**: [INSTALLATION.md](INSTALLATION.md)
- **Guide de la bibliothèque**: [LIBRARY_README.md](LIBRARY_README.md)
- **Démarrage rapide**: [docs/QUICKSTART.md](docs/QUICKSTART.md)
- **Documentation complète**: [docs/USAGE.md](docs/USAGE.md)

## 💡 Exemples

```bash
# Exemple basique
python examples/basic_usage.py

# Intégration avec scikit-learn
python examples/sklearn_integration.py

# Visualisations
python examples/visualization_example.py
```

---

**Version**: 1.0.0
**Copyright**: © 2025 Mike Amega
