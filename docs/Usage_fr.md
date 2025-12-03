# EARCP - Guide d'Utilisation Complet

**Ensemble Auto-Régulé par Cohérence et Performance**

Copyright © 2025 Mike Amega. Tous droits réservés.

---

## 📑 Table des Matières

1. [Installation](#installation)
2. [Démarrage Rapide](#démarrage-rapide)
3. [Concepts Fondamentaux](#concepts-fondamentaux)
4. [Configuration](#configuration)
5. [Utilisation Avancée](#utilisation-avancée)
6. [Intégration avec les Frameworks ML](#intégration-avec-les-frameworks-ml)
7. [Visualisation et Diagnostics](#visualisation-et-diagnostics)
8. [Cas d'Utilisation](#cas-dutilisation)
9. [API Reference](#api-reference)
10. [Troubleshooting](#troubleshooting)
11. [FAQ](#faq)

---

## 🚀 Installation

### Installation depuis PyPI (Recommandé)

EARCP est maintenant disponible sur PyPI ! Installation simple :

```bash
pip install earcp
```

### Installation avec dépendances optionnelles

```bash
# Avec support PyTorch
pip install earcp[torch]

# Avec support scikit-learn
pip install earcp[sklearn]

# Avec support TensorFlow/Keras
pip install earcp[tensorflow]

# Avec toutes les dépendances de visualisation
pip install earcp[viz]

# Installation complète (toutes les dépendances)
pip install earcp[full]
```

### Installation depuis GitHub

Pour la version de développement la plus récente :

```bash
# Installation directe
pip install git+https://github.com/Volgat/earcp.git@earcp-lib

# Ou cloner et installer localement
git clone -b earcp-lib https://github.com/Volgat/earcp.git
cd earcp
pip install -e .
```

### Dépendances

**Obligatoires :**
- numpy >= 1.20.0
- scipy >= 1.7.0

**Optionnelles :**
- matplotlib >= 3.3.0 (visualisation)
- torch >= 1.9.0 (support PyTorch)
- tensorflow >= 2.4.0 (support TensorFlow)
- scikit-learn >= 0.24.0 (wrappers et métriques)

### Vérification de l'installation

```python
import earcp
print(f"EARCP version: {earcp.__version__}")

# Test rapide
from earcp import EARCP
print("Installation réussie !")
```

---

## ⚡ Démarrage Rapide

### Exemple Minimal (30 secondes)

```python
from earcp import EARCP
import numpy as np

# 1. Définir des modèles experts (tout modèle avec .predict())
class SimpleExpert:
    def __init__(self, coefficient):
        self.coefficient = coefficient
    
    def predict(self, x):
        return self.coefficient * x

# 2. Créer vos experts
experts = [
    SimpleExpert(1.0),
    SimpleExpert(2.0),
    SimpleExpert(0.5),
]

# 3. Initialiser EARCP
ensemble = EARCP(experts=experts)

# 4. Utiliser en ligne
for t in range(100):
    x = np.random.randn(5)
    
    # Prédiction
    prediction, expert_preds = ensemble.predict(x)
    
    # Observer la cible
    target = 1.5 * x + np.random.randn(5) * 0.1
    
    # Mettre à jour
    metrics = ensemble.update(expert_preds, target)

print(f"Poids finaux: {ensemble.get_weights()}")
```

### Exemple Complet avec Données Réelles

```python
import numpy as np
from earcp import EARCP

# Générer des données synthétiques
np.random.seed(42)
T = 500  # Nombre d'étapes

def generate_data(t):
    """Fonction cible avec changement de régime."""
    x = t * 0.05
    if t < 250:
        # Régime linéaire
        return 2*x + np.random.normal(0, 0.1)
    else:
        # Régime sinusoïdal
        return 2*x + 3*np.sin(x) + np.random.normal(0, 0.1)

# Définir des experts avec différentes stratégies
class LinearExpert:
    """Expert basé sur une fonction linéaire."""
    def __init__(self, slope, intercept=0):
        self.slope = slope
        self.intercept = intercept
    
    def predict(self, x):
        return self.slope * x + self.intercept

class PolynomialExpert:
    """Expert basé sur un polynôme."""
    def __init__(self, coefficients):
        self.coefficients = coefficients
    
    def predict(self, x):
        return sum(c * x**i for i, c in enumerate(self.coefficients))

class SinusoidalExpert:
    """Expert basé sur une fonction sinusoïdale."""
    def __init__(self, amplitude, frequency, phase=0):
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
    
    def predict(self, x):
        return self.amplitude * np.sin(self.frequency * x + self.phase)

# Créer un ensemble d'experts diversifiés
experts = [
    LinearExpert(slope=2.0, intercept=0.5),
    LinearExpert(slope=1.8, intercept=1.0),
    PolynomialExpert([0, 2.0, 0.1]),
    SinusoidalExpert(amplitude=3.0, frequency=1.0),
]

# Initialiser EARCP avec paramètres optimaux
ensemble = EARCP(
    experts=experts,
    beta=0.7,        # Balance performance/cohérence
    eta_s=5.0,       # Sensibilité des poids
    alpha_P=0.9,     # Lissage de la performance
    alpha_C=0.85,    # Lissage de la cohérence
    w_min=0.05       # Poids minimum (évite l'exclusion)
)

# Apprentissage en ligne
predictions = []
targets = []
weights_history = []

for t in range(T):
    # Entrée
    x = np.array([t * 0.05])
    
    # Cible réelle
    target = np.array([generate_data(t)])
    
    # Prédiction d'ensemble
    pred, expert_preds = ensemble.predict(x)
    
    # Mise à jour des poids
    metrics = ensemble.update(expert_preds, target)
    
    # Sauvegarder pour analyse
    predictions.append(pred[0])
    targets.append(target[0])
    weights_history.append(metrics['weights'].copy())
    
    # Affichage périodique
    if (t + 1) % 100 == 0:
        current_weights = metrics['weights']
        print(f"Étape {t+1}:")
        print(f"  Poids: {[f'{w:.3f}' for w in current_weights]}")
        print(f"  Erreur: {metrics['ensemble_loss']:.4f}")

# Diagnostics finaux
diagnostics = ensemble.get_diagnostics()
print(f"\n{'='*50}")
print("RÉSULTATS FINAUX")
print(f"{'='*50}")
print(f"Poids finaux: {[f'{w:.3f}' for w in diagnostics['weights']]}")
print(f"Pertes cumulatives: {diagnostics['cumulative_loss']}")

# Calculer l'erreur RMSE
rmse = np.sqrt(np.mean((np.array(predictions) - np.array(targets))**2))
print(f"RMSE global: {rmse:.4f}")
```

---

## 📚 Concepts Fondamentaux

### Architecture EARCP

EARCP est un système d'ensemble qui combine intelligemment plusieurs modèles experts en utilisant un **mécanisme de pondération à double signal**.

#### Les Deux Signaux

1. **Performance (P)** 📈
   - Mesure la qualité individuelle des prédictions de chaque expert
   - Score élevé = expert performant
   - Utilise un lissage exponentiel pour la stabilité

2. **Cohérence (C)** 🤝
   - Mesure l'accord entre un expert et les autres
   - Score élevé = expert en phase avec le consensus
   - Favorise la diversité et la robustesse

#### Algorithme Étape par Étape

À chaque instant *t*, EARCP effectue les opérations suivantes :

**Étape 1 : Collection des Prédictions**
```
Obtenir p₁,ₜ, p₂,ₜ, ..., p_M,ₜ des M experts
```

**Étape 2 : Calcul de la Performance**
```
P_i,t = α_P · P_i,t-1 + (1 - α_P) · (-ℓ_i,t)

où ℓ_i,t = perte de l'expert i à l'instant t
```

**Étape 3 : Calcul de la Cohérence**
```
C_i,t = 1/(M-1) · Σⱼ≠ᵢ Agreement(pᵢ,ₜ, pⱼ,ₜ)

Agreement mesure la similarité entre prédictions
```

**Étape 4 : Fusion des Signaux**
```
s_i,t = β · P_i,t + (1 - β) · C_i,t

β contrôle la balance entre performance et cohérence
```

**Étape 5 : Mise à Jour des Poids**
```
w̃_i,t = exp(η_s · s_i,t)
w_i,t = max(w_min, w̃_i,t / Σⱼ w̃_j,t)

w_min garantit un poids minimum à chaque expert
```

**Étape 6 : Prédiction d'Ensemble**
```
ŷ_t = Σᵢ w_i,t · p_i,t
```

### Garanties Théoriques

**Théorème (Borne de Regret)** : Sous des hypothèses standards (pertes bornées dans [0,1], convexité), EARCP garantit :

```
Regret_T ≤ (1/β) · √(2T log M)
```

Pour β = 1 (performance pure) :
```
Regret_T ≤ √(2T log M)
```

**Implications pratiques :**
- Le regret par étape décroît en O(√(log M / T))
- Performance asymptotiquement optimale
- Comparable au meilleur expert en rétrospective

### Paramètres : Guide Complet

| Paramètre | Rôle | Défaut | Plage | Impact |
|-----------|------|--------|-------|--------|
| **alpha_P** | Mémoire de la performance | 0.9 | [0.8, 0.99] | ↑ = plus de mémoire, réaction lente<br>↓ = oubli rapide, adaptation rapide |
| **alpha_C** | Mémoire de la cohérence | 0.85 | [0.75, 0.95] | ↑ = cohérence stable<br>↓ = cohérence réactive |
| **beta** | Balance P/C | 0.7 | [0.5, 1.0] | ↑ = favorise performance<br>↓ = favorise diversité |
| **eta_s** | Taux d'apprentissage | 5.0 | [1.0, 10.0] | ↑ = changements rapides<br>↓ = stabilité |
| **w_min** | Poids plancher | 0.05 | [0.01, 0.2] | Évite l'exclusion complète |

#### Guide de Réglage de Beta

**β = 1.0** : Mode Performance Pure
- Équivalent à l'algorithme Hedge
- Utiliser quand : vous avez confiance absolue en vos métriques
- Avantage : convergence rapide vers le meilleur
- Inconvénient : peut sur-adapter, manque de robustesse

**β = 0.8-0.9** : Mode Performance Dominant
- Favorise fortement la performance
- Utiliser quand : environnement stable, métriques fiables
- Avantage : bonne performance, un peu de diversité
- Inconvénient : peut encore sur-adapter

**β = 0.6-0.7** : Mode Équilibré ⭐ (Recommandé)
- Balance optimale entre performance et cohérence
- Utiliser quand : environnement variable, incertitude modérée
- Avantage : robuste, adaptable, bon compromis
- Idéal pour la plupart des applications

**β = 0.4-0.5** : Mode Diversité
- Favorise fortement la cohérence
- Utiliser quand : environnement très bruité, métriques peu fiables
- Avantage : maximum de robustesse et diversité
- Inconvénient : convergence plus lente

**β = 0.0** : Mode Cohérence Pure (Expérimental)
- Pondération basée uniquement sur l'accord inter-experts
- Utiliser quand : aucune confiance dans les métriques individuelles
- Avantage : consensus pur
- Inconvénient : peut ignorer de bons experts isolés

#### Règles Pratiques de Réglage

**Pour eta_s (Sensibilité) :**
```python
# Environnement stable
eta_s = 3.0 - 4.0

# Environnement modéré (défaut)
eta_s = 5.0 - 6.0

# Environnement très dynamique
eta_s = 7.0 - 9.0
```

**Pour w_min (Poids minimum) :**
```python
# Règle générale : entre 0.5/M et 2.0/M
M = len(experts)
w_min = 1.0 / M  # Bon point de départ

# Beaucoup d'experts
if M > 10:
    w_min = 0.5 / M

# Peu d'experts
if M <= 3:
    w_min = 0.1  # Plus conservateur
```

**Pour alpha_P et alpha_C :**
```python
# Environnement stationnaire
alpha_P, alpha_C = 0.95, 0.90

# Environnement modérément variable (défaut)
alpha_P, alpha_C = 0.90, 0.85

# Environnement très variable (concept drift)
alpha_P, alpha_C = 0.85, 0.80
```

---

## ⚙️ Configuration

### Utiliser EARCPConfig

```python
from earcp import EARCP, EARCPConfig

# Créer une configuration personnalisée
config = EARCPConfig(
    # Paramètres de lissage
    alpha_P=0.9,
    alpha_C=0.85,
    
    # Balance et sensibilité
    beta=0.7,
    eta_s=5.0,
    
    # Contraintes
    w_min=0.05,
    epsilon=1e-10,
    
    # Mode de prédiction
    prediction_mode='regression',  # 'regression', 'classification', ou 'auto'
    
    # Fonctions personnalisées
    loss_fn=None,          # Fonction de perte personnalisée
    coherence_fn=None,     # Fonction de cohérence personnalisée
    
    # Options de suivi
    track_diagnostics=True,
    normalize_weights=True,
    
    # Reproductibilité
    random_state=42
)

# Utiliser la configuration
ensemble = EARCP(experts=experts, config=config)
```

### Configurations Prédéfinies

EARCP fournit plusieurs configurations optimisées pour différents cas d'usage :

```python
from earcp import get_preset_config

# 1. Configuration par défaut (équilibrée)
config = get_preset_config('default')
# beta=0.7, eta_s=5.0, adapté à la plupart des cas

# 2. Focus sur la performance
config = get_preset_config('performance_focused')
# beta=0.95, privilégie les experts performants

# 3. Focus sur la diversité
config = get_preset_config('diversity_focused')
# beta=0.5, favorise le consensus et la robustesse

# 4. Configuration équilibrée optimale
config = get_preset_config('balanced')
# beta=0.7, paramètres finement ajustés

# 5. Mode conservateur
config = get_preset_config('conservative')
# Changements lents, grande stabilité

# 6. Mode agressif
config = get_preset_config('aggressive')
# Changements rapides, adaptation dynamique

# 7. Mode robuste (pour données bruitées)
config = get_preset_config('robust')
# beta=0.6, w_min élevé, résiste au bruit

# 8. Mode haute performance
config = get_preset_config('high_performance')
# beta=0.9, convergence rapide

# Utilisation
ensemble = EARCP(experts=experts, config=config)
```

### Fonctions de Perte Personnalisées

```python
import numpy as np

# Fonction de perte pour régression
def custom_regression_loss(y_pred, y_true):
    """
    Fonction de perte personnalisée.
    Doit retourner une valeur dans [0, 1].
    """
    mse = np.mean((y_pred - y_true) ** 2)
    # Normaliser avec tanh ou sigmoid
    return np.tanh(mse)

# Fonction de perte pour classification
def custom_classification_loss(y_pred, y_true):
    """
    Cross-entropy pour classification.
    """
    # y_pred: probabilities, y_true: one-hot
    epsilon = 1e-10
    ce = -np.sum(y_true * np.log(y_pred + epsilon))
    return ce / len(y_true)

# Fonction de perte robuste (insensible aux outliers)
def huber_loss(y_pred, y_true, delta=1.0):
    """
    Perte de Huber : quadratique pour petites erreurs,
    linéaire pour grandes erreurs.
    """
    error = np.abs(y_pred - y_true)
    is_small = error <= delta
    
    small_loss = 0.5 * error**2
    large_loss = delta * error - 0.5 * delta**2
    
    loss = np.where(is_small, small_loss, large_loss)
    return np.mean(loss) / (2 * delta)  # Normaliser à [0, 1]

# Utilisation
config = EARCPConfig(loss_fn=huber_loss)
ensemble = EARCP(experts=experts, config=config)
```

### Fonctions de Cohérence Personnalisées

```python
import numpy as np

# Cohérence basée sur la corrélation
def correlation_coherence(pred_i, pred_j):
    """
    Mesure la corrélation entre deux prédictions.
    Retourne une valeur dans [0, 1].
    """
    # Aplatir les prédictions
    pi = pred_i.flatten()
    pj = pred_j.flatten()
    
    # Calculer la corrélation de Pearson
    if len(pi) > 1:
        correlation = np.corrcoef(pi, pj)[0, 1]
        # Mapper [-1, 1] à [0, 1]
        return (correlation + 1) / 2
    else:
        # Similarité directe pour prédictions scalaires
        return 1.0 / (1.0 + np.abs(pi - pj))

# Cohérence basée sur la distance
def distance_coherence(pred_i, pred_j):
    """
    Cohérence basée sur la distance euclidienne.
    Plus la distance est petite, plus la cohérence est élevée.
    """
    distance = np.linalg.norm(pred_i - pred_j)
    # Transformer distance en similarité
    return np.exp(-distance)

# Cohérence pour classification (accord sur les classes)
def classification_coherence(pred_i, pred_j):
    """
    Pour classification : pourcentage d'accord sur les classes prédites.
    """
    class_i = np.argmax(pred_i, axis=-1)
    class_j = np.argmax(pred_j, axis=-1)
    agreement = np.mean(class_i == class_j)
    return agreement

# Cohérence pondérée (favorise certaines dimensions)
def weighted_coherence(pred_i, pred_j, weights=None):
    """
    Cohérence avec pondération des dimensions.
    """
    if weights is None:
        weights = np.ones(pred_i.shape)
    
    diff = weights * (pred_i - pred_j) ** 2
    distance = np.sqrt(np.sum(diff))
    return np.exp(-distance)

# Utilisation
config = EARCPConfig(coherence_fn=correlation_coherence)
ensemble = EARCP(experts=experts, config=config)
```

---

## 🔬 Utilisation Avancée

### Sauvegarder et Charger l'État

```python
import pickle

# Sauvegarder l'état complet
ensemble.save_state('checkpoints/ensemble_step_1000.pkl')

# Charger l'état
ensemble_restored = EARCP(experts=experts)
ensemble_restored.load_state('checkpoints/ensemble_step_1000.pkl')

# Continuer l'apprentissage
for t in range(1000, 2000):
    pred, expert_preds = ensemble_restored.predict(x[t])
    ensemble_restored.update(expert_preds, y[t])
```

### Sauvegardes Périodiques

```python
# Sauvegarder tous les N pas
CHECKPOINT_INTERVAL = 100

for t in range(T):
    pred, expert_preds = ensemble.predict(x[t])
    metrics = ensemble.update(expert_preds, y[t])
    
    # Checkpoint
    if (t + 1) % CHECKPOINT_INTERVAL == 0:
        ensemble.save_state(f'checkpoints/step_{t+1}.pkl')
        print(f"Checkpoint sauvegardé à l'étape {t+1}")
```

### Réinitialiser l'Ensemble

```python
# Réinitialiser complètement (poids uniformes)
ensemble.reset()

# Réinitialiser avec des poids personnalisés
custom_weights = np.array([0.5, 0.3, 0.2])
ensemble.reset(initial_weights=custom_weights)
```

### Modifier les Paramètres Dynamiquement

```python
# Ajuster beta en cours d'exécution (exemple : décroissance)
for t in range(T):
    # Décroître beta progressivement de 0.9 à 0.6
    current_beta = 0.9 - 0.3 * (t / T)
    ensemble.weighting.set_beta(current_beta)
    
    pred, expert_preds = ensemble.predict(x[t])
    ensemble.update(expert_preds, y[t])

# Ajuster la sensibilité selon le contexte
if detect_concept_drift():
    ensemble.weighting.set_eta_s(8.0)  # Plus agressif
else:
    ensemble.weighting.set_eta_s(4.0)  # Plus conservateur

# Ajuster les facteurs de lissage
ensemble.performance_tracker.set_alpha(0.95)  # Plus de mémoire
ensemble.coherence_metrics.set_alpha(0.80)    # Moins de mémoire
```

### Accéder aux Composants Internes

```python
# Obtenir les scores de performance bruts
perf_scores = ensemble.performance_tracker.get_scores()
print(f"Scores de performance: {perf_scores}")

# Obtenir les scores de cohérence
coh_scores = ensemble.coherence_metrics.get_scores()
print(f"Scores de cohérence: {coh_scores}")

# Obtenir la matrice de cohérence complète
expert_predictions = [expert.predict(x) for expert in experts]
coh_matrix = ensemble.coherence_metrics.get_coherence_matrix(expert_predictions)
print(f"Matrice de cohérence:\n{coh_matrix}")

# Obtenir l'historique complet des pertes
loss_history = ensemble.performance_tracker.get_loss_history()

# Obtenir les statistiques avancées
stats = {
    'mean_weight': np.mean(ensemble.get_weights()),
    'weight_std': np.std(ensemble.get_weights()),
    'effective_experts': np.sum(ensemble.get_weights() > 0.1),
    'entropy': ensemble.get_diagnostics()['entropy']
}
```

### Gestion Multi-Objectifs

```python
from earcp.core.performance_tracker import MultiObjectivePerformanceTracker

# Créer un tracker pour plusieurs objectifs
# Exemple : trading avec profit ET risque
tracker = MultiObjectivePerformanceTracker(
    n_experts=3,
    n_objectives=2,
    objective_weights=[0.7, 0.3],  # 70% profit, 30% risque
    aggregation='weighted_sum'      # ou 'product', 'min', 'max'
)

# Dans la boucle d'apprentissage
for t in range(T):
    predictions = [expert.predict(state) for expert in experts]
    
    # Exécuter l'action
    profit, risk = execute_trade(predictions)
    
    # Mettre à jour avec les deux objectifs
    tracker.update(
        predictions=predictions,
        targets=[profit, -risk]  # Maximiser profit, minimiser risque
    )
    
    # Obtenir les scores agrégés
    aggregated_scores = tracker.get_aggregated_scores()
```

### Ensemble d'Ensembles (Méta-EARCP)

```python
# Créer plusieurs ensembles EARCP
ensemble_1 = EARCP(experts=experts_group_1, beta=0.8)
ensemble_2 = EARCP(experts=experts_group_2, beta=0.6)
ensemble_3 = EARCP(experts=experts_group_3, beta=0.7)

# Wrapper pour utiliser un EARCP comme expert
class EARCPExpert:
    def __init__(self, earcp_ensemble):
        self.ensemble = earcp_ensemble
    
    def predict(self, x):
        pred, _ = self.ensemble.predict(x)
        return pred

# Créer un méta-ensemble
meta_experts = [
    EARCPExpert(ensemble_1),
    EARCPExpert(ensemble_2),
    EARCPExpert(ensemble_3)
]

meta_ensemble = EARCP(experts=meta_experts, beta=0.75)

# Utilisation hiérarchique
for t in range(T):
    # Prédiction du méta-ensemble
    meta_pred, sub_preds = meta_ensemble.predict(x[t])
    
    # Mise à jour de tous les niveaux
    meta_ensemble.update(sub_preds, y[t])
    
    # Mise à jour des ensembles de base (optionnel)
    for i, sub_ensemble in enumerate([ensemble_1, ensemble_2, ensemble_3]):
        _, expert_preds = sub_ensemble.predict(x[t])
        sub_ensemble.update(expert_preds, y[t])
```

---

## 🔗 Intégration avec les Frameworks ML

### Avec scikit-learn

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from earcp import EARCP
from earcp.utils.wrappers import SklearnWrapper

# Créer et entraîner des modèles sklearn
models = {
    'ridge': Ridge(alpha=1.0),
    'lasso': Lasso(alpha=0.5),
    'elastic': ElasticNet(alpha=0.7, l1_ratio=0.5),
    'rf': RandomForestRegressor(n_estimators=100, max_depth=10),
    'gbm': GradientBoostingRegressor(n_estimators=100),
    'svr': SVR(kernel='rbf')
}

# Entraîner tous les modèles
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name} entraîné")

# Créer les wrappers pour EARCP
experts = [SklearnWrapper(model, name=name) 
           for name, model in models.items()]

# Créer l'ensemble
ensemble = EARCP(experts=experts, beta=0.7)

# Évaluation en ligne sur le test set
predictions = []
for i, (x, y) in enumerate(zip(X_test, y_test)):
    # Prédiction
    pred, expert_preds = ensemble.predict(x.reshape(1, -1))
    predictions.append(pred[0])
    
    # Mise à jour
    ensemble.update(expert_preds, y.reshape(1, -1))
    
    if (i + 1) % 100 == 0:
        current_rmse = np.sqrt(np.mean((np.array(predictions) - y_test[:i+1])**2))
        print(f"Étape {i+1}: RMSE = {current_rmse:.4f}")

# Résultats finaux
final_weights = ensemble.get_weights()
for name, weight in zip(models.keys(), final_weights):
    print(f"{name}: {weight:.3f}")
```

### Avec PyTorch

```python
import torch
import torch.nn as nn
from earcp import EARCP
from earcp.utils.wrappers import TorchWrapper

# Définir différentes architectures
class SmallCNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class DeepNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# Créer et entraîner les modèles
device = 'cuda' if torch.cuda.is_available() else 'cpu'
input_dim, output_dim = 10, 1

models = [
    SmallCNN(input_dim, output_dim).to(device),
    DeepNN(input_dim, output_dim).to(device),
]

# Entraînement (exemple simple)
for model in models:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(10):
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

# Créer les wrappers
experts = [TorchWrapper(model, device=device) for model in models]

# Mode évaluation pour les modèles PyTorch
for model in models:
    model.eval()

# Créer l'ensemble EARCP
ensemble = EARCP(experts=experts)

# Utilisation
with torch.no_grad():
    for x, y in test_loader:
        pred, expert_preds = ensemble.predict(x)
        ensemble.update(expert_preds, y)
```

### Avec TensorFlow/Keras

```python
from tensorflow import keras
from tensorflow.keras import layers
from earcp import EARCP
from earcp.utils.wrappers import KerasWrapper

# Créer différents modèles Keras
def create_dense_model(input_dim, output_dim):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(output_dim)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def create_deep_model(input_dim, output_dim):
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(output_dim)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Créer plusieurs modèles
models = [
    create_dense_model(10, 1),
    create_deep_model(10, 1),
    create_dense_model(10, 1),  # Même architecture, init différente
]

# Entraîner chaque modèle
for i, model in enumerate(models):
    print(f"Entraînement du modèle {i+1}...")
    model.fit(X_train, y_train, epochs=20, batch_size=32, 
              validation_split=0.2, verbose=0)

# Créer les wrappers
experts = [KerasWrapper(model) for model in models]

# Créer l'ensemble
ensemble = EARCP(experts=experts, beta=0.7)

# Utilisation
for i in range(len(X_test)):
    x = X_test[i:i+1]
    y = y_test[i:i+1]
    
    pred, expert_preds = ensemble.predict(x)
    ensemble.update(expert_preds, y)
```

### Wrapper Universel pour Fonctions

```python
from earcp.utils.wrappers import CallableWrapper

# Créer des experts à partir de fonctions simples
def moving_average(x, window=5):
    """Moyenne mobile."""
    return np.mean(x[-window:])

def exponential_smoothing(x, alpha=0.3):
    """Lissage exponentiel."""
    if len(x) == 0:
        return 0
    result = x[0]
    for val in x[1:]:
        result = alpha * val + (1 - alpha) * result
    return result

def trend_extrapolation(x):
    """Extrapolation de tendance."""
    if len(x) < 2:
        return x[-1] if len(x) > 0 else 0
    return 2 * x[-1] - x[-2]

# Encapsuler les fonctions
experts = [
    CallableWrapper(moving_average, name='MA'),
    CallableWrapper(exponential_smoothing, name='ES'),
    CallableWrapper(trend_extrapolation, name='Trend')
]

# Utiliser dans EARCP
ensemble = EARCP(experts=experts)
```

### Intégration avec LightGBM et XGBoost

```python
import lightgbm as lgb
import xgboost as xgb
from earcp.utils.wrappers import SklearnWrapper

# Créer des modèles de boosting
lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1)
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)

# Entraîner
lgb_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# Combiner avec autres modèles
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X_train, y_train)

# Créer l'ensemble
experts = [
    SklearnWrapper(lgb_model, name='LightGBM'),
    SklearnWrapper(xgb_model, name='XGBoost'),
    SklearnWrapper(rf_model, name='RandomForest')
]

ensemble = EARCP(experts=experts, beta=0.75)
```

---

## 📊 Visualisation et Diagnostics

### Obtenir les Diagnostics Complets

```python
# Obtenir tous les diagnostics
diagnostics = ensemble.get_diagnostics()

# Contenu disponible
print("Diagnostics disponibles:")
for key in diagnostics.keys():
    print(f"  - {key}: {type(diagnostics[key])}")

# Diagnostics typiques:
# - 'weights': Poids actuels [array]
# - 'performance_scores': Scores de performance [array]
# - 'coherence_scores': Scores de cohérence [array]
# - 'time_step': Étape temporelle actuelle [int]
# - 'weights_history': Historique complet des poids [list of arrays]
# - 'performance_history': Historique des performances [list of arrays]
# - 'coherence_history': Historique de la cohérence [list of arrays]
# - 'cumulative_loss': Perte cumulative par expert [array]
# - 'entropy': Entropie de la distribution des poids [float]
# - 'effective_experts': Nombre d'experts actifs [float]
```

### Visualisations Standard

```python
from earcp.utils.visualization import (
    plot_weights,
    plot_performance,
    plot_coherence,
    plot_diagnostics,
    plot_regret,
    plot_predictions
)
import matplotlib.pyplot as plt

# 1. Évolution des poids dans le temps
fig, ax = plt.subplots(figsize=(12, 6))
plot_weights(
    weights_history=diagnostics['weights_history'],
    expert_names=['Expert 1', 'Expert 2', 'Expert 3'],
    ax=ax,
    title='Évolution des Poids EARCP',
    save_path='figures/weights_evolution.png'
)
plt.show()

# 2. Performance et cohérence
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
plot_performance(
    performance_history=diagnostics['performance_history'],
    expert_names=['Expert 1', 'Expert 2', 'Expert 3'],
    ax=ax1,
    title='Scores de Performance'
)
plot_coherence(
    coherence_history=diagnostics['coherence_history'],
    expert_names=['Expert 1', 'Expert 2', 'Expert 3'],
    ax=ax2,
    title='Scores de Cohérence'
)
plt.tight_layout()
plt.savefig('figures/performance_coherence.png')
plt.show()

# 3. Dashboard complet
fig = plot_diagnostics(
    diagnostics=diagnostics,
    expert_names=['Expert 1', 'Expert 2', 'Expert 3'],
    figsize=(16, 10),
    save_path='figures/full_diagnostics.png'
)
plt.show()

# 4. Analyse du regret
fig, ax = plt.subplots(figsize=(10, 6))
ensemble_cumulative_loss = compute_ensemble_loss(predictions, targets)
plot_regret(
    expert_cumulative_losses=diagnostics['cumulative_loss'],
    ensemble_cumulative_loss=ensemble_cumulative_loss,
    expert_names=['Expert 1', 'Expert 2', 'Expert 3'],
    ax=ax,
    save_path='figures/regret_analysis.png'
)
plt.show()

# 5. Prédictions vs Réalité
fig, ax = plt.subplots(figsize=(12, 6))
plot_predictions(
    predictions=predictions,
    targets=targets,
    expert_predictions=expert_predictions_history,
    expert_names=['Expert 1', 'Expert 2', 'Expert 3'],
    ax=ax,
    title='Prédictions EARCP vs Réalité',
    save_path='figures/predictions.png'
)
plt.show()
```

### Visualisations Avancées

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Matrice de corrélation des poids
def plot_weight_correlation(weights_history):
    """Corrélation entre l'évolution des poids des experts."""
    weights_array = np.array(weights_history).T  # (n_experts, time_steps)
    correlation_matrix = np.corrcoef(weights_array)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',
                center=0, vmin=-1, vmax=1)
    plt.title('Corrélation de l\'évolution des poids')
    plt.xlabel('Expert')
    plt.ylabel('Expert')
    plt.tight_layout()
    plt.savefig('figures/weight_correlation.png')
    plt.show()

plot_weight_correlation(diagnostics['weights_history'])

# Heatmap de la matrice de cohérence
def plot_coherence_matrix(ensemble, expert_predictions):
    """Visualiser la matrice de cohérence entre experts."""
    coh_matrix = ensemble.coherence_metrics.get_coherence_matrix(expert_predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(coh_matrix, annot=True, cmap='viridis',
                vmin=0, vmax=1)
    plt.title('Matrice de Cohérence Inter-Experts')
    plt.xlabel('Expert')
    plt.ylabel('Expert')
    plt.tight_layout()
    plt.savefig('figures/coherence_matrix.png')
    plt.show()

# Distribution des poids à différents moments
def plot_weight_distributions(weights_history, time_points=[0, len//4, len//2, -1]):
    """Comparer les distributions de poids à différents moments."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, t in enumerate(time_points):
        weights = weights_history[t]
        axes[idx].bar(range(len(weights)), weights)
        axes[idx].set_title(f'Distribution à t={t}')
        axes[idx].set_xlabel('Expert')
        axes[idx].set_ylabel('Poids')
        axes[idx].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('figures/weight_distributions.png')
    plt.show()

len_history = len(diagnostics['weights_history'])
plot_weight_distributions(diagnostics['weights_history'])
```

### Métriques d'Évaluation

```python
from earcp.utils.metrics import (
    compute_regret,
    compute_diversity,
    evaluate_ensemble,
    theoretical_regret_bound,
    compute_stability,
    compute_adaptability
)

# 1. Calculer le regret
regret_metrics = compute_regret(
    expert_cumulative_losses=diagnostics['cumulative_loss'],
    ensemble_cumulative_loss=ensemble_cumulative_loss
)

print("=== ANALYSE DU REGRET ===")
print(f"Regret absolu: {regret_metrics['regret']:.4f}")
print(f"Regret relatif: {regret_metrics['relative_regret']:.2%}")
print(f"Meilleur expert: {regret_metrics['best_expert']}")
print(f"Pire expert: {regret_metrics['worst_expert']}")

# 2. Calculer la diversité
diversity_metrics = compute_diversity(diagnostics['weights_history'])

print("\n=== ANALYSE DE DIVERSITÉ ===")
print(f"Entropie moyenne: {diversity_metrics['mean_entropy']:.4f}")
print(f"Entropie max théorique: {diversity_metrics['max_entropy']:.4f}")
print(f"Ratio d'utilisation: {diversity_metrics['utilization_ratio']:.2%}")
print(f"Experts effectifs moyens: {diversity_metrics['mean_effective_experts']:.2f}")

# 3. Borne théorique
T = len(predictions)
M = len(experts)
theoretical_bound = theoretical_regret_bound(T=T, M=M, beta=0.7)

print("\n=== GARANTIES THÉORIQUES ===")
print(f"Borne de regret O(√T log M): {theoretical_bound:.4f}")
print(f"Regret observé: {regret_metrics['regret']:.4f}")
print(f"Ratio (observé/théorique): {regret_metrics['regret']/theoretical_bound:.2%}")

# 4. Évaluation de l'ensemble
eval_metrics = evaluate_ensemble(
    predictions=predictions,
    targets=targets,
    task_type='regression'
)

print("\n=== PERFORMANCE PRÉDICTIVE ===")
print(f"RMSE: {eval_metrics['rmse']:.4f}")
print(f"MAE: {eval_metrics['mae']:.4f}")
print(f"R²: {eval_metrics['r2']:.4f}")
print(f"Corrélation: {eval_metrics['correlation']:.4f}")

# 5. Stabilité et adaptabilité
stability = compute_stability(diagnostics['weights_history'])
adaptability = compute_adaptability(diagnostics['weights_history'])

print("\n=== COMPORTEMENT DYNAMIQUE ===")
print(f"Stabilité (variation moyenne): {stability:.4f}")
print(f"Adaptabilité (capacité de changement): {adaptability:.4f}")
```

### Export des Résultats

```python
import json
import pandas as pd

# Créer un rapport complet
def generate_report(ensemble, diagnostics, predictions, targets):
    """Génère un rapport JSON complet."""
    report = {
        'configuration': {
            'n_experts': len(ensemble.experts),
            'beta': ensemble.weighting.beta,
            'eta_s': ensemble.weighting.eta_s,
            'alpha_P': ensemble.performance_tracker.alpha,
            'alpha_C': ensemble.coherence_metrics.alpha,
            'w_min': ensemble.weighting.w_min
        },
        'final_state': {
            'weights': diagnostics['weights'].tolist(),
            'performance_scores': diagnostics['performance_scores'].tolist(),
            'coherence_scores': diagnostics['coherence_scores'].tolist(),
            'time_step': diagnostics['time_step']
        },
        'metrics': {
            'rmse': float(np.sqrt(np.mean((predictions - targets)**2))),
            'mae': float(np.mean(np.abs(predictions - targets))),
            'cumulative_losses': diagnostics['cumulative_loss'].tolist()
        },
        'diversity': {
            'entropy': float(diagnostics['entropy']),
            'effective_experts': float(diagnostics['effective_experts'])
        }
    }
    
    # Sauvegarder
    with open('reports/earcp_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

# Créer un DataFrame pour analyse
def create_results_dataframe(diagnostics, predictions, targets):
    """Crée un DataFrame avec tous les résultats."""
    df = pd.DataFrame({
        'step': range(len(predictions)),
        'prediction': predictions,
        'target': targets,
        'error': predictions - targets,
        'abs_error': np.abs(predictions - targets)
    })
    
    # Ajouter les poids
    weights_array = np.array(diagnostics['weights_history'])
    for i in range(weights_array.shape[1]):
        df[f'weight_expert_{i+1}'] = weights_array[:, i]
    
    # Sauvegarder
    df.to_csv('results/earcp_results.csv', index=False)
    
    return df

# Générer les rapports
report = generate_report(ensemble, diagnostics, predictions, targets)
df_results = create_results_dataframe(diagnostics, predictions, targets)

print("Rapports générés:")
print("  - reports/earcp_report.json")
print("  - results/earcp_results.csv")
```

---

## 💼 Cas d'Utilisation

### 1. Prédiction de Séries Temporelles

```python
import numpy as np
from earcp import EARCP

# Différents types d'experts pour séries temporelles
class MovingAverageExpert:
    """Expert basé sur moyenne mobile."""
    def __init__(self, window_size):
        self.window = window_size
        self.history = []
    
    def predict(self, x):
        self.history.append(x)
        if len(self.history) > self.window:
            self.history.pop(0)
        return np.mean(self.history, axis=0)

class ExponentialSmoothingExpert:
    """Expert basé sur lissage exponentiel."""
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.last_value = None
    
    def predict(self, x):
        if self.last_value is None:
            self.last_value = x
            return x
        
        prediction = self.alpha * x + (1 - self.alpha) * self.last_value
        self.last_value = prediction
        return prediction

class TrendExpert:
    """Expert basé sur extrapolation de tendance."""
    def __init__(self):
        self.history = []
    
    def predict(self, x):
        self.history.append(x)
        if len(self.history) < 3:
            return x
        
        # Régression linéaire sur les dernières valeurs
        recent = np.array(self.history[-5:])
        t = np.arange(len(recent))
        coeffs = np.polyfit(t, recent, deg=1)
        # Extrapoler
        next_t = len(recent)
        return coeffs[0] * next_t + coeffs[1]

# Créer l'ensemble
experts = [
    MovingAverageExpert(window_size=5),
    MovingAverageExpert(window_size=10),
    MovingAverageExpert(window_size=20),
    ExponentialSmoothingExpert(alpha=0.2),
    ExponentialSmoothingExpert(alpha=0.5),
    TrendExpert()
]

ensemble = EARCP(experts=experts, beta=0.7)

# Simulation de série temporelle
T = 1000
for t in range(T):
    # Valeur actuelle (avec saisonnalité et bruit)
    x = np.sin(t * 0.1) + 0.01 * t + np.random.normal(0, 0.1)
    
    # Prédiction
    pred, expert_preds = ensemble.predict(np.array([x]))
    
    # Vraie valeur future (t+1)
    target = np.sin((t+1) * 0.1) + 0.01 * (t+1) + np.random.normal(0, 0.1)
    
    # Mise à jour
    ensemble.update(expert_preds, np.array([target]))
```

### 2. Trading et Finance

```python
import numpy as np
from earcp import EARCP

# Stratégies de trading diversifiées
class MomentumStrategy:
    """Suit les tendances de marché."""
    def __init__(self, lookback=20):
        self.lookback = lookback
        self.price_history = []
    
    def predict(self, market_state):
        # market_state contient: prix, volume, indicateurs
        price = market_state['price']
        self.price_history.append(price)
        
        if len(self.price_history) < self.lookback:
            return 0  # Position neutre
        
        recent_prices = self.price_history[-self.lookback:]
        momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        # Signal: 1 (achat), 0 (neutre), -1 (vente)
        if momentum > 0.02:
            return 1
        elif momentum < -0.02:
            return -1
        return 0

class MeanReversionStrategy:
    """Parie sur le retour à la moyenne."""
    def __init__(self, lookback=30):
        self.lookback = lookback
        self.price_history = []
    
    def predict(self, market_state):
        price = market_state['price']
        self.price_history.append(price)
        
        if len(self.price_history) < self.lookback:
            return 0
        
        recent_prices = self.price_history[-self.lookback:]
        mean_price = np.mean(recent_prices)
        std_price = np.std(recent_prices)
        
        # Z-score
        z_score = (price - mean_price) / (std_price + 1e-10)
        
        # Signal inverse du z-score
        if z_score > 2:
            return -1  # Suracheté -> vendre
        elif z_score < -2:
            return 1   # Survendu -> acheter
        return 0

class VolatilityStrategy:
    """Adapte la position à la volatilité."""
    def __init__(self):
        self.returns_history = []
    
    def predict(self, market_state):
        if 'return' in market_state:
            self.returns_history.append(market_state['return'])
        
        if len(self.returns_history) < 20:
            return 0
        
        recent_returns = self.returns_history[-20:]
        volatility = np.std(recent_returns)
        mean_return = np.mean(recent_returns)
        
        # Plus la volatilité est faible, plus on peut être agressif
        if volatility < 0.01:
            return np.sign(mean_return)  # Signal directionnel
        return 0  # Rester neutre en haute volatilité

# Créer l'ensemble de stratégies
strategies = [
    MomentumStrategy(lookback=10),
    MomentumStrategy(lookback=30),
    MeanReversionStrategy(lookback=20),
    MeanReversionStrategy(lookback=50),
    VolatilityStrategy()
]

ensemble = EARCP(experts=strategies, beta=0.75)

# Simulation de trading
portfolio_value = 100000
position = 0
price_history = []

for day in range(500):
    # Simuler le marché
    price = 100 + 10 * np.sin(day * 0.1) + np.random.normal(0, 1)
    price_history.append(price)
    
    # État du marché
    market_state = {
        'price': price,
        'return': (price - price_history[-2]) / price_history[-2] if len(price_history) > 1 else 0
    }
    
    # Obtenir le signal d'ensemble
    signal, strategy_signals = ensemble.predict(market_state)
    
    # Agréger le signal (-1, 0, 1)
    aggregated_signal = np.mean(signal)
    
    # Exécuter le trade
    target_position = np.sign(aggregated_signal)
    if target_position != position:
        # Calculer P&L
        if position != 0:
            pnl = position * (price - entry_price)
            portfolio_value += pnl
        
        # Nouvelle position
        position = target_position
        entry_price = price if position != 0 else 0
    
    # Calculer le rendement réalisé pour mise à jour
    daily_return = (price - price_history[-2]) / price_history[-2] if len(price_history) > 1 else 0
    realized_return = position * daily_return
    
    # Mise à jour de l'ensemble (perte négative = gain)
    ensemble.update(strategy_signals, np.array([-realized_return]))  # Négatif car on minimise la perte
    
    if (day + 1) % 100 == 0:
        print(f"Jour {day+1}: Portfolio = ${portfolio_value:.2f}, Position = {position}")

print(f"\nValeur finale du portfolio: ${portfolio_value:.2f}")
print(f"Rendement: {((portfolio_value - 100000) / 100000 * 100):.2f}%")
```

### 3. Classification Multi-Classes

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from earcp import EARCP
from earcp.utils.wrappers import SklearnWrapper

# Générer des données de classification
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_classes=3,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Créer différents classifieurs
classifiers = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'SVM_linear': SVC(kernel='linear', probability=True),
    'SVM_rbf': SVC(kernel='rbf', probability=True),
    'RandomForest': RandomForestClassifier(n_estimators=100),
    'NaiveBayes': GaussianNB()
}

# Entraîner tous les classifieurs
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    print(f"{name} - Précision train: {train_acc:.3f}")

# Créer les wrappers
experts = [SklearnWrapper(clf, name=name) 
           for name, clf in classifiers.items()]

# Créer l'ensemble pour classification
ensemble = EARCP(
    experts=experts,
    beta=0.7,
    prediction_mode='classification'
)

# Évaluation en ligne sur le test set
correct = 0
predictions = []
true_labels = []

for i, (x, y_true) in enumerate(zip(X_test, y_test)):
    # Prédiction (retourne des probabilités)
    prob, expert_probs = ensemble.predict(x.reshape(1, -1))
    
    # Classe prédite
    y_pred = np.argmax(prob)
    predictions.append(y_pred)
    true_labels.append(y_true)
    
    # Vérifier si correct
    if y_pred == y_true:
        correct += 1
    
    # One-hot encoding de la vraie classe
    target_onehot = np.zeros(3)
    target_onehot[y_true] = 1
    
    # Mise à jour de l'ensemble
    ensemble.update(expert_probs, target_onehot.reshape(1, -1))
    
    if (i + 1) % 50 == 0:
        current_acc = correct / (i + 1)
        print(f"Échantillon {i+1}: Précision = {current_acc:.3f}")

# Résultats finaux
final_accuracy = correct / len(X_test)
print(f"\n{'='*50}")
print(f"Précision finale de l'ensemble: {final_accuracy:.3f}")
print(f"{'='*50}")

# Poids finaux des experts
final_weights = ensemble.get_weights()
for name, weight in zip(classifiers.keys(), final_weights):
    print(f"{name}: {weight:.3f}")

# Matrice de confusion
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(true_labels, predictions)
print("\nMatrice de confusion:")
print(cm)
print("\nRapport de classification:")
print(classification_report(true_labels, predictions))
```

### 4. Apprentissage par Renforcement

```python
import numpy as np
from earcp import EARCP

# Environnement simple: Multi-Armed Bandit
class SimpleBanditEnv:
    """Environnement de bandit à K bras."""
    def __init__(self, k=10):
        self.k = k
        # Vraies valeurs des bras (inconnues de l'agent)
        self.true_values = np.random.randn(k)
    
    def step(self, action):
        """Tirer sur un bras et obtenir une récompense."""
        reward = self.true_values[action] + np.random.randn() * 0.1
        return reward

# Agents RL avec différentes stratégies
class EpsilonGreedyAgent:
    """Agent epsilon-greedy."""
    def __init__(self, n_actions, epsilon=0.1):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.q_values = np.zeros(n_actions)
        self.action_counts = np.zeros(n_actions)
    
    def predict(self, state):
        """Retourne les Q-values."""
        return self.q_values.copy()
    
    def select_action(self):
        """Sélectionne une action."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_values)
    
    def update_internal(self, action, reward):
        """Mise à jour interne de l'agent."""
        self.action_counts[action] += 1
        alpha = 1.0 / self.action_counts[action]
        self.q_values[action] += alpha * (reward - self.q_values[action])

class UCBAgent:
    """Agent Upper Confidence Bound."""
    def __init__(self, n_actions, c=2.0):
        self.n_actions = n_actions
        self.c = c
        self.q_values = np.zeros(n_actions)
        self.action_counts = np.zeros(n_actions)
        self.t = 0
    
    def predict(self, state):
        """Retourne les Q-values avec bonus d'exploration."""
        if self.t == 0:
            return self.q_values
        
        ucb_values = self.q_values + self.c * np.sqrt(
            np.log(self.t) / (self.action_counts + 1e-10)
        )
        return ucb_values
    
    def select_action(self):
        """Sélectionne l'action avec le plus grand UCB."""
        return np.argmax(self.predict(None))
    
    def update_internal(self, action, reward):
        """Mise à jour interne."""
        self.t += 1
        self.action_counts[action] += 1
        alpha = 1.0 / self.action_counts[action]
        self.q_values[action] += alpha * (reward - self.q_values[action])

class ThompsonSamplingAgent:
    """Agent Thompson Sampling (Bayésien)."""
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.alpha = np.ones(n_actions)  # Succès
        self.beta = np.ones(n_actions)   # Échecs
    
    def predict(self, state):
        """Sample des distributions Beta."""
        samples = np.random.beta(self.alpha, self.beta)
        return samples
    
    def select_action(self):
        """Sélectionne selon le sample."""
        return np.argmax(self.predict(None))
    
    def update_internal(self, action, reward):
        """Mise à jour Bayésienne."""
        # Convertir reward en [0, 1]
        normalized_reward = (reward + 3) / 6  # Supposant reward in [-3, 3]
        self.alpha[action] += normalized_reward
        self.beta[action] += (1 - normalized_reward)

# Créer l'environnement et les agents
env = SimpleBanditEnv(k=10)
agents = [
    EpsilonGreedyAgent(n_actions=10, epsilon=0.1),
    EpsilonGreedyAgent(n_actions=10, epsilon=0.05),
    UCBAgent(n_actions=10, c=2.0),
    ThompsonSamplingAgent(n_actions=10)
]

# Créer l'ensemble RL
ensemble = EARCP(experts=agents, beta=0.8)

# Boucle d'apprentissage
T = 1000
total_reward = 0
optimal_actions = 0
optimal_action = np.argmax(env.true_values)

for t in range(T):
    # Obtenir les Q-values de tous les agents
    state = None  # Pas d'état dans les bandits
    ensemble_q_values, agent_q_values = ensemble.predict(state)
    
    # Sélectionner l'action (argmax de l'ensemble)
    action = np.argmax(ensemble_q_values)
    
    # Exécuter l'action
    reward = env.step(action)
    total_reward += reward
    
    if action == optimal_action:
        optimal_actions += 1
    
    # Mettre à jour l'ensemble (utilise les Q-values comme "prédictions")
    # Target: reward observé pour l'action choisie
    target_q_values = ensemble_q_values.copy()
    target_q_values[action] = reward
    
    ensemble.update(agent_q_values, target_q_values)
    
    # Mettre à jour les agents individuellement aussi
    for agent in agents:
        agent.update_internal(action, reward)
    
    if (t + 1) % 200 == 0:
        avg_reward = total_reward / (t + 1)
        optimality = optimal_actions / (t + 1)
        print(f"Étape {t+1}: Récompense moyenne = {avg_reward:.3f}, " 
              f"Optimalité = {optimality:.2%}")

print(f"\n{'='*50}")
print(f"Récompense totale: {total_reward:.2f}")
print(f"Pourcentage d'actions optimales: {optimal_actions/T:.2%}")
print(f"{'='*50}")

# Poids finaux
final_weights = ensemble.get_weights()
agent_names = ['ε-greedy (0.1)', 'ε-greedy (0.05)', 'UCB', 'Thompson']
for name, weight in zip(agent_names, final_weights):
    print(f"{name}: {weight:.3f}")
```

---

## 📖 API Reference

### Classe Principale: EARCP

```python
class EARCP(experts, config=None, **kwargs)
```

Ensemble Auto-Régulé par Cohérence et Performance.

**Paramètres:**

- `experts` (list): Liste de modèles experts. Chaque expert doit implémenter une méthode `.predict(x)`.
  
- `config` (EARCPConfig, optional): Objet de configuration. Si None, utilise les valeurs par défaut ou celles fournies dans `**kwargs`.
  
- `**kwargs`: Paramètres de configuration supplémentaires (alpha_P, alpha_C, beta, eta_s, w_min, etc.)

**Méthodes Principales:**

#### `predict(x, return_expert_predictions=True)`

Fait une prédiction d'ensemble pour l'entrée `x`.

**Paramètres:**
- `x` (array-like): Entrée pour la prédiction
- `return_expert_predictions` (bool): Si True, retourne aussi les prédictions individuelles

**Retourne:**
- `prediction` (np.ndarray): Prédiction pondérée de l'ensemble
- `expert_predictions` (list, optional): Liste des prédictions individuelles

**Exemple:**
```python
pred, expert_preds = ensemble.predict(x)
```

#### `update(expert_predictions, target)`

Met à jour l'ensemble après observation de la cible.

**Paramètres:**
- `expert_predictions` (list): Prédictions des experts (retournées par `predict`)
- `target` (array-like): Valeur cible observée

**Retourne:**
- `metrics` (dict): Dictionnaire contenant:
  - `'weights'`: Poids mis à jour
  - `'performance_scores'`: Scores de performance
  - `'coherence_scores'`: Scores de cohérence
  - `'ensemble_loss'`: Perte de l'ensemble

**Exemple:**
```python
metrics = ensemble.update(expert_preds, target)
```

#### `get_weights()`

Retourne les poids actuels des experts.

**Retourne:**
- `weights` (np.ndarray): Tableau des poids normalisés

**Exemple:**
```python
weights = ensemble.get_weights()
```

#### `get_diagnostics()`

Retourne les diagnostics complets de l'ensemble.

**Retourne:**
- `diagnostics` (dict): Dictionnaire contenant toutes les métriques et historiques

**Exemple:**
```python
diag = ensemble.get_diagnostics()
print(f"Entropie: {diag['entropy']:.3f}")
```

#### `reset(initial_weights=None)`

Réinitialise l'ensemble à l'état initial.

**Paramètres:**
- `initial_weights` (array-like, optional): Poids initiaux. Si None, utilise des poids uniformes.

**Exemple:**
```python
ensemble.reset()
```

#### `save_state(filepath)`

Sauvegarde l'état complet de l'ensemble.

**Paramètres:**
- `filepath` (str): Chemin du fichier de sauvegarde

**Exemple:**
```python
ensemble.save_state('checkpoint.pkl')
```

#### `load_state(filepath)`

Charge l'état de l'ensemble depuis un fichier.

**Paramètres:**
- `filepath` (str): Chemin du fichier à charger

**Exemple:**
```python
ensemble.load_state('checkpoint.pkl')
```

### Classe de Configuration: EARCPConfig

```python
class EARCPConfig(
    alpha_P=0.9,
    alpha_C=0.85,
    beta=0.7,
    eta_s=5.0,
    w_min=0.05,
    loss_fn=None,
    coherence_fn=None,
    prediction_mode='auto',
    epsilon=1e-10,
    normalize_weights=True,
    track_diagnostics=True,
    random_state=None
)
```

Configuration pour EARCP.

**Paramètres:**

- `alpha_P` (float): Facteur de lissage pour la performance [0, 1]
- `alpha_C` (float): Facteur de lissage pour la cohérence [0, 1]
- `beta` (float): Balance performance/cohérence [0, 1]
- `eta_s` (float): Taux d'apprentissage/sensibilité > 0
- `w_min` (float): Poids minimum [0, 1]
- `loss_fn` (callable, optional): Fonction de perte personnalisée
- `coherence_fn` (callable, optional): Fonction de cohérence personnalisée
- `prediction_mode` (str): 'auto', 'regression', ou 'classification'
- `epsilon` (float): Petite constante pour stabilité numérique
- `normalize_weights` (bool): Normaliser les poids à somme=1
- `track_diagnostics` (bool): Suivre l'historique complet
- `random_state` (int, optional): Seed pour reproductibilité

### Fonctions Utilitaires

#### `get_preset_config(preset_name)`

Obtient une configuration prédéfinie.

**Paramètres:**
- `preset_name` (str): Nom du preset ('default', 'performance_focused', etc.)

**Retourne:**
- `config` (EARCPConfig): Configuration prédéfinie

**Presets disponibles:**
- `'default'`: Configuration standard
- `'performance_focused'`: Focus sur la performance
- `'diversity_focused'`: Focus sur la diversité
- `'balanced'`: Équilibre optimal
- `'conservative'`: Changements prudents
- `'aggressive'`: Changements rapides
- `'robust'`: Robuste au bruit
- `'high_performance'`: Performance maximale

---

## 🔧 Troubleshooting

### Problème: Les poids convergent vers un seul expert

**Symptôme:** Un expert obtient un poids proche de 1.0, les autres proches de 0.

**Solutions:**
```python
# 1. Augmenter w_min
ensemble = EARCP(experts=experts, w_min=0.15)

# 2. Diminuer beta (favoriser la cohérence)
ensemble = EARCP(experts=experts, beta=0.5)

# 3. Diminuer eta_s (changements plus doux)
ensemble = EARCP(experts=experts, eta_s=3.0)

# 4. Utiliser un preset robuste
config = get_preset_config('diversity_focused')
ensemble = EARCP(experts=experts, config=config)
```

### Problème: Les poids oscillent beaucoup

**Symptôme:** Les poids changent de manière erratique.

**Solutions:**
```python
# 1. Augmenter alpha_P et alpha_C (plus de mémoire)
ensemble = EARCP(experts=experts, alpha_P=0.95, alpha_C=0.90)

# 2. Diminuer eta_s (moins de sensibilité)
ensemble = EARCP(experts=experts, eta_s=2.0)

# 3. Utiliser un preset conservateur
config = get_preset_config('conservative')
ensemble = EARCP(experts=experts, config=config)
```

### Problème: Performance inférieure au meilleur expert

**Symptôme:** EARCP fait pire que le meilleur expert individuel.

**Analyse et solutions:**
```python
# 1. Augmenter beta (plus de focus sur la performance)
ensemble = EARCP(experts=experts, beta=0.9)

# 2. Vérifier que les experts sont diversifiés
# Les experts trop similaires réduisent les bénéfices
from earcp.utils.analysis import check_expert_diversity
diversity_score = check_expert_diversity(experts, test_data)
if diversity_score < 0.3:
    print("Les experts sont trop similaires!")

# 3. Augmenter la période d'adaptation
# EARCP a besoin de temps pour apprendre
# Vérifier la performance après au moins T > 100 étapes

# 4. Vérifier les fonctions de perte et de cohérence
# S'assurer qu'elles retournent des valeurs dans [0, 1]
```

### Problème: NaN ou Inf dans les calculs

**Symptôme:** Erreurs avec valeurs NaN ou infinies.

**Solutions:**
```python
# 1. Ajuster epsilon
ensemble = EARCP(experts=experts, epsilon=1e-8)

# 2. Normaliser les prédictions
class NormalizedExpert:
    def __init__(self, base_expert):
        self.base_expert = base_expert
    
    def predict(self, x):
        pred = self.base_expert.predict(x)
        # Clip les valeurs extrêmes
        return np.clip(pred, -1e6, 1e6)

# 3. Vérifier les données d'entrée
assert not np.any(np.isnan(x))
assert not np.any(np.isinf(x))
```

### Problème: Mémoire insuffisante avec beaucoup d'experts

**Symptôme:** Out of memory avec M > 50 experts.

**Solutions:**
```python
# 1. Désactiver le tracking de l'historique
config = EARCPConfig(track_diagnostics=False)
ensemble = EARCP(experts=experts, config=config)

# 2. Sauvegarder périodiquement et libérer la mémoire
if t % 1000 == 0:
    ensemble.save_state(f'checkpoint_{t}.pkl')
    # Créer un nouvel ensemble depuis le checkpoint
    ensemble = EARCP(experts=experts)
    ensemble.load_state(f'checkpoint_{t}.pkl')

# 3. Utiliser un sous-ensemble d'experts
# Sélectionner les K meilleurs périodiquement
if t % 500 == 0:
    weights = ensemble.get_weights()
    top_k_indices = np.argsort(weights)[-10:]  # Garder top 10
    experts = [experts[i] for i in top_k_indices]
    ensemble = EARCP(experts=experts)
```

### Problème: Lent avec beaucoup d'experts

**Symptôme:** Calcul très lent avec M > 20 experts.

**Solutions:**
```python
# 1. Utiliser des calculs vectorisés
# S'assurer que les experts retournent des array NumPy

# 2. Désactiver certaines fonctionnalités
config = EARCPConfig(
    track_diagnostics=False,  # Pas d'historique
    normalize_weights=False   # Skip normalisation si pas nécessaire
)

# 3. Utiliser le multiprocessing pour les prédictions
from multiprocessing import Pool

def get_prediction(expert, x):
    return expert.predict(x)

with Pool(processes=4) as pool:
    expert_preds = pool.starmap(get_prediction, [(e, x) for e in experts])
```

---

## ❓ FAQ

### Questions Générales

**Q: Combien d'experts minimum/maximum puis-je utiliser ?**

**R:** Minimum 2 experts. Testé avec succès jusqu'à 50+ experts. Performance optimale avec 3-10 experts diversifiés. Au-delà de 20, considérez des approches hiérarchiques.

**Q: EARCP fonctionne-t-il avec des modèles pré-entraînés ?**

**R:** Oui, parfaitement ! Utilisez les wrappers (`SklearnWrapper`, `TorchWrapper`, `KerasWrapper`) pour intégrer n'importe quel modèle pré-entraîné sans réentraînement.

**Q: Dois-je réentraîner mes experts ?**

**R:** Non. EARCP ne modifie jamais les experts. Il apprend uniquement comment les combiner optimalement.

**Q: EARCP supporte-t-il le GPU ?**

**R:** EARCP lui-même tourne sur CPU (très léger). Mais vos experts (ex: réseaux de neurones) peuvent utiliser le GPU normalement.

### Hyperparamètres

**Q: Comment choisir beta ?**

**R:** 
- **β = 0.7-0.8** : Recommandé pour débuter
- **β > 0.8** : Si vous avez confiance en vos métriques
- **β < 0.7** : Pour favoriser la robustesse et la diversité
- **β = 1.0** : Mode performance pure (équivalent à Hedge)

**Q: Comment régler eta_s ?**

**R:**
- **eta_s = 3-4** : Environnement stable, changements lents
- **eta_s = 5-6** : Défaut, environnement modéré
- **eta_s = 7-9** : Environnement dynamique, adaptation rapide

**Q: Que fait w_min exactement ?**

**R:** `w_min` garantit qu'aucun expert ne sera complètement ignoré, même s'il performe mal. Cela permet une récupération si l'environnement change. Typiquement: `w_min = 1.0 / n_experts`.

### Performance

**Q: Quelle est la complexité temporelle ?**

**R:** O(M²) par étape pour M experts, principalement dû au calcul de la matrice de cohérence. Optimisations possibles pour M > 20.

**Q: EARCP est-il meilleur que le meilleur expert ?**

**R:** EARCP garantit un regret O(√(T log M)), ce qui signifie qu'il converge asymptotiquement vers le meilleur expert. Sur le long terme (T grand), EARCP est au moins aussi bon que le meilleur expert.

**Q: Combien de temps avant que EARCP converge ?**

**R:** Généralement 50-200 étapes pour voir une adaptation significative. La convergence dépend de:
- Diversité des experts
- Valeur de eta_s
- Stabilité de l'environnement

### Implémentation

**Q: EARCP supporte-t-il l'apprentissage par batch ?**

**R:** EARCP est conçu pour l'apprentissage en ligne séquentiel. Pour du batch, appelez simplement `update()` pour chaque échantillon dans le batch.

**Q: Puis-je ajouter/retirer des experts dynamiquement ?**

**R:** Actuellement non supporté directement. Vous devez créer un nouvel ensemble. Approche recommandée :
```python
# Créer un nouvel ensemble avec les nouveaux experts
new_ensemble = EARCP(experts=new_expert_list)
# Optionnel: transférer les poids si applicable
```

**Q: Comment gérer des experts qui retournent des formats différents ?**

**R:** Créez des classes wrapper pour standardiser les sorties:
```python
class OutputWrapper:
    def __init__(self, expert, transform_fn):
        self.expert = expert
        self.transform = transform_fn
    
    def predict(self, x):
        raw_output = self.expert.predict(x)
        return self.transform(raw_output)
```

### Cas d'Usage

**Q: EARCP fonctionne-t-il pour la classification ?**

**R:** Oui ! Utilisez `prediction_mode='classification'`. EARCP combinera les probabilités des classes et vous pouvez prendre l'argmax pour la prédiction finale.

**Q: Puis-je utiliser EARCP pour du clustering ?**

**R:** Oui, si vos experts produisent des assignations de clusters ou des probabilités d'appartenance. EARCP peut les combiner.

**Q: EARCP est-il adapté pour le traitement du langage naturel ?**

**R:** Oui, si vous avez plusieurs modèles de langage (BERT, GPT, etc.) produisant des embeddings ou des probabilités, EARCP peut les combiner intelligemment.

**Q: Puis-je utiliser EARCP dans un environnement de production ?**

**R:** Absolument ! EARCP est:
- Production-ready
- Testé extensivement
- Faible overhead
- Facile à monitorer

### Licence et Support

**Q: EARCP est-il gratuit ?**

**R:** 
- **Gratuit** : Recherche académique, projets personnels, entreprises < $100k de revenu
- **Commercial** : Licence requise pour entreprises > $100k
- **Open-source** : Devient Apache 2.0 le 13 novembre 2029

**Q: Où obtenir de l'aide ?**

**R:**
- Documentation: [README](https://github.com/Volgat/earcp)
- Issues GitHub: https://github.com/Volgat/earcp/issues
- Email: info@amewebstudio.com

**Q: Puis-je contribuer ?**

**R:** Oui ! Les contributions sont bienvenues. Voir [CONTRIBUTING.md](https://github.com/Volgat/earcp/blob/main/CONTRIBUTING.md).

---

## 📞 Support et Contact

### Obtenir de l'Aide

**Documentation:**
- README: https://github.com/Volgat/earcp
- Examples: https://github.com/Volgat/earcp/tree/main/examples
- Tutorials: https://github.com/Volgat/earcp/wiki

**Bugs et Issues:**
- GitHub Issues: https://github.com/Volgat/earcp/issues
- Inclure: version, code minimal reproductible, messages d'erreur

**Questions:**
- Discussions GitHub: https://github.com/Volgat/earcp/discussions
- Stack Overflow: Tag `earcp`

### Contact Direct

**Auteur:** Mike Amega  
**Email:** info@amewebstudio.com  
**LinkedIn:** https://www.linkedin.com/in/mike-amega-486329184/  
**GitHub:** [@Volgat](https://github.com/Volgat)  
**Location:** Windsor, Ontario, Canada

**Pour:**
- 🏢 Licences commerciales
- 🤝 Collaborations de recherche
- 💼 Consulting et support technique
- 🎓 Présentations et formations

---

## 📄 Citation

Si vous utilisez EARCP dans vos travaux, veuillez citer:

```bibtex
@article{amega2025earcp,
  title={EARCP: Ensemble Auto-Régulé par Cohérence et Performance},
  author={Amega, Mike},
  year={2025},
  journal={arXiv preprint},
  url={https://github.com/Volgat/earcp},
  note={Prior art established November 13, 2025}
}
```

---

## 📜 Licence

**Copyright © 2025 Mike Amega. Tous droits réservés.**

EARCP est distribué sous la **Business Source License 1.1**. Voir [LICENSE.md](https://github.com/Volgat/earcp/blob/main/LICENSE.md) pour les termes complets.

**Résumé:**
- ✅ Gratuit pour recherche, éducation, et usage interne (<$100k)
- 💼 Licence commerciale requise pour entreprises >$100k
- 🔓 Devient Apache 2.0 le 13 novembre 2029

---

**Dernière mise à jour:** 3 décembre 2025  
**Version du document:** 2.0  
**Version EARCP:** 1.0.0
