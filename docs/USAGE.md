# EARCP - Complete Usage Guide

**Ensemble Auto-Régulé par Cohérence et Performance**

Copyright © 2025 Mike Amega. All rights reserved.

---

## 📑 Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Configuration](#configuration)
5. [Advanced Usage](#advanced-usage)
6. [ML Framework Integration](#ml-framework-integration)
7. [Visualization & Diagnostics](#visualization--diagnostics)
8. [Use Cases](#use-cases)
9. [API Reference](#api-reference)
10. [Troubleshooting](#troubleshooting)
11. [FAQ](#faq)

---

## 🚀 Installation

### Install from PyPI (Recommended)

EARCP is now available on PyPI! Simple installation:

```bash
pip install earcp
```

### Install with Optional Dependencies

```bash
# With PyTorch support
pip install earcp[torch]

# With scikit-learn support
pip install earcp[sklearn]

# With TensorFlow/Keras support
pip install earcp[tensorflow]

# With all visualization dependencies
pip install earcp[viz]

# Full installation (all dependencies)
pip install earcp[full]
```

### Install from GitHub

For the latest development version:

```bash
# Direct installation
pip install git+https://github.com/Volgat/earcp.git@earcp-lib

# Or clone and install locally
git clone -b earcp-lib https://github.com/Volgat/earcp.git
cd earcp
pip install -e .
```

### Dependencies

**Required:**
- numpy >= 1.20.0
- scipy >= 1.7.0

**Optional:**
- matplotlib >= 3.3.0 (visualization)
- torch >= 1.9.0 (PyTorch support)
- tensorflow >= 2.4.0 (TensorFlow support)
- scikit-learn >= 0.24.0 (wrappers and metrics)

### Verify Installation

```python
import earcp
print(f"EARCP version: {earcp.__version__}")

# Quick test
from earcp import EARCP
print("Installation successful!")
```

---

## ⚡ Quick Start

### Minimal Example (30 seconds)

```python
from earcp import EARCP
import numpy as np

# 1. Define expert models (any model with .predict() method)
class SimpleExpert:
    def __init__(self, coefficient):
        self.coefficient = coefficient
    
    def predict(self, x):
        return self.coefficient * x

# 2. Create your experts
experts = [
    SimpleExpert(1.0),
    SimpleExpert(2.0),
    SimpleExpert(0.5),
]

# 3. Initialize EARCP
ensemble = EARCP(experts=experts)

# 4. Online learning loop
for t in range(100):
    x = np.random.randn(5)
    
    # Prediction
    prediction, expert_preds = ensemble.predict(x)
    
    # Observe target
    target = 1.5 * x + np.random.randn(5) * 0.1
    
    # Update
    metrics = ensemble.update(expert_preds, target)

print(f"Final weights: {ensemble.get_weights()}")
```

### Complete Example with Real Data

```python
import numpy as np
from earcp import EARCP

# Generate synthetic data
np.random.seed(42)
T = 500  # Number of time steps

def generate_data(t):
    """Target function with regime change."""
    x = t * 0.05
    if t < 250:
        # Linear regime
        return 2*x + np.random.normal(0, 0.1)
    else:
        # Sinusoidal regime
        return 2*x + 3*np.sin(x) + np.random.normal(0, 0.1)

# Define experts with different strategies
class LinearExpert:
    """Expert based on linear function."""
    def __init__(self, slope, intercept=0):
        self.slope = slope
        self.intercept = intercept
    
    def predict(self, x):
        return self.slope * x + self.intercept

class PolynomialExpert:
    """Expert based on polynomial."""
    def __init__(self, coefficients):
        self.coefficients = coefficients
    
    def predict(self, x):
        return sum(c * x**i for i, c in enumerate(self.coefficients))

class SinusoidalExpert:
    """Expert based on sinusoidal function."""
    def __init__(self, amplitude, frequency, phase=0):
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
    
    def predict(self, x):
        return self.amplitude * np.sin(self.frequency * x + self.phase)

# Create diverse ensemble of experts
experts = [
    LinearExpert(slope=2.0, intercept=0.5),
    LinearExpert(slope=1.8, intercept=1.0),
    PolynomialExpert([0, 2.0, 0.1]),
    SinusoidalExpert(amplitude=3.0, frequency=1.0),
]

# Initialize EARCP with optimal parameters
ensemble = EARCP(
    experts=experts,
    beta=0.7,        # Performance/coherence balance
    eta_s=5.0,       # Weight sensitivity
    alpha_P=0.9,     # Performance smoothing
    alpha_C=0.85,    # Coherence smoothing
    w_min=0.05       # Minimum weight (prevents exclusion)
)

# Online learning
predictions = []
targets = []
weights_history = []

for t in range(T):
    # Input
    x = np.array([t * 0.05])
    
    # True target
    target = np.array([generate_data(t)])
    
    # Ensemble prediction
    pred, expert_preds = ensemble.predict(x)
    
    # Update weights
    metrics = ensemble.update(expert_preds, target)
    
    # Save for analysis
    predictions.append(pred[0])
    targets.append(target[0])
    weights_history.append(metrics['weights'].copy())
    
    # Periodic display
    if (t + 1) % 100 == 0:
        current_weights = metrics['weights']
        print(f"Step {t+1}:")
        print(f"  Weights: {[f'{w:.3f}' for w in current_weights]}")
        print(f"  Error: {metrics['ensemble_loss']:.4f}")

# Final diagnostics
diagnostics = ensemble.get_diagnostics()
print(f"\n{'='*50}")
print("FINAL RESULTS")
print(f"{'='*50}")
print(f"Final weights: {[f'{w:.3f}' for w in diagnostics['weights']]}")
print(f"Cumulative losses: {diagnostics['cumulative_loss']}")

# Calculate overall RMSE
rmse = np.sqrt(np.mean((np.array(predictions) - np.array(targets))**2))
print(f"Overall RMSE: {rmse:.4f}")
```

---

## 📚 Core Concepts

### EARCP Architecture

EARCP is an ensemble system that intelligently combines multiple expert models using a **dual-signal weighting mechanism**.

#### The Two Signals

1. **Performance (P)** 📈
   - Measures individual prediction quality of each expert
   - High score = high performing expert
   - Uses exponential smoothing for stability

2. **Coherence (C)** 🤝
   - Measures agreement between an expert and others
   - High score = expert aligned with consensus
   - Promotes diversity and robustness

#### Algorithm Step by Step

At each time step *t*, EARCP performs the following operations:

**Step 1: Collect Predictions**
```
Get p₁,ₜ, p₂,ₜ, ..., p_M,ₜ from M experts
```

**Step 2: Compute Performance**
```
P_i,t = α_P · P_i,t-1 + (1 - α_P) · (-ℓ_i,t)

where ℓ_i,t = loss of expert i at time t
```

**Step 3: Compute Coherence**
```
C_i,t = 1/(M-1) · Σⱼ≠ᵢ Agreement(pᵢ,ₜ, pⱼ,ₜ)

Agreement measures similarity between predictions
```

**Step 4: Fuse Signals**
```
s_i,t = β · P_i,t + (1 - β) · C_i,t

β controls balance between performance and coherence
```

**Step 5: Update Weights**
```
w̃_i,t = exp(η_s · s_i,t)
w_i,t = max(w_min, w̃_i,t / Σⱼ w̃_j,t)

w_min ensures minimum weight for each expert
```

**Step 6: Ensemble Prediction**
```
ŷ_t = Σᵢ w_i,t · p_i,t
```

### Theoretical Guarantees

**Theorem (Regret Bound)**: Under standard assumptions (bounded losses in [0,1], convexity), EARCP guarantees:

```
Regret_T ≤ (1/β) · √(2T log M)
```

For β = 1 (pure performance):
```
Regret_T ≤ √(2T log M)
```

**Practical implications:**
- Regret per step decreases as O(√(log M / T))
- Asymptotically optimal performance
- Comparable to best expert in hindsight

### Parameters: Complete Guide

| Parameter | Role | Default | Range | Impact |
|-----------|------|---------|-------|--------|
| **alpha_P** | Performance memory | 0.9 | [0.8, 0.99] | ↑ = more memory, slow reaction<br>↓ = fast forgetting, quick adaptation |
| **alpha_C** | Coherence memory | 0.85 | [0.75, 0.95] | ↑ = stable coherence<br>↓ = reactive coherence |
| **beta** | P/C balance | 0.7 | [0.5, 1.0] | ↑ = favor performance<br>↓ = favor diversity |
| **eta_s** | Learning rate | 5.0 | [1.0, 10.0] | ↑ = fast changes<br>↓ = stability |
| **w_min** | Weight floor | 0.05 | [0.01, 0.2] | Prevents complete exclusion |

#### Beta Tuning Guide

**β = 1.0**: Pure Performance Mode
- Equivalent to Hedge algorithm
- Use when: absolute confidence in your metrics
- Advantage: fast convergence to the best
- Disadvantage: can overfit, lacks robustness

**β = 0.8-0.9**: Performance-Dominant Mode
- Strongly favors performance
- Use when: stable environment, reliable metrics
- Advantage: good performance with some diversity
- Disadvantage: still can overfit

**β = 0.6-0.7**: Balanced Mode ⭐ (Recommended)
- Optimal balance between performance and coherence
- Use when: variable environment, moderate uncertainty
- Advantage: robust, adaptable, good compromise
- Ideal for most applications

**β = 0.4-0.5**: Diversity Mode
- Strongly favors coherence
- Use when: very noisy environment, unreliable metrics
- Advantage: maximum robustness and diversity
- Disadvantage: slower convergence

**β = 0.0**: Pure Coherence Mode (Experimental)
- Weighting based solely on inter-expert agreement
- Use when: no confidence in individual metrics
- Advantage: pure consensus
- Disadvantage: can ignore good isolated experts

#### Practical Tuning Rules

**For eta_s (Sensitivity):**
```python
# Stable environment
eta_s = 3.0 - 4.0

# Moderate environment (default)
eta_s = 5.0 - 6.0

# Highly dynamic environment
eta_s = 7.0 - 9.0
```

**For w_min (Minimum weight):**
```python
# General rule: between 0.5/M and 2.0/M
M = len(experts)
w_min = 1.0 / M  # Good starting point

# Many experts
if M > 10:
    w_min = 0.5 / M

# Few experts
if M <= 3:
    w_min = 0.1  # More conservative
```

**For alpha_P and alpha_C:**
```python
# Stationary environment
alpha_P, alpha_C = 0.95, 0.90

# Moderately variable environment (default)
alpha_P, alpha_C = 0.90, 0.85

# Highly variable environment (concept drift)
alpha_P, alpha_C = 0.85, 0.80
```

---

## ⚙️ Configuration

### Using EARCPConfig

```python
from earcp import EARCP, EARCPConfig

# Create custom configuration
config = EARCPConfig(
    # Smoothing parameters
    alpha_P=0.9,
    alpha_C=0.85,
    
    # Balance and sensitivity
    beta=0.7,
    eta_s=5.0,
    
    # Constraints
    w_min=0.05,
    epsilon=1e-10,
    
    # Prediction mode
    prediction_mode='regression',  # 'regression', 'classification', or 'auto'
    
    # Custom functions
    loss_fn=None,          # Custom loss function
    coherence_fn=None,     # Custom coherence function
    
    # Tracking options
    track_diagnostics=True,
    normalize_weights=True,
    
    # Reproducibility
    random_state=42
)

# Use the configuration
ensemble = EARCP(experts=experts, config=config)
```

### Preset Configurations

EARCP provides several optimized configurations for different use cases:

```python
from earcp import get_preset_config

# 1. Default configuration (balanced)
config = get_preset_config('default')
# beta=0.7, eta_s=5.0, suitable for most cases

# 2. Performance-focused
config = get_preset_config('performance_focused')
# beta=0.95, prioritizes high-performing experts

# 3. Diversity-focused
config = get_preset_config('diversity_focused')
# beta=0.5, favors consensus and robustness

# 4. Optimal balanced configuration
config = get_preset_config('balanced')
# beta=0.7, finely tuned parameters

# 5. Conservative mode
config = get_preset_config('conservative')
# Slow changes, high stability

# 6. Aggressive mode
config = get_preset_config('aggressive')
# Fast changes, dynamic adaptation

# 7. Robust mode (for noisy data)
config = get_preset_config('robust')
# beta=0.6, high w_min, resists noise

# 8. High-performance mode
config = get_preset_config('high_performance')
# beta=0.9, fast convergence

# Usage
ensemble = EARCP(experts=experts, config=config)
```

### Custom Loss Functions

```python
import numpy as np

# Loss function for regression
def custom_regression_loss(y_pred, y_true):
    """
    Custom loss function.
    Must return a value in [0, 1].
    """
    mse = np.mean((y_pred - y_true) ** 2)
    # Normalize with tanh or sigmoid
    return np.tanh(mse)

# Loss function for classification
def custom_classification_loss(y_pred, y_true):
    """
    Cross-entropy for classification.
    """
    # y_pred: probabilities, y_true: one-hot
    epsilon = 1e-10
    ce = -np.sum(y_true * np.log(y_pred + epsilon))
    return ce / len(y_true)

# Robust loss (insensitive to outliers)
def huber_loss(y_pred, y_true, delta=1.0):
    """
    Huber loss: quadratic for small errors,
    linear for large errors.
    """
    error = np.abs(y_pred - y_true)
    is_small = error <= delta
    
    small_loss = 0.5 * error**2
    large_loss = delta * error - 0.5 * delta**2
    
    loss = np.where(is_small, small_loss, large_loss)
    return np.mean(loss) / (2 * delta)  # Normalize to [0, 1]

# Usage
config = EARCPConfig(loss_fn=huber_loss)
ensemble = EARCP(experts=experts, config=config)
```

### Custom Coherence Functions

```python
import numpy as np

# Correlation-based coherence
def correlation_coherence(pred_i, pred_j):
    """
    Measures correlation between two predictions.
    Returns a value in [0, 1].
    """
    # Flatten predictions
    pi = pred_i.flatten()
    pj = pred_j.flatten()
    
    # Calculate Pearson correlation
    if len(pi) > 1:
        correlation = np.corrcoef(pi, pj)[0, 1]
        # Map [-1, 1] to [0, 1]
        return (correlation + 1) / 2
    else:
        # Direct similarity for scalar predictions
        return 1.0 / (1.0 + np.abs(pi - pj))

# Distance-based coherence
def distance_coherence(pred_i, pred_j):
    """
    Coherence based on Euclidean distance.
    Smaller distance = higher coherence.
    """
    distance = np.linalg.norm(pred_i - pred_j)
    # Transform distance into similarity
    return np.exp(-distance)

# Coherence for classification (agreement on classes)
def classification_coherence(pred_i, pred_j):
    """
    For classification: percentage of agreement on predicted classes.
    """
    class_i = np.argmax(pred_i, axis=-1)
    class_j = np.argmax(pred_j, axis=-1)
    agreement = np.mean(class_i == class_j)
    return agreement

# Weighted coherence (favors certain dimensions)
def weighted_coherence(pred_i, pred_j, weights=None):
    """
    Coherence with dimension weighting.
    """
    if weights is None:
        weights = np.ones(pred_i.shape)
    
    diff = weights * (pred_i - pred_j) ** 2
    distance = np.sqrt(np.sum(diff))
    return np.exp(-distance)

# Usage
config = EARCPConfig(coherence_fn=correlation_coherence)
ensemble = EARCP(experts=experts, config=config)
```

---

## 🔬 Advanced Usage

### Save and Load State

```python
import pickle

# Save complete ensemble state
ensemble.save_state('checkpoints/ensemble_step_1000.pkl')

# Load state
ensemble_restored = EARCP(experts=experts)
ensemble_restored.load_state('checkpoints/ensemble_step_1000.pkl')

# Continue learning
for t in range(1000, 2000):
    pred, expert_preds = ensemble_restored.predict(x[t])
    ensemble_restored.update(expert_preds, y[t])
```

### Periodic Checkpoints

```python
# Save every N steps
CHECKPOINT_INTERVAL = 100

for t in range(T):
    pred, expert_preds = ensemble.predict(x[t])
    metrics = ensemble.update(expert_preds, y[t])
    
    # Checkpoint
    if (t + 1) % CHECKPOINT_INTERVAL == 0:
        ensemble.save_state(f'checkpoints/step_{t+1}.pkl')
        print(f"Checkpoint saved at step {t+1}")
```

### Reset Ensemble

```python
# Complete reset (uniform weights)
ensemble.reset()

# Reset with custom weights
custom_weights = np.array([0.5, 0.3, 0.2])
ensemble.reset(initial_weights=custom_weights)
```

### Modify Parameters Dynamically

```python
# Adjust beta during runtime (e.g., decay)
for t in range(T):
    # Gradually decrease beta from 0.9 to 0.6
    current_beta = 0.9 - 0.3 * (t / T)
    ensemble.weighting.set_beta(current_beta)
    
    pred, expert_preds = ensemble.predict(x[t])
    ensemble.update(expert_preds, y[t])

# Adjust sensitivity based on context
if detect_concept_drift():
    ensemble.weighting.set_eta_s(8.0)  # More aggressive
else:
    ensemble.weighting.set_eta_s(4.0)  # More conservative

# Adjust smoothing factors
ensemble.performance_tracker.set_alpha(0.95)  # More memory
ensemble.coherence_metrics.set_alpha(0.80)    # Less memory
```

### Access Internal Components

```python
# Get raw performance scores
perf_scores = ensemble.performance_tracker.get_scores()
print(f"Performance scores: {perf_scores}")

# Get coherence scores
coh_scores = ensemble.coherence_metrics.get_scores()
print(f"Coherence scores: {coh_scores}")

# Get full coherence matrix
expert_predictions = [expert.predict(x) for expert in experts]
coh_matrix = ensemble.coherence_metrics.get_coherence_matrix(expert_predictions)
print(f"Coherence matrix:\n{coh_matrix}")

# Get complete loss history
loss_history = ensemble.performance_tracker.get_loss_history()

# Get advanced statistics
stats = {
    'mean_weight': np.mean(ensemble.get_weights()),
    'weight_std': np.std(ensemble.get_weights()),
    'effective_experts': np.sum(ensemble.get_weights() > 0.1),
    'entropy': ensemble.get_diagnostics()['entropy']
}
```

### Multi-Objective Management

```python
from earcp.core.performance_tracker import MultiObjectivePerformanceTracker

# Create tracker for multiple objectives
# Example: trading with profit AND risk
tracker = MultiObjectivePerformanceTracker(
    n_experts=3,
    n_objectives=2,
    objective_weights=[0.7, 0.3],  # 70% profit, 30% risk
    aggregation='weighted_sum'      # or 'product', 'min', 'max'
)

# In learning loop
for t in range(T):
    predictions = [expert.predict(state) for expert in experts]
    
    # Execute action
    profit, risk = execute_trade(predictions)
    
    # Update with both objectives
    tracker.update(
        predictions=predictions,
        targets=[profit, -risk]  # Maximize profit, minimize risk
    )
    
    # Get aggregated scores
    aggregated_scores = tracker.get_aggregated_scores()
```

### Ensemble of Ensembles (Meta-EARCP)

```python
# Create multiple EARCP ensembles
ensemble_1 = EARCP(experts=experts_group_1, beta=0.8)
ensemble_2 = EARCP(experts=experts_group_2, beta=0.6)
ensemble_3 = EARCP(experts=experts_group_3, beta=0.7)

# Wrapper to use an EARCP as an expert
class EARCPExpert:
    def __init__(self, earcp_ensemble):
        self.ensemble = earcp_ensemble
    
    def predict(self, x):
        pred, _ = self.ensemble.predict(x)
        return pred

# Create meta-ensemble
meta_experts = [
    EARCPExpert(ensemble_1),
    EARCPExpert(ensemble_2),
    EARCPExpert(ensemble_3)
]

meta_ensemble = EARCP(experts=meta_experts, beta=0.75)

# Hierarchical usage
for t in range(T):
    # Meta-ensemble prediction
    meta_pred, sub_preds = meta_ensemble.predict(x[t])
    
    # Update all levels
    meta_ensemble.update(sub_preds, y[t])
    
    # Update base ensembles (optional)
    for i, sub_ensemble in enumerate([ensemble_1, ensemble_2, ensemble_3]):
        _, expert_preds = sub_ensemble.predict(x[t])
        sub_ensemble.update(expert_preds, y[t])
```

---

## 🔗 ML Framework Integration

### With scikit-learn

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from earcp import EARCP
from earcp.utils.wrappers import SklearnWrapper

# Create and train sklearn models
models = {
    'ridge': Ridge(alpha=1.0),
    'lasso': Lasso(alpha=0.5),
    'elastic': ElasticNet(alpha=0.7, l1_ratio=0.5),
    'rf': RandomForestRegressor(n_estimators=100, max_depth=10),
    'gbm': GradientBoostingRegressor(n_estimators=100),
    'svr': SVR(kernel='rbf')
}

# Train all models
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name} trained")

# Create wrappers for EARCP
experts = [SklearnWrapper(model, name=name) 
           for name, model in models.items()]

# Create ensemble
ensemble = EARCP(experts=experts, beta=0.7)

# Online evaluation on test set
predictions = []
for i, (x, y) in enumerate(zip(X_test, y_test)):
    # Prediction
    pred, expert_preds = ensemble.predict(x.reshape(1, -1))
    predictions.append(pred[0])
    
    # Update
    ensemble.update(expert_preds, y.reshape(1, -1))
    
    if (i + 1) % 100 == 0:
        current_rmse = np.sqrt(np.mean((np.array(predictions) - y_test[:i+1])**2))
        print(f"Step {i+1}: RMSE = {current_rmse:.4f}")

# Final results
final_weights = ensemble.get_weights()
for name, weight in zip(models.keys(), final_weights):
    print(f"{name}: {weight:.3f}")
```

### With PyTorch

```python
import torch
import torch.nn as nn
from earcp import EARCP
from earcp.utils.wrappers import TorchWrapper

# Define different architectures
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

# Create and train models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
input_dim, output_dim = 10, 1

models = [
    SmallCNN(input_dim, output_dim).to(device),
    DeepNN(input_dim, output_dim).to(device),
]

# Training (simple example)
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

# Create wrappers
experts = [TorchWrapper(model, device=device) for model in models]

# Set models to eval mode
for model in models:
    model.eval()

# Create EARCP ensemble
ensemble = EARCP(experts=experts)

# Usage
with torch.no_grad():
    for x, y in test_loader:
        pred, expert_preds = ensemble.predict(x)
        ensemble.update(expert_preds, y)
```

### With TensorFlow/Keras

```python
from tensorflow import keras
from tensorflow.keras import layers
from earcp import EARCP
from earcp.utils.wrappers import KerasWrapper

# Create different Keras models
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

# Create multiple models
models = [
    create_dense_model(10, 1),
    create_deep_model(10, 1),
    create_dense_model(10, 1),  # Same architecture, different init
]

# Train each model
for i, model in enumerate(models):
    print(f"Training model {i+1}...")
    model.fit(X_train, y_train, epochs=20, batch_size=32, 
              validation_split=0.2, verbose=0)

# Create wrappers
experts = [KerasWrapper(model) for model in models]

# Create ensemble
ensemble = EARCP(experts=experts, beta=0.7)

# Usage
for i in range(len(X_test)):
    x = X_test[i:i+1]
    y = y_test[i:i+1]
    
    pred, expert_preds = ensemble.predict(x)
    ensemble.update(expert_preds, y)
```

### Universal Wrapper for Functions

```python
from earcp.utils.wrappers import CallableWrapper

# Create experts from simple functions
def moving_average(x, window=5):
    """Moving average."""
    return np.mean(x[-window:])

def exponential_smoothing(x, alpha=0.3):
    """Exponential smoothing."""
    if len(x) == 0:
        return 0
    result = x[0]
    for val in x[1:]:
        result = alpha * val + (1 - alpha) * result
    return result

def trend_extrapolation(x):
    """Trend extrapolation."""
    if len(x) < 2:
        return x[-1] if len(x) > 0 else 0
    return 2 * x[-1] - x[-2]

# Wrap functions
experts = [
    CallableWrapper(moving_average, name='MA'),
    CallableWrapper(exponential_smoothing, name='ES'),
    CallableWrapper(trend_extrapolation, name='Trend')
]

# Use in EARCP
ensemble = EARCP(experts=experts)
```

### Integration with LightGBM and XGBoost

```python
import lightgbm as lgb
import xgboost as xgb
from earcp.utils.wrappers import SklearnWrapper

# Create boosting models
lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1)
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)

# Train
lgb_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# Combine with other models
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X_train, y_train)

# Create ensemble
experts = [
    SklearnWrapper(lgb_model, name='LightGBM'),
    SklearnWrapper(xgb_model, name='XGBoost'),
    SklearnWrapper(rf_model, name='RandomForest')
]

ensemble = EARCP(experts=experts, beta=0.75)
```

---

## 📊 Visualization & Diagnostics

### Get Complete Diagnostics

```python
# Get all diagnostics
diagnostics = ensemble.get_diagnostics()

# Available content
print("Available diagnostics:")
for key in diagnostics.keys():
    print(f"  - {key}: {type(diagnostics[key])}")

# Typical diagnostics:
# - 'weights': Current weights [array]
# - 'performance_scores': Performance scores [array]
# - 'coherence_scores': Coherence scores [array]
# - 'time_step': Current time step [int]
# - 'weights_history': Complete weight history [list of arrays]
# - 'performance_history': Performance history [list of arrays]
# - 'coherence_history': Coherence history [list of arrays]
# - 'cumulative_loss': Cumulative loss per expert [array]
# - 'entropy': Weight distribution entropy [float]
# - 'effective_experts': Number of active experts [float]
```

### Standard Visualizations

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

# 1. Weight evolution over time
fig, ax = plt.subplots(figsize=(12, 6))
plot_weights(
    weights_history=diagnostics['weights_history'],
    expert_names=['Expert 1', 'Expert 2', 'Expert 3'],
    ax=ax,
    title='EARCP Weight Evolution',
    save_path='figures/weights_evolution.png'
)
plt.show()

# 2. Performance and coherence
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
plot_performance(
    performance_history=diagnostics['performance_history'],
    expert_names=['Expert 1', 'Expert 2', 'Expert 3'],
    ax=ax1,
    title='Performance Scores'
)
plot_coherence(
    coherence_history=diagnostics['coherence_history'],
    expert_names=['Expert 1', 'Expert 2', 'Expert 3'],
    ax=ax2,
    title='Coherence Scores'
)
plt.tight_layout()
plt.savefig('figures/performance_coherence.png')
plt.show()

# 3. Complete dashboard
fig = plot_diagnostics(
    diagnostics=diagnostics,
    expert_names=['Expert 1', 'Expert 2', 'Expert 3'],
    figsize=(16, 10),
    save_path='figures/full_diagnostics.png'
)
plt.show()

# 4. Regret analysis
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

# 5. Predictions vs Reality
fig, ax = plt.subplots(figsize=(12, 6))
plot_predictions(
    predictions=predictions,
    targets=targets,
    expert_predictions=expert_predictions_history,
    expert_names=['Expert 1', 'Expert 2', 'Expert 3'],
    ax=ax,
    title='EARCP Predictions vs Reality',
    save_path='figures/predictions.png'
)
plt.show()
```

### Advanced Visualizations

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Weight correlation matrix
def plot_weight_correlation(weights_history):
    """Correlation between expert weight evolution."""
    weights_array = np.array(weights_history).T  # (n_experts, time_steps)
    correlation_matrix = np.corrcoef(weights_array)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',
                center=0, vmin=-1, vmax=1)
    plt.title('Weight Evolution Correlation')
    plt.xlabel('Expert')
    plt.ylabel('Expert')
    plt.tight_layout()
    plt.savefig('figures/weight_correlation.png')
    plt.show()

plot_weight_correlation(diagnostics['weights_history'])

# Coherence matrix heatmap
def plot_coherence_matrix(ensemble, expert_predictions):
    """Visualize inter-expert coherence matrix."""
    coh_matrix = ensemble.coherence_metrics.get_coherence_matrix(expert_predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(coh_matrix, annot=True, cmap='viridis',
                vmin=0, vmax=1)
    plt.title('Inter-Expert Coherence Matrix')
    plt.xlabel('Expert')
    plt.ylabel('Expert')
    plt.tight_layout()
    plt.savefig('figures/coherence_matrix.png')
    plt.show()

# Weight distributions at different times
def plot_weight_distributions(weights_history, time_points=None):
    """Compare weight distributions at different moments."""
    if time_points is None:
        len_history = len(weights_history)
        time_points = [0, len_history//4, len_history//2, -1]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, t in enumerate(time_points):
        weights = weights_history[t]
        axes[idx].bar(range(len(weights)), weights)
        axes[idx].set_title(f'Distribution at t={t}')
        axes[idx].set_xlabel('Expert')
        axes[idx].set_ylabel('Weight')
        axes[idx].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('figures/weight_distributions.png')
    plt.show()

plot_weight_distributions(diagnostics['weights_history'])
```

### Evaluation Metrics

```python
from earcp.utils.metrics import (
    compute_regret,
    compute_diversity,
    evaluate_ensemble,
    theoretical_regret_bound,
    compute_stability,
    compute_adaptability
)

# 1. Calculate regret
regret_metrics = compute_regret(
    expert_cumulative_losses=diagnostics['cumulative_loss'],
    ensemble_cumulative_loss=ensemble_cumulative_loss
)

print("=== REGRET ANALYSIS ===")
print(f"Absolute regret: {regret_metrics['regret']:.4f}")
print(f"Relative regret: {regret_metrics['relative_regret']:.2%}")
print(f"Best expert: {regret_metrics['best_expert']}")
print(f"Worst expert: {regret_metrics['worst_expert']}")

# 2. Calculate diversity
diversity_metrics = compute_diversity(diagnostics['weights_history'])

print("\n=== DIVERSITY ANALYSIS ===")
print(f"Mean entropy: {diversity_metrics['mean_entropy']:.4f}")
print(f"Max theoretical entropy: {diversity_metrics['max_entropy']:.4f}")
print(f"Utilization ratio: {diversity_metrics['utilization_ratio']:.2%}")
print(f"Mean effective experts: {diversity_metrics['mean_effective_experts']:.2f}")

# 3. Theoretical bound
T = len(predictions)
M = len(experts)
theoretical_bound = theoretical_regret_bound(T=T, M=M, beta=0.7)

print("\n=== THEORETICAL GUARANTEES ===")
print(f"Regret bound O(√T log M): {theoretical_bound:.4f}")
print(f"Observed regret: {regret_metrics['regret']:.4f}")
print(f"Ratio (observed/theoretical): {regret_metrics['regret']/theoretical_bound:.2%}")

# 4. Ensemble evaluation
eval_metrics = evaluate_ensemble(
    predictions=predictions,
    targets=targets,
    task_type='regression'
)

print("\n=== PREDICTIVE PERFORMANCE ===")
print(f"RMSE: {eval_metrics['rmse']:.4f}")
print(f"MAE: {eval_metrics['mae']:.4f}")
print(f"R²: {eval_metrics['r2']:.4f}")
print(f"Correlation: {eval_metrics['correlation']:.4f}")

# 5. Stability and adaptability
stability = compute_stability(diagnostics['weights_history'])
adaptability = compute_adaptability(diagnostics['weights_history'])

print("\n=== DYNAMIC BEHAVIOR ===")
print(f"Stability (mean variation): {stability:.4f}")
print(f"Adaptability (change capacity): {adaptability:.4f}")
```

### Export Results

```python
import json
import pandas as pd

# Create complete report
def generate_report(ensemble, diagnostics, predictions, targets):
    """Generate complete JSON report."""
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
    
    # Save
    with open('reports/earcp_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

# Create DataFrame for analysis
def create_results_dataframe(diagnostics, predictions, targets):
    """Create DataFrame with all results."""
    df = pd.DataFrame({
        'step': range(len(predictions)),
        'prediction': predictions,
        'target': targets,
        'error': predictions - targets,
        'abs_error': np.abs(predictions - targets)
    })
    
    # Add weights
    weights_array = np.array(diagnostics['weights_history'])
    for i in range(weights_array.shape[1]):
        df[f'weight_expert_{i+1}'] = weights_array[:, i]
    
    # Save
    df.to_csv('results/earcp_results.csv', index=False)
    
    return df

# Generate reports
report = generate_report(ensemble, diagnostics, predictions, targets)
df_results = create_results_dataframe(diagnostics, predictions, targets)

print("Reports generated:")
print("  - reports/earcp_report.json")
print("  - results/earcp_results.csv")
```

---

## 💼 Use Cases

### 1. Time Series Forecasting

```python
import numpy as np
from earcp import EARCP

# Different types of time series experts
class MovingAverageExpert:
    """Moving average based expert."""
    def __init__(self, window_size):
        self.window = window_size
        self.history = []
    
    def predict(self, x):
        self.history.append(x)
        if len(self.history) > self.window:
            self.history.pop(0)
        return np.mean(self.history, axis=0)

class ExponentialSmoothingExpert:
    """Exponential smoothing based expert."""
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
    """Trend extrapolation based expert."""
    def __init__(self):
        self.history = []
    
    def predict(self, x):
        self.history.append(x)
        if len(self.history) < 3:
            return x
        
        # Linear regression on recent values
        recent = np.array(self.history[-5:])
        t = np.arange(len(recent))
        coeffs = np.polyfit(t, recent, deg=1)
        # Extrapolate
        next_t = len(recent)
        return coeffs[0] * next_t + coeffs[1]

# Create ensemble
experts = [
    MovingAverageExpert(window_size=5),
    MovingAverageExpert(window_size=10),
    MovingAverageExpert(window_size=20),
    ExponentialSmoothingExpert(alpha=0.2),
    ExponentialSmoothingExpert(alpha=0.5),
    TrendExpert()
]

ensemble = EARCP(experts=experts, beta=0.7)

# Time series simulation
T = 1000
for t in range(T):
    # Current value (with seasonality and noise)
    x = np.sin(t * 0.1) + 0.01 * t + np.random.normal(0, 0.1)
    
    # Prediction
    pred, expert_preds = ensemble.predict(np.array([x]))
    
    # True future value (t+1)
    target = np.sin((t+1) * 0.1) + 0.01 * (t+1) + np.random.normal(0, 0.1)
    
    # Update
    ensemble.update(expert_preds, np.array([target]))
```

### 2. Trading & Finance

```python
import numpy as np
from earcp import EARCP

# Diversified trading strategies
class MomentumStrategy:
    """Follows market trends."""
    def __init__(self, lookback=20):
        self.lookback = lookback
        self.price_history = []
    
    def predict(self, market_state):
        # market_state contains: price, volume, indicators
        price = market_state['price']
        self.price_history.append(price)
        
        if len(self.price_history) < self.lookback:
            return 0  # Neutral position
        
        recent_prices = self.price_history[-self.lookback:]
        momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        # Signal: 1 (buy), 0 (neutral), -1 (sell)
        if momentum > 0.02:
            return 1
        elif momentum < -0.02:
            return -1
        return 0

class MeanReversionStrategy:
    """Bets on return to mean."""
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
        
        # Signal inverse to z-score
        if z_score > 2:
            return -1  # Overbought -> sell
        elif z_score < -2:
            return 1   # Oversold -> buy
        return 0

class VolatilityStrategy:
    """Adapts position to volatility."""
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
        
        # Lower volatility = can be more aggressive
        if volatility < 0.01:
            return np.sign(mean_return)  # Directional signal
        return 0  # Stay neutral in high volatility

# Create strategy ensemble
strategies = [
    MomentumStrategy(lookback=10),
    MomentumStrategy(lookback=30),
    MeanReversionStrategy(lookback=20),
    MeanReversionStrategy(lookback=50),
    VolatilityStrategy()
]

ensemble = EARCP(experts=strategies, beta=0.75)

# Trading simulation
portfolio_value = 100000
position = 0
price_history = []

for day in range(500):
    # Simulate market
    price = 100 + 10 * np.sin(day * 0.1) + np.random.normal(0, 1)
    price_history.append(price)
    
    # Market state
    market_state = {
        'price': price,
        'return': (price - price_history[-2]) / price_history[-2] if len(price_history) > 1 else 0
    }
    
    # Get ensemble signal
    signal, strategy_signals = ensemble.predict(market_state)
    
    # Aggregate signal (-1, 0, 1)
    aggregated_signal = np.mean(signal)
    
    # Execute trade
    target_position = np.sign(aggregated_signal)
    if target_position != position:
        # Calculate P&L
        if position != 0:
            pnl = position * (price - entry_price)
            portfolio_value += pnl
        
        # New position
        position = target_position
        entry_price = price if position != 0 else 0
    
    # Calculate realized return for update
    daily_return = (price - price_history[-2]) / price_history[-2] if len(price_history) > 1 else 0
    realized_return = position * daily_return
    
    # Update ensemble (negative loss = gain)
    ensemble.update(strategy_signals, np.array([-realized_return]))  # Negative because we minimize loss
    
    if (day + 1) % 100 == 0:
        print(f"Day {day+1}: Portfolio = ${portfolio_value:.2f}, Position = {position}")

print(f"\nFinal portfolio value: ${portfolio_value:.2f}")
print(f"Return: {((portfolio_value - 100000) / 100000 * 100):.2f}%")
```

### 3. Multi-Class Classification

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from earcp import EARCP
from earcp.utils.wrappers import SklearnWrapper

# Generate classification data
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

# Create different classifiers
classifiers = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'SVM_linear': SVC(kernel='linear', probability=True),
    'SVM_rbf': SVC(kernel='rbf', probability=True),
    'RandomForest': RandomForestClassifier(n_estimators=100),
    'NaiveBayes': GaussianNB()
}

# Train all classifiers
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    print(f"{name} - Train accuracy: {train_acc:.3f}")

# Create wrappers
experts = [SklearnWrapper(clf, name=name) 
           for name, clf in classifiers.items()]

# Create ensemble for classification
ensemble = EARCP(
    experts=experts,
    beta=0.7,
    prediction_mode='classification'
)

# Online evaluation on test set
correct = 0
predictions = []
true_labels = []

for i, (x, y_true) in enumerate(zip(X_test, y_test)):
    # Prediction (returns probabilities)
    prob, expert_probs = ensemble.predict(x.reshape(1, -1))
    
    # Predicted class
    y_pred = np.argmax(prob)
    predictions.append(y_pred)
    true_labels.append(y_true)
    
    # Check if correct
    if y_pred == y_true:
        correct += 1
    
    # One-hot encoding of true class
    target_onehot = np.zeros(3)
    target_onehot[y_true] = 1
    
    # Update ensemble
    ensemble.update(expert_probs, target_onehot.reshape(1, -1))
    
    if (i + 1) % 50 == 0:
        current_acc = correct / (i + 1)
        print(f"Sample {i+1}: Accuracy = {current_acc:.3f}")

# Final results
final_accuracy = correct / len(X_test)
print(f"\n{'='*50}")
print(f"Final ensemble accuracy: {final_accuracy:.3f}")
print(f"{'='*50}")

# Final expert weights
final_weights = ensemble.get_weights()
for name, weight in zip(classifiers.keys(), final_weights):
    print(f"{name}: {weight:.3f}")

# Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(true_labels, predictions)
print("\nConfusion matrix:")
print(cm)
print("\nClassification report:")
print(classification_report(true_labels, predictions))
```

### 4. Reinforcement Learning

```python
import numpy as np
from earcp import EARCP

# Simple environment: Multi-Armed Bandit
class SimpleBanditEnv:
    """K-armed bandit environment."""
    def __init__(self, k=10):
        self.k = k
        # True arm values (unknown to agent)
        self.true_values = np.random.randn(k)
    
    def step(self, action):
        """Pull an arm and get reward."""
        reward = self.true_values[action] + np.random.randn() * 0.1
        return reward

# RL agents with different strategies
class EpsilonGreedyAgent:
    """Epsilon-greedy agent."""
    def __init__(self, n_actions, epsilon=0.1):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.q_values = np.zeros(n_actions)
        self.action_counts = np.zeros(n_actions)
    
    def predict(self, state):
        """Returns Q-values."""
        return self.q_values.copy()
    
    def select_action(self):
        """Selects an action."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_values)
    
    def update_internal(self, action, reward):
        """Internal agent update."""
        self.action_counts[action] += 1
        alpha = 1.0 / self.action_counts[action]
        self.q_values[action] += alpha * (reward - self.q_values[action])

class UCBAgent:
    """Upper Confidence Bound agent."""
    def __init__(self, n_actions, c=2.0):
        self.n_actions = n_actions
        self.c = c
        self.q_values = np.zeros(n_actions)
        self.action_counts = np.zeros(n_actions)
        self.t = 0
    
    def predict(self, state):
        """Returns Q-values with exploration bonus."""
        if self.t == 0:
            return self.q_values
        
        ucb_values = self.q_values + self.c * np.sqrt(
            np.log(self.t) / (self.action_counts + 1e-10)
        )
        return ucb_values
    
    def select_action(self):
        """Selects action with highest UCB."""
        return np.argmax(self.predict(None))
    
    def update_internal(self, action, reward):
        """Internal update."""
        self.t += 1
        self.action_counts[action] += 1
        alpha = 1.0 / self.action_counts[action]
        self.q_values[action] += alpha * (reward - self.q_values[action])

class ThompsonSamplingAgent:
    """Thompson Sampling agent (Bayesian)."""
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.alpha = np.ones(n_actions)  # Successes
        self.beta = np.ones(n_actions)   # Failures
    
    def predict(self, state):
        """Sample from Beta distributions."""
        samples = np.random.beta(self.alpha, self.beta)
        return samples
    
    def select_action(self):
        """Selects according to sample."""
        return np.argmax(self.predict(None))
    
    def update_internal(self, action, reward):
        """Bayesian update."""
        # Convert reward to [0, 1]
        normalized_reward = (reward + 3) / 6  # Assuming reward in [-3, 3]
        self.alpha[action] += normalized_reward
        self.beta[action] += (1 - normalized_reward)

# Create environment and agents
env = SimpleBanditEnv(k=10)
agents = [
    EpsilonGreedyAgent(n_actions=10, epsilon=0.1),
    EpsilonGreedyAgent(n_actions=10, epsilon=0.05),
    UCBAgent(n_actions=10, c=2.0),
    ThompsonSamplingAgent(n_actions=10)
]

# Create RL ensemble
ensemble = EARCP(experts=agents, beta=0.8)

# Learning loop
T = 1000
total_reward = 0
optimal_actions = 0
optimal_action = np.argmax(env.true_values)

for t in range(T):
    # Get Q-values from all agents
    state = None  # No state in bandits
    ensemble_q_values, agent_q_values = ensemble.predict(state)
    
    # Select action (argmax of ensemble)
    action = np.argmax(ensemble_q_values)
    
    # Execute action
    reward = env.step(action)
    total_reward += reward
    
    if action == optimal_action:
        optimal_actions += 1
    
    # Update ensemble (uses Q-values as "predictions")
    # Target: observed reward for chosen action
    target_q_values = ensemble_q_values.copy()
    target_q_values[action] = reward
    
    ensemble.update(agent_q_values, target_q_values)
    
    # Update individual agents too
    for agent in agents:
        agent.update_internal(action, reward)
    
    if (t + 1) % 200 == 0:
        avg_reward = total_reward / (t + 1)
        optimality = optimal_actions / (t + 1)
        print(f"Step {t+1}: Avg reward = {avg_reward:.3f}, " 
              f"Optimality = {optimality:.2%}")

print(f"\n{'='*50}")
print(f"Total reward: {total_reward:.2f}")
print(f"Optimal action percentage: {optimal_actions/T:.2%}")
print(f"{'='*50}")

# Final weights
final_weights = ensemble.get_weights()
agent_names = ['ε-greedy (0.1)', 'ε-greedy (0.05)', 'UCB', 'Thompson']
for name, weight in zip(agent_names, final_weights):
    print(f"{name}: {weight:.3f}")
```

---

## 📖 API Reference

### Main Class: EARCP

```python
class EARCP(experts, config=None, **kwargs)
```

Ensemble Auto-Régulé par Cohérence et Performance.

**Parameters:**

- `experts` (list): List of expert models. Each expert must implement a `.predict(x)` method.
  
- `config` (EARCPConfig, optional): Configuration object. If None, uses default values or those provided in `**kwargs`.
  
- `**kwargs`: Additional configuration parameters (alpha_P, alpha_C, beta, eta_s, w_min, etc.)

**Main Methods:**

#### `predict(x, return_expert_predictions=True)`

Make ensemble prediction for input `x`.

**Parameters:**
- `x` (array-like): Input for prediction
- `return_expert_predictions` (bool): If True, also returns individual predictions

**Returns:**
- `prediction` (np.ndarray): Weighted ensemble prediction
- `expert_predictions` (list, optional): List of individual predictions

**Example:**
```python
pred, expert_preds = ensemble.predict(x)
```

#### `update(expert_predictions, target)`

Update ensemble after observing target.

**Parameters:**
- `expert_predictions` (list): Expert predictions (returned by `predict`)
- `target` (array-like): Observed target value

**Returns:**
- `metrics` (dict): Dictionary containing:
  - `'weights'`: Updated weights
  - `'performance_scores'`: Performance scores
  - `'coherence_scores'`: Coherence scores
  - `'ensemble_loss'`: Ensemble loss

**Example:**
```python
metrics = ensemble.update(expert_preds, target)
```

#### `get_weights()`

Returns current expert weights.

**Returns:**
- `weights` (np.ndarray): Array of normalized weights

**Example:**
```python
weights = ensemble.get_weights()
```

#### `get_diagnostics()`

Returns complete ensemble diagnostics.

**Returns:**
- `diagnostics` (dict): Dictionary containing all metrics and histories

**Example:**
```python
diag = ensemble.get_diagnostics()
print(f"Entropy: {diag['entropy']:.3f}")
```

#### `reset(initial_weights=None)`

Reset ensemble to initial state.

**Parameters:**
- `initial_weights` (array-like, optional): Initial weights. If None, uses uniform weights.

**Example:**
```python
ensemble.reset()
```

#### `save_state(filepath)`

Save complete ensemble state.

**Parameters:**
- `filepath` (str): Save file path

**Example:**
```python
ensemble.save_state('checkpoint.pkl')
```

#### `load_state(filepath)`

Load ensemble state from file.

**Parameters:**
- `filepath` (str): File path to load

**Example:**
```python
ensemble.load_state('checkpoint.pkl')
```

### Configuration Class: EARCPConfig

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

Configuration for EARCP.

**Parameters:**

- `alpha_P` (float): Smoothing factor for performance [0, 1]
- `alpha_C` (float): Smoothing factor for coherence [0, 1]
- `beta` (float): Performance/coherence balance [0, 1]
- `eta_s` (float): Learning rate/sensitivity > 0
- `w_min` (float): Minimum weight [0, 1]
- `loss_fn` (callable, optional): Custom loss function
- `coherence_fn` (callable, optional): Custom coherence function
- `prediction_mode` (str): 'auto', 'regression', or 'classification'
- `epsilon` (float): Small constant for numerical stability
- `normalize_weights` (bool): Normalize weights to sum=1
- `track_diagnostics` (bool): Track complete history
- `random_state` (int, optional): Seed for reproducibility

### Utility Functions

#### `get_preset_config(preset_name)`

Get a predefined configuration.

**Parameters:**
- `preset_name` (str): Preset name ('default', 'performance_focused', etc.)

**Returns:**
- `config` (EARCPConfig): Preset configuration

**Available presets:**
- `'default'`: Standard configuration
- `'performance_focused'`: Focus on performance
- `'diversity_focused'`: Focus on diversity
- `'balanced'`: Optimal balance
- `'conservative'`: Prudent changes
- `'aggressive'`: Fast changes
- `'robust'`: Robust to noise
- `'high_performance'`: Maximum performance

---

## 🔧 Troubleshooting

### Issue: Weights converge to single expert

**Symptom:** One expert gets weight close to 1.0, others close to 0.

**Solutions:**
```python
# 1. Increase w_min
ensemble = EARCP(experts=experts, w_min=0.15)

# 2. Decrease beta (favor coherence)
ensemble = EARCP(experts=experts, beta=0.5)

# 3. Decrease eta_s (softer changes)
ensemble = EARCP(experts=experts, eta_s=3.0)

# 4. Use robust preset
config = get_preset_config('diversity_focused')
ensemble = EARCP(experts=experts, config=config)
```

### Issue: Weights oscillate heavily

**Symptom:** Weights change erratically.

**Solutions:**
```python
# 1. Increase alpha_P and alpha_C (more memory)
ensemble = EARCP(experts=experts, alpha_P=0.95, alpha_C=0.90)

# 2. Decrease eta_s (less sensitivity)
ensemble = EARCP(experts=experts, eta_s=2.0)

# 3. Use conservative preset
config = get_preset_config('conservative')
ensemble = EARCP(experts=experts, config=config)
```

### Issue: Performance worse than best expert

**Symptom:** EARCP performs worse than best individual expert.

**Analysis and solutions:**
```python
# 1. Increase beta (more focus on performance)
ensemble = EARCP(experts=experts, beta=0.9)

# 2. Check expert diversity
# Too similar experts reduce benefits
from earcp.utils.analysis import check_expert_diversity
diversity_score = check_expert_diversity(experts, test_data)
if diversity_score < 0.3:
    print("Experts are too similar!")

# 3. Increase adaptation period
# EARCP needs time to learn
# Check performance after at least T > 100 steps

# 4. Verify loss and coherence functions
# Ensure they return values in [0, 1]
```

### Issue: NaN or Inf in calculations

**Symptom:** Errors with NaN or infinite values.

**Solutions:**
```python
# 1. Adjust epsilon
ensemble = EARCP(experts=experts, epsilon=1e-8)

# 2. Normalize predictions
class NormalizedExpert:
    def __init__(self, base_expert):
        self.base_expert = base_expert
    
    def predict(self, x):
        pred = self.base_expert.predict(x)
        # Clip extreme values
        return np.clip(pred, -1e6, 1e6)

# 3. Check input data
assert not np.any(np.isnan(x))
assert not np.any(np.isinf(x))
```

### Issue: Out of memory with many experts

**Symptom:** Out of memory with M > 50 experts.

**Solutions:**
```python
# 1. Disable history tracking
config = EARCPConfig(track_diagnostics=False)
ensemble = EARCP(experts=experts, config=config)

# 2. Periodic save and memory release
if t % 1000 == 0:
    ensemble.save_state(f'checkpoint_{t}.pkl')
    # Create new ensemble from checkpoint
    ensemble = EARCP(experts=experts)
    ensemble.load_state(f'checkpoint_{t}.pkl')

# 3. Use expert subset
# Periodically select top K experts
if t % 500 == 0:
    weights = ensemble.get_weights()
    top_k_indices = np.argsort(weights)[-10:]  # Keep top 10
    experts = [experts[i] for i in top_k_indices]
    ensemble = EARCP(experts=experts)
```

### Issue: Slow with many experts

**Symptom:** Very slow computation with M > 20 experts.

**Solutions:**
```python
# 1. Use vectorized calculations
# Ensure experts return NumPy arrays

# 2. Disable certain features
config = EARCPConfig(
    track_diagnostics=False,  # No history
    normalize_weights=False   # Skip normalization if unnecessary
)

# 3. Use multiprocessing for predictions
from multiprocessing import Pool

def get_prediction(expert, x):
    return expert.predict(x)

with Pool(processes=4) as pool:
    expert_preds = pool.starmap(get_prediction, [(e, x) for e in experts])
```

---

## ❓ FAQ

### General Questions

**Q: What's the minimum/maximum number of experts?**

**A:** Minimum 2 experts. Successfully tested up to 50+ experts. Optimal performance with 3-10 diverse experts. Beyond 20, consider hierarchical approaches.

**Q: Does EARCP work with pre-trained models?**

**A:** Yes, perfectly! Use wrappers (`SklearnWrapper`, `TorchWrapper`, `KerasWrapper`) to integrate any pre-trained model without retraining.

**Q: Do I need to retrain my experts?**

**A:** No. EARCP never modifies experts. It only learns how to combine them optimally.

**Q: Does EARCP support GPU?**

**A:** EARCP itself runs on CPU (very lightweight). But your experts (e.g., neural networks) can use GPU normally.

### Hyperparameters

**Q: How to choose beta?**

**A:** 
- **β = 0.7-0.8**: Recommended for starting
- **β > 0.8**: If you trust your metrics
- **β < 0.7**: To favor robustness and diversity
- **β = 1.0**: Pure performance mode (equivalent to Hedge)

**Q: How to tune eta_s?**

**A:**
- **eta_s = 3-4**: Stable environment, slow changes
- **eta_s = 5-6**: Default, moderate environment
- **eta_s = 7-9**: Dynamic environment, fast adaptation

**Q: What does w_min do exactly?**

**A:** `w_min` ensures no expert is completely ignored, even if performing poorly. This allows recovery if environment changes. Typically: `w_min = 1.0 / n_experts`.

### Performance

**Q: What's the time complexity?**

**A:** O(M²) per step for M experts, mainly due to coherence matrix computation. Optimizations possible for M > 20.

**Q: Is EARCP better than the best expert?**

**A:** EARCP guarantees O(√(T log M)) regret, meaning it asymptotically converges to the best expert. In the long run (large T), EARCP is at least as good as the best expert.

**Q: How long before EARCP converges?**

**A:** Generally 50-200 steps to see significant adaptation. Convergence depends on:
- Expert diversity
- eta_s value
- Environment stability

### Implementation

**Q: Does EARCP support batch learning?**

**A:** EARCP is designed for sequential online learning. For batch, simply call `update()` for each sample in the batch.

**Q: Can I add/remove experts dynamically?**

**A:** Not directly supported currently. You must create a new ensemble. Recommended approach:
```python
# Create new ensemble with new experts
new_ensemble = EARCP(experts=new_expert_list)
# Optional: transfer weights if applicable
```

**Q: How to handle experts with different output formats?**

**A:** Create wrapper classes to standardize outputs:
```python
class OutputWrapper:
    def __init__(self, expert, transform_fn):
        self.expert = expert
        self.transform = transform_fn
    
    def predict(self, x):
        raw_output = self.expert.predict(x)
        return self.transform(raw_output)
```

### Use Cases

**Q: Does EARCP work for classification?**

**A:** Yes! Use `prediction_mode='classification'`. EARCP will combine class probabilities and you can take argmax for final prediction.

**Q: Can I use EARCP for clustering?**

**A:** Yes, if your experts produce cluster assignments or membership probabilities. EARCP can combine them.

**Q: Is EARCP suitable for NLP?**

**A:** Yes, if you have multiple language models (BERT, GPT, etc.) producing embeddings or probabilities, EARCP can combine them intelligently.

**Q: Can I use EARCP in production?**

**A:** Absolutely! EARCP is:
- Production-ready
- Extensively tested
- Low overhead
- Easy to monitor

### License and Support

**Q: Is EARCP free?**

**A:** 
- **Free**: Academic research, personal projects, companies < $100k revenue
- **Commercial**: License required for companies > $100k
- **Open-source**: Becomes Apache 2.0 on November 13, 2029

**Q: Where to get help?**

**A:**
- Documentation: [README](https://github.com/Volgat/earcp)
- GitHub Issues: https://github.com/Volgat/earcp/issues
- Email: info@amewebstudio.com

**Q: Can I contribute?**

**A:** Yes! Contributions are welcome. See [CONTRIBUTING.md](https://github.com/Volgat/earcp/blob/main/CONTRIBUTING.md).

---

## 📞 Support and Contact

### Get Help

**Documentation:**
- README: https://github.com/Volgat/earcp
- Examples: https://github.com/Volgat/earcp/tree/main/examples
- Tutorials: https://github.com/Volgat/earcp/wiki

**Bugs and Issues:**
- GitHub Issues: https://github.com/Volgat/earcp/issues
- Include: version, minimal reproducible code, error messages

**Questions:**
- GitHub Discussions: https://github.com/Volgat/earcp/discussions
- Stack Overflow: Tag `earcp`

### Direct Contact

**Author:** Mike Amega  
**Email:** info@amewebstudio.com  
**LinkedIn:** https://www.linkedin.com/in/mike-amega-486329184/  
**GitHub:** [@Volgat](https://github.com/Volgat)  
**Location:** Windsor, Ontario, Canada

**For:**
- 🏢 Commercial licenses
- 🤝 Research collaborations
- 💼 Consulting and technical support
- 🎓 Presentations and training

---

## 📄 Citation

If you use EARCP in your work, please cite:

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

## 📜 License

**Copyright © 2025 Mike Amega. All rights reserved.**

EARCP is distributed under the **Business Source License 1.1**. See [LICENSE.md](https://github.com/Volgat/earcp/blob/main/LICENSE.md) for complete terms.

**Summary:**
- ✅ Free for research, education, and internal use (<$100k)
- 💼 Commercial license required for companies >$100k
- 🔓 Becomes Apache 2.0 on November 13, 2029

---

**Last updated:** December 3, 2025  
**Document version:** 2.0  
**EARCP version:** 1.0.0
