# EARCP - Quick Start Guide

**From zero to your first EARCP ensemble in 5 minutes**

---

## 🚀 Installation

### From PyPI (Recommended)

```bash
pip install earcp
```

### With Optional Dependencies

```bash
# With PyTorch support
pip install earcp[torch]

# With scikit-learn support
pip install earcp[sklearn]

# Full installation
pip install earcp[full]
```

### Verify Installation

```bash
python -c "import earcp; print(f'EARCP version: {earcp.__version__}')"
```

---

## ⚡ Minimal Example (30 seconds)

```python
from earcp import EARCP
import numpy as np

# 1. Define simple experts
class Expert:
    def __init__(self, factor):
        self.factor = factor
    
    def predict(self, x):
        return self.factor * x

# 2. Create ensemble
experts = [Expert(1.0), Expert(2.0), Expert(0.5)]
ensemble = EARCP(experts=experts)

# 3. Use it
for t in range(50):
    x = np.array([t * 0.1])
    target = np.array([2.0 * t * 0.1])  # Target function

    pred, expert_preds = ensemble.predict(x)
    ensemble.update(expert_preds, target)

# 4. Results
print(f"Final weights: {ensemble.get_weights()}")
# Output: [0.05, 0.9, 0.05] - Expert 2 (factor=2.0) dominates!
```

**That's it! You just created your first adaptive ensemble.**

---

## 📊 Complete Example (5 minutes)

### Step 1: Prepare Your Data

```python
import numpy as np

# Synthetic data
np.random.seed(42)
T = 200

def generate_data(t):
    """Generate data with linear trend + sine wave."""
    x = t * 0.1
    y = 2*x + np.sin(x) + np.random.normal(0, 0.1)
    return x, y
```

### Step 2: Create Diverse Experts

```python
# Linear expert
class LinearExpert:
    def __init__(self, slope, intercept):
        self.slope = slope
        self.intercept = intercept

    def predict(self, x):
        return self.slope * x + self.intercept

# Sinusoidal expert
class SinExpert:
    def __init__(self, amplitude, frequency):
        self.amplitude = amplitude
        self.frequency = frequency

    def predict(self, x):
        return self.amplitude * np.sin(self.frequency * x)

# Create multiple experts with different strategies
experts = [
    LinearExpert(slope=2.0, intercept=0.0),   # Good for trend
    LinearExpert(slope=1.5, intercept=0.5),   # Slightly off
    SinExpert(amplitude=1.0, frequency=1.0),  # Good for oscillations
]
```

### Step 3: Configure and Train EARCP

```python
from earcp import EARCP

# Initialize with configuration
ensemble = EARCP(
    experts=experts,
    beta=0.7,      # Performance/coherence balance
    eta_s=5.0,     # Weight sensitivity
    w_min=0.05     # Minimum weight floor
)

# Online training
print("Training...")
predictions = []
targets = []

for t in range(T):
    x, target = generate_data(t)
    x = np.array([x])
    target = np.array([target])

    # Predict and update
    pred, expert_preds = ensemble.predict(x)
    ensemble.update(expert_preds, target)
    
    predictions.append(pred[0])
    targets.append(target[0])

    # Progress update
    if (t + 1) % 50 == 0:
        weights = ensemble.get_weights()
        print(f"Step {t+1}: Weights = {[f'{w:.3f}' for w in weights]}")
```

### Step 4: Analyze Results

```python
# Get diagnostics
diagnostics = ensemble.get_diagnostics()

print("\n=== FINAL RESULTS ===")
print(f"Final weights: {diagnostics['weights']}")
print(f"Cumulative losses: {diagnostics['cumulative_loss']}")

# Best expert
best_expert = np.argmin(diagnostics['cumulative_loss'])
print(f"Best expert: Expert {best_expert + 1}")

# Calculate RMSE
rmse = np.sqrt(np.mean((np.array(predictions) - np.array(targets))**2))
print(f"Overall RMSE: {rmse:.4f}")

# Visualize (optional)
from earcp.utils.visualization import plot_diagnostics
plot_diagnostics(diagnostics, save_path='results.png')
print("\nVisualization saved to 'results.png'")
```

**Expected Output:**
```
Training...
Step 50: Weights = ['0.333', '0.333', '0.333']
Step 100: Weights = ['0.400', '0.250', '0.350']
Step 150: Weights = ['0.450', '0.200', '0.350']
Step 200: Weights = ['0.500', '0.150', '0.350']

=== FINAL RESULTS ===
Final weights: [0.50, 0.15, 0.35]
Cumulative losses: [15.234, 28.567, 18.901]
Best expert: Expert 1
Overall RMSE: 0.3245
```

---

## 🤖 With scikit-learn (2 more minutes)

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from earcp import EARCP
from earcp.utils.wrappers import SklearnWrapper

# Create sklearn models
models = {
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.5),
    'DecisionTree': DecisionTreeRegressor(max_depth=5),
    'RandomForest': RandomForestRegressor(n_estimators=50),
    'SVR': SVR(kernel='rbf')
}

# Train on your data
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name} trained")

# Wrap for EARCP
experts = [SklearnWrapper(model, name=name) 
           for name, model in models.items()]

# Create ensemble
ensemble = EARCP(experts=experts, beta=0.7)

# Use in online mode
print("\nOnline evaluation...")
for i, (x, y) in enumerate(zip(X_test, y_test)):
    pred, expert_preds = ensemble.predict(x.reshape(1, -1))
    ensemble.update(expert_preds, y.reshape(-1, 1))
    
    if (i + 1) % 50 == 0:
        print(f"Processed {i+1} samples")

# Results
print("\n=== FINAL WEIGHTS ===")
final_weights = ensemble.get_weights()
for name, weight in zip(models.keys(), final_weights):
    print(f"{name}: {weight:.3f}")
```

**Example Output:**
```
Ridge trained
Lasso trained
DecisionTree trained
RandomForest trained
SVR trained

Online evaluation...
Processed 50 samples
Processed 100 samples
Processed 150 samples

=== FINAL WEIGHTS ===
Ridge: 0.234
Lasso: 0.189
DecisionTree: 0.156
RandomForest: 0.312
SVR: 0.109
```

---

## 🔧 Preset Configurations

Use presets to get started quickly:

```python
from earcp import EARCP, get_preset_config

# Performance-focused (beta=0.95)
config = get_preset_config('performance_focused')
ensemble = EARCP(experts=experts, config=config)

# Diversity-focused (beta=0.5)
config = get_preset_config('diversity_focused')
ensemble = EARCP(experts=experts, config=config)

# Balanced (recommended, beta=0.7)
config = get_preset_config('balanced')
ensemble = EARCP(experts=experts, config=config)

# Conservative (slow adaptation)
config = get_preset_config('conservative')
ensemble = EARCP(experts=experts, config=config)

# Aggressive (fast adaptation)
config = get_preset_config('aggressive')
ensemble = EARCP(experts=experts, config=config)
```

### When to Use Each Preset

| Preset | Use When | Example |
|--------|----------|---------|
| `default` | Starting out | General purpose |
| `performance_focused` | Trust your metrics | Stable environment |
| `diversity_focused` | Noisy data | High uncertainty |
| `balanced` | Most cases | Moderate dynamics |
| `conservative` | Slow changes needed | Production systems |
| `aggressive` | Fast adaptation needed | Concept drift detection |

---

## 🔥 With PyTorch (Bonus)

```python
import torch
import torch.nn as nn
from earcp import EARCP
from earcp.utils.wrappers import TorchWrapper

# Define PyTorch models
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x)

# Create models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
models = [SimpleNN(10).to(device) for _ in range(3)]

# Train your models here...
# (training code omitted for brevity)

# Wrap for EARCP
experts = [TorchWrapper(model, device=device) for model in models]

# Create ensemble
ensemble = EARCP(experts=experts)

# Use
with torch.no_grad():
    for x, y in test_loader:
        pred, expert_preds = ensemble.predict(x)
        ensemble.update(expert_preds, y)
```

---

## 📈 Next Steps

### 1. Explore Examples

Check out the `examples/` folder for more use cases:
- Time series forecasting
- Financial trading
- Classification
- Reinforcement learning

### 2. Read Full Documentation

- [USAGE.md](USAGE.md) - Complete usage guide
- [API Reference](docs/api_reference.md) - Detailed API docs
- [GitHub](https://github.com/Volgat/earcp) - Source code and issues

### 3. Customize for Your Use Case

Experiment with:
- Different expert types
- Custom loss functions
- Custom coherence functions
- Hyperparameter tuning

### 4. Visualize Behavior

Use visualization tools to understand how EARCP adapts:

```python
from earcp.utils.visualization import (
    plot_weights,
    plot_performance,
    plot_diagnostics
)

diagnostics = ensemble.get_diagnostics()

# Weight evolution
plot_weights(diagnostics['weights_history'])

# Performance tracking
plot_performance(diagnostics['performance_history'])

# Complete dashboard
plot_diagnostics(diagnostics)
```

---

## 🎯 Quick Reference

### Key Parameters

| Parameter | Effect | When to Adjust |
|-----------|--------|----------------|
| `beta` | Performance/coherence balance | Favor performance (↑) or diversity (↓) |
| `eta_s` | Weight sensitivity | Fast updates (↑) or smooth (↓) |
| `w_min` | Minimum weight | Prevent expert exclusion (↑) |
| `alpha_P` | Performance smoothing | Favor history (↑) or recent (↓) |
| `alpha_C` | Coherence smoothing | Stable coherence (↑) or reactive (↓) |

### Recommended Values

```python
# Conservative (stable, slow adaptation)
EARCP(experts, beta=0.6, eta_s=3.0, alpha_P=0.95, alpha_C=0.90)

# Balanced (recommended default)
EARCP(experts, beta=0.7, eta_s=5.0, alpha_P=0.90, alpha_C=0.85)

# Aggressive (fast adaptation)
EARCP(experts, beta=0.8, eta_s=7.0, alpha_P=0.85, alpha_C=0.80)
```

### Common Issues and Solutions

#### **Weights concentrated on one expert**
```python
# Solution 1: Decrease eta_s
ensemble = EARCP(experts=experts, eta_s=3.0)

# Solution 2: Decrease beta
ensemble = EARCP(experts=experts, beta=0.5)

# Solution 3: Increase w_min
ensemble = EARCP(experts=experts, w_min=0.15)
```

#### **Weights too uniform**
```python
# Solution 1: Increase eta_s
ensemble = EARCP(experts=experts, eta_s=7.0)

# Solution 2: Increase beta (favor performance)
ensemble = EARCP(experts=experts, beta=0.9)
```

#### **Unstable weights (oscillating)**
```python
# Solution 1: Increase alpha_P and alpha_C
ensemble = EARCP(experts=experts, alpha_P=0.95, alpha_C=0.90)

# Solution 2: Decrease eta_s
ensemble = EARCP(experts=experts, eta_s=2.0)

# Solution 3: Use conservative preset
config = get_preset_config('conservative')
ensemble = EARCP(experts=experts, config=config)
```

---

## 💡 Pro Tips

### Tip 1: Start Simple
```python
# Begin with default parameters
ensemble = EARCP(experts=experts)

# Observe behavior for 100+ steps
# Then tune if needed
```

### Tip 2: Monitor Diagnostics
```python
# Check diagnostics periodically
if t % 100 == 0:
    diag = ensemble.get_diagnostics()
    print(f"Entropy: {diag['entropy']:.3f}")
    print(f"Effective experts: {diag['effective_experts']:.1f}")
```

### Tip 3: Save Checkpoints
```python
# Save periodically in long runs
if t % 500 == 0:
    ensemble.save_state(f'checkpoint_{t}.pkl')
```

### Tip 4: Ensure Expert Diversity
```python
# Bad: All experts are too similar
experts = [
    LinearExpert(2.0, 0.0),
    LinearExpert(2.1, 0.1),  # Too close to first
    LinearExpert(1.9, -0.1), # Too close to first
]

# Good: Diverse strategies
experts = [
    LinearExpert(2.0, 0.0),      # Linear
    SinExpert(1.0, 1.0),         # Periodic
    ExponentialExpert(0.1),      # Exponential
]
```

### Tip 5: Use Appropriate Loss Functions
```python
# For regression (default MSE works well)
ensemble = EARCP(experts=experts)

# For classification
ensemble = EARCP(experts=experts, prediction_mode='classification')

# For custom tasks
def custom_loss(y_pred, y_true):
    # Your loss here
    return loss_value

config = EARCPConfig(loss_fn=custom_loss)
ensemble = EARCP(experts=experts, config=config)
```

---

## 📚 Learning Path

### Beginner (You are here! ✅)
- [x] Installation
- [x] Minimal example
- [x] Complete example
- [x] sklearn integration

### Intermediate
- [ ] Explore all use cases in `examples/`
- [ ] Try different configurations
- [ ] Understand beta parameter effects
- [ ] Visualize ensemble behavior

### Advanced
- [ ] Custom loss functions
- [ ] Custom coherence functions
- [ ] Multi-objective optimization
- [ ] Hierarchical ensembles (meta-EARCP)

---

## 🆘 Getting Help

### Quick Questions
- Check [FAQ](USAGE.md#faq) in USAGE.md
- Browse [examples/](examples/)

### Issues & Bugs
- GitHub Issues: https://github.com/Volgat/earcp/issues
- Include: version, minimal code, error message

### Discussions
- GitHub Discussions: https://github.com/Volgat/earcp/discussions
- Stack Overflow: Tag `earcp`

### Direct Support
- **Email**: info@amewebstudio.com
- **LinkedIn**: [Mike Amega](https://www.linkedin.com/in/mike-amega-486329184/)

---

## 🎉 You're Ready!

You now know how to:
- ✅ Install EARCP
- ✅ Create your first ensemble
- ✅ Integrate with sklearn/PyTorch
- ✅ Use preset configurations
- ✅ Analyze results

**Next:** Try EARCP on your own data! 🚀

---

## 📖 Additional Resources

- **Full Documentation**: [USAGE.md](USAGE.md)
- **API Reference**: [docs/api_reference.md](docs/api_reference.md)
- **Academic Paper**: [EARCP_paper.pdf](EARCP_paper.pdf)
- **GitHub Repository**: https://github.com/Volgat/earcp
- **PyPI Package**: https://pypi.org/project/earcp/

---

## 💬 Share Your Success

Built something cool with EARCP? We'd love to hear about it!

- Tweet with #EARCP
- Post in GitHub Discussions
- Email us at info@amewebstudio.com

---

## 📜 License

**Free for:**
- 🎓 Academic research and education
- 💻 Personal projects
- 🏢 Companies with revenue < $100,000/year

**Commercial license required for companies > $100k/year**

See [LICENSE.md](LICENSE.md) for details.

---

**Happy Ensembling! 🎯**

Copyright © 2025 Mike Amega  
https://github.com/Volgat/earcp
