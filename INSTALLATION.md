# EARCP Library - Installation and Usage Guide

[![PyPI version](https://img.shields.io/pypi/v/earcp.svg)](https://pypi.org/project/earcp/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-BSL%201.1-blue.svg)](LICENSE.md)

**EARCP: Self-Regulating Coherence and Performance-Aware Ensemble**

---

## 🚀 Quick Installation

### Option 1: Install from PyPI (Recommended)

```bash
pip install earcp
```

### Option 2: Install from GitHub

```bash
# Install from the earcp-lib branch
pip install git+https://github.com/Volgat/earcp.git@earcp-lib
```

### Option 3: Local Development Installation

```bash
# Clone the earcp-lib branch
git clone -b earcp-lib https://github.com/Volgat/earcp.git
cd earcp

# Install in editable mode
pip install -e .
```

### Option 4: Install with All Dependencies

```bash
# Full installation (includes sklearn, torch, visualization tools)
pip install earcp[full]

# Or from source
pip install -e ".[full]"
```

---

## ✅ Verify Installation

```bash
# Quick test
python -c "from earcp import EARCP; print('EARCP installed successfully!')"
```

Or run the test suite:

```bash
# If installed from source
python tests/test_basic.py

# Expected output:
# ✓ ALL TESTS PASSED!
```

---

## 🎯 Quick Start Example

Create a file `test_earcp.py`:

```python
from earcp import EARCP
import numpy as np

# Define simple expert models
class Expert:
    def __init__(self, factor):
        self.factor = factor
    
    def predict(self, x):
        return self.factor * x

# Create ensemble with 3 experts
experts = [Expert(1.0), Expert(2.0), Expert(1.5)]
ensemble = EARCP(experts=experts)

# Display initial weights
print("Initial weights:", ensemble.get_weights())

# Simulate online learning
for t in range(50):
    x = np.array([t * 0.1])
    target = np.array([1.5 * t * 0.1])  # Target favors Expert 3
    
    pred, expert_preds = ensemble.predict(x)
    ensemble.update(expert_preds, target)

# Display final weights
print("Final weights:", ensemble.get_weights())
print("Expert with factor 1.5 should have highest weight!")
```

Run it:
```bash
python test_earcp.py
```

---

## 📚 Complete Examples

### Example 1: Basic Usage with Analysis

```python
from earcp import EARCP
import numpy as np

# Your expert models (any model with .predict() method)
experts = [model1, model2, model3, model4]

# Initialize EARCP
ensemble = EARCP(
    experts=experts,
    alpha_P=0.9,    # Performance smoothing factor
    alpha_C=0.85,   # Coherence smoothing factor
    beta=0.7,       # Balance between performance and coherence
    eta_s=5.0,      # Sensitivity parameter
    w_min=0.05      # Minimum weight floor
)

# Online learning loop
for x, y in zip(X_test, y_test):
    # Get ensemble prediction
    prediction, expert_preds = ensemble.predict(x)
    
    # Update weights based on performance
    metrics = ensemble.update(expert_preds, y)
    
    # Monitor (optional)
    diagnostics = ensemble.get_diagnostics()
    print(f"Weights: {diagnostics['weights']}")
```

### Example 2: Integration with Scikit-learn

```python
from earcp import EARCP
from earcp.utils.wrappers import SklearnWrapper
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

# Train your scikit-learn models
models = [
    Ridge(),
    RandomForestRegressor(),
    SVR()
]

for model in models:
    model.fit(X_train, y_train)

# Wrap for EARCP compatibility
experts = [SklearnWrapper(m) for m in models]

# Create EARCP ensemble
ensemble = EARCP(experts=experts, beta=0.7)

# Online learning on test set
for x, y in zip(X_test, y_test):
    pred, expert_preds = ensemble.predict(x.reshape(1, -1))
    ensemble.update(expert_preds, y.reshape(-1, 1))

# Get final results
print("Final weights:", ensemble.get_weights())
diagnostics = ensemble.get_diagnostics()
```

### Example 3: PyTorch Integration

```python
from earcp import EARCP
from earcp.utils.wrappers import TorchWrapper
import torch.nn as nn

# Your PyTorch models
cnn_model = MyCNNModel()
lstm_model = MyLSTMModel()
transformer_model = MyTransformerModel()

# Wrap for EARCP
experts = [
    TorchWrapper(cnn_model),
    TorchWrapper(lstm_model),
    TorchWrapper(transformer_model)
]

# Use EARCP
ensemble = EARCP(experts=experts)

# Training loop
for batch in data_loader:
    x, y = batch
    pred, expert_preds = ensemble.predict(x)
    ensemble.update(expert_preds, y)
```

---

## 📂 Repository Structure

```
earcp/
├── earcp/              # Library source code
│   ├── __init__.py
│   ├── core.py         # Core EARCP algorithm
│   ├── experts.py      # Expert base classes
│   └── utils/          # Utilities and wrappers
├── examples/           # Usage examples
│   ├── basic_usage.py
│   ├── sklearn_integration.py
│   ├── trading_example.py
│   └── visualization_example.py
├── tests/              # Unit tests
├── docs/               # Complete documentation
├── setup.py            # Installation configuration
├── requirements.txt    # Dependencies
└── LICENSE.md          # License (BSL 1.1)
```

---

## 📦 Dependencies

### Required Dependencies

```
numpy >= 1.20.0
scipy >= 1.7.0
matplotlib >= 3.3.0
```

### Optional Dependencies

**For scikit-learn integration:**
```bash
pip install earcp[sklearn]
# or
pip install scikit-learn>=0.24.0
```

**For PyTorch integration:**
```bash
pip install earcp[torch]
# or
pip install torch>=1.9.0
```

**For complete installation:**
```bash
pip install earcp[full]
```

This includes:
- scikit-learn
- torch
- pandas
- seaborn
- Additional visualization tools

---

## 🔧 Advanced Configuration

### Custom Loss Functions

```python
from earcp import EARCP
import numpy as np

def custom_loss(pred, target):
    """Custom loss function (must return values in [0,1])"""
    return np.abs(pred - target) / (1 + np.abs(target))

ensemble = EARCP(
    experts=experts,
    loss_fn=custom_loss  # Use your custom loss
)
```

### Expert Monitoring

```python
# Get detailed diagnostics
diagnostics = ensemble.get_diagnostics()

print("Current weights:", diagnostics['weights'])
print("Performance scores:", diagnostics['performance_scores'])
print("Coherence scores:", diagnostics['coherence_scores'])
print("Combined scores:", diagnostics['combined_scores'])
```

### Save/Load Ensemble State

```python
import pickle

# Save ensemble state
with open('ensemble_state.pkl', 'wb') as f:
    pickle.dump(ensemble.get_state(), f)

# Load ensemble state
with open('ensemble_state.pkl', 'rb') as f:
    state = pickle.load(f)
    ensemble.set_state(state)
```

---

## 📖 Documentation

- **Quick Start Guide**: `docs/QUICKSTART.md`
- **Complete Usage Guide**: `docs/USAGE.md`
- **API Reference**: `docs/API.md`
- **Academic Paper**: Available on main branch
- **Examples**: See `examples/` directory

---

## 🐛 Troubleshooting

### ImportError: cannot import name 'EARCP'

```bash
# Make sure the package is installed
pip install earcp

# Or reinstall
pip install --upgrade --force-reinstall earcp
```

### ModuleNotFoundError: No module named 'numpy'

```bash
# Install required dependencies
pip install numpy scipy matplotlib
```

### Tests Fail

```bash
# Reinstall with all dependencies
pip install earcp[full]

# Run tests again
python tests/test_basic.py
```

### PyTorch/Sklearn Import Errors

```bash
# Install optional dependencies
pip install earcp[full]

# Or install specifically what you need
pip install torch scikit-learn
```

---

## 📜 License

**Business Source License 1.1**

### Free Use Permitted:
- ✅ Academic research and education
- ✅ Personal projects
- ✅ Internal business use (revenue < $100,000/year)

### Commercial License Required:
For organizations with revenue ≥ $100,000/year or for:
- Embedding in commercial products
- Offering as a hosted service (SaaS)
- Commercial redistribution

**Contact for Commercial Licensing:**
- Email: info@amewebstudio.com
- Subject: "EARCP Commercial License Inquiry"

### Future License:
After November 13, 2029, EARCP will be available under Apache 2.0 license.

**Full license terms:** [LICENSE.md](LICENSE.md)

---

## 🤝 Contributing

Contributions are welcome! See the main branch for contribution guidelines.

**Areas for Contribution:**
- New expert wrappers (TensorFlow, JAX, etc.)
- Performance optimizations
- Additional examples
- Documentation improvements
- Bug fixes

---

## 📧 Support

- **GitHub Issues**: https://github.com/Volgat/earcp/issues
- **Email**: info@amewebstudio.com
- **Documentation**: https://github.com/Volgat/earcp/tree/main/docs
- **Commercial Inquiries**: info@amewebstudio.com

---

## 🔗 Links

- **Main Repository**: https://github.com/Volgat/earcp
- **PyPI Package**: https://pypi.org/project/earcp/
- **Documentation Branch**: https://github.com/Volgat/earcp/tree/main
- **Library Branch**: https://github.com/Volgat/earcp/tree/earcp-lib

---

## 📊 Performance

EARCP has been benchmarked against state-of-the-art ensemble methods:

| Method | Improvement over Hedge |
|--------|----------------------|
| RMSE   | 8.4% better         |
| Sharpe Ratio | 10.5% better   |
| Accuracy | Consistent gains    |

See the academic paper on the main branch for detailed benchmarks.

---

## 🎓 Citation

If you use EARCP in your work, please cite:

```bibtex
@article{amega2025earcp,
  title={EARCP: Self-Regulating Coherence and Performance-Aware Ensemble},
  author={Amega, Mike},
  year={2025},
  url={https://github.com/Volgat/earcp},
  note={Business Source License 1.1}
}
```

---

## ✨ Quick Reference

### Installation
```bash
pip install earcp
```

### Basic Usage
```python
from earcp import EARCP

ensemble = EARCP(experts=[model1, model2, model3])
pred, expert_preds = ensemble.predict(x)
ensemble.update(expert_preds, target)
```

### Get Diagnostics
```python
diagnostics = ensemble.get_diagnostics()
weights = ensemble.get_weights()
```

---

**Ready to use EARCP in your projects!** 🚀

**Questions?** Open an issue or contact info@amewebstudio.com

---

*Copyright © 2025 Mike Amega. Licensed under Business Source License 1.1*
