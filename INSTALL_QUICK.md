# Quick Install - EARCP Python Library

## ⚠️ Important: Branch Name

The Python library is on the branch:
```
earcp-lib
```

## 🚀 Simple Installation (3 Steps)

### Step 1: Clone
```bash
git clone https://github.com/Volgat/earcp.git
cd earcp
```

### Step 2: Switch to the Library Branch
```bash
git checkout earcp-lib
```

### Step 3: Install
```bash
pip install -e .
```

## 📦 Alternative: Direct Installation from GitHub

```bash
pip install git+https://github.com/Volgat/earcp.git@earcp-lib
```

## 📦 Easiest: Install from PyPI

```bash
pip install earcp
```

## ✅ Verify Installation

```bash
python -c "from earcp import EARCP; print('✓ Installation successful!')"
```

## 🧪 Quick Test

```python
from earcp import EARCP
import numpy as np

class Expert:
    def __init__(self, factor):
        self.factor = factor
    
    def predict(self, x):
        return self.factor * x

# Create ensemble
experts = [Expert(1.0), Expert(2.0), Expert(1.5)]
ensemble = EARCP(experts=experts)

# Test
for t in range(20):
    x = np.array([t * 0.1])
    target = np.array([1.5 * t * 0.1])
    pred, expert_preds = ensemble.predict(x)
    ensemble.update(expert_preds, target)

print("Final weights:", ensemble.get_weights())
# Expert with factor=1.5 should have the highest weight
```

## 🔧 Troubleshooting

### Error: "does not appear to be a Python project"
**Cause**: You are on the wrong branch  
**Solution**:
```bash
git checkout earcp-lib
```

### Error: "No module named 'numpy'"
**Solution**:
```bash
pip install numpy scipy matplotlib
```

### Error: "No module named 'earcp'"
**Solution**:
```bash
# Make sure you installed the package
pip install earcp

# Or if installing from source
cd earcp
git checkout earcp-lib
pip install -e .
```

## 📚 Complete Documentation

- **Detailed Installation**: [INSTALLATION.md](INSTALLATION.md)
- **Library Guide**: [LIBRARY_README.md](LIBRARY_README.md)
- **Quick Start**: [docs/QUICKSTART.md](docs/QUICKSTART.md)
- **Complete Documentation**: [docs/USAGE.md](docs/USAGE.md)

## 💡 Examples

```bash
# Basic example
python examples/basic_usage.py

# Integration with scikit-learn
python examples/sklearn_integration.py

# Visualizations
python examples/visualization_example.py
```

## 🌐 Installation Options Comparison

| Method | Command | Best For |
|--------|---------|----------|
| **PyPI** | `pip install earcp` | Production use |
| **GitHub Direct** | `pip install git+https://...@earcp-lib` | Latest version |
| **Local Development** | `git clone` + `pip install -e .` | Contributing |

## 📦 Optional Dependencies

### Full Installation (includes all features)
```bash
pip install earcp[full]
```

### Scikit-learn Integration
```bash
pip install earcp[sklearn]
```

### PyTorch Integration
```bash
pip install earcp[torch]
```

### Development Tools
```bash
pip install earcp[dev]
```

## 📜 License

**Business Source License 1.1**

- ✅ **Free** for academic research, personal projects, and businesses with revenue < $100,000/year
- 💼 **Commercial license required** for businesses with revenue ≥ $100,000/year

**Contact for commercial licensing**: info@amewebstudio.com

Full license: [LICENSE.md](LICENSE.md)

---

**Version**: 1.0.0  
**Copyright**: © 2025 Mike Amega  
**License**: Business Source License 1.1

**Need help?** 
- Open an issue: https://github.com/Volgat/earcp/issues
- Email: info@amewebstudio.com
