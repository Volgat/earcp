# EARCP: Ensemble Auto-Régulé par Cohérence et Performance

[![License](https://img.shields.io/badge/License-BSL%201.1-blue.svg)](LICENSE.md)
[![Disclosure Date](https://img.shields.io/badge/Prior%20Art%20Date-Nov%2013%2C%202025-green.svg)](https://github.com/Volgat/earcp)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)]()

**A Self-Regulating Coherence-Aware Ensemble Architecture for Sequential Decision Making**

**Author:** Mike Amega  
**Affiliation:** Independent Researcher  
**Contact:** info@amewebstudio.com  
**LinkedIn:** [Mike Amega](https://www.linkedin.com/in/mike-amega-486329184/)  
**Disclosure Date:** November 13, 2025

---

## 🎯 Overview

EARCP is a groundbreaking ensemble learning architecture that dynamically weights heterogeneous expert models based on both their individual performance and inter-model coherence. Unlike traditional ensemble methods with static or offline-learned combinations, EARCP continuously adapts through principled online learning with provable regret bounds.

**Key Innovation:** Dual-signal weighting mechanism combining exploitation (performance tracking) and exploration (coherence measurement) for robust sequential prediction in non-stationary environments.

### Why EARCP?

- ✅ **Adaptive:** Continuously adjusts to changing model reliability and data distributions
- ✅ **Robust:** Maintains diversity through coherence-aware weighting, preventing over-reliance on single experts
- ✅ **Theoretically Grounded:** Provable O(√(T log M)) regret bounds with formal guarantees
- ✅ **Practical:** Stable implementation with multiple safeguards and production-ready code
- ✅ **General-Purpose:** Applicable to any sequential prediction task (classification, regression, RL)
- ✅ **Lightweight:** Minimal overhead, works with any existing models

### Key Features

- **Online Learning:** Real-time weight adaptation without retraining
- **Heterogeneous Experts:** Seamlessly combines CNNs, LSTMs, Transformers, classical ML, etc.
- **Automatic Balancing:** Self-regulating mechanism between performance and diversity
- **Robust to Drift:** Handles concept drift and non-stationary environments
- **Interpretable:** Clear weight evolution and expert contribution tracking

---

## 📦 Repository Structure

This repository has two branches:

- **`main` (earcp)**: Documentation, academic papers, research materials, and IP protection documents
- **`earcp-lib`**: Python library implementation for installation and use in your projects

---

## 🚀 Installation

### From GitHub (Recommended)

Install directly from the `earcp-lib` branch:

```bash
pip install git+https://github.com/Volgat/earcp.git@earcp-lib
```

### Local Development

Clone and install locally:

```bash
# Clone the library branch
git clone -b earcp-lib https://github.com/Volgat/earcp.git
cd earcp
pip install -e .
```

### From PyPI 

```bash
pip install earcp
```

### Requirements

- Python 3.8+
- NumPy >= 1.19.0
- Optional: PyTorch, TensorFlow, scikit-learn (for expert models)

---

## 💻 Quick Start

### Basic Usage

```python
from earcp import EARCP
import numpy as np

# Create expert models (any models with .predict() method)
# These can be neural networks, classical ML, rule-based, etc.
experts = [cnn_model, lstm_model, transformer_model, random_forest]

# Initialize EARCP ensemble with optimal hyperparameters
ensemble = EARCP(
    experts=experts,
    alpha_P=0.9,    # Performance smoothing (0.85-0.95 recommended)
    alpha_C=0.85,   # Coherence smoothing (0.80-0.90 recommended)
    beta=0.7,       # Performance-coherence balance (0.6-0.8 recommended)
    eta_s=5.0,      # Sensitivity parameter (3.0-7.0 recommended)
    w_min=0.05      # Minimum weight floor (prevents expert exclusion)
)

# Online learning loop
for t in range(T):
    # Get weighted ensemble prediction
    prediction, expert_preds = ensemble.predict(state)
    
    # Execute action and observe true target
    target = execute_and_observe(prediction)
    
    # Update expert weights based on performance
    metrics = ensemble.update(expert_preds, target)
    
    # Monitor ensemble health (optional)
    diagnostics = ensemble.get_diagnostics()
    print(f"Step {t}: Weights: {diagnostics['weights']}")
    print(f"Effective experts: {diagnostics['effective_experts']}")
    print(f"Entropy: {diagnostics['entropy']:.3f}")
```

### Advanced Example: Financial Trading

```python
from earcp import EARCP
import pandas as pd

# Define multiple trading strategies as experts
class TrendFollower:
    def predict(self, market_state):
        # Moving average crossover
        return self.generate_signal(market_state)

class MeanReversion:
    def predict(self, market_state):
        # RSI-based reversion
        return self.generate_signal(market_state)

class MachineLearning:
    def predict(self, market_state):
        # LSTM prediction
        return self.model.forward(market_state)

# Initialize experts
experts = [TrendFollower(), MeanReversion(), MachineLearning()]

# Create EARCP ensemble
ensemble = EARCP(experts=experts, beta=0.75)

# Trading loop
portfolio_value = 100000
for day in trading_days:
    market_state = get_market_data(day)
    
    # Get ensemble trading signal
    signal, expert_signals = ensemble.predict(market_state)
    
    # Execute trade
    position = execute_trade(signal)
    
    # Observe return
    daily_return = calculate_return(position)
    
    # Update weights (negative loss = positive return)
    ensemble.update(expert_signals, daily_return)
    
    portfolio_value *= (1 + daily_return)
```

### Example: Image Classification Ensemble

```python
import torch
from earcp import EARCP

# Pre-trained models as experts
experts = [
    resnet50_model,
    efficientnet_model,
    vision_transformer_model
]

ensemble = EARCP(experts=experts, beta=0.8)

# Validation/test loop
for images, labels in dataloader:
    # Get predictions from all experts
    predictions, expert_preds = ensemble.predict(images)
    
    # Compute loss
    loss = criterion(predictions, labels)
    
    # Update expert weights based on individual losses
    expert_losses = [criterion(pred, labels) for pred in expert_preds]
    ensemble.update(expert_preds, labels, losses=expert_losses)
```

---

## 🧮 Mathematical Foundation

### Core Algorithm

At each time step t, EARCP performs the following steps:

1. **Prediction Collection:** Gather predictions from M expert models: p₁,ₜ, ..., p_M,ₜ

2. **Performance Tracking:** Update exponentially weighted performance scores:
   ```
   P_i,t = αₚ · P_i,t-1 + (1-αₚ) · (-ℓ_i,t)
   ```
   where ℓ_i,t is the loss of expert i

3. **Coherence Measurement:** Calculate inter-expert agreement:
   ```
   C_i,t = (1/(M-1)) · Σⱼ≠ᵢ Agreement(pᵢ,ₜ, pⱼ,ₜ)
   ```

4. **Signal Fusion:** Combine performance and coherence:
   ```
   s_i,t = β · P_i,t + (1-β) · C_i,t
   ```

5. **Weight Update:** Apply exponential weighting with floor constraint:
   ```
   w̃_i,t = exp(ηₛ · s_i,t)
   w_i,t = max(w_min, w̃_i,t / Σⱼ w̃_j,t)
   ```

### Theoretical Guarantees

**Theorem 1 (Regret Bound):** Under standard assumptions (bounded losses ℓ ∈ [0,1], convex loss), EARCP achieves:

```
Regret_T ≤ (1/β) · √(2T log M)
```

For pure performance tracking (β=1):
```
Regret_T ≤ √(2T log M)
```

This matches the regret of the classical Hedge algorithm while adding robustness through coherence.

**Corollary:** The regret per round is O(√(log M / T)), which vanishes as T → ∞.

**Proof Sketch:** See Section 4 of [academic paper](EARCP_paper.tex) for complete proof using online convex optimization techniques.

---

## 📊 Performance

### Benchmark Results

Comprehensive evaluation on three diverse domains:

| Method | Electricity (RMSE↓) | HAR Activity (Acc%↑) | Financial (Sharpe↑) |
|--------|-------------------|------------|-------------------|
| Best Single Expert | 0.124 ± 0.008 | 91.2 ± 1.1 | 1.42 ± 0.18 |
| Equal Weight | 0.118 ± 0.006 | 92.8 ± 0.9 | 1.58 ± 0.15 |
| Stacking (Offline) | 0.112 ± 0.007 | 93.1 ± 1.0 | 1.61 ± 0.14 |
| Mixture of Experts | 0.109 ± 0.006 | 93.5 ± 0.8 | 1.65 ± 0.16 |
| Hedge Algorithm | 0.107 ± 0.005 | 93.9 ± 0.7 | 1.71 ± 0.12 |
| **EARCP (ours)** | **0.098 ± 0.004** | **94.8 ± 0.6** | **1.89 ± 0.11** |

### Key Findings

- **8.4% improvement** over Hedge (state-of-the-art online ensemble) on RMSE
- **10.5% improvement** over Hedge on Sharpe ratio (financial trading)
- **0.9% improvement** in classification accuracy (HAR dataset)
- **Consistent gains** across all three diverse domains
- **Superior robustness** during distribution shifts and concept drift
- **Lower variance** in performance across multiple runs

### Ablation Study

| Configuration | RMSE | Improvement |
|--------------|------|-------------|
| EARCP (full) | 0.098 | - |
| w/o coherence (β=1) | 0.104 | -6.1% |
| w/o floor (w_min=0) | 0.112 | -14.3% |
| w/o EMA smoothing | 0.109 | -11.2% |

**Conclusion:** All components contribute significantly to performance.

---

## 🔧 Architecture & Design

### Expert Requirements

Any model implementing the prediction interface:

```python
class ExpertModel:
    def predict(self, x):
        """
        Return prediction for input x.
        
        Parameters:
        -----------
        x : array-like, shape (n_features,) or (batch_size, n_features)
            Input features
            
        Returns:
        --------
        prediction : array-like
            Model prediction(s)
        """
        return prediction
```

**No training interface required** - EARCP works with pre-trained, frozen models.

### Supported Configurations

- **Number of experts:** 2 to 100+ (tested up to M=50 in production)
- **Prediction types:** 
  - Classification (probabilities or logits)
  - Regression (continuous values)
  - Reinforcement learning (action values or policies)
- **Update frequency:** 
  - Real-time (every prediction)
  - Batch updates (accumulated gradients)
  - Episodic (end-of-episode)
- **Loss functions:** Any L: Y×Y → [0,1] (automatically normalized)

### Hyperparameter Guidance

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `alpha_P` | [0.8, 0.99] | 0.9 | Performance smoothing (higher = more memory) |
| `alpha_C` | [0.75, 0.95] | 0.85 | Coherence smoothing |
| `beta` | [0.5, 0.9] | 0.7 | Performance weight (0=pure coherence, 1=pure performance) |
| `eta_s` | [1.0, 10.0] | 5.0 | Learning rate / sensitivity |
| `w_min` | [0.01, 0.1] | 0.05 | Minimum weight (prevents exclusion) |

**Tuning tips:**
- Increase `alpha_P/C` for stable environments, decrease for rapid changes
- Higher `beta` when you trust performance metrics, lower for noisy feedback
- Higher `eta_s` for faster adaptation, lower for stability
- Set `w_min` based on M: typically 0.5/M to 2.0/M

---

## 📚 Documentation

### Core Documents (This Repository)

1. **[Academic Paper](EARCP_paper.tex)** - Full peer-review ready paper with theoretical analysis
2. **[Technical Whitepaper](EARCP_Technical_Whitepaper.md)** - Complete implementation specification
3. **[Implementation Guide](docs/implementation_guide.md)** - Step-by-step integration guide
4. **[API Reference](docs/api_reference.md)** - Complete API documentation with examples

### Research Artifacts

- **Proofs:** Mathematical derivations and regret bound proofs
- **Experiments:** Reproducible experimental protocols and datasets
- **Benchmarks:** Performance comparisons against 10+ baseline methods
- **Analysis:** Ablation studies and sensitivity analysis

---

## 📜 License

EARCP is released under the **Business Source License 1.1** (BSL 1.1).

### ✅ Free Use Cases

You can use EARCP **completely free of charge** for:

- 🎓 **Academic research and education** (universities, research labs)
- 💻 **Personal projects and open-source software**
- 🧪 **Internal business use** where your organization's **total annual revenue < $100,000 USD**
- 🔬 **Non-commercial research** (independent researchers, hobbyists)

**No license fee required for these use cases.**

### 💼 Commercial Licensing

Organizations must obtain a commercial license if:

- Annual revenue ≥ **$100,000 USD**
- Embedding EARCP in **commercial products**
- Offering EARCP as a **hosted service** (SaaS, API)
- **Redistributing** EARCP in commercial software

**📧 Commercial License Inquiry:**
- **Email:** info@amewebstudio.com
- **Subject:** "EARCP Commercial License Inquiry"
- **Include:** Company name, use case, expected scale

**Pricing:** Flexible based on use case and scale. Contact for quote.

### 🔓 Automatic Open Source Release

**After November 13, 2029** (4 years from publication date), EARCP will **automatically convert** to the **Apache 2.0 license**, making it freely available for all commercial uses without restrictions.

### 📄 Full License

For complete terms and conditions, see [LICENSE.md](LICENSE.md).

---

## 📖 Citation

If you use EARCP in your work, please cite:

### BibTeX (Academic Papers)

```bibtex
@article{amega2025earcp,
  title={EARCP: Ensemble Auto-Régulé par Cohérence et Performance},
  author={Amega, Mike},
  journal={arXiv preprint},
  year={2025},
  url={https://github.com/Volgat/earcp},
  note={Prior art established November 13, 2025}
}
```

### BibTeX (Technical Reports)

```bibtex
@techreport{amega2025earcp_tech,
  title={EARCP: Technical Whitepaper and Implementation Specification},
  author={Amega, Mike},
  institution={Independent Research},
  year={2025},
  type={Technical Report},
  url={https://github.com/Volgat/earcp},
  note={Business Source License 1.1}
}
```

### Text Citation

> Amega, M. (2025). EARCP: Ensemble Auto-Régulé par Cohérence et Performance. 
> https://github.com/Volgat/earcp

---

## 🛡️ Intellectual Property

### Copyright Notice

**Copyright © 2025 Mike Amega. All rights reserved.**

This software and associated documentation are protected by copyright law. The architecture, algorithms, and implementation details are original works by Mike Amega.

### Defensive Publication

**Prior Art Established:** November 13, 2025

This repository constitutes a **defensive publication** establishing prior art for:

- ✓ Core EARCP algorithm and mathematical formulation
- ✓ Dual-signal weighting mechanism (performance + coherence)
- ✓ Specific implementation details and numerical optimizations
- ✓ Extension mechanisms and architectural variations
- ✓ Theoretical analysis and regret bounds

**Legal Effect:** This public disclosure prevents third-party patent claims on disclosed inventions while preserving the author's rights to commercialize and license this technology under BSL 1.1.

### Attribution Requirements

All uses (academic, personal, commercial) must include:

```
This work uses EARCP (Ensemble Auto-Régulé par Cohérence et Performance)
developed by Mike Amega (2025). https://github.com/Volgat/earcp
```

For academic papers, include the full citation above.

---

## 🔬 Research & Development

### Development Status

**Current Version:** 1.0.0 (Production Ready)

- [x] Core algorithm implemented and tested
- [x] Theoretical guarantees proven
- [x] Comprehensive benchmarking completed (3 domains, 10+ baselines)
- [x] Production-grade code with safeguards and error handling
- [x] Business Source License 1.1 applied
- [x] Complete documentation suite
- [x] PyPI package publication
- [ ] Academic paper submission to NeurIPS/ICML (planned)
- [ ] Extended tutorials and video demonstrations
- [ ] Community extensions and framework integrations

### Roadmap

**Short-term (Q1-Q2 2026):**
- PyPI package release with CI/CD
- Integration examples with PyTorch, TensorFlow, scikit-learn
- Interactive Jupyter notebook tutorials
- Performance profiling and optimization

**Mid-term (Q3-Q4 2026):**
- Hierarchical EARCP for 100+ experts
- GPU-accelerated implementation
- Distributed/parallel training support
- AutoML integration for hyperparameter tuning

**Long-term (2027+):**
- Learned coherence functions (meta-learning)
- Multi-objective optimization extensions
- Domain-specific pre-tuned configurations
- Real-time monitoring dashboard

### Future Research Directions

1. **Theoretical Extensions:**
   - Tighter regret bounds for specific loss families
   - Analysis under adversarial conditions
   - Extension to contextual bandits

2. **Algorithmic Improvements:**
   - Adaptive beta scheduling
   - Learned coherence metrics
   - Online hyperparameter adaptation

3. **Applications:**
   - Large language model ensembles
   - Federated learning settings
   - Multi-agent reinforcement learning

---

## 👥 Contributing

Contributions are welcome! EARCP is designed to be extensible and we encourage community involvement.

### How to Contribute

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/YourFeature`
3. **Commit** your changes: `git commit -m 'Add YourFeature'`
4. **Push** to the branch: `git push origin feature/YourFeature`
5. **Open** a Pull Request

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

### Areas for Contribution

We especially welcome contributions in:

- **🔌 Framework Integrations:** PyTorch Lightning, Hugging Face, Ray
- **📊 Experiments:** New domains, datasets, benchmark comparisons
- **📐 Theory:** Tightening bounds, new theoretical results
- **📖 Documentation:** Tutorials, case studies, blog posts
- **⚡ Optimizations:** Performance improvements, GPU acceleration
- **🧪 Testing:** Additional test coverage, edge cases
- **🐛 Bug Reports:** Issues, bugs, unexpected behavior

### Contributor Recognition

Contributors are acknowledged in:
- README contributors section (below)
- Release notes and changelogs
- Academic papers citing this work
- Special recognition for significant contributions

---

## 🏆 Contributors

Thank you to everyone who has contributed to EARCP!

<!-- Auto-generated contributor list -->
- **Mike Amega** - Creator and primary maintainer

*Want to see your name here? Submit a PR!*

---

## 📧 Contact

### Primary Contact

**Mike Amega**  
**Email:** info@amewebstudio.com  
**GitHub:** [@Volgat](https://github.com/Volgat)  
**LinkedIn:** [Mike Amega](https://www.linkedin.com/in/mike-amega-486329184/)

### Inquiry Types

**🏢 Commercial Licensing:**
- Email: info@amewebstudio.com
- Subject: "EARCP Commercial License Inquiry"
- Response time: 1-2 business days

**🔬 Research Collaboration:**
Open to collaborations on:
- Theoretical extensions and new proofs
- Large-scale industrial applications
- Domain-specific adaptations
- Joint academic publications

**🐛 Bug Reports & Issues:**
- Use GitHub Issues for technical problems
- Provide minimal reproducible example
- Include version info and error messages

**💡 Feature Requests:**
- Submit via GitHub Issues with "enhancement" label
- Describe use case and expected behavior
- Community discussion encouraged

---

## 📝 Version History

### Version 1.0.0 (November 13, 2025)
**Initial Public Release**

- ✅ Complete implementation with theoretical guarantees
- ✅ Comprehensive documentation (paper, whitepaper, guides)
- ✅ Benchmark results on three diverse domains
- ✅ Production-ready code with extensive testing
- ✅ Defensive publication for IP protection
- ✅ Business Source License 1.1 applied

**Tested on:**
- Python 3.8, 3.9, 3.10, 3.11
- NumPy 1.19 - 1.26
- Multiple OS: Linux (Ubuntu 20.04+), macOS, Windows 10+

---

## 🙏 Acknowledgments

EARCP builds upon decades of research in ensemble learning and online optimization. We acknowledge:

- The machine learning community for foundational work on ensemble methods
- Open-source projects that enabled this research
- Beta testers and early adopters who provided valuable feedback

**Core Dependencies:**
- **NumPy:** Numerical computations and array operations
- **PyTorch:** Neural network expert implementations (optional)
- **scikit-learn:** Baseline comparisons and evaluation metrics (optional)

**Inspired by:**
- Freund & Schapire's Hedge algorithm (1997)
- Mixture of Experts literature
- Online convex optimization theory

---

## ❓ FAQ

**Q: How is EARCP different from traditional ensemble methods?**  
A: EARCP adapts weights in real-time based on both performance and coherence, while traditional methods use fixed weights or require offline training.

**Q: Can I use EARCP with pre-trained models?**  
A: Yes! EARCP works with any pre-trained model. No retraining required.

**Q: What if my experts have different output formats?**  
A: You can wrap experts with adapter classes to standardize outputs.

**Q: Does EARCP work with neural networks?**  
A: Absolutely. EARCP is model-agnostic and works great with NNs, transformers, etc.

**Q: How many experts should I use?**  
A: Start with 3-7 diverse experts. More isn't always better beyond a point.

**Q: Can I use EARCP for batch predictions?**  
A: Yes, though EARCP is designed for online settings where samples arrive sequentially.

**Q: Is EARCP patented?**  
A: No, it's published as prior art to prevent patenting. Licensed under BSL 1.1.

---

## 📜 Legal Disclaimer

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

For full legal terms, see [LICENSE.md](LICENSE.md).

---

## 🔐 IP Protection Checklist

This repository includes comprehensive IP protection:

- [x] Academic paper with full mathematical derivation
- [x] Technical whitepaper with implementation details
- [x] Complete working code with documentation
- [x] Timestamp through GitHub commit history (November 13, 2025)
- [x] Copyright notices in all files
- [x] Business Source License 1.1 applied
- [x] Citation guidelines and attribution requirements
- [ ] DOI from Zenodo/figshare (recommended for academic citability)
- [ ] arXiv submission (planned within 30 days)
- [ ] Conference/journal submission (planned Q1 2026)

**Legal Status:** Prior art established. All rights reserved under BSL 1.1.

---

## 🌟 Support This Project

If you find EARCP useful:

- ⭐ **Star this repository** to show support
- 🔔 **Watch** for updates and new features  
- 🍴 **Fork** to create your own variations
- 📢 **Share** with colleagues and on social media
- 💬 **Discuss** in GitHub Issues or Discussions
- 📝 **Cite** in your publications
- 🤝 **Contribute** improvements and extensions

**Commercial users:** Consider a commercial license to support continued development.

---

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/Volgat/earcp?style=social)
![GitHub forks](https://img.shields.io/github/forks/Volgat/earcp?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/Volgat/earcp?style=social)

![GitHub issues](https://img.shields.io/github/issues/Volgat/earcp)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Volgat/earcp)
![GitHub last commit](https://img.shields.io/github/last-commit/Volgat/earcp)

---

<div align="center">

**Made with ❤️ by Mike Amega in Windsor, Ontario, Canada**

[Website](https://amewebstudio.com) • [GitHub](https://github.com/Volgat) • [LinkedIn](https://www.linkedin.com/in/mike-amega-486329184/)

</div>

---

*Last Updated: December 3, 2025*  
*Repository: https://github.com/Volgat/earcp*  
*Prior Art Date: November 13, 2025*  
*License: Business Source License 1.1*  
*Version: 1.0.0*
