# EARCP: Ensemble Auto-Régulé par Cohérence et Performance

[![License](https://img.shields.io/badge/License-BSL%201.1-blue.svg)](LICENSE.md)
[![Disclosure Date](https://img.shields.io/badge/Prior%20Art%20Date-Nov%2013%2C%202025-green.svg)](https://github.com/Volgat/earcp)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)]()

**A Self-Regulating Coherence-Aware Ensemble Architecture for Sequential Decision Making**

**Author:** Mike Amega  
**Affiliation:** Independent Researcher  
**Contact:** amewebstudio35@gmail.com  
**private:** mikeamega910@gmail.com
**LinkedIn:** https://www.linkedin.com/in/mike-amega-486329184/  
**Disclosure Date:** November 13, 2025

---

## 🎯 Overview

EARCP is a novel ensemble learning architecture that dynamically weights heterogeneous expert models based on both their individual performance and inter-model coherence. Unlike traditional ensemble methods with static or offline-learned combinations, EARCP continuously adapts through principled online learning with provable regret bounds.

**Key Innovation:** Dual-signal weighting mechanism combining exploitation (performance) and exploration (coherence) for robust sequential prediction.

### Why EARCP?

- ✅ **Adaptive:** Continuously adjusts to changing model reliability
- ✅ **Robust:** Maintains diversity through coherence-aware weighting
- ✅ **Theoretically Grounded:** Provable O(√(T log M)) regret bounds
- ✅ **Practical:** Stable implementation with multiple safeguards
- ✅ **General-Purpose:** Applicable to any sequential prediction task

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

---

## 📚 Documentation

This repository contains complete documentation for academic recognition and IP protection:

### Core Documents

1. **[Academic Paper](EARCP_paper.tex)** - Full peer-review ready paper with theoretical analysis
2. **[Technical Whitepaper](EARCP_Technical_Whitepaper.md)** - Complete implementation specification
3. **[Implementation Guide](docs/implementation_guide.md)** - Step-by-step integration guide
4. **[API Reference](docs/api_reference.md)** - Complete API documentation

### Research Artifacts

- **Proofs:** Mathematical derivations and regret bound proofs
- **Experiments:** Reproducible experimental protocols and results
- **Benchmarks:** Performance comparisons against baselines

---

## 💻 Quick Start

### Basic Usage

```python
from earcp import EARCP

# Create expert models (any models with .predict() method)
experts = [cnn_model, lstm_model, transformer_model, dqn_model]

# Initialize EARCP ensemble
ensemble = EARCP(
    experts=experts,
    alpha_P=0.9,    # Performance smoothing
    alpha_C=0.85,   # Coherence smoothing
    beta=0.7,       # Performance-coherence balance
    eta_s=5.0,      # Sensitivity
    w_min=0.05      # Weight floor
)

# Online learning loop
for t in range(T):
    # Get predictions
    prediction, expert_preds = ensemble.predict(state)
    
    # Execute action and observe target
    target = execute_and_observe(prediction)
    
    # Update weights
    metrics = ensemble.update(expert_preds, target)
    
    # Monitor (optional)
    diagnostics = ensemble.get_diagnostics()
    print(f"Weights: {diagnostics['weights']}")
```

---

## 🧮 Mathematical Foundation

### Core Algorithm

At each time step t, EARCP:

1. **Collects predictions** from M expert models: p₁,ₜ, ..., p_M,ₜ
2. **Computes performance scores:** P_i,t = αₚ·P_i,t-1 + (1-αₚ)·(-ℓ_i,t)
3. **Calculates coherence:** C_i,t = (1/(M-1))·Σⱼ≠ᵢ Agreement(i,j)
4. **Combines signals:** s_i,t = β·P_i,t + (1-β)·C_i,t
5. **Updates weights:** w_i,t ∝ exp(ηₛ·s_i,t) with floor constraints

### Theoretical Guarantee

**Theorem:** Under standard assumptions (bounded losses, convexity), EARCP achieves:

```
Regret_T ≤ √(2T log M)
```

for pure performance (β=1), and:

```
Regret_T ≤ (1/β)·√(2T log M)
```

with coherence incorporation (β<1).

**Proof:** See Section 4 of [academic paper](EARCP_paper.tex).

---

## 📊 Performance

### Benchmark Results

| Method | Electricity (RMSE) | HAR (Acc.) | Financial (Sharpe) |
|--------|-------------------|------------|-------------------|
| Best Single | 0.124 ± 0.008 | 91.2 ± 1.1 | 1.42 ± 0.18 |
| Equal Weight | 0.118 ± 0.006 | 92.8 ± 0.9 | 1.58 ± 0.15 |
| Stacking | 0.112 ± 0.007 | 93.1 ± 1.0 | 1.61 ± 0.14 |
| Offline MoE | 0.109 ± 0.006 | 93.5 ± 0.8 | 1.65 ± 0.16 |
| Hedge | 0.107 ± 0.005 | 93.9 ± 0.7 | 1.71 ± 0.12 |
| **EARCP** | **0.098 ± 0.004** | **94.8 ± 0.6** | **1.89 ± 0.11** |

**Key Findings:**
- 8.4% improvement over Hedge on RMSE
- 10.5% improvement over Hedge on Sharpe ratio
- Consistent gains across diverse tasks
- Superior robustness during distribution shifts

---

## 🔧 Architecture

### Expert Requirements

Any model implementing:

```python
class ExpertModel:
    def predict(self, x):
        """Return prediction for input x."""
        return prediction  # array-like
```

### Supported Configurations

- **Number of experts:** 2 to 100+ (tested up to M=50)
- **Prediction types:** Classification, regression, reinforcement learning
- **Update frequency:** Real-time to batch updates
- **Loss functions:** Any L: Y×Y → [0,1]

---

## 📜 License

EARCP is released under the **Business Source License 1.1**.

### ✅ Free Use

You can use EARCP **for free** if:
- 🎓 **Academic research and education**
- 💻 **Personal and open-source projects**
- 🏢 **Internal business use** where your organization's total revenue is less than **USD $100,000 per year**

### 💼 Commercial Use

Organizations with revenue exceeding **$100,000/year** or those wishing to:
- Embed EARCP in commercial products
- Offer EARCP as a hosted service (SaaS)
- Redistribute EARCP commercially

...must obtain a **commercial license**.

**📧 Contact for Commercial Licensing:**
- **Email:** info@amewebstudio.com
- **Subject:** "EARCP Commercial License Inquiry"

### 🔓 Future License

After **November 13, 2029** (four years from publication), EARCP will automatically be released under the **Apache 2.0 license**, making it freely available for all uses.

### 📄 Full License Terms

For complete license terms, see [LICENSE.md](LICENSE.md)

---

## 📖 Citation

### Academic Citation

If you use EARCP in academic work, please cite:

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

### Technical Citation

For technical implementations:

```bibtex
@techreport{amega2025earcp_tech,
  title={EARCP: Technical Whitepaper and Implementation Specification},
  author={Amega, Mike},
  institution={Independent Research},
  year={2025},
  url={https://github.com/Volgat/earcp},
  note={Business Source License 1.1}
}
```

---

## 🛡️ Intellectual Property

### Copyright Notice

Copyright © 2025 Mike Amega. All rights reserved.

This software and associated documentation are protected by copyright law. The architecture, algorithms, and implementation details are original works by Mike Amega.

### Defensive Publication

**Prior Art Established:** November 13, 2025

This repository constitutes a defensive publication establishing prior art for:
- Core EARCP algorithm and mathematical formulation
- Dual-signal weighting mechanism (performance + coherence)
- Specific implementation details and optimizations
- Extension mechanisms and variations

**Legal Effect:** This public disclosure prevents third-party patent claims on disclosed inventions while preserving the author's rights to commercialize and license this technology.

### Attribution Requirements

All uses must include:
```
This work uses EARCP (Ensemble Auto-Régulé par Cohérence et Performance)
developed by Mike Amega (2025). See: https://github.com/Volgat/earcp
```

---

## 🔬 Research & Development

### Development Status

- [x] Core algorithm implemented and tested
- [x] Theoretical guarantees proven
- [x] Comprehensive benchmarking completed
- [x] Production-grade code with safeguards
- [x] Business Source License 1.1 applied
- [ ] PyPI package publication
- [ ] Academic paper submission to conference
- [ ] Extended documentation and tutorials
- [ ] Community extensions and contributions

### Future Directions

Planned enhancements:
1. Learned coherence functions
2. Hierarchical EARCP for large-scale ensembles
3. Multi-objective optimization extensions
4. Integration with popular ML frameworks (scikit-learn, PyTorch, TensorFlow)
5. Distributed/parallel implementations

---

## 👥 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Contribution

- **Implementations:** Integration with specific ML frameworks
- **Experiments:** Testing on new domains and benchmarks
- **Theory:** Tightening regret bounds, new guarantees
- **Documentation:** Tutorials, examples, case studies
- **Optimizations:** Performance improvements, GPU acceleration

### Contributor Recognition

Contributors will be acknowledged in:
- README contributors section
- Academic papers citing this work
- Release notes and documentation

---

## 📧 Contact

**Mike Amega**  
Email: mikeamega@yahoo.fr  
Location: Ontario, Canada  
GitHub: [@Volgat](https://github.com/Volgat)

### For Commercial Licensing Inquiries
**Email:** amewebstudio35@gmail.com  
**Subject:** "EARCP Commercial License Inquiry"

### For Research Collaboration
Open to collaborations on:
- Theoretical extensions
- Large-scale applications
- Domain-specific adaptations
- Academic publications

---

## 📝 Version History

### Version 1.0.0 (November 13, 2025)
- Initial public release
- Complete implementation with theoretical guarantees
- Comprehensive documentation
- Benchmark results on three domains
- Defensive publication for IP protection
- Business Source License 1.1 applied

---

## 🙏 Acknowledgments

Thanks to the open-source machine learning community for tools and datasets that enabled this research.

**Core Dependencies:**
- NumPy (numerical computations)
- PyTorch (neural network experts)
- scikit-learn (baseline comparisons)

---

## 📜 Legal Disclaimer

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

For full legal terms, see [LICENSE.md](LICENSE.md) file.

---

## 🔐 IP Protection Checklist

This repository includes the following for complete IP protection:

- [x] Academic paper with full mathematical derivation
- [x] Technical whitepaper with implementation details
- [x] Complete working code with documentation
- [x] Timestamp through GitHub commit history
- [x] Copyright notices in all files
- [x] Business Source License 1.1 applied
- [x] Citation guidelines
- [ ] DOI from Zenodo/figshare (recommended)
- [ ] arXiv submission (recommended within 30 days)

---

**🌟 Star this repository if you find EARCP useful!**

**🔔 Watch for updates and new features**

**🍴 Fork to create your own variations**

---

*Last Updated: December 3, 2025*  
*Repository: https://github.com/Volgat/earcp*  
*Prior Art Date: November 13, 2025*  
*License: Business Source License 1.1*
