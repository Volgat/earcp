# Contributing to EARCP

Thank you for your interest in contributing to EARCP (Self-Regulating Coherence and Performance-Aware Ensemble)! We welcome contributions from the community and appreciate your help in making EARCP better.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Contribution Workflow](#contribution-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Standards](#documentation-standards)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Pull Request Process](#pull-request-process)
- [License Considerations](#license-considerations)
- [Getting Help](#getting-help)

---

## 📜 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of:
- Experience level
- Gender identity and expression
- Sexual orientation
- Disability
- Personal appearance
- Body size
- Race
- Ethnicity
- Age
- Religion
- Nationality

### Our Standards

**Examples of behavior that contributes to a positive environment:**

✅ Using welcoming and inclusive language
✅ Being respectful of differing viewpoints and experiences
✅ Gracefully accepting constructive criticism
✅ Focusing on what is best for the community
✅ Showing empathy towards other community members

**Examples of unacceptable behavior:**

❌ Trolling, insulting/derogatory comments, and personal or political attacks
❌ Public or private harassment
❌ Publishing others' private information without explicit permission
❌ Other conduct which could reasonably be considered inappropriate in a professional setting

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project team at **info@amewebstudio.com**. All complaints will be reviewed and investigated promptly and fairly.

---

## 🤝 How Can I Contribute?

### Types of Contributions We Welcome

#### 1. 🐛 Bug Reports

Found a bug? Help us fix it!

**Before submitting a bug report:**
- Check the [existing issues](https://github.com/Volgat/earcp/issues) to avoid duplicates
- Verify the bug exists on the latest version
- Collect relevant information (version, OS, Python version, stack trace)

**What to include in your bug report:**
- **Clear title**: Describe the issue in one sentence
- **Steps to reproduce**: Numbered list of exact steps
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Environment details**: OS, Python version, EARCP version
- **Code sample**: Minimal reproducible example
- **Error messages**: Full stack trace if applicable

**Template:**
```markdown
## Bug Description
Brief description of the issue

## Steps to Reproduce
1. Step one
2. Step two
3. Step three

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., Ubuntu 22.04]
- Python Version: [e.g., 3.9.7]
- EARCP Version: [e.g., 1.0.0]
- Dependencies: [list relevant packages]

## Code Sample
```python
# Minimal code to reproduce the issue
```

## Error Message
```
Full stack trace here
```
```

---

#### 2. 💡 Feature Requests

Have an idea for improvement?

**Before submitting a feature request:**
- Check if the feature already exists
- Review [existing feature requests](https://github.com/Volgat/earcp/issues?q=is%3Aissue+is%3Aopen+label%3Aenhancement)
- Consider if it fits EARCP's core philosophy

**What to include:**
- **Use case**: Why is this feature needed?
- **Proposed solution**: How should it work?
- **Alternatives**: What other approaches did you consider?
- **Impact**: Who benefits from this feature?

---

#### 3. 📝 Documentation Improvements

Documentation is crucial! You can contribute by:
- Fixing typos or grammatical errors
- Clarifying confusing explanations
- Adding examples and tutorials
- Translating documentation
- Improving API documentation

---

#### 4. 🧪 Adding Examples

Help others learn EARCP by contributing:
- Domain-specific examples (new industries)
- Integration tutorials (new ML frameworks)
- Benchmark implementations
- Case studies

---

#### 5. 🔧 Code Contributions

Enhance EARCP by:
- Fixing bugs
- Implementing new features
- Optimizing performance
- Adding new expert wrappers
- Improving existing algorithms

---

#### 6. 🧪 Testing

Improve code quality by:
- Writing unit tests
- Adding integration tests
- Creating benchmark tests
- Testing edge cases
- Improving test coverage

---

## 🛠️ Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- pip or conda

### Fork and Clone

1. **Fork the repository** on GitHub:
   - Go to https://github.com/Volgat/earcp
   - Click "Fork" button

2. **Clone your fork:**
```bash
git clone https://github.com/YOUR_USERNAME/earcp.git
cd earcp
```

3. **Add upstream remote:**
```bash
git remote add upstream https://github.com/Volgat/earcp.git
```

### Create Development Environment

**Option 1: Using venv**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

**Option 2: Using conda**
```bash
conda create -n earcp python=3.9
conda activate earcp
pip install -e ".[dev]"
```

### Install Pre-commit Hooks (Recommended)

```bash
pip install pre-commit
pre-commit install
```

This ensures code quality checks run automatically before each commit.

### Verify Installation

```bash
# Run tests
pytest

# Check code style
flake8 earcp/

# Type checking
mypy earcp/
```

---

## 🔄 Contribution Workflow

### 1. Create a Feature Branch

**Always create a new branch for your work:**

```bash
# Update your fork
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

**Branch naming conventions:**
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `test/` - Test additions/improvements
- `refactor/` - Code refactoring
- `perf/` - Performance improvements

---

### 2. Make Your Changes

**Guidelines:**
- Keep changes focused and atomic
- Write clear, self-documenting code
- Add comments for complex logic
- Update documentation as needed
- Add tests for new functionality

---

### 3. Test Your Changes

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_core.py

# Run with coverage
pytest --cov=earcp --cov-report=html

# Check code style
flake8 earcp/

# Type checking
mypy earcp/
```

---

### 4. Commit Your Changes

See [Commit Message Guidelines](#commit-message-guidelines) below.

```bash
git add .
git commit -m "feat: add new coherence calculation method"
```

---

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then go to GitHub and create a Pull Request from your branch to `main`.

---

## 📐 Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

**Line Length:**
- Maximum 88 characters (Black default)
- Maximum 100 for docstrings

**Imports:**
```python
# Standard library
import os
import sys

# Third-party
import numpy as np
import torch

# Local
from earcp.core import EARCP
from earcp.utils import compute_coherence
```

**Naming Conventions:**
```python
# Classes: PascalCase
class ExpertModel:
    pass

# Functions/methods: snake_case
def compute_weights(experts):
    pass

# Constants: UPPER_SNAKE_CASE
MAX_EXPERTS = 100
DEFAULT_ALPHA = 0.9

# Private: leading underscore
def _internal_helper():
    pass
```

---

### Type Hints

**Always use type hints for public APIs:**

```python
from typing import List, Optional, Tuple
import numpy as np

def predict(
    self,
    x: np.ndarray,
    return_expert_preds: bool = False
) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
    """
    Make prediction using ensemble.
    
    Args:
        x: Input data of shape (batch_size, features)
        return_expert_preds: Whether to return individual expert predictions
        
    Returns:
        prediction: Ensemble prediction
        expert_predictions: Optional list of expert predictions
    """
    pass
```

---

### Docstring Format

We use **NumPy-style docstrings:**

```python
def update_weights(
    self,
    expert_predictions: List[np.ndarray],
    target: np.ndarray,
    loss_fn: Optional[Callable] = None
) -> Dict[str, float]:
    """
    Update expert weights based on performance.
    
    This method implements the core EARCP weight update mechanism,
    combining performance and coherence signals.
    
    Parameters
    ----------
    expert_predictions : List[np.ndarray]
        List of predictions from each expert, each of shape (batch_size, output_dim)
    target : np.ndarray
        Ground truth targets of shape (batch_size, output_dim)
    loss_fn : Optional[Callable], default=None
        Custom loss function. If None, uses squared error.
        
    Returns
    -------
    metrics : Dict[str, float]
        Dictionary containing:
        - 'ensemble_loss': Loss of ensemble prediction
        - 'avg_expert_loss': Average loss across experts
        - 'coherence': Average coherence score
        - 'weight_entropy': Entropy of weight distribution
        
    Examples
    --------
    >>> ensemble = EARCP(experts=[model1, model2])
    >>> predictions = [model1.predict(x), model2.predict(x)]
    >>> metrics = ensemble.update_weights(predictions, y_true)
    >>> print(metrics['ensemble_loss'])
    0.023
    
    Notes
    -----
    The weight update follows the exponential weights algorithm with
    coherence-aware modifications. See [1] for theoretical details.
    
    References
    ----------
    .. [1] Amega, M. (2025). "EARCP: Self-Regulating Coherence and 
           Performance-Aware Ensemble". ArXiv preprint.
    """
    pass
```

---

### Code Quality Tools

**We use:**
- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **isort**: Import sorting

**Configuration in `pyproject.toml`:**
```toml
[tool.black]
line-length = 88
target-version = ['py38', 'py39', 'py310', 'py311']

[tool.isort]
profile = "black"
line_length = 88

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

**Run all checks:**
```bash
# Format code
black earcp/ tests/

# Sort imports
isort earcp/ tests/

# Lint
flake8 earcp/ tests/

# Type check
mypy earcp/
```

---

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── test_core.py           # Core EARCP functionality
├── test_experts.py        # Expert base classes
├── test_wrappers.py       # Framework wrappers
├── test_utils.py          # Utility functions
├── test_integration.py    # Integration tests
└── test_benchmarks.py     # Benchmark tests
```

### Writing Tests

**Use pytest conventions:**

```python
import pytest
import numpy as np
from earcp import EARCP

class TestEARCPCore:
    """Test suite for core EARCP functionality."""
    
    @pytest.fixture
    def simple_experts(self):
        """Create simple test experts."""
        class DummyExpert:
            def __init__(self, bias):
                self.bias = bias
            
            def predict(self, x):
                return x + self.bias
        
        return [DummyExpert(0.0), DummyExpert(1.0), DummyExpert(2.0)]
    
    def test_initialization(self, simple_experts):
        """Test EARCP initialization."""
        ensemble = EARCP(experts=simple_experts)
        
        # Check initial weights are uniform
        weights = ensemble.get_weights()
        assert len(weights) == 3
        assert np.allclose(weights, 1/3)
    
    def test_prediction_shape(self, simple_experts):
        """Test prediction output shape."""
        ensemble = EARCP(experts=simple_experts)
        x = np.array([[1.0, 2.0]])
        
        pred, expert_preds = ensemble.predict(x)
        
        assert pred.shape == (1, 2)
        assert len(expert_preds) == 3
    
    @pytest.mark.parametrize("alpha_P,alpha_C", [
        (0.9, 0.85),
        (0.5, 0.5),
        (0.99, 0.99)
    ])
    def test_parameter_ranges(self, simple_experts, alpha_P, alpha_C):
        """Test EARCP with different parameter values."""
        ensemble = EARCP(
            experts=simple_experts,
            alpha_P=alpha_P,
            alpha_C=alpha_C
        )
        assert ensemble.alpha_P == alpha_P
        assert ensemble.alpha_C == alpha_C
```

### Test Coverage Requirements

- **Minimum coverage**: 80% for new code
- **Core modules**: 90%+ coverage required
- **Critical paths**: 100% coverage

**Check coverage:**
```bash
pytest --cov=earcp --cov-report=html
# Open htmlcov/index.html in browser
```

---

### Integration Tests

**Test real-world scenarios:**

```python
def test_sklearn_integration():
    """Test EARCP with scikit-learn models."""
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from earcp.utils.wrappers import SklearnWrapper
    
    # Train models
    X_train, y_train = load_data()
    models = [Ridge(), RandomForestRegressor()]
    for model in models:
        model.fit(X_train, y_train)
    
    # Create EARCP ensemble
    experts = [SklearnWrapper(m) for m in models]
    ensemble = EARCP(experts=experts)
    
    # Test online learning
    X_test, y_test = load_test_data()
    for x, y in zip(X_test, y_test):
        pred, _ = ensemble.predict(x.reshape(1, -1))
        ensemble.update(_, y.reshape(-1, 1))
    
    # Verify weights adapted
    weights = ensemble.get_weights()
    assert not np.allclose(weights, 0.5)  # Should have diverged from uniform
```

---

## 📚 Documentation Standards

### Adding Documentation

**When to add documentation:**
- New features or APIs
- Complex algorithms
- Configuration options
- Usage examples

**Where to add documentation:**
- **Docstrings**: In-code documentation
- **README**: High-level overview
- **docs/**: Detailed guides and tutorials
- **examples/**: Working code examples

---

### Documentation Structure

```
docs/
├── QUICKSTART.md          # 5-minute getting started
├── USAGE.md               # Complete usage guide
├── API.md                 # API reference
├── THEORY.md              # Mathematical foundations
├── BENCHMARKS.md          # Performance benchmarks
├── FAQ.md                 # Frequently asked questions
└── CONTRIBUTING.md        # This file
```

---

### Writing Style

**Guidelines:**
- Use clear, concise language
- Provide concrete examples
- Include code snippets
- Add diagrams where helpful
- Link to related documentation

**Example:**
```markdown
## Using EARCP with PyTorch Models

EARCP seamlessly integrates with PyTorch models through the `TorchWrapper`:

```python
import torch.nn as nn
from earcp import EARCP
from earcp.utils.wrappers import TorchWrapper

# Define your PyTorch models
cnn = MyCNNModel()
lstm = MyLSTMModel()

# Wrap for EARCP
experts = [TorchWrapper(cnn), TorchWrapper(lstm)]

# Create ensemble
ensemble = EARCP(experts=experts)
```

For more details, see [PyTorch Integration Guide](pytorch_integration.md).
```

---

## 📝 Commit Message Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, no logic change)
- **refactor**: Code refactoring
- **perf**: Performance improvements
- **test**: Adding or updating tests
- **chore**: Maintenance tasks (dependencies, build, etc.)
- **ci**: CI/CD changes

### Examples

**Simple commit:**
```
feat: add coherence threshold parameter
```

**Detailed commit:**
```
feat(core): implement adaptive learning rate for weight updates

Add support for adaptive learning rates that adjust based on
the stability of expert predictions. This improves convergence
in highly dynamic environments.

- Add `adaptive_lr` parameter to EARCP constructor
- Implement learning rate decay schedule
- Add tests for adaptive behavior

Closes #123
```

**Breaking change:**
```
feat!: change default beta parameter from 0.5 to 0.7

BREAKING CHANGE: The default beta parameter has changed from 0.5
to 0.7 based on benchmark results showing better performance.
Users relying on the old default should explicitly set beta=0.5.
```

**Bug fix:**
```
fix(wrappers): handle None predictions in SklearnWrapper

Previously, if a sklearn model returned None, the wrapper would
crash. Now it properly handles this edge case by returning zeros.

Fixes #456
```

---

## 🔀 Pull Request Process

### Before Submitting

**Checklist:**
- [ ] Code follows style guidelines
- [ ] All tests pass (`pytest`)
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Type hints added
- [ ] Docstrings written
- [ ] Commits follow convention
- [ ] No merge conflicts with main

---

### Pull Request Template

When you create a PR, please include:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update

## Related Issues
Closes #(issue number)

## Checklist
- [ ] My code follows the style guidelines
- [ ] I have performed a self-review
- [ ] I have commented my code where needed
- [ ] I have updated the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix/feature works
- [ ] New and existing tests pass locally
- [ ] Any dependent changes have been merged

## Screenshots (if applicable)
Add screenshots or GIFs

## Additional Context
Add any other context about the PR here
```

---

### Review Process

**What reviewers look for:**
1. ✅ Code correctness and quality
2. ✅ Test coverage
3. ✅ Documentation completeness
4. ✅ Performance implications
5. ✅ API design consistency
6. ✅ Backward compatibility

**Timeline:**
- Initial review: Within 3-5 business days
- Follow-up reviews: Within 2 business days
- Complex PRs may take longer

**Approval requirements:**
- At least 1 maintainer approval
- All CI checks passing
- No unresolved conversations

---

## ⚖️ License Considerations

### Business Source License 1.1

EARCP is licensed under the **Business Source License 1.1** (BSL).

**Important for contributors:**

✅ **You can contribute if:**
- Your contribution is your original work
- You agree to license your contribution under BSL 1.1
- You're contributing for free/open source use cases

⚠️ **Note:**
- Commercial use of EARCP requires a separate license
- After 2029-11-13, EARCP will become Apache 2.0
- See [LICENSE.md](LICENSE.md) for full details

### Contributor License Agreement (CLA)

By submitting a contribution, you agree that:

1. **You have the right** to grant the license for your contribution
2. **You grant** Mike Amega a perpetual, worldwide, non-exclusive, royalty-free license to use, modify, and distribute your contribution
3. **Your contribution** is provided "as-is" without warranties
4. **You understand** that EARCP uses BSL 1.1 and your contribution will be under the same license

**No formal CLA signature required** - submission of a PR constitutes agreement.

---

## 🆘 Getting Help

### Communication Channels

**For questions and discussions:**
- 💬 **GitHub Discussions**: https://github.com/Volgat/earcp/discussions
- 📧 **Email**: amewebstudio35@gmail.com
- 🐛 **Bug Reports**: https://github.com/Volgat/earcp/issues

**For urgent security issues:**
- 🔒 **Security Email**: amewebstudio35@gmail.com (mark subject as "SECURITY")

---

### FAQ

**Q: I'm new to contributing to open source. Where do I start?**

A: Great! Start with:
1. Read this guide thoroughly
2. Look for issues labeled `good-first-issue` or `help-wanted`
3. Join GitHub Discussions to introduce yourself
4. Start small (typo fixes, documentation improvements)

**Q: How long does it take for my PR to be reviewed?**

A: We aim to review PRs within 3-5 business days. Complex PRs may take longer.

**Q: Can I work on an issue that's already assigned?**

A: Please check with the assignee first. They may welcome collaboration or may prefer to complete it themselves.

**Q: My PR was rejected. What now?**

A: Don't be discouraged! Read the feedback, learn from it, and try again. Every contributor faces rejections.

**Q: Can I use EARCP commercially if I contribute?**

A: Contributions don't grant automatic commercial licenses. Commercial use still requires a separate license. Contact info@amewebstudio.com for details.

**Q: What if I disagree with a maintainer's decision?**

A: You can:
1. Provide additional context and rationale
2. Request a second opinion from another maintainer
3. Escalate to project lead (info@amewebstudio.com)

---

## 🎯 Priority Areas

We especially welcome contributions in:

### High Priority
- 🔥 **Performance optimizations** (GPU acceleration, parallel processing)
- 🔥 **New expert wrappers** (TensorFlow, JAX, scikit-learn extensions)
- 🔥 **Real-world examples** (more domains and applications)
- 🔥 **Documentation improvements** (tutorials, guides, translations)

### Medium Priority
- 📊 **Visualization tools** (weight evolution, performance tracking)
- 🧪 **Additional tests** (edge cases, stress tests)
- 📖 **API enhancements** (convenience methods, utilities)

### Future Directions
- 🌟 **Distributed EARCP** (multi-node coordination)
- 🌟 **Hierarchical ensembles** (ensembles of ensembles)
- 🌟 **Auto-ML integration** (automatic expert selection)
- 🌟 **Neurogenesis** (dynamic expert creation)

---

## 🙏 Recognition

### Hall of Contributors

All contributors are recognized in:
- **README.md** contributors section
- **Release notes**
- **Academic papers** (for significant contributions)

### Becoming a Maintainer

Outstanding contributors may be invited to become maintainers. Criteria:
- Consistent high-quality contributions
- Deep understanding of EARCP
- Active participation in community
- Adherence to project values

---

## 📄 Additional Resources

### Learning Materials
- [EARCP Academic Paper](EARCP_paper.tex)
- [Technical Whitepaper](EARCP_Technical_Whitepaper.md)
- [Mathematical Foundations](docs/THEORY.md)

### Development Tools
- [Black](https://black.readthedocs.io/)
- [Flake8](https://flake8.pycqa.org/)
- [MyPy](https://mypy.readthedocs.io/)
- [Pytest](https://docs.pytest.org/)

### Git Resources
- [Pro Git Book](https://git-scm.com/book)
- [GitHub Flow](https://guides.github.com/introduction/flow/)
- [Conventional Commits](https://www.conventionalcommits.org/)

---

## 📞 Contact

**Project Maintainer:** Mike Amega

- **Email:** info@amewebstudio.com
- **GitHub:** [@Volgat](https://github.com/Volgat)
- **LinkedIn:** [Mike Amega](https://www.linkedin.com/in/mike-amega-486329184/)

---

## 🎉 Thank You!

Thank you for taking the time to contribute to EARCP! Every contribution, no matter how small, helps make EARCP better for everyone.

**Happy coding!** 🚀

---

*Last Updated: December 3, 2025*
*Version: 1.0.0*
*EARCP © 2025 Mike Amega - Business Source License 1.1*
