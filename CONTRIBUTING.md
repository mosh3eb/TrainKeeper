# 🤝 Contributing to TrainKeeper

First off, thank you for considering contributing to Project TrainKeeper! It's people like you that make this tool faster, safer, and more robust for everyone.

## 🛠️ Development Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/mosh3eb/TrainKeeper.git
   cd TrainKeeper
   ```

2. **Install dependencies**
   ```bash
   pip install -e .[dev,torch,all]
   ```

3. **Run the test suite**
   ```bash
   pytest
   tk system-check
   ```

## 📐 Pull Request Process

1.  **Scope**: Keep changes focused. One feature/fix per PR.
2.  **Tests**: Add unit tests for new features. Ensure `pytest` is green.
3.  **Docs**: Update docstrings and `README.md` if user-facing behavior changes.
4.  **Style**: We follow standard Python conventions (PEP8).

## 📝 Code Style Guidelines

-   **Explicit is better than implicit**: Avoid magic.
-   **Public API Stability**: Don't break `run_reproducible` or `distributed_training` interfaces without major version bumps.
-   **Type Hints**: Use them where helpful for user clarity.

## 🐛 Found a Bug?

-   **Ensure the bug was not already reported** by searching on GitHub under [Issues](https://github.com/mosh3eb/TrainKeeper/issues).
-   If you're unable to find an open issue addressing the problem, **open a new one**. Be sure to include a **title and clear description**, as well as a code sample or an executable test case demonstrating the expected behavior that is not occurring.

Thank you for building with us! 🚀
