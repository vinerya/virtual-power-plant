# Contributing to Virtual Power Plant Library

We love your input! We want to make contributing to the Virtual Power Plant library as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## We Develop with Github
We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

## Development Process
We use GitHub Flow, so all code changes happen through pull requests:

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code lints
6. Issue that pull request!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/username/virtual-power-plant.git
cd virtual-power-plant
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

## Code Style
- We use [Black](https://github.com/psf/black) for code formatting
- Type hints are required for all functions and classes
- Docstrings should follow Google style
- Maximum line length is 88 characters (Black default)

## Testing
- All new code should include tests
- Tests are written using pytest
- Run tests with: `pytest tests/`
- Aim for 90%+ test coverage

## Documentation
- Documentation is written in Markdown and reStructuredText
- API documentation is auto-generated from docstrings
- Examples should be included for new features
- Build docs with: `cd docs && make html`

## Pull Request Process
1. Update the README.md with details of changes if needed
2. Update the documentation with any new features
3. The PR will be merged once you have the sign-off of two maintainers

## Any contributions you make will be under the MIT Software License
In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using Github's [issue tracker](https://github.com/username/virtual-power-plant/issues)
We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/username/virtual-power-plant/issues/new).

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## License
By contributing, you agree that your contributions will be licensed under its MIT License.

## References
This document was adapted from the open-source contribution guidelines for [Facebook's Draft](https://github.com/facebook/draft-js/blob/a9316a723f9e918afde44dea68b5f9f39b7d9b00/CONTRIBUTING.md).
