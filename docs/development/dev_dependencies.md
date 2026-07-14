# Developer Dependencies

> _"Development tools shape the quality and clarity of the codebase."_

The following packages are required only for development and testing. They are
not needed for running Word Forge in production environments.

The `dev` project extra is the single source of truth for development tools.
`requirements.txt` installs the package in editable mode with that extra, while
`requirements-all.txt` additionally installs every optional runtime feature.

```text
black
build
isort
mypy
pre-commit
pytest
pytest-cov
ruff
```

Each tool serves a distinct purpose:

- **black** – Enforces a consistent code style across the project.
- **build** – Produces isolated source and wheel distributions.
- **isort** – Automatically sorts imports to reduce merge conflicts.
- **mypy** – Provides optional static type checking.
- **pre-commit** – Runs repository checks before a commit is created.
- **pytest** and **pytest-cov** – Run tests and measure code coverage.
- **ruff** – Performs fast static linting.

Install the lightweight development environment:

```bash
python -m pip install -r requirements.txt
```

Install all optional integrations as well:

```bash
python -m pip install -r requirements-all.txt
```
