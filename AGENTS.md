# Repository Guidelines

## Project Structure & Module Organization
- Python package lives in `src/vibelab` (new code goes here). Import as `from vibelab...`.
- Tests mirror `src/` in `tests/` (e.g., `src/vibelab/utils/` → `tests/utils/`).
- External code in `ext/` (editable installs), configs in `configs/`, utilities in `scripts/`, docs in `docs/`.
- Java module at `java/SuperHero` (Maven project).

## Build, Test, and Development Commands
- Setup: `pip install -r requirements.txt` for the broad repo dependency set.
- Editable install: `pip install -e .` once the Python 3.11 environment is ready.
- Lint: `ruff check .`  Format: `ruff format .`  (configured via `pyproject.toml`).
- Tests: `pytest` for full suite, or `pytest -k voc -q` for a subset.
- Coverage: `pytest --cov --cov-report=term-missing`.
- Example run (YOLO):
  `python src/vibelab/models/ext/yolov8/train_yolov8.py --config src/vibelab/models/ext/yolov8/configs/voc_finetune_config.yaml`.
- Java: `mvn -f java/SuperHero/pom.xml test` or `... package`.

## Coding Style & Naming Conventions
- Python 3.11. Use Ruff for lint/format: line length 100, double quotes, spaces for indent.
- Modules and functions: `snake_case`; classes: `PascalCase`; private helpers: `_leading_underscore`.
- Keep functions small; prefer reusable logic in `src/vibelab/...` over long notebook cells.
- Keep experiment data and generated media outside the repo tree; notebooks should point at `/data/...` or env-configured paths rather than checking in artifacts.

## Testing Guidelines
- Framework: `pytest` (see `pyproject.toml` for patterns: `test_*.py`, `Test*`, `test_*`).
- Place tests next to mirrored package paths under `tests/`.
- Aim for unit tests that run CPU-only; add fixtures for I/O and model stubs.
- Add/adjust tests when changing public behavior; target meaningful coverage.

## Commit & Pull Request Guidelines
- Use clear, atomic commits. Conventional style is encouraged (e.g., `feat(models): add VOC metrics`).
- PRs should include: concise description, rationale, linked issues, before/after metrics or logs (for ML), and docs updates.
- Pre-submit: run `ruff check`, `ruff format`, and `pytest` locally; ensure paths/imports use `vibelab`.

## Security & Configuration Tips
- Use a local `.env` for secrets and paths; never commit real keys. Treat `sky.env` as an example only.
- Do not commit datasets or large artifacts; reference paths via env vars (`DATA_ROOT`, etc.).
- After cloning: `git submodule update --init --recursive` to sync `ext/` contents.
