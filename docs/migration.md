# Migration Guide: from `src.*` to `vibelab.*`

This document explains **how to migrate** all importable code that currently lives directly under `src/` (imported via `from src.…`) into the new, namespaced package **`vibelab`**.

> **Goal:** after migration every runtime import will be `from vibelab.…`, and no top‑level module named `src` will be required in production or in tests.

# Detailed Migration Tasks (2024-05)

## Overview

This section tracks the step-by-step plan for the migration of `src/models/ext/yolov11` to the `vibelab` namespace, and the prerequisite migration of `src/utils/metrics`. The goal is to ensure a clean, DRY, and robust codebase with minimal disruption.

---

## Pre-Migration Tasks

### 1. Migrate `src/utils/metrics` to `src/vibelab/utils/metrics`
- [x] Move only the source code (not tests or docs) using `git mv`.
- [x] Update all codebase imports from `src.utils.metrics` to `vibelab.utils.metrics`.
- [x] Update all test and doc references to point to the new source location, but keep the test/docs files in their current place for now.
- [x] Run tests to confirm nothing is broken.
- [x] Update documentation in `docs/utils/metrics_compute.md` and `docs/utils/metrics_detection.md` to reflect the new source location and migration status.
- [x] **Result:** All tests passed and documentation updated as of 2024-05.

### 2. Refactor/Merge yolov11 Training Scripts
- Compare `train_detect.py` and `train_segment.py` in detail.
- If possible, merge into a single script (with a `--task` argument or config-based task detection); otherwise, refactor to maximize shared logic.
- Ensure robust, DRY, and maintainable training orchestration.
- Update CLI usage and documentation as needed.
- Test both detection and segmentation training.

---

## Main yolov11 Migration

### 3. Migrate yolov11 to vibelab
- Move the entire `src/models/ext/yolov11` folder to `src/vibelab/models/ext/yolov11` using `git mv`.
- Update all imports in the moved files and elsewhere in the codebase from `src.models.ext.yolov11` to `vibelab.models.ext.yolov11`.
- Ensure all parent directories have `__init__.py`.
- Update any project root detection logic as needed.
- Run tests and linting to confirm the migration is successful.
- Update documentation and migration notes.
- Remove the old folder after confirming success.

---

## Special Notes

- All moves should use `git mv` to preserve history.
- Each step should be tested before proceeding to the next.
- Document any issues or deviations from this plan in this file.

---

## 1. Why change?

| Issue with `src.*` | Benefit of `vibelab.*` |
|--------------------|------------------------|
| `src` is a generic name → risk of collision with other projects when installed together. | Unique namespace → safe to publish wheels / upload to internal PyPI. |
| Running tests from repo root silently works (because `.` is on `PYTHONPATH`) but breaks in any other working dir. | Editable install ensures `vibelab` is always on `sys.path` regardless of working directory. |
| Many IDEs need special config to index `src/` properly. | Standard package structure is much better supported by tooling. |
| Complex imports with multiple levels of relative imports. | Clear absolute imports from the root module. |

## 2. Migration Process

### Module Migration Checklist

✅ = Migrated, tests pass, old module still exists
🚀 = Fully completed (old module deleted)
⬜ = Not started

| Module                | Status | Notes |
|-----------------------|--------|-------|
| `utils.common`        | ✅     | Migrated in first phase |
| `utils.logging`       | ✅     | Migrated in first phase |
| `dataops.common`      | ✅     | Migrated with s3_fetcher |
| `dataops.cov_segm`    | ✅     | Migrated with all submodules |
| `utils.data_converter` | ⬜     | |
| `utils.data_loaders`  | ⬜     | |
| `utils.metrics`       | ✅     | Migrated, tests/docs updated |
| `utils.visualization` | ⬜     | |
| `models.ext`          | ⬜     | |
| `models.py`           | ⬜     | |

### Remaining Tasks

Once testing confirms proper functionality, the following clean-up tasks are required:

1. Remove original modules that have been migrated:
   - [ ] Delete src/utils/common
   - [ ] Delete src/utils/logging
   - [ ] Delete src/dataops/common
   - [ ] Delete src/dataops/cov_segm

2. Update any remaining imports in other parts of the codebase that might still reference the old modules.

---
## 3. Incremental Migration Strategy

We will **not** perform a big‑bang rename. Instead we move modules gradually while keeping the codebase green at every commit.

### 3.1 One‑time preparation (already done)

```bash
mkdir -p src/vibelab
touch src/vibelab/__init__.py
```

The empty `__init__.py` makes `vibelab` a valid package immediately.

### 3.2 Conventional commit sequence

| Commit | Contents |
|--------|----------|
| **A**  | *Submodule refactor* – merged to main already. |
| **B**  | Add `src/vibelab/` package, leave code untouched. CI should stay green because legacy imports still work. |
| **C…N**| For each logical unit (e.g. `utils.geometry`, `models.yolo`):<br>1. **Move file/dir** to `src/vibelab/…`.<br>2. Fix all imports in code, tests, scripts.<br>3. Run `pytest` and `ruff` locally.<br>4. Push PR – small, reviewable, easy to revert. |
| **Z**  | Delete `src/__init__.py` and any remaining loose modules; run `grep -R "from src\."` to ensure zero matches. |

### 3.3 Finding remaining legacy imports

```bash
grep -R "from src\." src tests scripts | cut -c1-120
```
(or use ripgrep `rg` for speed.)

CI pipeline will fail if this list is non‑empty once we reach commit **Z** (add a check‑step script).

---
## 4. Practical tips

### 4.1 IDE assisted rename

* **VS Code**: F2 rename symbol → choose "Move to new file" if offered.
* **PyCharm**: Refactor → Move… keeps imports in sync across project.

### 4.2 Keep imports sorted

Run `ruff format` (or `black + isort`) after each move to avoid noisy diffs.

### 4.3 Editable install stays valid

Because we keep `-e .` in `requirements.txt`, moved code is immediately importable without reinstalling anything.

### 4.4 Testing during transition

Run:

```bash
pytest -q
python -m vibelab.quickcheck   # any quick smoke scripts
```

after each batch move.

---
## 5. Updating external references

* **Scripts inside repo** – update their shebang or import lines to `vibelab.…` once the target module has been moved.
* **Notebooks** – use *Find & Replace* or run unit tests embedded in notebooks.
* **Down‑stream repos** (if any) – coordinate a branch where they bump the dependency and update imports.

---
## 6. Done checklist

- [ ] No `grep -R "from src\."` matches (except this doc).
- [ ] All modules in the migration checklist completed.
- [ ] Old modules in src have been removed (`src/utils/common`, `src/utils/logging`, etc.).
- [ ] `pytest` passes in clean virtual env created from `requirements.txt`.
- [ ] `pip wheel .` builds a wheel; installing that wheel into a fresh env runs CLI entry-points successfully.
- [ ] Old tags / tutorials updated or marked deprecated.

### 6.1 Final cleanup tasks

After all modules have been migrated and all tests pass with the new imports:

1. Remove the old module directories from `src/`:
   ```bash
   rm -rf src/utils/common
   rm -rf src/utils/logging
   # ... and so on for all migrated modules
   ```

2. Update any remaining documentation or references.

3. Commit the final cleanup as the last migration step.

When all checkboxes are ticked, announce in `CHANGELOG.md`:

> **[BREAKING]** All imports are now under `vibelab.*`; `src.*` is no longer supported.

---
Happy migrating! 🎉

