```markdown
# 🧪 VibeLab Coding Playground (`vibelab`)

Welcome to **VibeLab’s** playground!
This repository is a sandbox for experimenting with ideas, learning new tech, and storing useful code snippets.

---

## 🎯 Purpose

* Prototype new concepts quickly
* Explore different tool-chains and libraries
* Practise coding techniques
* Keep reference examples for future projects

---

## 🚀 Getting Started

### 1 Create a virtual environment (conda recommended)

```bash
conda create -n vbl python=3.12
conda activate vbl
```

### 2 Clone and install

```bash
git clone --recursive https://github.com/wvibe/vibe_coding.git
cd vibe_coding

# Installs every dependency *and* sets up editable packages
pip install -r requirements.txt
```

`requirements.txt` contains

```
-e .                   # editable install of src/vibelab
-e ./ext/ultralytics   # editable install of our forked Ultralytics submodule
<all third-party packages>
```

---

## 🗂 Project Structure

```
vibe_coding/
├─ configs/            # dataset / training / model YAML
├─ docs/               # design docs & guides
├─ ext/ultralytics/    # the single submodule (fork)
├─ notebooks/          # Jupyter experiments
├─ ref/                # read-only reference code
├─ scripts/            # CLI utilities (not part of the package)
├─ src/
│   ├─ vibelab/        # ← top-level Python package (new)
│   │   ├─ models/
│   │   └─ utils/
│   └─ …legacy code    # to be moved into `vibelab`
├─ tests/              # mirrors src/ hierarchy
├─ requirements.txt
└─ setup.py
```

*New code should live under **`src/vibelab/…`** and be imported as, for example:*

```python
from vibelab.utils.geometry import mask_to_yolo_polygons
```

Legacy `from src.…` imports will be migrated gradually.

---

## ⚙️ Environment Variables

Create a `.env` file in the repo root (example):

```dotenv
VIBE_ROOT=/abs/path/to/vibe
VHUB_ROOT=${VIBE_ROOT}/vhub
DATA_ROOT=${VHUB_ROOT}/data
```

---

## 🖥️ Cursor IDE Setup

For an enhanced experience with the Cursor IDE, follow the guide:

➡️ **docs/cursor/setup-guide.md**

---

## 👥 Contributing

* Work on a feature branch
* Keep commits focused and documented
* Feel free to add new folders for distinct experiments
* Update docs / notebooks with your findings

---

## YOLO Training Scripts

The training scripts now live under **`vibelab/models/ext/…`**.
Update your command lines accordingly, e.g.:

```bash
python src/vibelab/models/ext/yolov8/train_yolov8.py \
  --config src/vibelab/models/ext/yolov8/configs/voc_finetune_config.yaml \
  --name yolov8l_voc_finetune_run1
```

(Full instructions remain the same as before.)

---

## 📝 Note

This is an experimental playground—break things, fix them, and learn!

Happy coding! ✨
```