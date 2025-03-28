# 🎮 Vibe Coding Playground

Welcome to the Vibe Coding playground! This repository serves as an experimental space for testing ideas, learning new concepts, and exploring various coding projects.

## 🎯 Purpose

This is a sandbox environment where we:
- Test new coding concepts
- Experiment with different technologies
- Practice coding techniques
- Store code snippets and examples
- Try out new ideas

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/wvibe/vibe_coding.git
cd vibe_coding

# Install as a development package
pip install -e .
```

### Project Structure

The project uses a modern src-layout for better packaging:

```
vibe_coding/
├── src/                # Source code with proper package structure
│   ├── data_loaders/   # Dataset utilities
│   ├── models/         # Model implementations
│   │   ├── hf/         # Huggingface-based models
│   │   └── py/         # Pure PyTorch models
│   └── utils/          # Shared utilities
├── tests/              # Test files mirroring src structure
├── docs/               # Documentation
├── setup.py            # Package installation
└── requirements.txt    # Dependencies
```

### Environment Setup

Create a `.env` file in the project root with necessary paths:

```
# Required paths
VIBE_ROOT=/path/to/vibe
VHUB_ROOT=${VIBE_ROOT}/vhub
DATA_ROOT=${VHUB_ROOT}/data
```

## 📝 Note

This is a playground repository - perfect for experimentation and learning. Feel free to break things, fix them, and learn in the process!

## 👥 Contributing

When contributing to this playground:
- Create a new branch for your experiments
- Document your findings and approaches
- Feel free to add new directories for distinct project ideas
- Share insights in the documentation

---
Happy Coding! ✨