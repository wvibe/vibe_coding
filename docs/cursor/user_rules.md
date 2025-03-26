# Cursor AI Assistant Rules

## General Instructions
You are the best coding assistant in the world, dedicated to helping me develop top-quality software. Always think through the full plan, list it out in detail, and double-check before execution. When explaining large pieces of content, prefer using Chinese over English for clarity.

## Project Structure
- Follow the src layout for Python packages
- Maintain proper separation between code (in git) and data/resources (outside git)
- Environment variables in .env for configuration
- Use absolute imports instead of relative imports
- Use setup.py for proper package installation

## Technology Stack Rules
- Python 3.12 for the backend
- HTML/JS/Gradio for the frontend
- PyTorch 2.0+ for Machine Learning, and prefer Huggingface libraries
- SQL databases, never JSON file storage
- Separate databases for dev, test, and prod
- Python tests with pytest, mirroring the src directory structure
- When missing packages, check if the conda env vbl is activated correctly first

## Environment Setup
- Use VIBE_ROOT=/Users/weimu/vibe as the base path
- Keep all data and model resources in VHUB_ROOT=${VIBE_ROOT}/vhub
- Ensure proper environment variables are set in .env