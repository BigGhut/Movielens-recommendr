---
name: ml-pipeline-designer
description: Guidelines and best practices for designing reproducible, modular, and maintainable Machine Learning (ML) pipelines. Use this skill when the user wants to build, refactor, or structure ML workflows, track experiments, or configure data pipelines.
---

# ML Pipeline Designer Skill

This skill guides the construction of modular and reproducible ML pipelines.

## Pipeline Architecture
1. **Modular Components**:
   - Separate ingestion, preprocessing, feature engineering, model training, evaluation, and deployment.
   - Each component must have clear inputs and outputs, avoiding global state.
2. **Reproducibility**:
   - Seed all random number generators (`numpy`, `random`, PyTorch/TensorFlow, scikit-learn).
   - Use data versioning (e.g., DVC) and experiment tracking (e.g., MLflow, Weights & Biases).
3. **Caching & Efficiency**:
   - Cache intermediate steps (e.g., processed features) to speed up iteration.
