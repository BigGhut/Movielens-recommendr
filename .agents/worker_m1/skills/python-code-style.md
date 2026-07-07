---
name: python-code-style
description: Python programming style guide enforcing idiomatic code, strict type hinting, docstrings, and performance best practices. Use this skill when writing, reviewing, or refactoring Python code.
---

# Python Code Style Skill

Enforces high-quality Pythonic code development.

## Guidelines
1. **Type Hints**: Always use Python type hints (PEP 484) for function signatures and class definitions.
2. **Pydantic**: Use Pydantic models for data parsing and validation at boundaries.
3. **Code Quality**: Follow PEP 8 guidelines. Use Ruff for linting.
4. **Performance**: Vectorize operations in Pandas and NumPy; avoid iterating over rows whenever possible.
