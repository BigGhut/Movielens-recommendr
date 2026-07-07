---
name: python-debug-expert
description: Techniques and instructions for troubleshooting, debugging, and profiling Python applications. Use this skill to resolve exceptions, performance bottlenecks, memory leaks, and configure application logging.
---

# Python Debug Expert Skill

Guides interactive debugging and profiling of Python scripts.

## Debugging Strategy
1. **Interactive Debugging**: Use `import pdb; pdb.set_trace()` or IDE breakpoints to inspect state.
2. **Traceback Analysis**: Examine nested exceptions and capture debug-level state variables.
3. **Structured Logging**: Use the standard `logging` library instead of `print` to debug production code.
4. **Profiling**: Use `cProfile` for execution bottlenecks and `memory_profiler` for memory usage analysis.
