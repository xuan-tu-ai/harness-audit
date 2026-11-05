---
name: python-bugfix
description: Procedure for fixing a bug in a small Python function so that it matches its docstring. Use it whenever you are asked to fix a Python function.
---

# Fixing a Python function against its docstring

1. Read the whole function and its docstring. Treat every sentence of the docstring as a requirement.
2. Before editing, list the edge cases the docstring implies. Check at least: empty input; a single
   element; whitespace runs (tabs, newlines, repeated spaces); upper and lower case; accented or other
   non-ASCII characters; negative numbers and zero; ties when rounding; boundaries (`<` versus `<=`);
   invalid arguments that must raise; and whether the input must stay unmodified.
3. Fix the function. Keep its name and signature.
4. Verify: run `python test_public.py`, then run one `python -c` command whose `assert` statements
   check every edge case from step 2.
5. If a check fails, fix the code and run both checks again until they pass.
