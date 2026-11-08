---
name: spec-to-tests
description: Procedure for implementing code from a written specification (such as INSTRUCTIONS.md) that hidden unit tests will check. Use it whenever you implement an exercise or feature from a written spec.
---

# Implementing from a written specification

Hidden tests check the exact behaviour the specification describes: names, return types, exact
strings, error types and error messages. Most lost points come from details that were read once and
then forgotten. Work in this order.

1. **Read everything.** Read the whole specification and the stub file. Note every function, class,
   method and property name and signature the stub defines; hidden tests import them, so keep them.
2. **Extract a checklist.** Before writing code, list every requirement as a short line:
   - each worked example (input and exact expected output);
   - each error case: the exception type and the *exact* message text when the spec gives one
     (`raise ValueError("...")` with that wording);
   - edge cases the spec implies: empty input, zero, one element, negative numbers, duplicates,
     upper/lower case, whitespace, very large input, invalid input;
   - output format details: list vs tuple vs string, ordering, trailing newlines, punctuation.
3. **Write your own tests first.** Create `test_spec.py` in the current directory with one `assert`
   (or `assertRaises`-style `try/except`) per checklist line, using plain Python so it runs with
   `python test_spec.py`. End it with `print("all spec tests passed")`.
4. **Implement** the code in the stub file. Use only the standard library.
5. **Run your tests** with `python test_spec.py`.
6. **If a test fails**, decide which side is wrong by re-reading the specification: fix the code, or fix
   the test if the test misread the spec. Never delete a test just to make the run pass. Run the tests
   again until they all pass.
7. **Final review.** Re-read the specification line by line against your checklist, and check that
   every name from the stub still exists with the same signature.
