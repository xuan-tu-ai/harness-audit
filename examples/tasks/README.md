# Example tasks

Twenty small Python bug-fix tasks. Each directory has:

- `task.yaml`: id, `group` (used for task distance), `split`, and the prompt;
- `workspace/`: what the agent sees (a buggy module and `test_public.py`);
- `evaluator/`: what the agent never sees (`test_hidden.py`, scored by `harness_audit.evaluator`,
  and `reference.py`, a reference fix used only by the test suite to check that the evaluator is sound).

| split | tasks | groups |
|---|---|---|
| proposal | slugify, title_case, truncate | strings |
| admission | count_vowels, is_palindrome, wrap_text, median, round_half_up | strings, numbers |
| report | camel_to_snake, initials, parse_kv, reverse_words, clamp, percent_change, chunk, flatten | strings, numbers, collections |
| retention | fizzbuzz, factorial, sum_digits, leap_year | legacy |

The `python-bugfix` skill in `harness_b` was written by hand for this example, with the three proposal
tasks in mind (edge cases such as whitespace runs, accents and boundaries). The proposal split is listed
so that Selection and Improvement can check that it does not overlap the report split; it is not run.
