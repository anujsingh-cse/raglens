# Contributing to RagLens

Thanks for considering a contribution. RagLens is a young project; small focused
PRs are much easier to review than big-bang rewrites.

## First-time contributors

Issues labeled `good first issue` are scoped for new contributors. If none are
open, propose one in a Discussion thread before opening a PR.

## Development setup

```bash
git clone https://github.com/anujsingh-cse/raglens
cd raglens
pip install -e .[dev]
pytest -ra                       # pure-logic tests, no API key required
```

## Test suite — two layers, no mock data

RagLens never ships mock LLM responses. Tests split into two layers:

* **Pure-logic** (default, runs in CI): the failure-mode classifier, claim
  extraction, judge-score parser, conflict-format parser, dataset I/O, HTML/JSON
  rendering on synthetic `Report` objects, and the `_coerce_to_trace`
  normalizer. These need no LLM. The early-return paths of
  `CounterfactualAttribution` (empty chunks / empty answer) are also pure because
  they return before the LLM is touched — these tests use a `_NoopLLM` that
  raises `AssertionError` if its `.complete()` is called, *proving* the
  early-return path is hit (this is a contract test, not mock data).
* **Live integration** (`@pytest.mark.integration`, auto-skipped unless
  `NVIDIA_API_KEY` is set): real end-to-end `probe()` against NVIDIA NIM. These
  tests assert **structural invariants** — `faithfulness in [0,1]`,
  `attribution_score in [-1,1]`, `failure_mode in allowed set`,
  `n_judge_calls > 0` — never specific numbers, because real LLM output is
  non-deterministic.

```bash
# Pure-logic (default CI):
pytest

# Live integration against NVIDIA NIM (requires key):
export NVIDIA_API_KEY=nvapi-...
pytest -m integration
```

Anyone running `pytest` with no key sees all pure-logic tests pass and
integration tests auto-skip. **No mock LLM is in the codebase.**

## Coding standards

- **Python 3.10+**. Use modern syntax: `X | Y` unions, `match`/`case` where it
  reads cleanly, `@dataclass(slots=True)`.
- **Zero hard dependencies** in core. Provider plumbing may import optional
  packages (litellm) lazily, inside the function that needs them.
- **No comments unless they explain a non-obvious decision.** Docstrings are
  welcome and expected on public API.
- **Type-hint every public surface.** Run `ruff check raglens` before pushing.
- **One PR per logical change.** A new metric, a new provider adapter, and a
  README tweak are three PRs.

## Tests

Add a test under `tests/` for any behaviour you add or change. Prefer
**pure-logic tests** (no LLM) wherever possible — the classifier, the parsers
(`extract_claims`, `_parse_judge_score`, `_parse_conflict`), dataset I/O, and
HTML/JSON rendering can all be tested without an LLM. Tests that exercise
the LLM path must be marked `@pytest.mark.integration` so they auto-skip
without `NVIDIA_API_KEY`. Never depend on wall-clock or unseeded randomness.

When asserting against integration test output, assert **structural
invariants** (ranges, allowed enum values, call counts > 0) — never specific
verdicts, because real LLM output is non-deterministic.

## Attribution algorithm changes

The counterfactual attribution algorithm in `raglens/attribution.py` is the
heart of the project. Changes there warrant extra scrutiny:

1. Add a unit test that pins the new behaviour on a synthetic Trace.
2. Update `ARCHITECTURE.md` if the algorithm's contract changes.
3. Do not silently swap the default strategy — provide an opt-in flag and keep
   the existing one as a regression baseline.

## Reporting bugs

Open an issue with:
1. RagLens version (`python -c "import raglens; print(raglens.__version__)"`)
2. A minimal reproducible Trace + Dataset (you can pickle a trace and attach)
3. Expected vs actual attribution report

## Reporting security issues

Do not open public issues for security vulnerabilities. Email
`security@your-domain.example` (replace with the maintainer address listed in
your fork). For now, the address is in the LICENSE author block.
