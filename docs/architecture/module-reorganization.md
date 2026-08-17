# Proposal: reorganize the module architecture

This document proposes a staged response to [issue #280](https://github.com/alcides/GeneticEngine/issues/280).
It is intentionally a design proposal, not a request for a one-shot directory
rewrite. The goal is to make the ownership and dependency rules obvious while
preserving the current public API during migration.

## Motivation

The repository currently has three different kinds of code at its top level:

| Area | Current role |
| --- | --- |
| `geneticengine/` | Core search, grammar, representation, evaluation, and solution APIs |
| `geml/` | Scikit-learn-style estimators, symbolic-regression grammars, and coding grammars |
| `examples/` | Runnable examples, datasets, and benchmark definitions |

The code already has a useful dependency direction: `geml` builds on
`geneticengine`, while the core package does not need to import `geml`.
However, that boundary is implicit, the public entry points are spread across
many implementation modules, and optional integrations are mixed into the
same install surface as the core library. This makes it harder to answer basic
questions such as:

- Which modules are stable user-facing APIs?
- Which code is an algorithm implementation versus an algorithm building block?
- Which dependencies are required by the core package?
- Where should a new representation, integration, or benchmark be added?

## Proposed target layout

The first phase should establish ownership without moving every file:

```text
geneticengine/
  __init__.py                 # small, documented public facade
  algorithms/                 # complete search algorithms and GP steps
  evaluation/                 # budgets, evaluators, trackers, recorders
  grammar/                    # grammar extraction, decorators, metahandlers
  problems/                   # Problem and fitness abstractions
  random/                     # RandomSource implementations
  representations/            # genotype/phenotype representations and operators
  solutions/                  # Individual and tree solution types
  exceptions.py

geml/                         # optional integrations built on geneticengine
  estimators/                 # sklearn-compatible classifiers/regressors
  grammars/                   # reusable domain grammars
  simplegp.py                 # compatibility facade during migration

examples/
  ...                         # runnable examples only
  benchmarks/                 # benchmark definitions and benchmark runners
tests/
docs/
```

The important change is conceptual rather than cosmetic: `geneticengine` is
the engine, `geml` is an integration layer, and `examples` is never imported by
the engine. Existing top-level package names should remain available while
the migration is in progress.

## Dependency rules

The following rules should be checked in CI:

```text
geneticengine.grammar       ─┐
geneticengine.problems       │
geneticengine.random         ├── may depend on core utilities only
geneticengine.solutions      │
geneticengine.evaluation    ─┘

geneticengine.representations ──> grammar, solutions, random
geneticengine.algorithms      ──> evaluation, problems, representations, random
geml                          ──> geneticengine
examples and tests            ──> geneticengine and geml
```

More specifically:

1. `geneticengine` must never import `geml`, `examples`, or `tests`.
2. `geml` may import `geneticengine`, but the reverse dependency is forbidden.
3. Algorithms should depend on interfaces (`Problem`, `Representation`,
   `Evaluator`, and `RandomSource`) rather than concrete implementations where
   possible.
4. Grammar metahandlers should not import algorithms or evaluators.
5. Examples must construct the public API instead of importing private helper
   modules.
6. New optional integrations should be separate packages or extras rather than
   adding their dependencies to the core installation.

## Public API proposal

Each stable area should expose a small facade. For example:

```python
from geneticengine.algorithms import GeneticProgramming, RandomSearch
from geneticengine.evaluation import EvaluationBudget, TimeBudget
from geneticengine.grammar import Grammar, extract_grammar
from geneticengine.problems import Problem, SingleObjectiveProblem
from geneticengine.random import NativeRandomSource
from geneticengine.representations import TreeBasedRepresentation
```

The facades should re-export existing objects first. Moving implementation
details behind these facades can then happen independently, without requiring
users to update imports on every release.

## Migration plan

### Phase 1: document and measure

- Add the dependency rules above to contributor documentation.
- Publish the intended public imports for each area.
- Generate a dependency graph in CI using `tach` or an equivalent tool.
- Add a CI check that fails when `geneticengine` imports `geml`, `examples`, or
  `tests`, or when a new forbidden cycle is introduced.
- Inventory imports used by documentation, examples, and downstream tests.

Deliverable: a baseline graph and a checked-in list of supported imports.

### Phase 2: introduce compatibility facades

- Add `__init__.py` exports for algorithms, evaluation, grammar, problems,
  random, representations, and solutions.
- Keep existing module paths working.
- Add deprecation warnings only for paths that are genuinely being removed;
  do not warn for the current stable paths merely because a facade exists.
- Add import tests for every documented facade.

Deliverable: users can adopt the target imports before files move.

### Phase 3: separate optional integrations

- Keep `geml` as a compatibility package, but define it explicitly as an
  integration layer over the core package.
- Move sklearn-specific code into `geml.estimators` and retain forwarding
  modules for `geml.classifiers` and `geml.regressors`.
- Move reusable domain grammars into `geml.grammars` and keep examples out of
  the import path.
- Evaluate a `geml` optional dependency extra so users of the core engine do
  not install the full ML stack unnecessarily.

Deliverable: a minimal core installation and a clearly scoped integration
installation.

### Phase 4: move implementation modules incrementally

- Move one subsystem at a time, starting with the least coupled utilities.
- Replace old paths with forwarding modules that import from the new location.
- Update internal imports to use the target facades.
- Remove a forwarding module only after at least one deprecation cycle and a
  migration note.

Deliverable: implementation layout matches the documented architecture while
existing users receive a controlled migration path.

### Phase 5: enforce the architecture

- Run the dependency checker in pull requests.
- Require new public APIs to be exported from a facade and documented.
- Add a lightweight architecture test for forbidden imports and cycles.
- Review the package extras and dependency lockfile after each integration is
  split out.

## Compatibility and release policy

This should be released as a compatibility-preserving sequence:

- Minor release: add facades, checks, and deprecation notices.
- Minor release: move implementation modules behind forwarding imports.
- Major release: remove paths that have been deprecated for at least one full
  release cycle.

The first two phases should not require changes for existing examples or
downstream projects. The project should also publish a short import migration
table whenever a path is deprecated.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Import cycles appear during moves | Move one subsystem at a time; run the dependency check in CI |
| Users depend on implementation paths | Keep forwarding modules and document the supported facades |
| Core installation becomes heavier | Isolate `geml` and ML dependencies behind an optional extra |
| Large, review-hostile PR | Keep each phase and subsystem in a separate PR |
| Examples hide architectural regressions | Run `run_examples.sh` and import-facade tests in CI |

## Success criteria

The proposal is complete when:

- the dependency graph has one-way flow from core to integrations and examples;
- the supported public imports are documented and tested;
- `geneticengine` can be installed without optional ML integration dependencies;
- old imports have a documented compatibility path;
- architecture checks run automatically on pull requests; and
- adding an algorithm, representation, or integration has an obvious home.

## Suggested follow-up issues

This proposal should be implemented as separate issues or PRs:

1. Add public API facades and import tests.
2. Add dependency-graph CI checks.
3. Split `geml` estimators from reusable grammars.
4. Add optional dependency extras for integrations.
5. Migrate one core subsystem at a time behind compatibility modules.
