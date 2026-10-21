# Agent Protocol - HawkesPlasticNetworks

This is an early-stage Julia package for spiking network simulations.

## Current Scope

- Prefer simple, explicit implementations over broad abstractions. The package
  is still finding its core API.
- Keep README/docs claims aligned with the current focus when editing
  user-facing documentation.

## Julia Style

- Write small functions with one clear responsibility.
- Prefer Julia multiple dispatch over large conditional branches when behavior
  depends on model/input/recorder types.
- Use mutating functions with a `!` suffix and return `nothing` unless there is a
  clear reason to return a value.
- Keep allocations visible and intentional. Reuse provided buffers such as
  `input_alloc`, `utility_alloc`, or recorder storage where that matches the
  local design.
- Keep public names, struct fields, and function signatures boring and explicit;
  avoid clever API layers while the model set is small.
- Favor deterministic behavior in tests and examples. Pass an RNG or seed local
  randomness when random behavior matters.
- Always use explicit `if` statements for conditional control flow. Do not use
  short-circuit expressions such as `condition || return`, `condition &&
  continue`, or `condition || throw(...)`.

## Other conventions

When dealing with connectivity and recursion, use a "post <- pre" ordering, meaning that the w_ij element of the weight matrix is the weight from neuron j (pre) to neuron i (post). Functions also follow this same convention, with arguments referring to postsynaptic populations first, then presynaptic populations.

## Documentation

- Keep executable documentation examples in `docs/literate/` and write them as
  Literate.jl source files that can be run from top to bottom.
- Put executable setup, validation, or cleanup code that should not appear in
  the generated documentation between `## #src` marker lines. Keep code visible
  when it is part of the explanation readers should follow.
- Suppress uninformative output in generated documentation. In particular, add
  a trailing semicolon to the final line of a code block before the next text
  block unless displaying that value is useful to the example.
- Keep prose close to the code it explains, and make examples deterministic by
  seeding or explicitly passing an RNG whenever randomness affects the result.
- When changing package behavior used by an example, update the corresponding
  Literate source rather than editing generated documentation.

## Testing

- Keep package tests lightweight and independent of documentation. Files in
  `test/` must test the package API directly; do not load, parse, or extract code
  from `docs/literate/` or other documentation sources.
- Do not add plotting, rendering, or documentation packages (such as `Plots`,
  `Literate`, or `Documenter`) to the root project or its test dependencies for
  example validation. Keep documentation dependencies in `docs/Project.toml`.
  Any new test-only dependency must be necessary for package behavior tests.
- Put checks of documentation examples in `docs/test/` and run them with
  `julia --project=docs docs/test/runtests.jl` in the documentation CI job,
  separately from `Pkg.test()`. Keep example parameter assertions there too.
- For package implementation changes, use the standard Julia package test flow:

  ```sh
  julia --project=. -e 'using Pkg; Pkg.test()'
  ```

- Every new package function should have a corresponding unit test. Small helper
  functions are not exempt.
- When changing an existing package function, add or update tests that capture the
  changed behavior directly.
- Documentation examples are validated by executing or rendering the affected
  example. Changes confined to example parameters, plots, or prose do not require
  the full package test suite, even when the example contains executable Julia.
- Do not add package unit tests solely to pin documentation parameter choices.
  Update and run the affected documentation checks when example parameters
  change; this alone does not require a full package test run.
- Organize tests into focused `@testset`s by component or behavior. It is fine
  to split tests into additional files and `include` them from
  `test/runtests.jl` as coverage grows.
- Test numerical behavior with explicit tolerances using `isapprox`
  rather than exact equality when floating-point roundoff is possible.
- Cover edge cases that define the API contract, such as empty populations,
  mismatched dimensions, recorder boundaries, and deterministic random inputs.

## Benchmarks

When requested, generate benchmarks as separate test files in the `benchmark/` directory.
Always compare the proposed version of a function with the previous implementations. For
consistency copy the code entirely into the benchmark script, so that it contains:
- previous implementation
- improved implementation
- current implementation in the HawkesPlasticNetworks package.

Expose workload sizes, model parameters, initial values, and BenchmarkTools settings
as clearly named constants at the top of each benchmark script so they are easy to
control.


## Development Workflow

- Before editing, check the worktree with `git status --short` and avoid
  overwriting unrelated user changes.
- Keep changes scoped to the requested behavior. Do not refactor inactive functions
  or files.
- Run the full package test command before finishing when package implementation
  changes are made.
  If tests cannot run because dependencies or the environment are unavailable,
  report that clearly.
- For documentation-only edits, including executable example parameter changes,
  validate the affected example or page without running the full package suite.
  For prose-only edits, check accuracy; execution or rendering is optional.
