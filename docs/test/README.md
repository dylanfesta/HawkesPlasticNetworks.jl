These checks exercise documentation example definitions and short seeded
simulations. They run separately from the package unit tests because examples
use the documentation environment.

From the repository root:

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs docs/test/runtests.jl
```

CI runs these checks in the documentation job before building the site.
Package tests remain available with `julia --project=. -e 'using Pkg; Pkg.test()'`.
