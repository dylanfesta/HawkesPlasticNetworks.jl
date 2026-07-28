# Benchmarks

From the repository root, prepare the benchmark environment with:

```sh
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

Then run the asymmetric STDP comparison:

```sh
julia --project=benchmark benchmark/plasticity_asymmetric_stdp.jl
```

Workload sizes, model parameters, initial values, and BenchmarkTools settings
are constants at the top of the benchmark file.
