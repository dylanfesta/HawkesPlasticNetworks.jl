# Benchmarks

From the repository root, prepare the benchmark environment with:

```sh
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

Then run the asymmetric STDP comparison:

```sh
julia --project=benchmark benchmark/plasticity_asymmetric_stdp.jl
```

To compare trace propagation implementations, run:

```sh
julia --project=benchmark benchmark/01_traces_propagation.jl
```

To compare the current monolithic asymmetric STDP update with concretely typed
internal weight-update kernels, run:

```sh
julia --project=benchmark benchmark/plasticity_typed_kernels.jl
```

To compare the previous generic heterosynaptic normalization kernel, the
target-specialized improvement, and the current package implementation, run:

```sh
julia --project=benchmark benchmark/plasticity_heterosynaptic_normalization.jl
```

Workload sizes, model parameters, initial values, and BenchmarkTools settings
are constants at the top of the benchmark file.
