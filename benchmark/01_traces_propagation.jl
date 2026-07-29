using BenchmarkTools
using HawkesPlasticNetworks

const HPN = HawkesPlasticNetworks

const TRACE_SIZES = (1_000, 5_000, 10_000)
const TRACE_TAU = 100e-3
const TIME_NOW = 1.0
const INITIAL_TRACE_VALUE = 1.0
const BENCHMARK_SAMPLES = 10_000
const BENCHMARK_SECONDS = 1.0

# Previous implementation: decay the trace directly in the propagation method.
@inline function propagate_nokernel!(tnow::Real, trace::HPN.Trace)
    Δt = tnow - trace.t_last
    trace.val .*= exp(-Δt / trace.τ)
    trace.t_last = tnow
    return nothing
end

@inline function trace_decay_kernel(tnow::Real, trace::HPN.Trace)
    return exp(-(tnow - trace.t_last) / trace.τ)
end

@inline function propagate_with_kernel!(tnow::Real, trace::HPN.Trace)
    trace.val .*= trace_decay_kernel(tnow, trace)
    trace.t_last = tnow
    return nothing
end

function make_trace(n::Integer)
    trace = HPN.Trace(TRACE_TAU, n)
    fill!(trace.val, INITIAL_TRACE_VALUE)
    return trace
end

function propagation_suite()
    suite = BenchmarkGroup()

    for n in TRACE_SIZES
        group = suite[string(n)]
        group["previous (no kernel)"] = @benchmarkable(
            propagate_nokernel!($TIME_NOW, trace),
            setup = (trace = make_trace($n)),
            evals = 1,
        )
        group["improved (kernel)"] = @benchmarkable(
            propagate_with_kernel!($TIME_NOW, trace),
            setup = (trace = make_trace($n)),
            evals = 1,
        )
        group["current package"] = @benchmarkable(
            HPN.propagate!($TIME_NOW, trace),
            setup = (trace = make_trace($n)),
            evals = 1,
        )
    end

    return suite
end

suite = propagation_suite()
results = run(
    suite;
    verbose = true,
    samples = BENCHMARK_SAMPLES,
    seconds = BENCHMARK_SECONDS,
)
show(stdout, MIME("text/plain"), results)
println()
