using BenchmarkTools
using HawkesPlasticNetworks
using LinearAlgebra

const HPN = HawkesPlasticNetworks

# Workload sizes and initial values
const N_POST = 512
const N_EXC = 768
const N_INH = 256
const EXTERNAL_INPUT = 4.0
const EXCITATORY_WEIGHT = 0.002
const INHIBITORY_WEIGHT = 0.004
const EXCITATORY_TRACE_VALUE = 3.0
const INHIBITORY_TRACE_VALUE = 5.0
const EVALUATION_TIME = 0.25

# Model parameters
const TAU_EXC = 0.02
const TAU_INH = 0.01

# BenchmarkTools settings
const BENCHMARK_SAMPLES = 100
const BENCHMARK_EVALS = 1
const BENCHMARK_SECONDS = 10.0

function make_benchmark_state()
    post = HPN.PopulationExpKernelExcitatory(
        N_POST,TAU_EXC;label="benchmark_post")
    pre_exc = HPN.PopulationExpKernelExcitatory(
        N_EXC,TAU_EXC;label="benchmark_exc")
    pre_inh = HPN.PopulationExpKernelInhibitory(
        N_INH,TAU_INH;label="benchmark_inh")
    weights_exc = fill(EXCITATORY_WEIGHT,N_POST,N_EXC)
    weights_inh = fill(INHIBITORY_WEIGHT,N_POST,N_INH)
    connection_exc = HPN.ConnectionWithWeights(post,weights_exc,pre_exc)
    connection_inh = HPN.ConnectionWithWeights(post,weights_inh,pre_inh)
    connected = HPN.ConnectedPopulationExpKernel(
        post,fill(EXTERNAL_INPUT,N_POST),
        (connection_exc,pre_exc),(connection_inh,pre_inh))
    HPN.set_initial_rates!(pre_exc,EXCITATORY_TRACE_VALUE)
    HPN.set_initial_rates!(pre_inh,INHIBITORY_TRACE_VALUE)
    return (; connected,weights_exc,weights_inh,pre_exc,pre_inh)
end

# Previous implementation: both signs were accumulated into the proposal rate,
# so inhibition could make this value lower than a later instantaneous rate.
function previous_compute_rates_upper!(rates::Vector{Float64},t_now::Float64,state)
    copy!(rates,state.connected.input)
    decay_exc = HPN.trace_decay(t_now,state.pre_exc.trace)
    mul!(rates,state.weights_exc,state.pre_exc.trace.val,decay_exc,1.0)
    decay_inh = HPN.trace_decay(t_now,state.pre_inh.trace)
    mul!(rates,state.weights_inh,state.pre_inh.trace.val,-decay_inh,1.0)
    @inbounds for idx in eachindex(rates)
        rates[idx] = max(rates[idx],eps(Float64))
    end
    return nothing
end

# Improved implementation: only the current positive excitation is needed.
function improved_compute_rates_upper!(rates::Vector{Float64},t_now::Float64,state)
    copy!(rates,state.connected.input)
    decay_exc = HPN.trace_decay(t_now,state.pre_exc.trace)
    mul!(rates,state.weights_exc,state.pre_exc.trace.val,decay_exc,1.0)
    @inbounds for idx in eachindex(rates)
        rates[idx] = max(rates[idx],eps(Float64))
    end
    return nothing
end

function current_compute_rates_upper!(rates::Vector{Float64},t_now::Float64,state)
    HPN.compute_rates_upper!(rates,t_now,state.connected)
    return nothing
end

function benchmark_upper_bounds()
    state = make_benchmark_state()
    previous_rates = zeros(N_POST)
    improved_rates = zeros(N_POST)
    current_rates = zeros(N_POST)
    improved_compute_rates_upper!(improved_rates,EVALUATION_TIME,state)
    current_compute_rates_upper!(current_rates,EVALUATION_TIME,state)
    if improved_rates != current_rates
        error("improved and current package upper bounds differ")
    end

    suite = BenchmarkGroup()
    suite["previous"] = @benchmarkable previous_compute_rates_upper!(
        $previous_rates,$EVALUATION_TIME,$state) samples=BENCHMARK_SAMPLES evals=BENCHMARK_EVALS seconds=BENCHMARK_SECONDS
    suite["improved"] = @benchmarkable improved_compute_rates_upper!(
        $improved_rates,$EVALUATION_TIME,$state) samples=BENCHMARK_SAMPLES evals=BENCHMARK_EVALS seconds=BENCHMARK_SECONDS
    suite["current package"] = @benchmarkable current_compute_rates_upper!(
        $current_rates,$EVALUATION_TIME,$state) samples=BENCHMARK_SAMPLES evals=BENCHMARK_EVALS seconds=BENCHMARK_SECONDS
    return run(suite;verbose=true)
end

display(benchmark_upper_bounds())
