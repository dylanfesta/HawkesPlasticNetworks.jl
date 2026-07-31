using BenchmarkTools
using HawkesPlasticNetworks

const HPN = HawkesPlasticNetworks

# Workload sizes and initial values
const POPULATION_SIZES = (100,500,1000)
const INITIAL_WEIGHT = 1.0
const NONVIOLATING_WEIGHT = 0.25
const ZERO_STRIDE = 0 # Set above zero to mask every ZERO_STRIDE-th synapse.
const VIOLATION_CASES = (:all,:none)

# Model parameters
const WEIGHT_SUM_LIMIT_FRACTION = 0.5
const WEIGHT_SUM_TOLERANCE_FRACTION = 0.05
const WEIGHT_MIN = 0.0
const WEIGHT_MAX = Inf
const TARGETS = (
    HPN.HeterosynapticIncoming(),
    HPN.HeterosynapticOutgoing())
const METHODS = (
    HPN.HeterosynapticSubtractive(),
    HPN.HeterosynapticDivisive())

# BenchmarkTools settings
const BENCHMARK_SAMPLES = 50
const BENCHMARK_EVALS = 1
const BENCHMARK_SECONDS = 3.0

struct HeterosynapticBenchmarkState{
    HT<:HPN.HeterosynapticTarget,
    HM<:HPN.HeterosynapticMethod}
    weights::Matrix{Float64}
    mask::Matrix{Bool}
    workspace::Vector{Float64}
    active_counts::Vector{Int}
    groups_to_normalize::Vector{Bool}
    weight_sum_limit::Float64
    tolerance::Float64
    weight_min::Float64
    weight_max::Float64
    target::HT
    method::HM
end

function group_count(
        weights::Matrix{Float64},::HPN.HeterosynapticIncoming)
    return size(weights,1)
end

function group_count(
        weights::Matrix{Float64},::HPN.HeterosynapticOutgoing)
    return size(weights,2)
end

function group_width(
        weights::Matrix{Float64},::HPN.HeterosynapticIncoming)
    return size(weights,2)
end

function group_width(
        weights::Matrix{Float64},::HPN.HeterosynapticOutgoing)
    return size(weights,1)
end

function group_index(
        post_idx::Int,::Int,::HPN.HeterosynapticIncoming)
    return post_idx
end

function group_index(
        ::Int,pre_idx::Int,::HPN.HeterosynapticOutgoing)
    return pre_idx
end

function make_state(
        n::Int,target::HPN.HeterosynapticTarget,
        method::HPN.HeterosynapticMethod)
    weights = fill(INITIAL_WEIGHT,n,n)
    if ZERO_STRIDE > 0
        @inbounds for idx in eachindex(weights)
            if idx % ZERO_STRIDE == 0
                weights[idx] = 0.0
            end
        end
    end
    mask = Matrix{Bool}(undef,size(weights))
    @inbounds for idx in eachindex(weights,mask)
        mask[idx] = iszero(weights[idx])
    end
    n_groups = group_count(weights,target)
    active_counts = zeros(Int,n_groups)
    @inbounds for pre_idx in axes(weights,2)
        for post_idx in axes(weights,1)
            if !mask[post_idx,pre_idx]
                idx = group_index(post_idx,pre_idx,target)
                active_counts[idx] += 1
            end
        end
    end
    workspace = zeros(n_groups)
    groups_to_normalize = fill(false,n_groups)
    weight_sum_limit =
        WEIGHT_SUM_LIMIT_FRACTION*INITIAL_WEIGHT*group_width(weights,target)
    tolerance =
        WEIGHT_SUM_TOLERANCE_FRACTION*INITIAL_WEIGHT*group_width(weights,target)
    return HeterosynapticBenchmarkState(
        weights,mask,workspace,active_counts,groups_to_normalize,
        weight_sum_limit,tolerance,WEIGHT_MIN,WEIGHT_MAX,target,method)
end

function reset_weights!(state::HeterosynapticBenchmarkState,value::Float64)
    @inbounds for idx in eachindex(state.weights,state.mask)
        if state.mask[idx]
            state.weights[idx] = 0.0
        else
            state.weights[idx] = value
        end
    end
    return nothing
end

function prepare_correction!(
        state::HeterosynapticBenchmarkState{
            HT,HPN.HeterosynapticSubtractive}) where HT
    @inbounds for idx in eachindex(state.workspace)
        weight_sum = state.workspace[idx]
        normalize = weight_sum > state.weight_sum_limit+state.tolerance
        state.groups_to_normalize[idx] = normalize
        if normalize
            state.workspace[idx] =
                (weight_sum-state.weight_sum_limit)/state.active_counts[idx]
        else
            state.workspace[idx] = 0.0
        end
    end
    return nothing
end

function prepare_correction!(
        state::HeterosynapticBenchmarkState{
            HT,HPN.HeterosynapticDivisive}) where HT
    @inbounds for idx in eachindex(state.workspace)
        weight_sum = state.workspace[idx]
        normalize = weight_sum > state.weight_sum_limit+state.tolerance
        state.groups_to_normalize[idx] = normalize
        if normalize
            state.workspace[idx] = state.weight_sum_limit/weight_sum
        else
            state.workspace[idx] = 1.0
        end
    end
    return nothing
end

function corrected_weight(
        weight::Float64,correction::Float64,
        state::HeterosynapticBenchmarkState{
            HT,HPN.HeterosynapticSubtractive}) where HT
    return min(state.weight_max,max(weight-correction,state.weight_min))
end

function corrected_weight(
        weight::Float64,correction::Float64,
        state::HeterosynapticBenchmarkState{
            HT,HPN.HeterosynapticDivisive}) where HT
    return min(state.weight_max,max(weight*correction,state.weight_min))
end

# Previous implementation: generic target indexing and two complete matrix
# passes, copied from the package version before target specialization.
function previous_normalize!(state::HeterosynapticBenchmarkState)
    fill!(state.workspace,0.0)
    @inbounds for pre_idx in axes(state.weights,2)
        for post_idx in axes(state.weights,1)
            if !state.mask[post_idx,pre_idx]
                idx = group_index(post_idx,pre_idx,state.target)
                state.workspace[idx] += state.weights[post_idx,pre_idx]
            end
        end
    end
    prepare_correction!(state)
    @inbounds for pre_idx in axes(state.weights,2)
        for post_idx in axes(state.weights,1)
            if !state.mask[post_idx,pre_idx]
                idx = group_index(post_idx,pre_idx,state.target)
                if state.groups_to_normalize[idx]
                    state.weights[post_idx,pre_idx] = corrected_weight(
                        state.weights[post_idx,pre_idx],state.workspace[idx],state)
                end
            end
        end
    end
    return nothing
end

# Improved implementation: copied target-specialized kernels.
function improved_normalize!(
        state::HeterosynapticBenchmarkState{
            HPN.HeterosynapticIncoming})
    fill!(state.workspace,0.0)
    @inbounds for pre_idx in axes(state.weights,2)
        for post_idx in axes(state.weights,1)
            if !state.mask[post_idx,pre_idx]
                state.workspace[post_idx] += state.weights[post_idx,pre_idx]
            end
        end
    end
    prepare_correction!(state)
    any_group = false
    @inbounds for post_idx in eachindex(state.groups_to_normalize)
        if state.groups_to_normalize[post_idx]
            any_group = true
            break
        end
    end
    if !any_group
        return nothing
    end
    @inbounds for pre_idx in axes(state.weights,2)
        for post_idx in axes(state.weights,1)
            if state.groups_to_normalize[post_idx]
                if !state.mask[post_idx,pre_idx]
                    state.weights[post_idx,pre_idx] = corrected_weight(
                        state.weights[post_idx,pre_idx],
                        state.workspace[post_idx],state)
                end
            end
        end
    end
    return nothing
end

function improved_normalize!(
        state::HeterosynapticBenchmarkState{
            HPN.HeterosynapticOutgoing})
    @inbounds for pre_idx in axes(state.weights,2)
        weight_sum = 0.0
        for post_idx in axes(state.weights,1)
            if !state.mask[post_idx,pre_idx]
                weight_sum += state.weights[post_idx,pre_idx]
            end
        end
        state.workspace[pre_idx] = weight_sum
    end
    prepare_correction!(state)
    any_group = false
    @inbounds for pre_idx in eachindex(state.groups_to_normalize)
        if state.groups_to_normalize[pre_idx]
            any_group = true
            break
        end
    end
    if !any_group
        return nothing
    end
    @inbounds for pre_idx in axes(state.weights,2)
        if state.groups_to_normalize[pre_idx]
            correction = state.workspace[pre_idx]
            for post_idx in axes(state.weights,1)
                if !state.mask[post_idx,pre_idx]
                    state.weights[post_idx,pre_idx] = corrected_weight(
                        state.weights[post_idx,pre_idx],correction,state)
                end
            end
        end
    end
    return nothing
end

# Current implementation in HawkesPlasticNetworks.
function current_normalize!(state::HeterosynapticBenchmarkState)
    HPN._normalize_heterosynaptic!(
        state.weights,state.mask,state.workspace,state.active_counts,
        state.groups_to_normalize,state.weight_sum_limit,
        state.tolerance,
        state.weight_min,state.weight_max,state.target,state.method)
    return nothing
end

function benchmark_case(
        n::Int,target::HPN.HeterosynapticTarget,
        method::HPN.HeterosynapticMethod,violation_case::Symbol)
    value = if violation_case == :all
        INITIAL_WEIGHT
    else
        NONVIOLATING_WEIGHT
    end
    previous_state = make_state(n,target,method)
    improved_state = make_state(n,target,method)
    current_state = make_state(n,target,method)

    reset_weights!(previous_state,value)
    reset_weights!(improved_state,value)
    reset_weights!(current_state,value)
    previous_normalize!(previous_state)
    improved_normalize!(improved_state)
    current_normalize!(current_state)
    @assert previous_state.weights ≈ improved_state.weights
    @assert improved_state.weights ≈ current_state.weights

    previous_benchmark = @benchmarkable previous_normalize!(
        $previous_state) setup=(
        reset_weights!($previous_state,$value)
    ) samples=BENCHMARK_SAMPLES evals=BENCHMARK_EVALS seconds=BENCHMARK_SECONDS
    improved_benchmark = @benchmarkable improved_normalize!(
        $improved_state) setup=(
        reset_weights!($improved_state,$value)
    ) samples=BENCHMARK_SAMPLES evals=BENCHMARK_EVALS seconds=BENCHMARK_SECONDS
    current_benchmark = @benchmarkable current_normalize!(
        $current_state) setup=(
        reset_weights!($current_state,$value)
    ) samples=BENCHMARK_SAMPLES evals=BENCHMARK_EVALS seconds=BENCHMARK_SECONDS
    previous_trial = run(previous_benchmark)
    improved_trial = run(improved_benchmark)
    current_trial = run(current_benchmark)

    println(
        "n=",n,
        " target=",nameof(typeof(target)),
        " method=",nameof(typeof(method)),
        " violations=",violation_case)
    println("previous")
    display(median(previous_trial))
    println("improved")
    display(median(improved_trial))
    println("package")
    display(median(current_trial))
    return nothing
end

for n in POPULATION_SIZES
    for target in TARGETS
        for method in METHODS
            for violation_case in VIOLATION_CASES
                benchmark_case(n,target,method,violation_case)
            end
        end
    end
end
