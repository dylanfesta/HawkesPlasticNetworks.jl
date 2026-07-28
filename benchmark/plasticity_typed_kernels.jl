using BenchmarkTools
using HawkesPlasticNetworks

const HPN = HawkesPlasticNetworks

# Workload sizes and initial values
const N_POST = 512
const N_PRE = 768
const N_SELF = 512
const ZERO_STRIDE = 11
const INITIAL_WEIGHT = 0.5
const EVENT_TIME = 1.25
const EVENT_INDEX = 127

# Model parameters
const ETA = 1.0e-3
const BIAS_BALANCE = -0.1
const ALPHA_PRE = 0.02
const ALPHA_POST = -0.01
const TAU_PLUS = 0.02
const GAMMA = 1.5
const WEIGHT_MIN = 0.0
const WEIGHT_MAX = 2.0

# BenchmarkTools settings
const BENCHMARK_SAMPLES = 100
const BENCHMARK_EVALS = 1
const BENCHMARK_SECONDS = 10.0

struct BenchmarkPlasticity{
    TP<:HPN.Trace,TM<:HPN.Trace}
    η::Float64
    B::Float64
    αpre::Float64
    αpost::Float64
    weight_min::Float64
    weight_max::Float64
    trace_pre_plus::TP
    trace_post_minus::TM
    zero_weight_mask::Matrix{Bool}
end

function make_weights(n_post::Int,n_pre::Int)
    weights = fill(INITIAL_WEIGHT,n_post,n_pre)
    @inbounds for idx in eachindex(weights)
        if idx % ZERO_STRIDE == 0
            weights[idx] = 0.0
        end
    end
    return weights
end

function seed_traces!(trace_pre::HPN.Trace,trace_post::HPN.Trace)
    @inbounds for idx in eachindex(trace_pre.val)
        trace_pre.val[idx] = 0.05*(idx % 7)
    end
    @inbounds for idx in eachindex(trace_post.val)
        trace_post.val[idx] = 0.04*(idx % 5)
    end
    trace_pre.t_last = 0.25
    trace_post.t_last = 0.25
    return nothing
end

function make_benchmark_plasticity(weights::Matrix{Float64})
    n_post,n_pre = size(weights)
    mask = Matrix{Bool}(undef,n_post,n_pre)
    @inbounds for idx in eachindex(weights,mask)
        mask[idx] = iszero(weights[idx])
    end
    trace_pre = HPN.Trace(TAU_PLUS,n_pre)
    trace_post = HPN.Trace(GAMMA*TAU_PLUS,n_post)
    seed_traces!(trace_pre,trace_post)
    return BenchmarkPlasticity(
        ETA,BIAS_BALANCE,ALPHA_PRE,ALPHA_POST,WEIGHT_MIN,WEIGHT_MAX,
        trace_pre,trace_post,mask)
end

function make_local_state(
    n_post::Int,n_pre::Int,post_label::Symbol,pre_label::Symbol)
    weights = make_weights(n_post,n_pre)
    plast = make_benchmark_plasticity(weights)
    return (; weights,plast,post_label,pre_label)
end

function make_package_state(
    n_post::Int,n_pre::Int,post_label::Symbol,pre_label::Symbol)
    weights = make_weights(n_post,n_pre)
    plast = HPN.PlasticityAsymmetricSTDP(
        ETA,BIAS_BALANCE,ALPHA_PRE,ALPHA_POST,TAU_PLUS,GAMMA,weights;
        weight_min=WEIGHT_MIN,weight_max=WEIGHT_MAX)
    seed_traces!(plast.trace_pre_plus,plast.trace_post_minus)
    connection = HPN.ConnectionWithWeights(
        weights,post_label,pre_label,(plast,),true)
    return (; weights,plast,connection)
end

# Previous implementation: the complete current monolithic implementation,
# copied here to isolate it from package dispatch and connection handling.
function previous_apply!(
    state,t_fire::Float64,label::Symbol,neuron::Int)
    plast = state.plast
    weights = state.weights
    post_fired = label == state.post_label
    pre_fired = label == state.pre_label
    if !(post_fired || pre_fired)
        return nothing
    end

    mask = plast.zero_weight_mask
    if size(weights) != size(mask)
        throw(DimensionMismatch(
            "connection weights and zero-weight mask must have the same size"))
    end
    η = plast.η

    if pre_fired
        decay_post = HPN.trace_decay(t_fire,plast.trace_post_minus)
        trace_scale = η*((plast.B-1.0)/2.0)*decay_post
        rate_term = η*plast.αpre
        @inbounds for post_idx in axes(weights,1)
            if !mask[post_idx,neuron]
                weight = weights[post_idx,neuron]
                Δweight =
                    rate_term + trace_scale*plast.trace_post_minus.val[post_idx]
                weights[post_idx,neuron] = HPN.hardbounds(
                    weight+Δweight,plast.weight_min,plast.weight_max)
            end
        end
    end

    if post_fired
        decay_pre = HPN.trace_decay(t_fire,plast.trace_pre_plus)
        trace_scale = η*((plast.B+1.0)/2.0)*decay_pre
        rate_term = η*plast.αpost
        @inbounds for pre_idx in axes(weights,2)
            if !mask[neuron,pre_idx]
                weight = weights[neuron,pre_idx]
                Δweight =
                    rate_term + trace_scale*plast.trace_pre_plus.val[pre_idx]
                weights[neuron,pre_idx] = HPN.hardbounds(
                    weight+Δweight,plast.weight_min,plast.weight_max)
            end
        end
    end

    if post_fired
        HPN.propagate!(t_fire,plast.trace_post_minus)
        HPN.add_firing_event_now!(plast.trace_post_minus,neuron)
    end
    if pre_fired
        HPN.propagate!(t_fire,plast.trace_pre_plus)
        HPN.add_firing_event_now!(plast.trace_pre_plus,neuron)
    end
    return nothing
end

function typed_pre_kernel!(
    weights::Matrix{Float64},
    mask::Matrix{Bool},
    trace::Vector{Float64},
    neuron::Int,
    rate_term::Float64,
    trace_scale::Float64,
    weight_min::Float64,
    weight_max::Float64)
    @inbounds for post_idx in axes(weights,1)
        if !mask[post_idx,neuron]
            weight = weights[post_idx,neuron]
            Δweight = rate_term + trace_scale*trace[post_idx]
            weights[post_idx,neuron] =
                HPN.hardbounds(weight+Δweight,weight_min,weight_max)
        end
    end
    return nothing
end

function typed_post_kernel!(
    weights::Matrix{Float64},
    mask::Matrix{Bool},
    trace::Vector{Float64},
    neuron::Int,
    rate_term::Float64,
    trace_scale::Float64,
    weight_min::Float64,
    weight_max::Float64)
    @inbounds for pre_idx in axes(weights,2)
        if !mask[neuron,pre_idx]
            weight = weights[neuron,pre_idx]
            Δweight = rate_term + trace_scale*trace[pre_idx]
            weights[neuron,pre_idx] =
                HPN.hardbounds(weight+Δweight,weight_min,weight_max)
        end
    end
    return nothing
end

# Improved implementation: the same wrapper and trace handling as the current
# implementation, with the weight loops delegated to concretely typed kernels.
function improved_apply!(
    state,t_fire::Float64,label::Symbol,neuron::Int)
    plast = state.plast
    weights = state.weights
    post_fired = label == state.post_label
    pre_fired = label == state.pre_label
    if !(post_fired || pre_fired)
        return nothing
    end

    mask = plast.zero_weight_mask
    if size(weights) != size(mask)
        throw(DimensionMismatch(
            "connection weights and zero-weight mask must have the same size"))
    end
    η = plast.η

    if pre_fired
        decay_post = HPN.trace_decay(t_fire,plast.trace_post_minus)
        trace_scale = η*((plast.B-1.0)/2.0)*decay_post
        rate_term = η*plast.αpre
        typed_pre_kernel!(
            weights,mask,plast.trace_post_minus.val,neuron,rate_term,trace_scale,
            plast.weight_min,plast.weight_max)
    end

    if post_fired
        decay_pre = HPN.trace_decay(t_fire,plast.trace_pre_plus)
        trace_scale = η*((plast.B+1.0)/2.0)*decay_pre
        rate_term = η*plast.αpost
        typed_post_kernel!(
            weights,mask,plast.trace_pre_plus.val,neuron,rate_term,trace_scale,
            plast.weight_min,plast.weight_max)
    end

    if post_fired
        HPN.propagate!(t_fire,plast.trace_post_minus)
        HPN.add_firing_event_now!(plast.trace_post_minus,neuron)
    end
    if pre_fired
        HPN.propagate!(t_fire,plast.trace_pre_plus)
        HPN.add_firing_event_now!(plast.trace_pre_plus,neuron)
    end
    return nothing
end

function package_apply!(state,t_fire::Float64,label::Symbol,neuron::Int)
    HPN.apply_plasticity!(
        state.plast,state.connection,t_fire,label,neuron)
    return nothing
end

function validate_case(
    n_post::Int,n_pre::Int,post_label::Symbol,pre_label::Symbol,
    event_label::Symbol)
    previous = make_local_state(n_post,n_pre,post_label,pre_label)
    improved = make_local_state(n_post,n_pre,post_label,pre_label)
    package = make_package_state(n_post,n_pre,post_label,pre_label)

    for step in 0:3
        t_fire = EVENT_TIME + 0.1*step
        neuron = mod1(EVENT_INDEX+step,event_label == pre_label ? n_pre : n_post)
        previous_apply!(previous,t_fire,event_label,neuron)
        improved_apply!(improved,t_fire,event_label,neuron)
        package_apply!(package,t_fire,event_label,neuron)
    end

    @assert previous.weights ≈ improved.weights
    @assert previous.weights ≈ package.weights
    @assert previous.plast.trace_pre_plus.val ≈
        improved.plast.trace_pre_plus.val
    @assert previous.plast.trace_pre_plus.val ≈
        package.plast.trace_pre_plus.val
    @assert previous.plast.trace_post_minus.val ≈
        improved.plast.trace_post_minus.val
    @assert previous.plast.trace_post_minus.val ≈
        package.plast.trace_post_minus.val
    return nothing
end

const CASES = (
    ("pre",N_POST,N_PRE,:post,:pre,:pre),
    ("post",N_POST,N_PRE,:post,:pre,:post),
    ("self",N_SELF,N_SELF,:self,:self,:self),
)

for (_,n_post,n_pre,post_label,pre_label,event_label) in CASES
    validate_case(n_post,n_pre,post_label,pre_label,event_label)
end

const SUITE = BenchmarkGroup()

for (case_name,n_post,n_pre,post_label,pre_label,event_label) in CASES
    case_group = SUITE[case_name] = BenchmarkGroup()
    case_group["previous"] = @benchmarkable previous_apply!(
        state,EVENT_TIME,$event_label,EVENT_INDEX) setup=(
        state=make_local_state($n_post,$n_pre,$post_label,$pre_label)
    ) samples=BENCHMARK_SAMPLES evals=BENCHMARK_EVALS seconds=BENCHMARK_SECONDS
    case_group["improved"] = @benchmarkable improved_apply!(
        state,EVENT_TIME,$event_label,EVENT_INDEX) setup=(
        state=make_local_state($n_post,$n_pre,$post_label,$pre_label)
    ) samples=BENCHMARK_SAMPLES evals=BENCHMARK_EVALS seconds=BENCHMARK_SECONDS
    case_group["package"] = @benchmarkable package_apply!(
        state,EVENT_TIME,$event_label,EVENT_INDEX) setup=(
        state=make_package_state($n_post,$n_pre,$post_label,$pre_label)
    ) samples=BENCHMARK_SAMPLES evals=BENCHMARK_EVALS seconds=BENCHMARK_SECONDS
end

results = run(SUITE;verbose=true)
show(stdout,MIME("text/plain"),results)
println()

println("\nMedian time and allocations:")
for (case_name,_,_,_,_,_) in CASES
    for variant in ("previous","improved","package")
        estimate = median(results[case_name][variant])
        println(
            rpad("$case_name/$variant",22),
            lpad(string(estimate.time," ns"),14),
            lpad(string(estimate.memory," bytes"),14),
            lpad(string(estimate.allocs," allocs"),12))
    end
end
