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

function make_benchmark_plasticity(weights::Matrix{Float64})
    n_post,n_pre = size(weights)
    mask = Matrix{Bool}(undef,n_post,n_pre)
    @inbounds for idx in eachindex(weights,mask)
        mask[idx] = iszero(weights[idx])
    end
    return BenchmarkPlasticity(
        ETA,BIAS_BALANCE,ALPHA_PRE,ALPHA_POST,WEIGHT_MIN,WEIGHT_MAX,
        HPN.Trace(TAU_PLUS,n_pre),HPN.Trace(GAMMA*TAU_PLUS,n_post),mask)
end

function make_local_state(n_post::Int,n_pre::Int,post_label::Symbol,pre_label::Symbol)
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
    connection = HPN.ConnectionWithWeights(
        weights,post_label,pre_label,(plast,),true)
    return (; plast,connection)
end

# Previous implementation: eagerly materialize the trace used by each weight
# update, then propagate the target trace again when writing the event.
function previous_apply!(
    state,t_fire::Float64,label::Symbol,neuron::Int)
    plast = state.plast
    weights = state.weights
    post_fired = label == state.post_label
    pre_fired = label == state.pre_label
    if !(post_fired || pre_fired)
        return nothing
    end

    if pre_fired
        HPN.propagate!(t_fire,plast.trace_post_minus)
        scale = plast.η*((plast.B-1.0)/2.0)
        rate_term = plast.η*plast.αpre
        @inbounds for post_idx in axes(weights,1)
            if !plast.zero_weight_mask[post_idx,neuron]
                weight = weights[post_idx,neuron]
                Δweight = rate_term + scale*plast.trace_post_minus.val[post_idx]
                weights[post_idx,neuron] = HPN.hardbounds(
                    weight+Δweight,plast.weight_min,plast.weight_max)
            end
        end
    end
    if post_fired
        HPN.propagate!(t_fire,plast.trace_pre_plus)
        scale = plast.η*((plast.B+1.0)/2.0)
        rate_term = plast.η*plast.αpost
        @inbounds for pre_idx in axes(weights,2)
            if !plast.zero_weight_mask[neuron,pre_idx]
                weight = weights[neuron,pre_idx]
                Δweight = rate_term + scale*plast.trace_pre_plus.val[pre_idx]
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

# Proposed implementation copied here so its loop cost can be separated from
# package dispatch and connection handling.
function improved_apply!(
    state,t_fire::Float64,label::Symbol,neuron::Int)
    plast = state.plast
    weights = state.weights
    post_fired = label == state.post_label
    pre_fired = label == state.pre_label
    if !(post_fired || pre_fired)
        return nothing
    end

    if pre_fired
        decay = HPN.trace_decay(t_fire,plast.trace_post_minus)
        scale = plast.η*((plast.B-1.0)/2.0)*decay
        rate_term = plast.η*plast.αpre
        @inbounds for post_idx in axes(weights,1)
            if !plast.zero_weight_mask[post_idx,neuron]
                weight = weights[post_idx,neuron]
                Δweight = rate_term + scale*plast.trace_post_minus.val[post_idx]
                weights[post_idx,neuron] = HPN.hardbounds(
                    weight+Δweight,plast.weight_min,plast.weight_max)
            end
        end
    end
    if post_fired
        decay = HPN.trace_decay(t_fire,plast.trace_pre_plus)
        scale = plast.η*((plast.B+1.0)/2.0)*decay
        rate_term = plast.η*plast.αpost
        @inbounds for pre_idx in axes(weights,2)
            if !plast.zero_weight_mask[neuron,pre_idx]
                weight = weights[neuron,pre_idx]
                Δweight = rate_term + scale*plast.trace_pre_plus.val[pre_idx]
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

function package_apply!(state,t_fire::Float64,label::Symbol,neuron::Int)
    HPN.apply_plasticity!(
        state.plast,state.connection,t_fire,label,neuron)
    return nothing
end

const SUITE = BenchmarkGroup()

for (case_name,n_post,n_pre,post_label,pre_label,event_label) in (
    ("pre",N_POST,N_PRE,:post,:pre,:pre),
    ("post",N_POST,N_PRE,:post,:pre,:post),
    ("self",N_SELF,N_SELF,:self,:self,:self),
)
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
