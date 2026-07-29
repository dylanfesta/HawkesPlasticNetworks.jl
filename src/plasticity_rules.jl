#=
This is a sub-section of HawkesPlasticNetworks.jl for
monosynaptic plasticity rules

Monosynaptic plasticity rules follow post <- pre matrix ordering,
and act on a single synapse depending on pre/post spike times.

General signature

@inline function apply_plasticity!(
     plasticity::AbstractPlasticityRule,
     connection::AbstractConnection,
     t_now::Real,
     pop_fire_idx::Integer,
     pop_fire_label::Symbol,
     neuron_fire_idx::Integer)
  return nothing
end
=#





abstract type AbstractPlasticitySTDP <: AbstractPlasticityRule end

"""
    refresh_mask!(plasticity, weights)

Replace the structural-zero mask with the positions that are exactly zero in
`weights`. The matrix shape must match the shape used to construct the rule.
"""
function refresh_mask!(
    plast::AbstractPlasticitySTDP,weights::Matrix{Float64})
  if size(weights) != size(plast.zero_weight_mask)
    throw(DimensionMismatch("weights and zero-weight mask must have the same size"))
  end
  @inbounds for idx in eachindex(weights,plast.zero_weight_mask)
    plast.zero_weight_mask[idx] = iszero(weights[idx])
  end
  return nothing
end

#=
Helper function to apply STDP additively
with a single trace, either positive or negative.

It updates the weight according to:

Δw(t) = η (α + trace_scale*trace(t))
w = w + w*Δw(t)

The kernel updates column neuron_fire_idx of the provided matrices.
Pass transposed matrices to update a row of the original matrices.
Masked weights are unchanged. Updated weights are restricted to
the interval [weight_min,weight_max].
=#
function _apply_stdp_single_trace!(
    weights::AbstractMatrix{Float64},
    mask::AbstractMatrix{Bool},
    trace::Vector{Float64},
    neuron_fire_idx::Int,
    η::Float64,
    α::Float64,
    trace_scale::Float64,
    weight_min::Float64,
    weight_max::Float64)
  @inbounds for post_idx in axes(weights,1)
    if !mask[post_idx,neuron_fire_idx]
      weight = weights[post_idx,neuron_fire_idx]
      Δweight = η*(α + trace_scale*trace[post_idx])
      weights[post_idx,neuron_fire_idx] =
        hardbounds(weight+Δweight,weight_min,weight_max)
    end
  end
  return nothing
end

#=
Helper function to apply STDP additively
with two traces, one positive and one negative.

It updates the weight according to:

Δw(t) = η (α + trace_scale_1*trace_1(t) +
                trace_scale_2*trace_2(t))
w = w + w*Δw(t)

The kernel updates column neuron_fire_idx of the provided matrices.
Pass transposed matrices to update a row of the original matrices.
Updated weights are restricted to
the interval [weight_min,weight_max].
=#
function _apply_stdp_two_traces!(
    weights::AbstractMatrix{Float64},
    mask::AbstractMatrix{Bool},
    trace_1::Vector{Float64},
    trace_2::Vector{Float64},
    neuron_fire_idx::Int,
    η::Float64,
    α::Float64,
    trace_scale_1::Float64,
    trace_scale_2::Float64,
    weight_min::Float64,
    weight_max::Float64)
  @inbounds for post_idx in axes(weights,1)
    if !mask[post_idx,neuron_fire_idx]
      weight = weights[post_idx,neuron_fire_idx]
      Δweight = η*(α +
        trace_scale_1*trace_1[post_idx] +
        trace_scale_2*trace_2[post_idx])
      weights[post_idx,neuron_fire_idx] =
        hardbounds(weight+Δweight,weight_min,weight_max)
    end
  end
  return nothing
end



"""
    PlasticityVogelsSprekeler(
        η::Real, r_target::Real,
        τ::Real,
        weights::Matrix{Float64};
        weight_min::Real=0.0, weight_max::Real=Inf)

Constructs the symmetric Vogels-Spreckeler inhibitory homeostatic rule
for `weights`. Positions that are exactly zero at construction are not
changed by plasticity. Use [`refresh_mask!`](@ref)
to update the plastic synapses.

`η` is the global learning rate: it scales all terms of plasticity

`r_target` is the postsynaptic target firing rate

`τ` sets the time constant

"""
struct PlasticityVogelsSprekeler{
    TP<:Trace,TM<:Trace} <: AbstractPlasticitySTDP
  η::Float64
  r_target::Float64
  τ::Float64
  weight_min::Float64
  weight_max::Float64
  trace_pre_plus::TP
  trace_post_plus::TM
  zero_weight_mask::Matrix{Bool}
end

function PlasticityVogelsSprekeler(
    η::Real, r_target::Real, τ::Real,
    weights::Matrix{Float64};
    weight_min::Real=0.0,weight_max::Real=Inf)
  if !(τ > 0)
    throw(ArgumentError("τ must be positive"))
  end
  if !(r_target > 0)
    throw(ArgumentError("r_target must be positive"))
  end
  if !(weight_min <= weight_max)
    throw(ArgumentError("weight_min must not exceed weight_max"))
  end
  n_post,n_pre = size(weights)
  zero_weight_mask = Matrix{Bool}(undef,n_post,n_pre)
  @inbounds for idx in eachindex(weights,zero_weight_mask)
    zero_weight_mask[idx] = iszero(weights[idx])
  end
  trace_pre = Trace(Float64(τ),n_pre)
  trace_post = Trace(Float64(τ),n_post)

  return PlasticityVogelsSprekeler(
    Float64(η),Float64(r_target),Float64(τ),
    Float64(weight_min),Float64(weight_max),
    trace_pre,trace_post,zero_weight_mask)
end

function reset!(plast::PlasticityVogelsSprekeler)
  reset!(plast.trace_pre_plus)
  reset!(plast.trace_post_plus)
  return nothing
end


function apply_plasticity!(
    plast::PlasticityVogelsSprekeler,
    connection::AbstractConnectionWithWeights,
    t_fire::Real,
    pop_fire_label::Symbol,
    neuron_fire_idx::Integer)
  post_fired = pop_fire_label == connection.post_pop_label
  pre_fired = pop_fire_label == connection.pre_pop_label
  if !(post_fired || pre_fired)
    return nothing
  end

  α = - plast.r_target
  weights = connection.weights
  mask = plast.zero_weight_mask
  if size(weights) != size(mask)
    throw(DimensionMismatch("connection weights and zero-weight mask must have the same size"))
  end
  η = plast.η

  # Read the traces lazily. The trace vectors are advanced only below, when
  # the current spike is written into them.
  if pre_fired
    decay_plus = trace_decay(t_fire,plast.trace_post_plus)
    trace_scale_plus = 0.5*decay_plus # 0.5 factor so that B = 1
    _apply_stdp_single_trace!(
      weights,mask,plast.trace_post_plus.val,
      neuron_fire_idx,η,α,trace_scale_plus,
      plast.weight_min,plast.weight_max)
  end

  if post_fired
    decay_plus = trace_decay(t_fire,plast.trace_pre_plus)
    trace_scale_plus = 0.5*decay_plus # 0.5 factor so that B = 1
    _apply_stdp_single_trace!(
      transpose(weights),transpose(mask),plast.trace_pre_plus.val,
      neuron_fire_idx,η,α,trace_scale_plus,
      plast.weight_min,plast.weight_max)
  end

  if post_fired
    propagate!(t_fire,plast.trace_post_plus)
    add_firing_event_now!(plast.trace_post_plus,neuron_fire_idx)
  end
  if pre_fired
    propagate!(t_fire,plast.trace_pre_plus)
    add_firing_event_now!(plast.trace_pre_plus,neuron_fire_idx)
  end
  return nothing
end


"""
    PlasticityAsymmetricSTDP(
        η::Real, B::Real, αpre::Real, αpost::Real,
        τ_plus::Real, γ::Real, weights::Matrix{Float64};
        weight_min::Real=0.0, weight_max::Real=Inf)

Construct an asymmetric pair-based STDP rule for `weights`, whose shape is
`n_post × n_pre`. Positions that are exactly zero at construction are not 
changed by plasticity. Use [`refresh_mask!`](@ref) 
to update the plastic synapses.

`η` is the global learning rate: it scales all terms of plasticity

`B` controls the balance between potentiation and depression,
it must be in the range [-1,1], and is defined as
- B = 1 : pure potentiation
- B = -1 : pure depression
- B = 0 : symmetric potentiation and depression

`αpre` and `αpost` are the rate constants, applied at 
each pre- and post-synaptic spike, respectively.

`τ_plus` sets the time constant for potentiation.

`γ` sets the relative time constant for depression `γ τ_plus`.

"""
struct PlasticityAsymmetricSTDP{
    TP<:Trace,TM<:Trace} <: AbstractPlasticitySTDP
  η::Float64
  B::Float64
  αpre::Float64
  αpost::Float64
  τ_plus::Float64
  γ::Float64
  weight_min::Float64
  weight_max::Float64
  trace_pre_plus::TP
  trace_post_minus::TM
  zero_weight_mask::Matrix{Bool}
end

function PlasticityAsymmetricSTDP(
    η::Real,B::Real,αpre::Real,αpost::Real,τ_plus::Real,γ::Real,
    weights::Matrix{Float64};
    weight_min::Real=0.0,weight_max::Real=Inf)
  if !(-1 <= B <= 1)
    throw(ArgumentError("B must be between -1 and 1"))
  end
  if !(τ_plus > 0)
    throw(ArgumentError("τ_plus must be positive"))
  end
  if !(γ > 0)
    throw(ArgumentError("γ must be positive"))
  end
  if !(η >= 0)
    throw(ArgumentError("η must be non-negative"))
  end
  if !(weight_min <= weight_max)
    throw(ArgumentError("weight_min must not exceed weight_max"))
  end

  n_post,n_pre = size(weights)
  zero_weight_mask = Matrix{Bool}(undef,n_post,n_pre)
  @inbounds for idx in eachindex(weights,zero_weight_mask)
    zero_weight_mask[idx] = iszero(weights[idx])
  end

  return PlasticityAsymmetricSTDP(
    Float64(η),Float64(B),Float64(αpre),Float64(αpost),
    Float64(τ_plus),Float64(γ),Float64(weight_min),Float64(weight_max),
    Trace(Float64(τ_plus),n_pre),
    Trace(Float64(γ*τ_plus),n_post),
    zero_weight_mask)
end

function reset!(plast::PlasticityAsymmetricSTDP)
  reset!(plast.trace_pre_plus)
  reset!(plast.trace_post_minus)
  return nothing
end


function apply_plasticity!(
    plast::PlasticityAsymmetricSTDP,
    connection::AbstractConnectionWithWeights,
    t_fire::Real,
    pop_fire_label::Symbol,
    neuron_fire_idx::Integer)
  post_fired = pop_fire_label == connection.post_pop_label
  pre_fired = pop_fire_label == connection.pre_pop_label
  if !(post_fired || pre_fired)
    return nothing
  end

  weights = connection.weights
  mask = plast.zero_weight_mask
  if size(weights) != size(mask)
    throw(DimensionMismatch("connection weights and zero-weight mask must have the same size"))
  end
  η = plast.η

  # Read the traces lazily. The trace vectors are advanced only below, when
  # the current spike is written into them.
  if pre_fired
    decay_post = trace_decay(t_fire,plast.trace_post_minus)
    trace_scale = ((plast.B-1.0)/2.0)*decay_post
    _apply_stdp_single_trace!(
      weights,mask,plast.trace_post_minus.val,neuron_fire_idx,η,plast.αpre,
      trace_scale,plast.weight_min,plast.weight_max)
  end

  if post_fired
    decay_pre = trace_decay(t_fire,plast.trace_pre_plus)
    trace_scale = ((plast.B+1.0)/2.0)*decay_pre
    _apply_stdp_single_trace!(
      transpose(weights),transpose(mask),plast.trace_pre_plus.val,
      neuron_fire_idx,η,plast.αpost,
      trace_scale,plast.weight_min,plast.weight_max)
  end

  # Both weight directions above see the traces before the current event.
  if post_fired
    propagate!(t_fire,plast.trace_post_minus)
    add_firing_event_now!(plast.trace_post_minus,neuron_fire_idx)
  end
  if pre_fired
    propagate!(t_fire,plast.trace_pre_plus)
    add_firing_event_now!(plast.trace_pre_plus,neuron_fire_idx)
  end
  return nothing
end


"""
    PlasticitySymmetricSTDP(
        η::Real, B::Real, αpre::Real, αpost::Real,
        τ_plus::Real, γ::Real, weights::Matrix{Float64};
        weight_min::Real=0.0, weight_max::Real=Inf)

Construct a symmetric STDP rule for `weights`, whose shape is
`n_post × n_pre`. Positions that are exactly zero at construction are not
changed by plasticity. Use [`refresh_mask!`](@ref) to update the plastic synapses.

The rule stores positive and negative traces for both populations. A
presynaptic spike combines the two postsynaptic traces, while a postsynaptic
spike combines the two presynaptic traces:

```math
\\Delta w_{ij}^{\\mathrm{pre}} =
\\eta\\left[\\alpha_{\\mathrm{pre}} +
A_+ x_{i,\\mathrm{post}}^+ + A_- x_{i,\\mathrm{post}}^-\\right],
```

```math
\\Delta w_{ij}^{\\mathrm{post}} =
\\eta\\left[\\alpha_{\\mathrm{post}} +
A_+ x_{j,\\mathrm{pre}}^+ + A_- x_{j,\\mathrm{pre}}^-\\right],
```

where `i` is the postsynaptic index, `j` is the presynaptic index,
`A_plus = (B + 1) / 2`, and `A_minus = (B - 1) / 2`.

`η` is the global learning rate: it scales all terms of plasticity.

`B` controls the balance between potentiation and depression. It must be in
the range `[-1, 1]`:

- `B = 1`: pure potentiation
- `B = -1`: pure depression
- `B = 0`: balanced potentiation and depression

`αpre` and `αpost` are the rate constants applied at each pre- and
postsynaptic spike, respectively.

`τ_plus` sets the time constant of the positive traces.

`γ` sets the relative time constant of the negative traces, `γ τ_plus`.

Updated weights are restricted to the interval `[weight_min, weight_max]`.
"""
struct PlasticitySymmetricSTDP <: AbstractPlasticitySTDP
  η::Float64
  B::Float64
  αpre::Float64
  αpost::Float64
  τ_plus::Float64
  γ::Float64
  weight_min::Float64
  weight_max::Float64
  trace_pre_plus::Trace
  trace_pre_minus::Trace
  trace_post_plus::Trace
  trace_post_minus::Trace
  zero_weight_mask::Matrix{Bool}
end

function PlasticitySymmetricSTDP(
    η::Real,B::Real,αpre::Real,αpost::Real,τ_plus::Real,γ::Real,
    weights::Matrix{Float64};
    weight_min::Real=0.0,weight_max::Real=Inf)
  if !(-1 <= B <= 1)
    throw(ArgumentError("B must be between -1 and 1"))
  end
  if !(τ_plus > 0)
    throw(ArgumentError("τ_plus must be positive"))
  end
  if !(γ > 0)
    throw(ArgumentError("γ must be positive"))
  end
  if !(weight_min <= weight_max)
    throw(ArgumentError("weight_min must not exceed weight_max"))
  end

  n_post,n_pre = size(weights)
  zero_weight_mask = Matrix{Bool}(undef,n_post,n_pre)
  @inbounds for idx in eachindex(weights,zero_weight_mask)
    zero_weight_mask[idx] = iszero(weights[idx])
  end
  trace_pre_plus = Trace(Float64(τ_plus),n_pre)
  trace_pre_minus = Trace(Float64(γ*τ_plus),n_pre)
  trace_post_plus = Trace(Float64(τ_plus),n_post)
  trace_post_minus = Trace(Float64(γ*τ_plus),n_post)

  return PlasticitySymmetricSTDP(
    Float64(η),Float64(B),Float64(αpre),Float64(αpost),
    Float64(τ_plus),Float64(γ),Float64(weight_min),Float64(weight_max),
    trace_pre_plus,trace_pre_minus,trace_post_plus,trace_post_minus,
    zero_weight_mask)
end

function reset!(plast::PlasticitySymmetricSTDP)
  reset!(plast.trace_pre_plus)
  reset!(plast.trace_pre_minus)
  reset!(plast.trace_post_plus)
  reset!(plast.trace_post_minus)
  return nothing
end


function apply_plasticity!(
    plast::PlasticitySymmetricSTDP,
    connection::AbstractConnectionWithWeights,
    t_fire::Real,
    pop_fire_label::Symbol,
    neuron_fire_idx::Integer)
  post_fired = pop_fire_label == connection.post_pop_label
  pre_fired = pop_fire_label == connection.pre_pop_label
  if !(post_fired || pre_fired)
    return nothing
  end

  weights = connection.weights
  mask = plast.zero_weight_mask
  if size(weights) != size(mask)
    throw(DimensionMismatch("connection weights and zero-weight mask must have the same size"))
  end
  η = plast.η

  # Read the traces lazily. The trace vectors are advanced only below, when
  # the current spike is written into them.
  if pre_fired
    decay_plus = trace_decay(t_fire,plast.trace_post_plus)
    decay_minus = trace_decay(t_fire,plast.trace_post_minus)
    trace_scale_plus = ((plast.B+1.0)/2.0)*decay_plus
    trace_scale_minus = ((plast.B-1.0)/2.0)*decay_minus
    _apply_stdp_two_traces!(
      weights,mask,plast.trace_post_plus.val,plast.trace_post_minus.val,
      neuron_fire_idx,η,plast.αpre,trace_scale_plus,trace_scale_minus,
      plast.weight_min,plast.weight_max)
  end

  if post_fired
    decay_plus = trace_decay(t_fire,plast.trace_pre_plus)
    decay_minus = trace_decay(t_fire,plast.trace_pre_minus)
    trace_scale_plus = ((plast.B+1.0)/2.0)*decay_plus
    trace_scale_minus = ((plast.B-1.0)/2.0)*decay_minus
    _apply_stdp_two_traces!(
      transpose(weights),transpose(mask),
      plast.trace_pre_plus.val,plast.trace_pre_minus.val,
      neuron_fire_idx,η,plast.αpost,trace_scale_plus,trace_scale_minus,
      plast.weight_min,plast.weight_max)
  end

  if post_fired
    propagate!(t_fire,plast.trace_post_plus)
    propagate!(t_fire,plast.trace_post_minus)
    add_firing_event_now!(plast.trace_post_plus,neuron_fire_idx)
    add_firing_event_now!(plast.trace_post_minus,neuron_fire_idx)
  end
  if pre_fired
    propagate!(t_fire,plast.trace_pre_plus)
    propagate!(t_fire,plast.trace_pre_minus)
    add_firing_event_now!(plast.trace_pre_plus,neuron_fire_idx)
    add_firing_event_now!(plast.trace_pre_minus,neuron_fire_idx)
  end
  return nothing
end


"""
    PlasticityHomeostaticScaling(
        η::Real,
        α::Real,
        s::Real,
        τ::Real,
        weights::Matrix{Float64};
        weight_min::Real=0.0, weight_max::Real=Inf)

Constructs the homeostatic scaling rule (STDHS)
for `weights`. Positions that are exactly zero at construction are not
changed by plasticity. Use [`refresh_mask!`](@ref)
to update the plastic synapses.

`η` is the global learning rate: it scales all terms of plasticity

`α` is the target rate in isolation.

`s` should be +1 for inhibitory populations and -1 for excitatory populations.

`τ` sets the time constant

"""
struct PlasticityHomeostaticScaling{
    TP<:Trace,TM<:Trace} <: AbstractPlasticitySTDP
  η::Float64
  α::Float64
  s::Float64
  τ::Float64
  weight_min::Float64
  weight_max::Float64
  trace_post::TP
  zero_weight_mask::Matrix{Bool}
end

function PlasticityHomeostaticScaling(
    η::Real, α::Real, s::Real, τ::Real,
    weights::Matrix{Float64};
    weight_min::Real=0.0,weight_max::Real=Inf)
  if !(τ > 0)
    throw(ArgumentError("τ must be positive"))
  end
  if !(α > 0)
    throw(ArgumentError("α must be positive"))
  end
  if !(s == 1 || s == -1)
    throw(ArgumentError("s must be 1 or -1"))
  end
  if !(weight_min <= weight_max)
    throw(ArgumentError("weight_min must not exceed weight_max"))
  end
  n_post,n_pre = size(weights)
  zero_weight_mask = Matrix{Bool}(undef,n_post,n_pre)
  @inbounds for idx in eachindex(weights,zero_weight_mask)
    zero_weight_mask[idx] = iszero(weights[idx])
  end
  trace_post = Trace(Float64(τ),n_post)

  return PlasticityHomeostaticScaling(
    Float64(η),Float64(α),Float64(s),Float64(τ),
    Float64(weight_min),Float64(weight_max),
    trace_post,zero_weight_mask)
end



#=
Helper function to apply STDHS multiplicatively
with a single trace.

It updates the weight according to:

Δw(t) = η (α + trace_scale*trace(t))
w = w + w*Δw(t)

The kernel updates column neuron_fire_idx of the provided matrices.
Pass transposed matrices to update a row of the original matrices.
Masked weights are unchanged. Updated weights are restricted to
the interval [weight_min,weight_max].
=#
function _apply_stdhs_single_trace!(
    weights::AbstractMatrix{Float64},
    mask::AbstractMatrix{Bool},
    trace::Vector{Float64},
    neuron_fire_idx::Int,
    η::Float64,
    α::Float64,
    trace_scale::Float64,
    weight_min::Float64,
    weight_max::Float64)
  @inbounds for post_idx in axes(weights,1)
    if !mask[post_idx,neuron_fire_idx]
      weight = weights[post_idx,neuron_fire_idx]
      Δweight = η*(α + trace_scale*trace[post_idx])
      weights[post_idx,neuron_fire_idx] =
        hardbounds(weight+weight*Δweight,weight_min,weight_max) # the only change vs additive case is here!
    end
  end
  return nothing
end


function apply_plasticity!(
    plast::PlasticityHomeostaticScaling,
    connection::AbstractConnectionWithWeights,
    t_fire::Real,
    pop_fire_label::Symbol,
    neuron_fire_idx::Integer)
  post_fired = pop_fire_label == connection.post_pop_label
  if !(post_fired)
    return nothing
  end

  weights = connection.weights
  mask = plast.zero_weight_mask
  if size(weights) != size(mask)
    throw(DimensionMismatch("connection weights and zero-weight mask must have the same size"))
  end

  decay = trace_decay(t_fire,plast.trace_post)
  trace_scale = decay
  _apply_stdhs_single_trace!(
      transpose(weights),transpose(mask),
      plast.trace_post.val,
      neuron_fire_idx,plast.η,plast.α,trace_scale,
      plast.weight_min,plast.weight_max)
  propagate!(t_fire,plast.trace_post)
  add_firing_event_now!(plast.trace_post,neuron_fire_idx)

  return nothing
end
