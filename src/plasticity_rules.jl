# Monosynaptic plasticity rules follow post <- pre matrix ordering.

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


function _apply_asymmetric_stdp_pre!(
    weights::Matrix{Float64},
    mask::Matrix{Bool},
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

function _apply_asymmetric_stdp_post!(
    weights::Matrix{Float64},
    mask::Matrix{Bool},
    trace::Vector{Float64},
    neuron_fire_idx::Int,
    η::Float64,
    α::Float64,
    trace_scale::Float64,
    weight_min::Float64,
    weight_max::Float64)
  @inbounds for pre_idx in axes(weights,2)
    if !mask[neuron_fire_idx,pre_idx]
      weight = weights[neuron_fire_idx,pre_idx]
      Δweight = η*(α + trace_scale*trace[pre_idx])
      weights[neuron_fire_idx,pre_idx] =
        hardbounds(weight+Δweight,weight_min,weight_max)
    end
  end
  return nothing
end

function apply_plasticity!(
    plast::PlasticityAsymmetricSTDP,
    connection::ConnectionWithWeights,
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
    _apply_asymmetric_stdp_pre!(
      weights,mask,plast.trace_post_minus.val,neuron_fire_idx,η,plast.αpre,
      trace_scale,plast.weight_min,plast.weight_max)
  end

  if post_fired
    decay_pre = trace_decay(t_fire,plast.trace_pre_plus)
    trace_scale = ((plast.B+1.0)/2.0)*decay_pre
    _apply_asymmetric_stdp_post!(
      weights,mask,plast.trace_pre_plus.val,neuron_fire_idx,η,plast.αpost,
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


function _apply_symmetric_stdp_pre!(
    weights::Matrix{Float64},
    mask::Matrix{Bool},
    trace_plus::Vector{Float64},
    trace_minus::Vector{Float64},
    neuron_fire_idx::Int,
    η::Float64,
    α::Float64,
    trace_scale_plus::Float64,
    trace_scale_minus::Float64,
    weight_min::Float64,
    weight_max::Float64)
  @inbounds for post_idx in axes(weights,1)
    if !mask[post_idx,neuron_fire_idx]
      weight = weights[post_idx,neuron_fire_idx]
      Δweight = η*(α +
        trace_scale_plus*trace_plus[post_idx] +
        trace_scale_minus*trace_minus[post_idx])
      weights[post_idx,neuron_fire_idx] =
        hardbounds(weight+Δweight,weight_min,weight_max)
    end
  end
  return nothing
end

function _apply_symmetric_stdp_post!(
    weights::Matrix{Float64},
    mask::Matrix{Bool},
    trace_plus::Vector{Float64},
    trace_minus::Vector{Float64},
    neuron_fire_idx::Int,
    η::Float64,
    α::Float64,
    trace_scale_plus::Float64,
    trace_scale_minus::Float64,
    weight_min::Float64,
    weight_max::Float64)
  @inbounds for pre_idx in axes(weights,2)
    if !mask[neuron_fire_idx,pre_idx]
      weight = weights[neuron_fire_idx,pre_idx]
      Δweight = η*(α +
        trace_scale_plus*trace_plus[pre_idx] +
        trace_scale_minus*trace_minus[pre_idx])
      weights[neuron_fire_idx,pre_idx] =
        hardbounds(weight+Δweight,weight_min,weight_max)
    end
  end
  return nothing
end

function apply_plasticity!(
    plast::PlasticitySymmetricSTDP,
    connection::ConnectionWithWeights,
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
    _apply_symmetric_stdp_pre!(
      weights,mask,plast.trace_post_plus.val,plast.trace_post_minus.val,
      neuron_fire_idx,η,plast.αpre,trace_scale_plus,trace_scale_minus,
      plast.weight_min,plast.weight_max)
  end

  if post_fired
    decay_plus = trace_decay(t_fire,plast.trace_pre_plus)
    decay_minus = trace_decay(t_fire,plast.trace_pre_minus)
    trace_scale_plus = ((plast.B+1.0)/2.0)*decay_plus
    trace_scale_minus = ((plast.B-1.0)/2.0)*decay_minus
    _apply_symmetric_stdp_post!(
      weights,mask,plast.trace_pre_plus.val,plast.trace_pre_minus.val,
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
