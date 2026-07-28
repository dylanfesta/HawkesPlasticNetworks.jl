# Monosynaptic plasticity rules follow post <- pre matrix ordering.

abstract type AbstractPlasticitySTDP <: AbstractPlasticityRule end

"""
    PlasticityAsymmetricSTDP(
        η, B, αpre, αpost, τ_plus, γ, weights;
        weight_min=0.0, weight_max=Inf)

Construct an asymmetric pair-based STDP rule for `weights`, whose shape is
`n_post × n_pre`. Positions that are exactly zero at construction are treated
as structural zeros and are not changed by plasticity. Use
[`refresh_mask!`](@ref) after changing that topology externally.

`B` controls the balance between potentiation and depression:
`A_plus = (B + 1) / 2` and `A_minus = (B - 1) / 2`.
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
  -1 <= B <= 1 || throw(ArgumentError("B must be between -1 and 1"))
  τ_plus > 0 || throw(ArgumentError("τ_plus must be positive"))
  γ > 0 || throw(ArgumentError("γ must be positive"))
  weight_min <= weight_max ||
    throw(ArgumentError("weight_min must not exceed weight_max"))

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

"""
    refresh_mask!(plasticity, weights)

Replace the structural-zero mask with the positions that are exactly zero in
`weights`. The matrix shape must match the shape used to construct the rule.
"""
function refresh_mask!(
    plast::PlasticityAsymmetricSTDP,weights::Matrix{Float64})
  size(weights) == size(plast.zero_weight_mask) ||
    throw(DimensionMismatch("weights and zero-weight mask must have the same size"))
  @inbounds for idx in eachindex(weights,plast.zero_weight_mask)
    plast.zero_weight_mask[idx] = iszero(weights[idx])
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
  (post_fired || pre_fired) || return nothing

  weights = connection.weights
  mask = plast.zero_weight_mask
  size(weights) == size(mask) ||
    throw(DimensionMismatch("connection weights and zero-weight mask must have the same size"))
  η = plast.η

  # Read the traces lazily. The trace vectors are advanced only below, when
  # the current spike is written into them.
  if pre_fired
    decay_post = trace_decay(t_fire,plast.trace_post_minus)
    trace_scale = η*((plast.B-1.0)/2.0)*decay_post
    bias = η*plast.αpre
    @inbounds for post_idx in axes(weights,1)
      if !mask[post_idx,neuron_fire_idx]
        weight = weights[post_idx,neuron_fire_idx]
        Δweight = bias + trace_scale*plast.trace_post_minus.val[post_idx]
        weights[post_idx,neuron_fire_idx] =
          hardbounds(weight+Δweight,plast.weight_min,plast.weight_max)
      end
    end
  end

  if post_fired
    decay_pre = trace_decay(t_fire,plast.trace_pre_plus)
    trace_scale = η*((plast.B+1.0)/2.0)*decay_pre
    bias = η*plast.αpost
    @inbounds for pre_idx in axes(weights,2)
      if !mask[neuron_fire_idx,pre_idx]
        weight = weights[neuron_fire_idx,pre_idx]
        Δweight = bias + trace_scale*plast.trace_pre_plus.val[pre_idx]
        weights[neuron_fire_idx,pre_idx] =
          hardbounds(weight+Δweight,plast.weight_min,plast.weight_max)
      end
    end
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


struct PlasticitySTDPSymmetric <: AbstractPlasticityRule
  θ::Float64
  αpre::Float64
  αpost::Float64
  τ_plus::Float64
  γ::Float64  # τ_minus = γ*τ_plus
  weight_min::Float64
  weight_max::Float64
  trace_pre_plus::Trace
  trace_pre_minus::Trace
  trace_post_plus::Trace
  trace_post_minus::Trace
end

