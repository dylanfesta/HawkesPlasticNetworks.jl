#=
This is a sub-section of HawkesPlasticNetworks.jl for
heterosynaptic plasticity rules.

Weights follow post <- pre ordering. Rows therefore contain incoming weights
and columns contain outgoing weights.
=#

abstract type HeterosynapticTarget end

struct HeterosynapticIncoming <: HeterosynapticTarget end
struct HeterosynapticOutgoing <: HeterosynapticTarget end
struct HeterosynapticBoth <: HeterosynapticTarget end

abstract type HeterosynapticMethod end

struct HeterosynapticSubtractive <: HeterosynapticMethod end
struct HeterosynapticDivisive <: HeterosynapticMethod end

"""
    PlasticityHeterosynapticNormalization(
        weight_sum_target::Real,
        tolerance::Real,
        Δt::Real,
        weights::Matrix{Float64};
        target::HeterosynapticTarget,
        method::HeterosynapticMethod,
        weight_min::Real=0.0,
        weight_max::Real=Inf)

Construct a rule that periodically restricts sums of incoming rows, outgoing
columns, or both. `HeterosynapticBoth()` normalizes incoming rows first and
outgoing columns second. A group is changed only when its sum exceeds
`weight_sum_target + tolerance`.

`HeterosynapticSubtractive()` subtracts the excess equally from all plastic
weights in the group. `HeterosynapticDivisive()` scales all plastic weights in
the group by `weight_sum_target/sum`. Each changed weight is finally restricted
to `[weight_min,weight_max]`, so the resulting sum may differ from the target.

Positions that are exactly zero at construction are not changed and do not
contribute to the sums. Use [`refresh_mask!`](@ref) to update the plastic
synapses. The first update occurs at the first plasticity callback at or after
`Δt`.
"""
mutable struct PlasticityHeterosynapticNormalization{
    HT<:HeterosynapticTarget,
    HM<:HeterosynapticMethod} <: AbstractPlasticityRule
  weight_sum_target::Float64
  tolerance::Float64
  Δt::Float64
  weight_min::Float64
  weight_max::Float64
  target::HT
  method::HM
  t_last::Float64
  zero_weight_mask::Matrix{Bool}
  group_workspace::Vector{Float64}
  active_counts::Vector{Int}
  groups_to_normalize::Vector{Bool}
end

@inline function _heterosynaptic_group_count(
    weights::AbstractMatrix,::HeterosynapticIncoming)
  return size(weights,1)
end

@inline function _heterosynaptic_group_count(
    weights::AbstractMatrix,::HeterosynapticOutgoing)
  return size(weights,2)
end

@inline function _heterosynaptic_group_count(
    weights::AbstractMatrix,::HeterosynapticBoth)
  return max(size(weights,1),size(weights,2))
end

@inline function _heterosynaptic_group_index(
    post_idx::Int,::Int,::HeterosynapticIncoming)
  return post_idx
end

@inline function _heterosynaptic_group_index(
    ::Int,pre_idx::Int,::HeterosynapticOutgoing)
  return pre_idx
end

function _count_heterosynaptic_active!(
    counts::Vector{Int},mask::Matrix{Bool},target::HeterosynapticTarget)
  fill!(counts,0)
  @inbounds for pre_idx in axes(mask,2)
    for post_idx in axes(mask,1)
      if !mask[post_idx,pre_idx]
        group_idx = _heterosynaptic_group_index(post_idx,pre_idx,target)
        counts[group_idx] += 1
      end
    end
  end
  return nothing
end

function _count_heterosynaptic_active!(
    counts::Vector{Int},mask::Matrix{Bool},::HeterosynapticBoth)
  _count_heterosynaptic_active!(counts,mask,HeterosynapticIncoming())
  return nothing
end

function PlasticityHeterosynapticNormalization(
    weight_sum_target::Real,
    tolerance::Real,
    Δt::Real,
    weights::Matrix{Float64};
    target::HeterosynapticTarget,
    method::HeterosynapticMethod,
    weight_min::Real=0.0,
    weight_max::Real=Inf)
  if !(isfinite(weight_sum_target) && weight_sum_target > 0)
    throw(ArgumentError("weight_sum_target must be finite and positive"))
  end
  if !(isfinite(tolerance) && tolerance > 0)
    throw(ArgumentError("tolerance must be finite and positive"))
  end
  if !(isfinite(Δt) && Δt > 0)
    throw(ArgumentError("Δt must be finite and positive"))
  end
  _validate_plasticity_weight_bounds(weight_min,weight_max)

  zero_weight_mask = Matrix{Bool}(undef,size(weights))
  n_groups = _heterosynaptic_group_count(weights,target)
  group_workspace = zeros(n_groups)
  active_counts = zeros(Int,n_groups)
  groups_to_normalize = fill(false,n_groups)
  plast = PlasticityHeterosynapticNormalization(
    Float64(weight_sum_target),Float64(tolerance),Float64(Δt),
    Float64(weight_min),Float64(weight_max),target,method,0.0,
    zero_weight_mask,group_workspace,active_counts,groups_to_normalize)
  refresh_mask!(plast,weights)
  return plast
end

function refresh_mask!(
    plast::PlasticityHeterosynapticNormalization,
    weights::Matrix{Float64})
  if size(weights) != size(plast.zero_weight_mask)
    throw(DimensionMismatch("weights and zero-weight mask must have the same size"))
  end

  fill!(plast.group_workspace,0.0)
  fill!(plast.groups_to_normalize,false)
  mask = plast.zero_weight_mask
  @inbounds for pre_idx in axes(weights,2)
    for post_idx in axes(weights,1)
      mask[post_idx,pre_idx] = iszero(weights[post_idx,pre_idx])
    end
  end
  _count_heterosynaptic_active!(plast.active_counts,mask,plast.target)
  return nothing
end

function reset!(plast::PlasticityHeterosynapticNormalization)
  plast.t_last = 0.0
  fill!(plast.group_workspace,0.0)
  fill!(plast.groups_to_normalize,false)
  return nothing
end

function _prepare_heterosynaptic_correction!(
    workspace::Vector{Float64},
    groups_to_normalize::Vector{Bool},
    active_counts::Vector{Int},
    weight_sum_target::Float64,
    tolerance::Float64,
    ::HeterosynapticSubtractive)
  @inbounds for group_idx in eachindex(workspace)
    weight_sum = workspace[group_idx]
    normalize = weight_sum > weight_sum_target+tolerance
    groups_to_normalize[group_idx] = normalize
    if normalize
      workspace[group_idx] =
        (weight_sum-weight_sum_target)/active_counts[group_idx]
    else
      workspace[group_idx] = 0.0
    end
  end
  return nothing
end

function _prepare_heterosynaptic_correction!(
    workspace::Vector{Float64},
    groups_to_normalize::Vector{Bool},
    ::Vector{Int},
    weight_sum_target::Float64,
    tolerance::Float64,
    ::HeterosynapticDivisive)
  @inbounds for group_idx in eachindex(workspace)
    weight_sum = workspace[group_idx]
    normalize = weight_sum > weight_sum_target+tolerance
    groups_to_normalize[group_idx] = normalize
    if normalize
      workspace[group_idx] = weight_sum_target/weight_sum
    else
      workspace[group_idx] = 1.0
    end
  end
  return nothing
end

@inline function _apply_heterosynaptic_correction(
    weight::Float64,correction::Float64,
    weight_min::Float64,weight_max::Float64,
    ::HeterosynapticSubtractive)
  return hardbounds(weight-correction,weight_min,weight_max)
end

@inline function _apply_heterosynaptic_correction(
    weight::Float64,correction::Float64,
    weight_min::Float64,weight_max::Float64,
    ::HeterosynapticDivisive)
  return hardbounds(weight*correction,weight_min,weight_max)
end

function _normalize_heterosynaptic!(
    weights::Matrix{Float64},
    mask::Matrix{Bool},
    workspace::Vector{Float64},
    active_counts::Vector{Int},
    groups_to_normalize::Vector{Bool},
    weight_sum_target::Float64,
    tolerance::Float64,
    weight_min::Float64,
    weight_max::Float64,
    ::HeterosynapticIncoming,
    method::HeterosynapticMethod)
  fill!(workspace,0.0)
  @inbounds for pre_idx in axes(weights,2)
    for post_idx in axes(weights,1)
      if !mask[post_idx,pre_idx]
        workspace[post_idx] += weights[post_idx,pre_idx]
      end
    end
  end

  _prepare_heterosynaptic_correction!(
    workspace,groups_to_normalize,active_counts,
    weight_sum_target,tolerance,method)

  any_group_to_normalize = false
  @inbounds for post_idx in eachindex(groups_to_normalize)
    if groups_to_normalize[post_idx]
      any_group_to_normalize = true
      break
    end
  end
  if !any_group_to_normalize
    return nothing
  end

  @inbounds for pre_idx in axes(weights,2)
    for post_idx in axes(weights,1)
      if groups_to_normalize[post_idx]
        if !mask[post_idx,pre_idx]
          weights[post_idx,pre_idx] = _apply_heterosynaptic_correction(
            weights[post_idx,pre_idx],workspace[post_idx],
            weight_min,weight_max,method)
        end
      end
    end
  end
  return nothing
end

function _normalize_heterosynaptic!(
    weights::Matrix{Float64},
    mask::Matrix{Bool},
    workspace::Vector{Float64},
    active_counts::Vector{Int},
    groups_to_normalize::Vector{Bool},
    weight_sum_target::Float64,
    tolerance::Float64,
    weight_min::Float64,
    weight_max::Float64,
    ::HeterosynapticOutgoing,
    method::HeterosynapticMethod)
  fill!(workspace,0.0)
  @inbounds for pre_idx in axes(weights,2)
    weight_sum = 0.0
    for post_idx in axes(weights,1)
      if !mask[post_idx,pre_idx]
        weight_sum += weights[post_idx,pre_idx]
      end
    end
    workspace[pre_idx] = weight_sum
  end

  _prepare_heterosynaptic_correction!(
    workspace,groups_to_normalize,active_counts,
    weight_sum_target,tolerance,method)

  any_group_to_normalize = false
  @inbounds for pre_idx in eachindex(groups_to_normalize)
    if groups_to_normalize[pre_idx]
      any_group_to_normalize = true
      break
    end
  end
  if !any_group_to_normalize
    return nothing
  end

  @inbounds for pre_idx in axes(weights,2)
    if groups_to_normalize[pre_idx]
      correction = workspace[pre_idx]
      for post_idx in axes(weights,1)
        if !mask[post_idx,pre_idx]
          weights[post_idx,pre_idx] = _apply_heterosynaptic_correction(
            weights[post_idx,pre_idx],correction,
            weight_min,weight_max,method)
        end
      end
    end
  end
  return nothing
end

function _normalize_heterosynaptic!(
    weights::Matrix{Float64},
    mask::Matrix{Bool},
    workspace::Vector{Float64},
    active_counts::Vector{Int},
    groups_to_normalize::Vector{Bool},
    weight_sum_target::Float64,
    tolerance::Float64,
    weight_min::Float64,
    weight_max::Float64,
    ::HeterosynapticBoth,
    method::HeterosynapticMethod)
  incoming = HeterosynapticIncoming()
  _count_heterosynaptic_active!(active_counts,mask,incoming)
  _normalize_heterosynaptic!(
    weights,mask,workspace,active_counts,groups_to_normalize,
    weight_sum_target,tolerance,weight_min,weight_max,incoming,method)

  outgoing = HeterosynapticOutgoing()
  _count_heterosynaptic_active!(active_counts,mask,outgoing)
  _normalize_heterosynaptic!(
    weights,mask,workspace,active_counts,groups_to_normalize,
    weight_sum_target,tolerance,weight_min,weight_max,outgoing,method)
  return nothing
end

function apply_plasticity!(
    plast::PlasticityHeterosynapticNormalization,
    connection::AbstractConnectionWithWeights,
    t_fire::Real,
    ::Symbol,
    ::Integer)
  weights = connection.weights
  if size(weights) != size(plast.zero_weight_mask)
    throw(DimensionMismatch(
      "connection weights and zero-weight mask must have the same size"))
  end
  if t_fire-plast.t_last < plast.Δt
    return nothing
  end

  plast.t_last = Float64(t_fire)
  _normalize_heterosynaptic!(
    weights,plast.zero_weight_mask,
    plast.group_workspace,plast.active_counts,plast.groups_to_normalize,
    plast.weight_sum_target,plast.tolerance,plast.weight_min,plast.weight_max,
    plast.target,plast.method)
  return nothing
end
