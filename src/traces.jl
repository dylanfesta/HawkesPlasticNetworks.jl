#= 
This file is a sub-section of HawkesPlasticNetworks.jl.
It deals with the internal traces for dynamics and plasticity.
=#

import Base.length

mutable struct Trace{V<:AbstractVector,R<:Real} <: AbstractTrace
  val::V
  τ::R
  t_last::R
  function Trace(τ::Real,n::Integer)
    if !isfinite(τ)
      throw(ArgumentError("τ must be finite and positive"))
    end
    if !(τ > 0)
      throw(ArgumentError("τ must be finite and positive"))
    end
    if n < 0
      throw(ArgumentError("trace length must be nonnegative"))
    end
    val = fill(0.0,n)
    t_last = 0.0
    return new{typeof(val),typeof(t_last)}(val,τ,t_last)
  end
end

function Base.length(tra::Trace)
  return length(tra.val)
end

function reset!(tra::Trace)
  fill!(tra.val,0.0)
  tra.t_last=0.0
  return nothing
end

# decay factor that affects a trace at time tnow
@inline function trace_decay(tnow::Real,tra::Trace)
  return exp(-(tnow-tra.t_last)/tra.τ)
end

@inline function propagate!(tnow::Real,tra::Trace)
  tra.val .*= trace_decay(tnow,tra)
  tra.t_last=tnow
  return nothing
end

@inline function add_firing_event_now!(tra::Trace,idx_update::Int64)
  tra.val[idx_update] += inv(tra.τ)
  return nothing
end

# # updates single element of trace, unless it is nothing
# function update_now!(tra::Trace{P,R},idx_update::Int64,up_val::R=1.0) where {P,R}
#   if !iszero(idx_update)
#     tra.val[idx_update] += up_val
#   end
#   return nothing
# end

# # updates only if the trace is specifically for dynamics
# function update_for_dynamics!(::Trace,::Int64)
#   return nothing
# end
# function update_for_dynamics!(tra::Trace{ForDynamics,Float64},idx_update::Int64)
#   tra.val[idx_update] += inv(tra.τ)
#   return nothing
# end
# # same with propagate
# function propagate_for_dynamics!(::Float64,::Trace)
#   return nothing
# end
# function propagate_for_dynamics!(tnow::Float64,tra::Trace{ForDynamics,Float64})
#   Δt = tnow - tra.t_last
#   tra.val .*= exp(-Δt/tra.τ)
#   tra.t_last=tnow
#   return nothing
# end


# @inline function trace_decay(tnow::Real,tra::Trace)
#   return exp(-(tnow-tra.t_last)/tra.τ)
# end

# # proposal of future trace. Useful to compute quantities without advancing the trace
# function trace_proposal!(proposal::Vector{R},tnow::R,tra::Trace{P,R}) where {P,R}
#   copy!(proposal,tra.val)
#   rmul!(proposal,trace_decay(tnow,tra))
#   return nothing
# end
