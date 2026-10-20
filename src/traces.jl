#= 
This file is a sub-section of HawkesPlasticNetworks.jl.
It deals with the internal traces for dynamics and plasticity.
=#

import Base: length, getindex

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

"""
    CalciumTrace(τ, n; r_min, r_max)

Track exponentially smoothed firing rates for `n` units with time constant `τ`.
The underlying `trace::Trace` retains the full rate estimate; readout through
`trace_value(tra, i)` or `tra[i]` clamps it to `[r_min, r_max]`.
Bounds must be finite and satisfy `0 ≤ r_min ≤ r_max`.

Call `propagate!(t, tra)` before adding spikes at `t` with
`add_firing_event_now!(tra, i)`. Each spike adds `1/τ`, so the underlying
estimate is already normalized to spikes per time unit (Hz when time is in
seconds). Readout uses the last propagated time and does not advance the trace
or correct for startup transients; allow warmup when estimating steady rates.

Note: I called it CalciumTrace just to make it sound fancy.
Any relation to calcium ions or calcum channels is purely coincidential.
"""
struct CalciumTrace{Tr<:Trace,R<:Real} <: AbstractTrace
  trace::Tr
  r_min::R
  r_max::R
  function CalciumTrace(τ::Real,n::Integer; r_min::Real,r_max::Real)
    if !isfinite(r_min) || !isfinite(r_max)
      throw(ArgumentError("rate bounds must be finite"))
    end
    if !(0 <= r_min <= r_max)
      throw(ArgumentError("rate bounds must satisfy 0 ≤ r_min ≤ r_max"))
    end
    trace = Trace(τ,n)
    lower,upper = promote(r_min,r_max)
    return new{typeof(trace),typeof(lower)}(trace,lower,upper)
  end
end

function Base.length(tra::CalciumTrace)
  return length(tra.trace)
end

function reset!(tra::CalciumTrace)
  reset!(tra.trace)
  return nothing
end

@inline function trace_decay(tnow::Real,tra::CalciumTrace)
  return trace_decay(tnow,tra.trace)
end

@inline function propagate!(tnow::Real,tra::CalciumTrace)
  propagate!(tnow,tra.trace)
  return nothing
end

@inline function add_firing_event_now!(tra::CalciumTrace,idx_update::Int64)
  add_firing_event_now!(tra.trace,idx_update)
  return nothing
end

"""
    trace_value(tra, i)

Return unit `i`'s rate estimate at the last propagated time, in spikes per time
unit. A `Trace` returns its stored rate; a `CalciumTrace` clamps that rate to
its bounds. Equivalent to `tra[i]`; neither form mutates the trace.
"""
@inline function trace_value(tra::Trace,i::Integer)
  return tra.val[i]
end

@inline function trace_value(tra::CalciumTrace,i::Integer)
  return clamp(trace_value(tra.trace,i),tra.r_min,tra.r_max)
end

@inline function Base.getindex(tra::Union{Trace,CalciumTrace},i::Integer)
  return trace_value(tra,i)
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
