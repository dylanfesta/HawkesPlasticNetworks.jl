module HawkesPlasticNetworks

using StatsBase,Statistics,Distributions,LinearAlgebra,Random
using FFTW


# Populations define neuron types, internal states and traces
abstract type AbstractPopulation end

# A connected population includes pre- and post-synaptic populations and connections
abstract type AbstractConnectedPopulation end

# A connection represents the interaction between pre- and post-synaptic populations
# It includes the population themselves, and also possible plasticity rules!
abstract type AbstractConnection end

"""
Supertype for connections that store a weight matrix and plasticity rules.
"""
abstract type AbstractConnectionWithWeights <: AbstractConnection end
# Plasticity rules change connection weights
abstract type AbstractPlasticityRule end

# Traces keep track of internal activity in time
abstract type AbstractTrace end

# These are meant to save things to memory
abstract type AbstractRecorder end


struct NoPlasticity <: AbstractPlasticityRule end

#### Small general utility functions
@inline hardbounds(x::R,low::R,high::R) where R = min(high,max(x,low))
@inline rand_label() = Symbol(randstring(3))

# Now include different sub-components
include("analytics.jl")
include("weight_matrix_utilities.jl")
include("traces.jl")


struct ConnectionWithWeights{
    PR<:Tuple{Vararg{AbstractPlasticityRule}}} <: AbstractConnectionWithWeights
  weights::Matrix{Float64}
  post_pop_label::Symbol
  pre_pop_label::Symbol
  plasticity_rules::PR
  is_plastic::Base.RefValue{Bool}
end

function _connection_with_weights(
    connection_type::Type{<:AbstractConnectionWithWeights},
    weights::Matrix{Float64},
    post_pop_label::Symbol,
    pre_pop_label::Symbol,
    plasticity_rules::PR,
    is_plastic::Bool) where {PR<:Tuple{Vararg{AbstractPlasticityRule}}}
  return connection_type(
    weights,post_pop_label,pre_pop_label,plasticity_rules,Ref(is_plastic))
end

function _connection_with_weights(
    connection_type::Type{<:AbstractConnectionWithWeights},
    pop_post::AbstractPopulation,weights::Matrix{Float64},pop_pre::AbstractPopulation;
    plasticity_rules::Tuple{Vararg{AbstractPlasticityRule}}=Tuple{}(),
    is_plastic::Union{Nothing,Bool}=nothing)
  _is_plastic = something(is_plastic, ! isempty(plasticity_rules))
  return _connection_with_weights(
    connection_type,weights,pop_post.label,pop_pre.label,
    plasticity_rules,_is_plastic)
end

function ConnectionWithWeights(
    weights::Matrix{Float64},
    post_pop_label::Symbol,
    pre_pop_label::Symbol,
    plasticity_rules::PR,
    is_plastic::Bool) where {PR<:Tuple{Vararg{AbstractPlasticityRule}}}
  return _connection_with_weights(
    ConnectionWithWeights,weights,post_pop_label,pre_pop_label,
    plasticity_rules,is_plastic)
end

function ConnectionWithWeights(
    pop_post::AbstractPopulation,weights::Matrix{Float64},pop_pre::AbstractPopulation;
    plasticity_rules::Tuple{Vararg{AbstractPlasticityRule}}=Tuple{}(),
    is_plastic::Union{Nothing,Bool}=nothing)
  return _connection_with_weights(
    ConnectionWithWeights,pop_post,weights,pop_pre;
    plasticity_rules=plasticity_rules,is_plastic=is_plastic)
end

"""
Weighted connection that participates in plasticity but transmits no signal.
"""
struct ConnectionNonInteracting{
    PR<:Tuple{Vararg{AbstractPlasticityRule}}} <: AbstractConnectionWithWeights
  weights::Matrix{Float64}
  post_pop_label::Symbol
  pre_pop_label::Symbol
  plasticity_rules::PR
  is_plastic::Base.RefValue{Bool}
end

function ConnectionNonInteracting(
    weights::Matrix{Float64},
    post_pop_label::Symbol,
    pre_pop_label::Symbol,
    plasticity_rules::PR,
    is_plastic::Bool) where {PR<:Tuple{Vararg{AbstractPlasticityRule}}}
  return _connection_with_weights(
    ConnectionNonInteracting,weights,post_pop_label,pre_pop_label,
    plasticity_rules,is_plastic)
end

function ConnectionNonInteracting(
    pop_post::AbstractPopulation,weights::Matrix{Float64},pop_pre::AbstractPopulation;
    plasticity_rules::Tuple{Vararg{AbstractPlasticityRule}}=Tuple{}(),
    is_plastic::Union{Nothing,Bool}=nothing)
  return _connection_with_weights(
    ConnectionNonInteracting,pop_post,weights,pop_pre;
    plasticity_rules=plasticity_rules,is_plastic=is_plastic)
end


"""
    plasticity_on!(connection)

Activate plasticity for a connection (all rules included).
"""
function plasticity_on!(connection::AbstractConnectionWithWeights)
  connection.is_plastic[] = true
  return nothing
end

"""
    plasticity_off!(connection)

Deactivate plasticity for a connection (all rules included).
"""
function plasticity_off!(connection::AbstractConnectionWithWeights)
  connection.is_plastic[] = false
  return nothing
end

# this is enough to include the plasticity rules

include("plasticity_rules.jl")
include("plasticity_heterosynaptic_rules.jl")


struct PopulationExpKernelExcitatory{Tr<:Trace}  <: AbstractPopulation
  label::Symbol
  n::Int64
  trace::Tr
  spike_proposals::Vector{Float64} # memory allocation 
end
nneurons(ps::PopulationExpKernelExcitatory) = ps.n

function PopulationExpKernelExcitatory(n::Integer,trace::Trace;
    label::Union{String,Nothing}=nothing)
  label = something(label,rand_label()) 
  label = Symbol(label)
  spike_proposals = fill(Inf,n)
  return PopulationExpKernelExcitatory(label,n,trace,spike_proposals)
end
function PopulationExpKernelExcitatory(n::Integer,τ_kernel::Real;
    label::Union{String,Nothing}=nothing)
  trace = Trace(τ_kernel,n)
  label = something(label,rand_label())
  label = Symbol(label)
  spike_proposals = fill(Inf,n)
  return PopulationExpKernelExcitatory(label,n,trace,spike_proposals)
end

struct PopulationExpKernelInhibitory{Tr<:Trace}  <: AbstractPopulation
  label::Symbol
  n::Int64
  trace::Tr
  spike_proposals::Vector{Float64} # memory allocation 
end
nneurons(ps::PopulationExpKernelInhibitory) = ps.n

function PopulationExpKernelInhibitory(n::Integer,trace::Trace;
    label::Union{String,Nothing}=nothing)
  label = something(label,rand_label()) 
  label = Symbol(label)
  spike_proposals = fill(Inf,n)
  return PopulationExpKernelInhibitory(label,n,trace,spike_proposals)
end
function PopulationExpKernelInhibitory(n::Integer,τ_kernel::Real;
    label::Union{String,Nothing}=nothing)
  trace = Trace(τ_kernel,n)
  label = something(label,rand_label())
  label = Symbol(label)
  spike_proposals = fill(Inf,n)
  return PopulationExpKernelInhibitory(label,n,trace,spike_proposals)
end

include("recorders.jl")

function reset!(ps::Union{PopulationExpKernelExcitatory,PopulationExpKernelInhibitory})
  reset!(ps.trace)
  fill!(ps.spike_proposals,Inf)
  return nothing
end


function set_initial_rates!(pop::Union{PopulationExpKernelExcitatory,PopulationExpKernelInhibitory},
    rates::Union{Vector{<:Real},Real})
  _rates = if isa(rates,Real)
    fill(rates,nneurons(pop))
  else
    typeassert(rates,Vector{<:Real})
  end
  @assert nneurons(pop) == length(_rates) "Dimensions wrong!"
  pop.trace.val .= _rates
  return nothing
end


struct ConnectedPopulationExpKernel{N,
    PS<:Union{PopulationExpKernelExcitatory,PopulationExpKernelInhibitory},
    TC<:NTuple{N,AbstractConnection},
    TP<:NTuple{N,AbstractPopulation}} <: AbstractPopulation
  population::PS
  connections::TC
  pre_populations::TP
  input::Vector{Float64} # input for fixed external currents 
end

# more convenient constructor, with connection arguments such as 
# (connection_1,pop_pre_1), (connection_2,pop_pre_2), ...
function ConnectedPopulationExpKernel(
    state::Union{PopulationExpKernelExcitatory,PopulationExpKernelInhibitory},
    input::Vector{Float64},
    (conn_pre::Tuple{C,PS} where {C<:AbstractConnection,PS<:AbstractPopulation})...)
  connections = Tuple(getindex.(conn_pre,1))
  pre_states = Tuple(getindex.(conn_pre,2))
  return ConnectedPopulationExpKernel(state,connections,pre_states,input) 
end

function set_initial_rates!(pop::ConnectedPopulationExpKernel,rates::Union{Vector{<:Real},Real})
  set_initial_rates!(pop.population,rates)
  return nothing
end

function reset!(pop::ConnectedPopulationExpKernel)
  reset!(pop.population)
  return nothing
end

struct RecurrentNetworkExpKernel{N,TP<:NTuple{N,AbstractPopulation},NR,TR<:NTuple{NR,AbstractRecorder}}
  populations::TP
  recorders::TR
  function RecurrentNetworkExpKernel(populations::TP,recorders::TR) where {TP,TR}
    npops = length(populations)
    nrec = length(recorders)
    return new{npops,TP,nrec,TR}(populations,recorders)
  end
end
# one population constructor
function RecurrentNetworkExpKernel(pop::AbstractPopulation,recorders...)
  if isempty(recorders)
    recorders=(RecNothing(),)
  end
  return RecurrentNetworkExpKernel((pop,),recorders)
end
# multiple pops, but no recorders
function RecurrentNetworkExpKernel(pops::Tuple)
  recorders=(RecNothing(),)
  return RecurrentNetworkExpKernel(pops,recorders)
end

function reset!(network::RecurrentNetworkExpKernel)
  reset_iterator!(network.populations)
  reset_iterator!(network.recorders)
  return nothing
end

function reset_iterator!(items::Tuple)
  reset!(first(items))
  reset_iterator!(Base.tail(items))
  return nothing
end

function reset_iterator!(::Tuple{})
  return nothing
end


# This is the function that computes the total input at t_now, from all possible sources 
function compute_rates!(r_alloc::Vector{Float64},t_now::Real,conn_pop::ConnectedPopulationExpKernel)
  copy!(r_alloc,conn_pop.input)
  accumulate_signal_iterator!(r_alloc,t_now,conn_pop.population,
    conn_pop.connections,conn_pop.pre_populations)
  return nothing
end

# The upper limit is the same as the default, except it floors 
# the result at zero.
function compute_rates_upper!(r_alloc::Vector{Float64},t_now::Real,
    conn_pop::ConnectedPopulationExpKernel)
  copy!(r_alloc,conn_pop.input)
  accumulate_signal_iterator!(r_alloc,t_now,conn_pop.population,conn_pop.connections,conn_pop.pre_populations)
  _eps = eps(Float64)
  r_alloc .= max.(r_alloc,_eps)
  return nothing
end


# when there is content, accumulate the signal iteratively
function accumulate_signal_iterator!(rates::Vector{Float64},t_now::Real,
    ps_post::AbstractPopulation,connections,pre_states)
  # call first
  accumulate_signal!(rates,t_now,ps_post,first(connections),first(pre_states))
  # move to all others
  accumulate_signal_iterator!(rates,t_now,ps_post,Base.tail(connections),Base.tail(pre_states))
  return nothing
end
# When iterator empty return nothing and exit
function accumulate_signal_iterator!(::Vector{Float64},::Real,
    ::AbstractPopulation,::Tuple{},::Tuple{})
  return nothing
end


function accumulate_signal!(rates::Vector{Float64},t_now::Real,
    ::AbstractPopulation,connection::ConnectionWithWeights,
    pop_pre::PopulationExpKernelExcitatory)
  # accumulates the signal from the presynaptic population
  # decay = exp(-(tnow-t_last)/tau)
  # traces_now = traces_before .* decay
  # input_now = weights * traces_now
  # rates <- rates + input_now
  decay = trace_decay(t_now,pop_pre.trace)
  mul!(rates,connection.weights,pop_pre.trace.val,decay,1.0)
  return nothing
end

function accumulate_signal!(rates::Vector{Float64},t_now::Real,
    ::AbstractPopulation,connection::ConnectionWithWeights,
    pop_pre::PopulationExpKernelInhibitory)
  decay = trace_decay(t_now,pop_pre.trace)
  mul!(rates,connection.weights,pop_pre.trace.val,-decay,1.0)
  return nothing
end

# here is where the ConnectionNonInteracting does nothing
function accumulate_signal!(::Vector{Float64},::Real,
    ::AbstractPopulation,::ConnectionNonInteracting,::AbstractPopulation)
  return nothing
end

# multivariate thinning algorithm. From Y. Chen, 2016
function compute_next_spike(t_now::Real,
    connected_pop::ConnectedPopulationExpKernel;Tmax::Real=100.0)
  t_start = t_now
  t = t_now
  rates = connected_pop.population.spike_proposals # recycle & reuse  # Vector{Float64}(undef,n)
  while (t-t_start)<Tmax 
    compute_rates_upper!(rates,t,connected_pop)
    R_up = sum(rates)
    Δt = -log(rand())/R_up
    t = t+Δt
    u = rand()*R_up # random between 0 and R_up
    compute_rates!(rates,t,connected_pop)
    cumsum!(rates,rates)
    if u < rates[end] # else, continue
      k = searchsortedfirst(rates,u)
      return (t,k)
    end
  end
  @warn "Population did not spike ! Returning fake spike at t=$(Tmax+t_start) (is this a test? or too much inh?)" maxlog=20
  return (Tmax + t_start,1)
end

# This applies compute_next_spike recursively over multiple populations of neurons
# and select the absolute best spike over the whole network. 
# it also specify which neuron in which population it's firing
# To avoid memory leaks, it's an interative function, that carries with it its
# best current selection, releasing it when it runs out of populations to consider.

function compute_next_spike_population_iterator(
    t_now::Real,connected_populations::Tuple;
    previous_pop_idx::Integer=0,
    bestspiketime::Real=Inf,
    bestpop_idx::Integer=0,
    bestpoplabel::Symbol=:nope,
    bestneuron::Int=-1)
  conn_pop = first(connected_populations)
  this_pop_idx = previous_pop_idx + 1
  bestspiketime_here,bestneuron_here = compute_next_spike(t_now,conn_pop)
  if bestspiketime_here < bestspiketime
    bestspiketime = bestspiketime_here
    bestpop_idx = this_pop_idx
    bestpoplabel = conn_pop.population.label
    bestneuron = bestneuron_here
  end
  # iterate again, keeping the best values
  return compute_next_spike_population_iterator(
    t_now,Base.tail(connected_populations);
    previous_pop_idx = this_pop_idx,
    bestspiketime = bestspiketime,
    bestpop_idx = bestpop_idx,
    bestpoplabel = bestpoplabel,
    bestneuron = bestneuron)
end

function compute_next_spike_population_iterator(
    ::Real,::Tuple{};
    previous_pop_idx::Integer=0,
    bestspiketime::Real=Inf,
    bestpop_idx::Integer=0,
    bestpoplabel::Symbol=:nope,
    bestneuron::Int=-1)
  return (bestspiketime,bestpop_idx,bestpoplabel,bestneuron)
end


# As usual, to avoid leaks I iterate a "spike-updater" across
# the population. The neuron that fired receives a special treatment, all others
# are just moved forward.
# Question: is it really needed to propagate every population forward?
# since the information of last updated time is stored in the traces anyway...

function burn_spike_iterator_legacy!(
    tfire::Real, pop_fire_idx::Integer,
    neuron_fire_idx::Integer,connected_populations::Tuple;
    past_population_idx::Integer=0)
  this_connected_population = first(connected_populations)
  current_population_idx = past_population_idx + 1
  if current_population_idx == pop_fire_idx
    burn_spike!(tfire,this_connected_population.population.trace,neuron_fire_idx)
  else
    burn_spike!(tfire,this_connected_population.population.trace)
  end
  return burn_spike_iterator_legacy!(
    tfire,pop_fire_idx,neuron_fire_idx,
    Base.tail(connected_populations);
    past_population_idx=current_population_idx)
end

function burn_spike_iterator_legacy!(
    ::Real,::Integer,::Integer,::Tuple{};
    past_population_idx::Integer=0)
  return nothing
end

# like above, but I update ONLY the population that fired
function burn_spike_iterator!(
    tfire::Real, pop_fire_idx::Integer,
    neuron_fire_idx::Integer,connected_populations::Tuple;
    past_population_idx::Integer=0)
  this_connected_population = first(connected_populations)
  current_population_idx = past_population_idx + 1
  if current_population_idx == pop_fire_idx
    burn_spike!(tfire,this_connected_population.population.trace,neuron_fire_idx)
  end
  return burn_spike_iterator!(
    tfire,pop_fire_idx,neuron_fire_idx,
    Base.tail(connected_populations);
    past_population_idx=current_population_idx)
end

function burn_spike_iterator!(
    ::Real,::Integer,::Integer,::Tuple{};
    past_population_idx::Integer=0)
  return nothing
end

function burn_spike!(t_spike::Real,
    trace::Trace,
    idx_update::Integer)
  # take care of traces  
  propagate!(t_spike,trace) # update full trace to t_spike
  add_firing_event_now!(trace,idx_update) # add pulse at index
  return nothing
end
function burn_spike!(t_spike::Real,trace::Trace)
  propagate!(t_spike,trace) # update full trace to t_spike 
  return nothing
end


function recorders_iterator!(
    t_fire::Real,
    pop_fire_idx::Integer,
    pop_fire_label::Symbol,
    neuron_fire_idx::Integer,
    recorders::Tuple)
  record_stuff!(
    first(recorders),t_fire,pop_fire_idx,pop_fire_label,neuron_fire_idx)
  recorders_iterator!(
    t_fire,pop_fire_idx,pop_fire_label,neuron_fire_idx,Base.tail(recorders))
  return nothing
end

function recorders_iterator!(
    ::Real,::Integer,::Symbol,::Integer,::Tuple{})
  return nothing
end


"""
    apply_plasticity!(t_fire, population_fire_label, neuron_fire_idx,
                      connected_populations)

Apply every plasticity rule on every connection after a neuron fires.

Rules implement
`apply_plasticity!(rule, connection, t_fire, population_fire_label,
neuron_fire_idx)`. The tuple recursion keeps the concrete connection and rule
types visible to the compiler and does not allocate intermediate collections.
"""
function apply_plasticity!(
    t_fire::Real,population_fire_label::Symbol,neuron_fire_idx::Integer,
    connected_populations::Tuple)
  _apply_plasticity_to_populations!(
    connected_populations,t_fire,population_fire_label,neuron_fire_idx)
  return nothing
end

function _apply_plasticity_to_populations!(
    connected_populations::Tuple,t_fire::Real,
    population_fire_label::Symbol,neuron_fire_idx::Integer)
  _apply_plasticity_rules_to_connections!(
    first(connected_populations).connections,t_fire,
    population_fire_label,neuron_fire_idx)
  _apply_plasticity_to_populations!(
    Base.tail(connected_populations),t_fire,
    population_fire_label,neuron_fire_idx)
  return nothing
end

function _apply_plasticity_to_populations!(
    ::Tuple{},::Real,::Symbol,::Integer)
  return nothing
end

function _apply_plasticity_rules_to_connections!(
    connections::Tuple,t_fire::Real,
    population_fire_label::Symbol,neuron_fire_idx::Integer)
  connection = first(connections)
  if connection.is_plastic[]
    _apply_plasticity_rules!(
      connection,t_fire,population_fire_label,neuron_fire_idx)
  end
  _apply_plasticity_rules_to_connections!(
    Base.tail(connections),t_fire,population_fire_label,neuron_fire_idx)
  return nothing
end

function _apply_plasticity_rules_to_connections!(
    ::Tuple{},::Real,::Symbol,::Integer)
  return nothing
end

function _apply_plasticity_rules!(
    connection::AbstractConnection,t_fire::Real,
    population_fire_label::Symbol,neuron_fire_idx::Integer)
  _apply_plasticity_rules!(
    connection.plasticity_rules,connection,t_fire,
    population_fire_label,neuron_fire_idx)
  return nothing
end

function _apply_plasticity_rules!(
    rules::Tuple,connection::AbstractConnection,t_fire::Real,
    population_fire_label::Symbol,neuron_fire_idx::Integer)
  apply_plasticity!(
    first(rules),connection,t_fire,population_fire_label,neuron_fire_idx)
  _apply_plasticity_rules!(
    Base.tail(rules),connection,t_fire,population_fire_label,neuron_fire_idx)
  return nothing
end

function _apply_plasticity_rules!(
    ::Tuple{},::AbstractConnection,::Real,::Symbol,::Integer)
  return nothing
end

function apply_plasticity!(
    ::NoPlasticity,::AbstractConnection,::Real,::Symbol,::Integer)
  return nothing
end

function dynamics_step_legacy!(t_now::Real,ntw::RecurrentNetworkExpKernel)
  tfire,popfire,labelfire,neufire =
    compute_next_spike_population_iterator(t_now,ntw.populations)
  # update stuff for that specific neuron/population state :
  burn_spike_iterator_legacy!(tfire,popfire,neufire,ntw.populations)
  # apply plasticity rules ! Each rule in each connection in each population
  apply_plasticity!(tfire,labelfire,neufire,ntw.populations)
  # now trigger recorders. 
  # Recorder objects will take care of which population to target
  recorders_iterator!(tfire,popfire,labelfire,neufire,ntw.recorders)
  # update t_now 
  return tfire
end

function dynamics_step!(t_now::Real,ntw::RecurrentNetworkExpKernel)
  tfire,popfire,labelfire,neufire =
    compute_next_spike_population_iterator(t_now,ntw.populations)
  # update stuff for that specific neuron/population state :
  burn_spike_iterator!(tfire,popfire,neufire,ntw.populations)
  # apply plasticity rules ! Each rule in each connection in each population
  apply_plasticity!(tfire,labelfire,neufire,ntw.populations)
  # now trigger recorders. 
  # Recorder objects will take care of which population to target
  recorders_iterator!(tfire,popfire,labelfire,neufire,ntw.recorders)
  # update t_now 
  return tfire
end

end # of module
