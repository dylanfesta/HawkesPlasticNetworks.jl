#= 
This is a sub-section of HawkesPlasticNetworks.jl for 
recorders, that is the components that save stuff during the simulation.

General principle: recorder called at ever spike time
with all the available information on the spike event
The recorder should point internally at what it needs to record.

@inline function record_stuff!(
     the_recorder::AbstractRecorder,
     t_now::Real,
     pop_fire_idx::Integer,
     pop_fire_label::Symbol,
     neuron_fire_idx::Integer)
  return nothing  
end
=#



mutable struct RecorderPopulationTrain <: AbstractRecorder
  population_label::Symbol
  n_max_spikes::Int64
  times::Vector{Float64}
  neurons::Vector{Int64}
  Tstart::Float64
  Tend::Float64
  k_write::Integer
end
function RecorderPopulationTrain(population_label::Symbol,n_max_spikes::Integer;
    Tstart::Real=0.0,
    Tend::Real=Inf)
  times = fill(NaN,n_max_spikes)
  neurons = fill(-1,n_max_spikes)
  RecorderPopulationTrain(population_label,n_max_spikes,times,neurons,Tstart,Tend,0)
end
function reset!(rec::RecorderPopulationTrain)
  fill!(rec.times,NaN)
  fill!(rec.neurons,-1)
  rec.k_write=0
  return nothing
end

struct RecorderPopulationTrainContent
  population_label::Symbol
  n_recorded_spikes::Int64
  times::Vector{Float64}
  neurons::Vector{Int64}
end
function RecorderPopulationTrainContent(rec::RecorderPopulationTrain)
  idx_keep = isfinite.(rec.times)
  _times = rec.times[idx_keep]
  _neus = rec.neurons[idx_keep]
  n_recorded_spikes = length(_times)
  return RecorderPopulationTrainContent(rec.population_label,n_recorded_spikes,_times,_neus)
end

function get_content(rec::RecorderPopulationTrain)
  return RecorderPopulationTrainContent(rec)
end



function record_stuff!(rec::RecorderPopulationTrain, 
    t_fire::Real,
    pop_fire_idx::Integer,
    pop_fire_label::Symbol,
    neuron_fire_idx::Integer)

  if (rec.Tstart <= t_fire <= rec.Tend) && (rec.population_label == pop_fire_label)
      k = rec.k_write + 1
      if k <= rec.n_max_spikes
        rec.times[k] = t_fire
        rec.neurons[k] = neuron_fire_idx
      rec.k_write = k
    end
  end
  return nothing  
end



struct RecorderPopulationRate <: AbstractRecorder
  population_label::Symbol
  n_neurons::Int64
  Δt::Float64
  times::Vector{Float64}
  rates::Vector{Float64}
  Tstart::Float64
  Tend::Float64
end

function RecorderPopulationRate(population_label::Symbol,n_neurons::Integer,Tend::Real;
    Tstart::Real=0.0,
    Δt::Real=10.0)
  if !(n_neurons > 0)
    throw(ArgumentError("n_neurons must be positive"))
  end
  if !(Δt > 0)
    throw(ArgumentError("Δt must be positive"))
  end
  if !(Tend >= Tstart)
    throw(ArgumentError("Tend must not be smaller than Tstart"))
  end

  n_bins = ceil(Int,(Tend-Tstart)/Δt)
  times = Tstart .+ ((1:n_bins) .- 0.5) .* Δt
  rates = zeros(n_bins)
  return RecorderPopulationRate(
    population_label,n_neurons,Δt,times,rates,Tstart,Tend)
end

function RecorderPopulationRate(
    population::Union{PopulationExpKernelExcitatory,PopulationExpKernelInhibitory},
    Tend::Real;
    Tstart::Real=0.0,
    Δt::Real=10.0)
  return RecorderPopulationRate(
    population.label,nneurons(population),Tend; Tstart=Tstart,Δt=Δt)
end

function reset!(rec::RecorderPopulationRate)
  fill!(rec.rates,0.0)
  return nothing
end

struct RecorderPopulationRateContent
  population_label::Symbol
  times::Vector{Float64}
  rates::Vector{Float64}
end

function get_content(rec::RecorderPopulationRate)
  return RecorderPopulationRateContent(
    rec.population_label,copy(rec.times),copy(rec.rates))
end

function record_stuff!(rec::RecorderPopulationRate,
    t_fire::Real,
    pop_fire_idx::Integer,
    pop_fire_label::Symbol,
    neuron_fire_idx::Integer)

  if (rec.Tstart <= t_fire < rec.Tend) && (rec.population_label == pop_fire_label)
    k = floor(Int,(t_fire-rec.Tstart)/rec.Δt) + 1
    rec.rates[k] += 1.0/(rec.n_neurons*rec.Δt)
  end
  return nothing  
end
