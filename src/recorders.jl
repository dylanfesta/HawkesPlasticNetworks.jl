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


struct RecorderPopulationTrain <: AbstractRecorder
  population_label::Symbol
  n_max_spikes::Int64
  times::Vector{Float64}
  neurons::Vector{Int64}
  Tstart::Float64
  Tend::Float64
  k_write::Base.RefValue{Int}
end
function RecorderPopulationTrain(population_label::Symbol,n_max_spikes::Integer;
    Tstart::Real=0.0,
    Tend::Real=Inf)
  times = fill(NaN,n_max_spikes)
  neurons = fill(-1,n_max_spikes)
  RecorderPopulationTrain(
    population_label,n_max_spikes,times,neurons,Tstart,Tend,Ref(0))
end
function reset!(rec::RecorderPopulationTrain)
  fill!(rec.times,NaN)
  fill!(rec.neurons,-1)
  rec.k_write[] = 0
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
      k = rec.k_write[] + 1
      if k <= rec.n_max_spikes
        rec.times[k] = t_fire
        rec.neurons[k] = neuron_fire_idx
      rec.k_write[] = k
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
  t_last::Base.RefValue{Float64}
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

  n_bins = ceil(Int,(Tend-Tstart)/Δt) + 1
  times = Tstart .+ ((1:n_bins) .- 0.5) .* Δt
  rates = zeros(n_bins)
  return RecorderPopulationRate(
    population_label,n_neurons,Δt,times,rates,Tstart,Tend,Ref(-Inf))
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
  rec.t_last[] = -Inf
  return nothing
end

struct RecorderPopulationRateContent
  population_label::Symbol
  times::Vector{Float64}
  rates::Vector{Float64}
end

function get_content(rec::RecorderPopulationRate)
  n_complete_bins = 0
  if isfinite(rec.t_last[])
    n_complete_bins = floor(Int,(rec.t_last[]-rec.Tstart)/rec.Δt)
    n_complete_bins = clamp(n_complete_bins,0,length(rec.times)-1)
  end
  idxs_keep = 1:n_complete_bins
  return RecorderPopulationRateContent(
    rec.population_label,
    copy(rec.times[idxs_keep]),
    copy(rec.rates[idxs_keep]))
end

function record_stuff!(rec::RecorderPopulationRate,
    t_fire::Real,
    pop_fire_idx::Integer,
    pop_fire_label::Symbol,
    neuron_fire_idx::Integer)

  if rec.Tstart <= t_fire < rec.Tend + rec.Δt
    rec.t_last[] = t_fire
    if rec.population_label == pop_fire_label
      k = floor(Int,(t_fire-rec.Tstart)/rec.Δt) + 1
      rec.rates[k] += 1.0/(rec.n_neurons*rec.Δt)
    end
  end
  return nothing  
end



function time_to_record(t_last::Real,t_now::Real,Δt::Real)
  return t_now-t_last >= Δt
end

struct DoEveryDt{F} <: AbstractRecorder
  thing_to_do::F
  Δt::Float64
  Tstart::Float64
  Tend::Float64
  t_last::Base.RefValue{Float64}
end

function DoEveryDt(
    thing_to_do,
    Δt::Real;
    Tstart::Real=0.0,
    Tend::Real=Inf)
  if !(Δt > 0)
    throw(ArgumentError("Δt must be positive"))
  end
  if !(Tend >= Tstart)
    throw(ArgumentError("Tend must not be smaller than Tstart"))
  end
  return DoEveryDt(
    thing_to_do,Float64(Δt),Float64(Tstart),Float64(Tend),Ref(-Inf))
end

function reset!(rec::DoEveryDt)
  rec.t_last[] = -Inf
  return nothing
end

function record_stuff!(rec::DoEveryDt,t_fire::Real,other_args...)
  if !time_to_record(rec.t_last[],t_fire,rec.Δt)
    return nothing
  end
  if t_fire < rec.Tstart || t_fire > rec.Tend
    return nothing
  end

  rec.t_last[] = t_fire
  rec.thing_to_do(t_fire,other_args...)
  return nothing
end



struct WeightMatrixRecorder{D<:DoEveryDt} <: AbstractRecorder
  times::Vector{Float64}
  weights::Array{Float64,3}
  k_write::Base.RefValue{Int}
  do_every::D
end

function WeightMatrixRecorder(
    weight_matrix_pointer::Matrix{Float64},
    Δt::Real,
    Tend::Real;
    Tstart::Real=0.0)
  if !(Δt > 0)
    throw(ArgumentError("Δt must be positive"))
  end
  if !(Tend >= Tstart)
    throw(ArgumentError("Tend must not be smaller than Tstart"))
  end
  if !isfinite(Tstart) || !isfinite(Tend)
    throw(ArgumentError("Tstart and Tend must be finite"))
  end

  n_times = floor(Int,(Tend-Tstart)/Δt) + 1
  n_post,n_pre = size(weight_matrix_pointer)
  times = fill(NaN,n_times)
  weights = fill(NaN,n_times,n_post,n_pre)
  k_write = Ref(0)
  record_weights! = function(t_fire,other_args...)
    k = k_write[] + 1
    times[k] = t_fire
    @views copyto!(weights[k,:,:],weight_matrix_pointer)
    k_write[] = k
    return nothing
  end
  do_every = DoEveryDt(record_weights!,Δt;Tstart=Tstart,Tend=Tend)
  return WeightMatrixRecorder(times,weights,k_write,do_every)
end

function reset!(rec::WeightMatrixRecorder)
  fill!(rec.times,NaN)
  fill!(rec.weights,NaN)
  rec.k_write[] = 0
  reset!(rec.do_every)
  return nothing
end

struct WeightMatrixRecorderContent
  times::Vector{Float64}
  weights::Array{Float64,3}
end

function get_content(rec::WeightMatrixRecorder)
  idxs_keep = 1:rec.k_write[]
  return WeightMatrixRecorderContent(
    copy(rec.times[idxs_keep]),copy(rec.weights[idxs_keep,:,:]))
end

function record_stuff!(rec::WeightMatrixRecorder,t_fire::Real,other_args...)
  record_stuff!(rec.do_every,t_fire,other_args...)
  return nothing
end
