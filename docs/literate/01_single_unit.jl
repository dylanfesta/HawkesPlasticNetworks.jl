
# # Single self-interacting unit

#=
This example considers the simplest non-trivial configuration:
a single unit interacting with itself, without plasticity.
=#

# ## Initialization

using Random
using Statistics
using Plots
Random.seed!(0)

using HawkesPlasticNetworks; global const H = HawkesPlasticNetworks

# ## Define the parameters

n_units = 1
τ_kernel = 10.0
w_self = 0.5
h_in = 0.6
initial_rate = 0.0001
n_spikes = 20_000
bin_width = 2.0

# For a stable linear Hawkes process with a normalized exponential kernel, the
# stationary rate is `h_in / (1 - w_self)`.
theoretical_rate = h_in/(1-w_self)

# ## Define the population and connection

population = H.PopulationExpKernelExcitatory(
    n_units,τ_kernel; label="unit")
connection = H.ConnectionWithWeights(
    population,fill(w_self,n_units,n_units),population)

# A network contains connected populations, rather than bare populations.
# The argument order is always post-synaptic first, then pre-synaptic.
connected_population = H.ConnectedPopulationExpKernel(
    population,fill(h_in,n_units),(connection,population))

# ## Record the spike train and mean firing rate

rec_fulltrain = H.RecorderPopulationTrain(
    population.label,n_spikes)

# Rate recorders allocate their bins in advance. Twice the expected duration
# leaves ample room for this seeded example.
rate_recording_end = 2*n_spikes/theoretical_rate
rec_binnedrate = H.RecorderPopulationRate(
    population,rate_recording_end; Δt=bin_width)

# ## Define the network

network = H.RecurrentNetworkExpKernel(
    (connected_population,),(rec_fulltrain,rec_binnedrate))

# ## Run the simulation

function run_simulation!(network,n_spikes,initial_rate)
  t_now = 0.0
  H.reset!(network)
  H.set_initial_rates!(connected_population,initial_rate)
  for _ in 1:n_spikes
    t_now = H.dynamics_step!(t_now,network)
  end
  return t_now
end

last_spike_time = run_simulation!(network,n_spikes,initial_rate)
train = H.get_content(rec_fulltrain)
binned_rate = H.get_content(rec_binnedrate)

empirical_rate = train.n_recorded_spikes/last_spike_time
reached_bins = binned_rate.times .< last_spike_time

(; theoretical_rate,empirical_rate,last_spike_time,
   n_recorded_spikes=train.n_recorded_spikes,
   mean_binned_rate=mean(binned_rate.rates[reached_bins]))

# ## Relaxation toward the asymptotic rate

# The trace starts at `initial_rate`, while the external input makes the initial
# conditional intensity `h_in + w_self * initial_rate`. Its expected value
# relaxes with time constant `τ_kernel/(1-w_self)`.
initial_intensity = h_in + w_self*initial_rate
relaxation_time = τ_kernel/(1-w_self)
expected_rate(t) = theoretical_rate +
    (initial_intensity-theoretical_rate)*exp(-t/relaxation_time)

plot_end = 8*relaxation_time
idx_plot = reached_bins .& (binned_rate.times .<= plot_end)
plot(
    binned_rate.times[idx_plot],binned_rate.rates[idx_plot];
    label="2 s binned rate",color=:steelblue,alpha=0.6,
    xlabel="time",ylabel="rate",ylim=(0,1.8theoretical_rate))
plot!(
    t -> expected_rate(t),0,plot_end;
    label="expected relaxation",color=:black,linewidth=3)
hline!(
    [theoretical_rate];
    label="asymptotic rate",color=:firebrick,linestyle=:dash,linewidth=2)
