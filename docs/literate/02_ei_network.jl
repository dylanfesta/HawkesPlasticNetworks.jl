# # One excitatory and one inhibitory unit

#=
This example considers two mutually connected units: one excitatory and one
inhibitory. Each postsynaptic unit receives a connection from both units, for a
total of four connections.
=#

# ## Initialization
## #src
using LinearAlgebra
using Plots
using Random
Random.seed!(0)

using HawkesPlasticNetworks; global const H = HawkesPlasticNetworks
## #src
# ## Define the parameters

τ_kernel = [20.0,10.0]
weights = [1.3 -1.2; 1.2 -1.0]
input = [40.0,10.0]
simulation_end = 1_000.0
bin_width = 10.0

# The matrix uses `post <- pre` ordering. Its first column is excitatory and its
# second column is inhibitory. Connections store only positive magnitudes; the
# presynaptic population type determines whether a contribution is added or
# subtracted.

to_matrix(x::Real) = cat(x;dims=2)

weight_exc_exc = to_matrix(abs(weights[1,1]))
weight_exc_inh = to_matrix(abs(weights[1,2]))
weight_inh_exc = to_matrix(abs(weights[2,1]))
weight_inh_inh = to_matrix(abs(weights[2,2]))

expected_rates = (I-weights)\input

println("Expected excitatory rate: ",expected_rates[1])
println("Expected inhibitory rate: ",expected_rates[2])

##

# ## Define the populations and connections

population_exc = H.PopulationExpKernelExcitatory(
    1,τ_kernel[1];label="exc")
population_inh = H.PopulationExpKernelInhibitory(
    1,τ_kernel[2];label="inh")

connection_exc_exc = H.ConnectionWithWeights(
    population_exc,weight_exc_exc,population_exc)
connection_exc_inh = H.ConnectionWithWeights(
    population_exc,weight_exc_inh,population_inh)
connection_inh_exc = H.ConnectionWithWeights(
    population_inh,weight_inh_exc,population_exc)
connection_inh_inh = H.ConnectionWithWeights(
    population_inh,weight_inh_inh,population_inh)

connected_exc = H.ConnectedPopulationExpKernel(
    population_exc,[input[1]],
    (connection_exc_exc,population_exc),
    (connection_exc_inh,population_inh))
connected_inh = H.ConnectedPopulationExpKernel(
    population_inh,[input[2]],
    (connection_inh_exc,population_exc),
    (connection_inh_inh,population_inh))

# ## Record the population rates

rec_rate_exc = H.RecorderPopulationRate(
    population_exc,simulation_end;Δt=bin_width)
rec_rate_inh = H.RecorderPopulationRate(
    population_inh,simulation_end;Δt=bin_width)

network = H.RecurrentNetworkExpKernel(
    (connected_exc,connected_inh),(rec_rate_exc,rec_rate_inh))

# ## Run the simulation

function run_simulation!(network,simulation_end)
  t_now = 0.0
  H.reset!(network)
  while t_now < simulation_end
    t_now = H.dynamics_step!(t_now,network)
  end
  return t_now
end

last_spike_time = run_simulation!(network,simulation_end)
rate_exc = H.get_content(rec_rate_exc)
rate_inh = H.get_content(rec_rate_inh)

# ## Convergence toward the expected rates

my_line_width=2

plot_exc = plot(
    rate_exc.times,rate_exc.rates;
    label="$(bin_width) s binned rate",color=:blue,linewidth=my_line_width,
    xlabel="",ylabel="excitatory rate (Hz)",
    ylim=(0,1.1maximum(rate_exc.rates)))
hline!(
    plot_exc,[expected_rates[1]];
    label="expected rate",color=:grey,linestyle=:dash,linewidth=2)

plot_inh = plot(
    rate_inh.times,rate_inh.rates;
    label="$(bin_width) s binned rate",color=:red,linewidth=my_line_width,
    xlabel="time (s)",ylabel="inhibitory rate (Hz)",
    ylim=(0,1.1maximum(rate_inh.rates)))
hline!(
    plot_inh,[expected_rates[2]];
    label="expected rate",color=:grey,linestyle=:dash,linewidth=2)

plot(plot_exc,plot_inh;layout=(2,1),size=(800,600))
