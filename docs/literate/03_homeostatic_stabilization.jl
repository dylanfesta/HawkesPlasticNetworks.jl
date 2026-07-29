# # Stabilization of large E/I recurrent networks through homeostatic plasticity

#=

In this example, I show how two flavors of homeostatic
inhibitory-to-excitatory plasticity can stabilize
a large E/I network, so that the exc population
reaches a target firing rate determined below.

=#

# ## Initialization
## #src
using Random
using Statistics

## #src

using Distributions
using Plots, Colors
Random.seed!(0)

using HawkesPlasticNetworks; global const H = HawkesPlasticNetworks
## #src
# ## Define the parameters
ne = 100       # number of excitatory neurons
ni = 20        # number of inhibitory neurons
τ_e = 20E-3    # time constant for excitatory neurons
τ_i = 10E-3    # time constant for inhibitory neurons

n_spikes = 1_000_000  # total number of spikes to simulate (for all neurons!)

w_ee = 1.295       # excitatory-to-excitatory connection weight
w_ie = 1.2         # excitatory-to-inhibitory connection weight
w_ei_example = 1.2 # inhibitory-to-excitatory connection weight (this will actually come from plasticity)
w_ii = 1.0         # inhibitory-to-inhibitory connection weight

h_e = 40.0     # external input to excitatory neurons
h_i = 10.0     # external input to inhibitory neurons

p_ee = 0.4     # sparseness for E->E
p_ei = 1.0     # sparseness for I->E
p_ie = 0.6     # sparseness for E->I
p_ii = 0.8     # sparseness for I->I

bin_width = 5.0 # bin width for rate recording

n_tot = ne + ni
idx_e = 1:ne
idx_i = ne+1:n_tot;

## #src

# ## Check the expected rates

rate_example_e,rate_example_i = H.expected_rate_EI(
    w_ee,w_ie,w_ei_example,w_ii,
    h_e,h_i)

re_target = rate_example_e # the target rate will be the same we calculate here
ri_target = rate_example_i

println("Expected excitatory rate: ",rate_example_e)
println("Expected inhibitory rate: ",rate_example_i)


## #src

# ## Create expanded weight matrices

wmat_ee = H.generate_sparse_matrix_fixed_input(ne,ne,w_ee,p_ee;self_connections=false)
wmat_ie = H.generate_sparse_matrix_fixed_input(ni,ne,w_ie,p_ie)
wmat_ii = H.generate_sparse_matrix_fixed_input(ni,ni,w_ii,p_ii;self_connections=false)

# We still need to initialize wmat_ei, the plastic part
# let's use a very weak long-tailed distribution for the weights
# with this choice, we expect the network to be unstable at first
wmat_ei_start = H.generate_sparse_matrix_fixed_input(ne,ni,0.01,p_ei)
ei_distr = Exponential(3.0)
for i in eachindex(wmat_ei_start)
  if wmat_ei_start[i] != 0.0
    wmat_ei_start[i] *= rand(ei_distr)
  end
end
ei_connection_mask = .!iszero.(wmat_ei_start)


# And a full version for plotting purposes

wmat_full = Matrix{Float64}(undef,n_tot,n_tot)
wmat_full[idx_e,idx_e] = wmat_ee
wmat_full[idx_e,idx_i] = .- wmat_ei_start
wmat_full[idx_i,idx_e] = wmat_ie
wmat_full[idx_i,idx_i] = .- wmat_ii

function do_colormap(minval::Real,maxval::Real;
    cminus=:red,czero=:white,cplus=:blue,
    ncolors::Integer=500)
  if minval >= 0
    return cgrad([czero,cplus], [0,1])
  end
  if maxval <= 0
    return cgrad([cminus,czero], [0,1])
  end
  _mid = minval/(minval-maxval)
  vcols = [cminus,fill(czero,ncolors-2)...,cplus]
  midrange = range(_mid-1E-10,_mid+1E-10,length=ncolors-2)
  vals = [0.,midrange...,1.]
  return cgrad(vcols,vals)
end

function Wplot(W::Matrix{<:Real},bounds::Tuple{<:Real,<:Real}=extrema(W);
    title::String="")
  N=size(W,1)
  _cmap = do_colormap(bounds...)
  return heatmap(W;ratio=1,
    xlims=(0,N).+0.5,
    ylims=(0,N).+0.5,
    xlabel="pre",
    ylabel="post",
    color=_cmap,
    clims=bounds,
    title=title)
end;
nothing #hide

Wplot(wmat_full;title="Initial recurrent weights")

## #src

# ## Network 1, Vogels-Sprekeler rate homeostatic rule

# Plasticity parameters
τplast = 20E-3 # time constant in line with the kernel
ηplast = 1E-6; # learning rate

# Define populations, one excitatory one inhibitory

population_exc = H.PopulationExpKernelExcitatory(
    ne,τ_e;label="exc")
population_inh = H.PopulationExpKernelInhibitory(
    ni,τ_i;label="inh");

# Define fixed connections
connection_ee = H.ConnectionWithWeights(
    population_exc,wmat_ee,population_exc)
connection_ie = H.ConnectionWithWeights(
    population_inh,wmat_ie,population_exc)
connection_ii = H.ConnectionWithWeights(
    population_inh,wmat_ii,population_inh);

# Now the plasticity rule for the inhibitory-to-excitatory weights 

plast_vog_sprek = H.PlasticityVogelsSprekeler(ηplast,
  re_target,τplast,wmat_ei_start;
  weight_min=0.0);

connection_ei = H.ConnectionWithWeights(
    population_exc,wmat_ei_start,population_inh;
    plasticity_rules=(plast_vog_sprek,))

connected_exc = H.ConnectedPopulationExpKernel(
    population_exc,fill(h_e,ne),
    (connection_ee,population_exc),
    (connection_ei,population_inh))
connected_inh = H.ConnectedPopulationExpKernel(
    population_inh,fill(h_i,ni),
    (connection_ie,population_exc),
    (connection_ii,population_inh));

# ## Record the population rates and weights

# Rate recorders allocate their bins in advance. 
# I expect the network to be unstable at first, so 
# I set a very long recorder duration.

target_total_rate = ne*re_target + ni*ri_target
t_end_recorders = 100.0*n_spikes/target_total_rate 

rec_rate_exc = H.RecorderPopulationRate(
    population_exc,t_end_recorders;Δt=bin_width)
rec_rate_inh = H.RecorderPopulationRate(
    population_inh,t_end_recorders;Δt=bin_width)

rec_w_ei = H.WeightMatrixRecorder(
    connection_ei.weights,bin_width,t_end_recorders)

network = H.RecurrentNetworkExpKernel(
    (connected_exc,connected_inh),
    (rec_rate_exc,rec_rate_inh,rec_w_ei));

# ## Run the simulation

function run_simulation!(network,n_spikes)
  t_now = 0.0
  H.reset!(network)
  H.set_initial_rates!(connected_exc,h_e)
  H.set_initial_rates!(connected_inh,h_i)
  for _ in 1:n_spikes
    t_now = H.dynamics_step!(t_now,network)
  end
  return t_now
end

last_spike_time = run_simulation!(network,n_spikes)
println("Network run completed!")
println("Last spike time: ",round(last_spike_time;digits=2)," s")

rates_exc = H.get_content(rec_rate_exc)
rates_inh = H.get_content(rec_rate_inh)
weights_ei_content = H.get_content(rec_w_ei)

## #src
# ## Convergence toward the target rates

my_line_width = 2

plot_exc = plot(
    rates_exc.times,rates_exc.rates;
    label="$(bin_width) s binned rate",color=:blue,
    linewidth=my_line_width,xlabel="",ylabel="excitatory rate (Hz)",
    legend=:bottomright)
hline!(
    plot_exc,[re_target];
    label="target rate",color=:grey,linestyle=:dash,linewidth=2)

plot_inh = plot(
    rates_inh.times,rates_inh.rates;
    label="$(bin_width) s binned rate",color=:red,
    linewidth=my_line_width,xlabel="time (s)",
    ylabel="inhibitory rate (Hz)",legend=:bottomright)
hline!(
    plot_inh,[ri_target];
    label="target rate",color=:grey,linestyle=:dash,linewidth=2)

plot(plot_exc,plot_inh;layout=(2,1),size=(800,600))

## #src

# ## Evolution of inhibitory-to-excitatory weights

# Structural zeros represent absent connections and are excluded from the
# summary. The shaded region spans the 0.25 to 0.75 quantiles.
n_weight_times = length(weights_ei_content.times)
weights_ei_connected = reshape(
    weights_ei_content.weights,n_weight_times,:)[:,vec(ei_connection_mask)]
mean_weights_ei = vec(mean(weights_ei_connected;dims=2))
quantiles_weights_ei = reduce(
    hcat,quantile(row,[0.25,0.75]) for row in eachrow(weights_ei_connected))
lower_weights_ei = vec(quantiles_weights_ei[1,:])
upper_weights_ei = vec(quantiles_weights_ei[2,:])

plot(
    weights_ei_content.times,mean_weights_ei;
    ribbon=(
        mean_weights_ei .- lower_weights_ei,
        upper_weights_ei .- mean_weights_ei),
    fillalpha=0.25,
    label="mean with 0.25–0.75 quantiles",
    color=:purple,
    linewidth=my_line_width,
    xlabel="time (s)",
    ylabel="inhibitory-to-excitatory weight")
