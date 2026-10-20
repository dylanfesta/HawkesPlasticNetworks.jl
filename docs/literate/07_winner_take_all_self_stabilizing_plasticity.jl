# # Winner-take-all self-stabilizing plasticity

# ## Excitatory winner-take-all

# Ten independent excitatory neurons fire at 50 Hz and project to one
# postsynaptic neuron with a nominal target of 50 Hz. Its external drive is
# only 5 Hz. All ten plastic weights start at 0.001; there is no feedback or
# recurrent connectivity. Spike fluctuations break the initial symmetry.
#
# Run from the repository with the documentation environment:
#
# ```sh
# julia --project=docs docs/literate/07_winner_take_all_self_stabilizing_plasticity.jl
# ```
#
# Times are in seconds, rates in Hz, and weights are integrated kernel strengths.
# Each example defines its own population, input, plasticity, initialization,
# and simulation parameters. Later examples do not inherit earlier settings.

using Random
using HawkesPlasticNetworks
const H = HawkesPlasticNetworks

# All parameters below belong to this excitatory-only example.
const N = 10
const r_pre = 50.0
const r_target = 50.0
const h_post = r_target/10
const τ_syn = 30e-3
const τ_plus = 20e-3
const γ = 10.0
const τ_minus = γ*τ_plus
const A_plus = 0.2
const A_minus = 0.8
const B = A_plus-A_minus
const η = 1e-6
const αpre = -B*r_target
const αpost = 0.0
const weight_start = 0.001
const weight_min = 0.0
const weight_max = Inf
const simulation_seed = 0
const simulation_duration = 16*3600.0
const recording_interval = 60.0
const shared_weight_target = (r_target-h_post)/(N*r_pre)
const M = A_plus/(τ_syn+τ_plus)
const winner_weight = (αpre+B*h_post)/(-B*r_pre-M)
const winner_rate = h_post+r_pre*winner_weight;

# ### STDP kernel

# This is the asymmetric STDP window used in the simulation, with the overall
# learning rate ``\eta=10^{-6}`` factored out. For
# ``\Delta t=t_{\mathrm{post}}-t_{\mathrm{pre}}``, causal pairs potentiate and
# reverse-order pairs depress:
#
# ```math
# L(\Delta t)=
# \begin{cases}
# -\dfrac{0.8}{0.2}\exp(\Delta t/0.2), & \Delta t<0,\\[4pt]
# \dfrac{0.2}{0.02}\exp(-\Delta t/0.02), & \Delta t>0.
# \end{cases}
# ```
#
# Depression lasts ten times longer and has four times the area of potentiation.
# The integrated window is ``B=A_+-A_-=-0.6``. The plot uses milliseconds.

using Plots

negative_lags = range(-6τ_minus,0.0;length=1201)
positive_lags = range(0.0,6τ_minus;length=1201)
depression = @. -A_minus/τ_minus*exp(negative_lags/τ_minus)
potentiation = @. A_plus/τ_plus*exp(-positive_lags/τ_plus)
stdp_figure = plot(
    1000 .* negative_lags,depression;
    color=:red,fillrange=0,fillalpha=0.25,linewidth=3,label=false,
    xlabel="Spike lag Δt = t_post − t_pre (ms)",
    ylabel="Learning window L(Δt) (s⁻¹)",
    title="Asymmetric STDP: broad depression, brief potentiation",
    xlims=(-250,250),legend=false,size=(850,450),
    left_margin=6Plots.mm,bottom_margin=3Plots.mm)
plot!(stdp_figure,1000 .* positive_lags,potentiation;
    color=:blue,fillrange=0,fillalpha=0.25,linewidth=3,label=false)
hline!(stdp_figure,[0.0];color=:black,linewidth=1,label=false);

## #src
@assert isapprox(τ_minus/τ_plus,10.0;rtol=1e-12) #src
@assert isapprox(A_minus/A_plus,4.0;rtol=1e-12) #src
@assert isapprox(A_plus+A_minus,1.0;atol=1e-12) #src
@assert all(depression .< 0.0) #src
@assert all(potentiation .> 0.0) #src
display(stdp_figure) #src
## #src
stdp_figure

# ### Rate regulation and competition

# With ``\alpha_{\mathrm{pre}}=-B r_{\mathrm{target}}=30`` and
# ``\alpha_{\mathrm{post}}=0``, a first-order approximation gives
#
# ```math
# \langle\dot w_i\rangle\approx\eta r_{\mathrm{pre}}
# \left[B(r_{\mathrm{post}}-r_{\mathrm{target}})+M w_i\right],
# \qquad M=\frac{A_+}{\tau_{\mathrm{syn}}+\tau_+}=4\,\mathrm{s}^{-1}.
# ```
#
# The negative ``B`` regulates total excitation. The causal term ``M w_i``
# favors stronger synapses, amplifying differences until weaker weights approach
# zero. Here ``|B|r_{\mathrm{pre}}>M``, so total excitation remains stable even
# with a single active synapse. Correlations can shift the learned rate above
# the nominal target. The difference timescale is about 1.39 hours; we run for 16.
#
# Once only one synapse remains, ``r_{\mathrm{post}}=h+r_{\mathrm{pre}}w``.
# Setting its rate and causal-correlation drift to zero gives
#
# ```math
# 0=\alpha_{\mathrm{pre}}+Bh+(Br_{\mathrm{pre}}+M)w_*,
# \qquad
# w_* = \frac{\alpha_{\mathrm{pre}}+Bh}{-Br_{\mathrm{pre}}-M}
# = \frac{27}{26}\approx1.03846.
# ```
#
# This slow-learning fixed point is stable because ``Br_{\mathrm{pre}}+M=-26<0``.
# Its predicted postsynaptic rate is ``h+r_{\mathrm{pre}}w_*\approx56.92`` Hz.
# The rule is self-regulating: no upper weight saturation is needed, only the
# nonnegative lower bound.

# ### Build the feedforward network

# In `post <- pre` order, the plastic matrix has one row and ten columns.
# The presynaptic population has constant input and no incoming connections.
# Weights stay nonnegative with no upper cap. Their nonzero initialization
# keeps all ten synapses eligible for plasticity, including after reaching zero.

function simulate_competition(;
    simulation_duration::Real=simulation_duration,
    recording_interval::Real=recording_interval,seed::Integer=simulation_seed)
  Random.seed!(seed)
  population_post = H.PopulationExpKernelExcitatory(1,τ_syn;label="post")
  population_pre = H.PopulationExpKernelExcitatory(N,τ_syn;label="pre")
  weights = fill(weight_start,1,N)
  rule = H.PlasticityAsymmetricSTDP(
      η,B,αpre,αpost,τ_plus,γ,weights;weight_min=weight_min,weight_max=weight_max)
  connection = H.ConnectionWithWeights(
      population_post,weights,population_pre;plasticity_rules=(rule,))
  connected_post = H.ConnectedPopulationExpKernel(
      population_post,[h_post],(connection,population_pre))
  connected_pre = H.ConnectedPopulationExpKernel(
      population_pre,fill(r_pre,N))
  recorder_weights = H.WeightMatrixRecorder(
      connection.weights,recording_interval,simulation_duration)
  recorder_post = H.RecorderPopulationRate(
      population_post,simulation_duration;Δt=recording_interval)
  recorder_pre = H.RecorderPopulationRate(
      population_pre,simulation_duration;Δt=recording_interval)
  network = H.RecurrentNetworkExpKernel(
      (connected_post,connected_pre),
      (recorder_weights,recorder_post,recorder_pre))

  H.reset!(network)
  H.reset!(rule)
  initial_weights = copy(connection.weights)
  time_now = 0.0
  while time_now < simulation_duration
    time_now = H.dynamics_step!(time_now,network)
  end

  ## Weight snapshots are event-triggered; keep their own time vector.
  ## Prepend the exact initial state rather than assigning it a spike time.
  content_weights = H.get_content(recorder_weights)
  content_post = H.get_content(recorder_post)
  content_pre = H.get_content(recorder_pre)
  return (
      weight_times=vcat(0.0,content_weights.times),
      weights=vcat(initial_weights,content_weights.weights[:,1,:]),
      final_weights=copy(connection.weights),
      rate_times=content_post.times,rates_post=content_post.rates,
      rates_pre=content_pre.rates,
      final_input_pre=copy(connected_pre.input),
      final_input_post=copy(connected_post.input),
      weight_min=rule.weight_min,weight_max=rule.weight_max)
end;

# ### Run the network and follow the ten weights

# Navy curves show the ten weights; the green line marks the single-winner
# fixed point ``w_*\approx1.03846``. The gray dashed line marks equal sharing
# at the nominal target: ``(r_{\mathrm{target}}-h)/(Nr_{\mathrm{pre}})=0.09``.
# The lower panel shows the postsynaptic rate in 60-second bins, with the
# nominal target and the predicted single-winner rate as horizontal references.

execution_seconds = @elapsed begin
  result = simulate_competition()
end
weight_figure = plot(
    result.weight_times./3600,result.weights;
    color=:navy,linewidth=3,label=false,legend=false,
    xlabel="time (hours)",ylabel="excitatory weight",
    title="Ten initially identical synapses",
    size=(1000,550),left_margin=6Plots.mm,bottom_margin=4Plots.mm)
hline!(weight_figure,[winner_weight];
    color=:green,linewidth=3,label=false)
hline!(weight_figure,[shared_weight_target];
    color=:gray,linestyle=:dash,linewidth=2,label=false)
rate_figure = plot(result.rate_times./3600,result.rates_post;
    color=:black,linewidth=1.5,label="postsynaptic rate",
    xlabel="time (hours)",ylabel="rate (Hz)",ylims=(0.0,Inf),legend=false)
hline!(rate_figure,[r_target];
    color=:gray,linestyle=:dash,linewidth=2,label="target (50 Hz)")
hline!(rate_figure,[winner_rate];
    color=:green,linestyle=:dot,linewidth=2,label="single-winner prediction")
competition_figure = plot(weight_figure,rate_figure;
    layout=grid(2,1;heights=[2/3,1/3]),link=:x,xlims=(0.0,simulation_duration/3600),
    size=(1000,825),left_margin=6Plots.mm,bottom_margin=4Plots.mm);

## #src
@assert size(result.weights,2) == N #src
@assert all(isfinite,result.weights) #src
@assert all(result.weights .>= 0.0) #src
@assert all(result.weights[1,:] .== weight_start) #src
@assert h_post <= r_target/10 #src
@assert isapprox(αpre+B*winner_rate+M*winner_weight,0.0;atol=1e-12) #src
@assert isapprox(h_post+r_pre*N*shared_weight_target,r_target;rtol=1e-12) #src
@assert length(result.rate_times) == length(result.rates_post) #src
@assert all(isfinite,result.rates_post) #src
display(competition_figure) #src
## #src
competition_figure

# Simulation execution time, including first-use compilation and excluding plotting:

println("Execution time: ",round(execution_seconds/60;digits=3)," minutes.") #hide


# ## Inhibitory winner-take-all

# Ten independent inhibitory neurons now fire at 25 Hz, with constant external
# inputs and no connections from E. The postsynaptic drive remains 100 Hz,
# above the 50 Hz target: inhibition must remove the excess activity.
# The higher inhibitory rate keeps the final winning weight near 3.
# The initial weights, recording interval, and 16-hour duration are unchanged.

# All parameters below belong to this inhibitory-only example.
const N_inhibitory = 10
const r_I = 25.0
const r_target_inhibitory = 50.0
const h_post_inhibitory = 100.0
const τ_syn_inhibitory = 30e-3
const τ_plus_inhibitory = 20e-3
const γ_inhibitory = 10.0
const τ_minus_inhibitory = γ_inhibitory*τ_plus_inhibitory
const αpost_inhibitory = 0.0
const weight_start_inhibitory = 0.001
const weight_min_inhibitory = 0.0
const weight_max_inhibitory = Inf
const simulation_seed_inhibitory = 0
const simulation_duration_inhibitory = 16*3600.0
const recording_interval_inhibitory = 60.0
const η_inhibitory = 5e-6
const B_inhibitory = 0.6
const αpre_inhibitory = -B_inhibitory*r_target_inhibitory
const A_left_inhibitory = (1+B_inhibitory)/2
const A_right_inhibitory = (1-B_inhibitory)/2
const M_inhibitory = A_right_inhibitory/(τ_syn_inhibitory+τ_plus_inhibitory)
const shared_target_inhibitory = (h_post_inhibitory-r_target_inhibitory)/(N_inhibitory*r_I);

# ### Inhibitory STDP kernel

# Invert the asymmetric window: post-before-pre pairs potentiate over 200 ms,
# while pre-before-post pairs depress over 20 ms. The positive left area is
# 0.8 and the negative right area is 0.2, giving integrated area ``B=0.6``.
# Negative ``\alpha_{\mathrm{pre}}=-30`` balances that positive rate bias.
# `inverted=true` changes the timing signs while keeping `B` equal to the area:
#
# ```math
# L_I(\Delta t)=
# \begin{cases}
# \dfrac{0.8}{0.2}\exp(\Delta t/0.2), & \Delta t<0,\\[4pt]
# -\dfrac{0.2}{0.02}\exp(-\Delta t/0.02), & \Delta t>0.
# \end{cases}
# ```

negative_lags_inhibitory = range(-6τ_minus_inhibitory,0.0;length=1201)
positive_lags_inhibitory = range(0.0,6τ_minus_inhibitory;length=1201)
inhibitory_stdp_figure = plot(
    1000 .* negative_lags_inhibitory,
    A_left_inhibitory/τ_minus_inhibitory .* exp.(negative_lags_inhibitory./τ_minus_inhibitory);
    color=:blue,fillrange=0,fillalpha=0.25,linewidth=3,label=false,legend=false,
    xlabel="Spike lag Δt = t_post − t_pre (ms)",
    ylabel="Learning window Lᵢ(Δt) (s⁻¹)",
    title="Inverted STDP: broad potentiation, brief depression",
    xlims=(-250,250),size=(850,450),
    left_margin=6Plots.mm,bottom_margin=3Plots.mm)
plot!(inhibitory_stdp_figure,1000 .* positive_lags_inhibitory,
    -A_right_inhibitory/τ_plus_inhibitory .* exp.(-positive_lags_inhibitory./τ_plus_inhibitory);
    color=:red,fillrange=0,fillalpha=0.25,linewidth=3,label=false)
hline!(inhibitory_stdp_figure,[0.0];color=:black,linewidth=1,label=false);

## #src
display(inhibitory_stdp_figure) #src
## #src
inhibitory_stdp_figure

# ### Regulation and competition

# Stronger inhibition suppresses causal postsynaptic spikes and therefore
# receives less timing-dependent depression. The causal coefficient is now
# positive, ``M_I=0.2/(\tau_{\mathrm{syn}}+\tau_+)=4\,\mathrm{s}^{-1}``:
#
# ```math
# \langle\dot w_i\rangle\approx\eta_I r_I
# \left[B(r_E-r_{\mathrm{target}})+M_Iw_i\right],
# \qquad r_E\approx h_E-r_I\sum_i w_i.
# ```
#
# Above target inhibition strengthens; below target it weakens. Meanwhile,
# differences grow as ``d(w_i-w_j)/dt=\eta_I r_IM_I(w_i-w_j)``: fluctuations
# select stronger synapses instead of being averaged away. The sum is stable
# in this approximation when ``NBr_I>M_I``. We impose only a zero lower bound;
# no upper weight saturation is needed because the rate feedback is self-regulating.
# At 25 Hz, ``\eta_I=5\times10^{-6}`` gives a difference timescale of about
# 33.3 minutes.
# Strong single inhibitory inputs can rectify E, so the linear approximation
# does not give a reliable numerical single-winner weight at this input rate.

# ### Single-winner estimate with rate rectification

# For one surviving input, retain the clipped intensity ``[h_E-wx_I]_+``.
# Set ``q=r_I\tau_{\mathrm{syn}}``, ``a=\tau_{\mathrm{syn}}/\tau_+``, and
# ``z=h_E\tau_{\mathrm{syn}}/w``. For ``z<1``, averaging over the stationary
# Poisson-driven exponential trace gives
#
# ```math
# r_E(w)=Kz^q,\qquad K=\frac{h_Ee^{-\gamma_{\mathrm{Euler}}q}}{\Gamma(q+2)},
# \qquad C=a\,\mathrm{Beta}(a,q+2).
# ```
#
# The broad positive lobe samples the mean rate, while the short negative lobe
# samples the rate after an inhibitory spike. The resulting drift equation is
#
# ```math
# 0=\alpha_{\mathrm{pre}}+Kz^q\left[A_{\mathrm{left}}-
# A_{\mathrm{right}}Cz^a\right],\qquad
# w_*\approx3.03,\quad r_E^*\approx40.05\,\mathrm{Hz}.
# ```
#
# Solve this scalar equation by bisection. This is a slow-learning estimate
# with one fixed synapse; it includes rate clipping but neglects losing weights
# and fluctuations of the winning weight.

using SpecialFunctions: gamma, beta

function inhibitory_single_winner_prediction()
  q = r_I*τ_syn_inhibitory
  a = τ_syn_inhibitory/τ_plus_inhibitory
  K = h_post_inhibitory*exp(-Float64(Base.MathConstants.eulergamma)*q)/gamma(q+2)
  C = a*beta(a,q+2)
  if αpre_inhibitory >= 0.0
    throw(ArgumentError("the single-winner estimate requires negative αpre"))
  end
  if αpre_inhibitory+K*(A_left_inhibitory-A_right_inhibitory*C) < 0.0
    throw(ArgumentError("this estimate requires a fixed point with w ≥ h_E*τ_syn_inhibitory"))
  end
  lower = 0.0
  upper = 1.0
  for _ in 1:100
    z = (lower+upper)/2
    drift = αpre_inhibitory+K*z^q*(A_left_inhibitory-A_right_inhibitory*C*z^a)
    if drift > 0.0
      upper = z
    else
      lower = z
    end
  end
  z = (lower+upper)/2
  return (weight=h_post_inhibitory*τ_syn_inhibitory/z,rate=K*z^q)
end;

# ### Build the inhibitory feedforward network

# The `E <- I` matrix is plastic. I has only constant external input: there is
# no reciprocal connection and no adaptive-input callback.

function simulate_inhibitory_competition(;
    inhibitory_rate::Real=r_I,learning_rate::Real=η_inhibitory,
    simulation_duration::Real=simulation_duration_inhibitory,
    recording_interval::Real=recording_interval_inhibitory,
    seed::Integer=simulation_seed_inhibitory)
  Random.seed!(seed)
  population_e = H.PopulationExpKernelExcitatory(1,τ_syn_inhibitory;label="e")
  population_i = H.PopulationExpKernelInhibitory(N_inhibitory,τ_syn_inhibitory;label="i")
  weights = fill(weight_start_inhibitory,1,N_inhibitory)
  rule = H.PlasticityAsymmetricSTDP(
      learning_rate,B_inhibitory,αpre_inhibitory,αpost_inhibitory,τ_plus_inhibitory,γ_inhibitory,weights;
      weight_min=weight_min_inhibitory,weight_max=weight_max_inhibitory,inverted=true)
  connection_ei = H.ConnectionWithWeights(
      population_e,weights,population_i;plasticity_rules=(rule,))
  connected_e = H.ConnectedPopulationExpKernel(
      population_e,[h_post_inhibitory],(connection_ei,population_i))
  connected_i = H.ConnectedPopulationExpKernel(
      population_i,fill(Float64(inhibitory_rate),N_inhibitory))
  recorder_weights = H.WeightMatrixRecorder(
      connection_ei.weights,recording_interval,simulation_duration)
  recorder_e = H.RecorderPopulationRate(
      population_e,simulation_duration;Δt=recording_interval)
  recorder_i = H.RecorderPopulationRate(
      population_i,simulation_duration;Δt=recording_interval)

  network = H.RecurrentNetworkExpKernel(
      (connected_e,connected_i),
      (recorder_weights,recorder_e,recorder_i))
  H.reset!(network)
  H.reset!(rule)
  initial_weights = copy(connection_ei.weights)
  time_now = 0.0
  while time_now < simulation_duration
    time_now = H.dynamics_step!(time_now,network)
  end

  content_weights = H.get_content(recorder_weights)
  content_e = H.get_content(recorder_e)
  content_i = H.get_content(recorder_i)
  return (
      weight_times=vcat(0.0,content_weights.times),
      weights=vcat(initial_weights,content_weights.weights[:,1,:]),
      final_weights=copy(connection_ei.weights),
      rate_times=content_e.times,rates_post=content_e.rates,rates_pre=content_i.rates,
      final_input_pre=copy(connected_i.input),
      final_input_post=copy(connected_e.input),
      weight_min=rule.weight_min,weight_max=rule.weight_max)
end;

# ### Run the inhibitory network

# Red curves show the ten inhibitory weights; green marks the clipping-aware
# single-winner estimate ``w_*\approx3.03``. The gray dashed line is the
# nominal equal-sharing reference ``(h_E-r_{\mathrm{target}})/(Nr_I)=0.2``.
# The lower panel shows the postsynaptic rate in 60-second bins, with the
# nominal target and the clipping-aware rate prediction as references.

prediction_inhibitory = inhibitory_single_winner_prediction()
execution_seconds_inhibitory = @elapsed begin
  result_inhibitory = simulate_inhibitory_competition()
end
inhibitory_weight_figure = plot(
    result_inhibitory.weight_times./3600,result_inhibitory.weights;
    color=:red,linewidth=3,label=false,legend=false,
    xlabel="time (hours)",ylabel="inhibitory weight",
    title="Ten inhibitory synapses with inverted STDP",
    size=(1000,550),left_margin=6Plots.mm,bottom_margin=4Plots.mm)
hline!(inhibitory_weight_figure,[prediction_inhibitory.weight];
    color=:green,linewidth=3,label=false)
hline!(inhibitory_weight_figure,[shared_target_inhibitory];
    color=:gray,linestyle=:dash,linewidth=2,label=false)
inhibitory_rate_figure = plot(
    result_inhibitory.rate_times./3600,result_inhibitory.rates_post;
    color=:black,linewidth=1.5,label="postsynaptic rate",
    xlabel="time (hours)",ylabel="rate (Hz)",ylims=(0.0,Inf),legend=false)
hline!(inhibitory_rate_figure,[r_target_inhibitory];
    color=:gray,linestyle=:dash,linewidth=2,label="target (50 Hz)")
hline!(inhibitory_rate_figure,[prediction_inhibitory.rate];
    color=:green,linestyle=:dot,linewidth=2,label="single-winner prediction")
inhibitory_competition_figure = plot(inhibitory_weight_figure,inhibitory_rate_figure;
    layout=grid(2,1;heights=[2/3,1/3]),link=:x,xlims=(0.0,simulation_duration_inhibitory/3600),
    size=(1000,825),left_margin=6Plots.mm,bottom_margin=4Plots.mm);

## #src
@assert all(isfinite,result_inhibitory.weights) #src
@assert all(result_inhibitory.weights .>= 0.0) #src
@assert all(result_inhibitory.weights[1,:] .== weight_start_inhibitory) #src
@assert isinf(result_inhibitory.weight_max) #src
@assert 2.0 <= maximum(result_inhibitory.final_weights) <= 5.0 #src
@assert length(result_inhibitory.rate_times) == length(result_inhibitory.rates_post) #src
@assert all(isfinite,result_inhibitory.rates_post) #src
display(inhibitory_competition_figure) #src
## #src
inhibitory_competition_figure

# Simulation execution time, including first-use compilation and excluding plotting:

println("Execution time: ",round(execution_seconds_inhibitory/60;digits=3)," minutes.") #hide

# ## Both excitatory and inhibitory

# Five excitatory neurons at 50 Hz and five inhibitory neurons at 60 Hz now
# project to the same postsynaptic neuron. Both presynaptic populations have
# constant inputs and no incoming connections. The postsynaptic external drive
# is only 5 Hz. All ten weights start at 0.001.
#
# Excitation uses asymmetric STDP with a fixed target of 10 Hz. Inhibition
# uses adaptive inverted asymmetric STDP: its postsynaptic calcium trace has
# a 30-second time constant, with a target readout bounded between 10 and
# 30 Hz. Both inhibitory alpha terms are zero, so calcium supplies the target.
# These bounds apply to the adaptive target, not to instantaneous firing rates.
#
# Initially the 5 Hz postsynaptic drive is below both targets, favoring stronger
# excitation and weaker inhibition on average. Excitatory rate feedback becomes
# depressive above 10 Hz. The inhibitory target follows the slower calcium
# estimate within 10–30 Hz; correlations can still affect both weight drifts.
# Both excitatory and inhibitory weights are capped at 10.0.

# All parameters below belong to this mixed E/I example.
const N_e_combined = 5
const N_i_combined = 5
const r_e_combined = 50.0
const r_i_combined = 60.0
const h_post_combined = 5.0
const τ_syn_combined = 30e-3
const τ_plus_e_combined = 20e-3
const τ_plus_i_combined = 20e-3
const γ_e_combined = 10.0
const γ_i_combined = 10.0
const A_plus_e_combined = 0.2
const A_minus_e_combined = 0.8
const B_e_combined = A_plus_e_combined-A_minus_e_combined
const B_i_combined = 0.6
const η_e_combined = 5e-6
const η_i_combined = 3.3325e-5
const r_min_i_combined = 10.0
const r_max_i_combined = 30.0
const γ_calcium_i_combined = 1500.0
const r_target_e_combined = 10.0
const αpre_e_combined = -B_e_combined*r_target_e_combined
const αpost_e_combined = 0.0
const αpre_i_combined = 0.0
const αpost_i_combined = 0.0
const weight_start_combined = 0.001
const weight_min_e_combined = 0.0
const weight_min_i_combined = 0.0
const weight_max_e_combined = 10.0
const weight_max_i_combined = 10.0
const simulation_seed_combined = 0
const simulation_duration_combined = 16*3600.0
const recording_interval_combined = 60.0;

# ### Build the combined feedforward network

# Each plastic matrix has shape `1 × 5`, in `post <- pre` order. The inhibitory
# weights are positive magnitudes; the inhibitory population supplies their
# negative effect on the postsynaptic input. Each matrix gets its own rule and
# recorder, retaining its own event-triggered recording times.

function simulate_combined_competition(;
    simulation_duration::Real=simulation_duration_combined,
    recording_interval::Real=recording_interval_combined,seed::Integer=simulation_seed_combined)
  Random.seed!(seed)
  population_post = H.PopulationExpKernelExcitatory(1,τ_syn_combined;label="post")
  population_e = H.PopulationExpKernelExcitatory(N_e_combined,τ_syn_combined;label="pre_e")
  population_i = H.PopulationExpKernelInhibitory(N_i_combined,τ_syn_combined;label="pre_i")
  weights_e = fill(weight_start_combined,1,N_e_combined)
  weights_i = fill(weight_start_combined,1,N_i_combined)
  rule_e = H.PlasticityAsymmetricSTDP(
      η_e_combined,B_e_combined,αpre_e_combined,αpost_e_combined,
      τ_plus_e_combined,γ_e_combined,weights_e;
      weight_min=weight_min_e_combined,weight_max=weight_max_e_combined)
  rule_i = H.PlasticityAdaptiveAsymmetric(
      η_i_combined,B_i_combined,τ_plus_i_combined,γ_i_combined,γ_calcium_i_combined,weights_i;
      r_min=r_min_i_combined,r_max=r_max_i_combined,αpre=αpre_i_combined,αpost=αpost_i_combined,
      weight_min=weight_min_i_combined,weight_max=weight_max_i_combined,inverted=true)
  ## #src
  @assert isapprox(rule_e.αpre+rule_e.B*r_target_e_combined,0.0;atol=1e-12) #src
  @assert rule_i.inverted #src
  @assert rule_i.trace_post_calcium.r_min == 10.0 #src
  @assert rule_i.trace_post_calcium.r_max == 30.0 #src
  @assert isapprox(rule_i.trace_post_calcium.trace.τ,30.0;atol=1e-12,rtol=1e-12) #src
  ## #src
  connection_e = H.ConnectionWithWeights(
      population_post,weights_e,population_e;plasticity_rules=(rule_e,))
  connection_i = H.ConnectionWithWeights(
      population_post,weights_i,population_i;plasticity_rules=(rule_i,))
  connected_post = H.ConnectedPopulationExpKernel(
      population_post,[h_post_combined],
      (connection_e,population_e),(connection_i,population_i))
  connected_e = H.ConnectedPopulationExpKernel(population_e,fill(r_e_combined,N_e_combined))
  connected_i = H.ConnectedPopulationExpKernel(population_i,fill(r_i_combined,N_i_combined))
  recorder_e = H.WeightMatrixRecorder(
      connection_e.weights,recording_interval,simulation_duration)
  recorder_i = H.WeightMatrixRecorder(
      connection_i.weights,recording_interval,simulation_duration)
  recorder_post = H.RecorderPopulationRate(
      population_post,simulation_duration;Δt=recording_interval)
  network = H.RecurrentNetworkExpKernel(
      (connected_post,connected_e,connected_i),
      (recorder_e,recorder_i,recorder_post))

  H.reset!(network)
  H.reset!(rule_e)
  H.reset!(rule_i)
  initial_weights_e = copy(connection_e.weights)
  initial_weights_i = copy(connection_i.weights)
  time_now = 0.0
  while time_now < simulation_duration
    time_now = H.dynamics_step!(time_now,network)
  end

  content_e = H.get_content(recorder_e)
  content_i = H.get_content(recorder_i)
  content_post = H.get_content(recorder_post)
  return (
      weight_times_e=vcat(0.0,content_e.times),
      weight_times_i=vcat(0.0,content_i.times),
      weights_e=vcat(initial_weights_e,content_e.weights[:,1,:]),
      weights_i=vcat(initial_weights_i,content_i.weights[:,1,:]),
      final_weights_e=copy(connection_e.weights),
      final_weights_i=copy(connection_i.weights),
      rate_times=content_post.times,rates_post=content_post.rates)
end;

# ### Follow all ten weights

# Navy curves show the five excitatory weights; red curves show the five
# inhibitory weight magnitudes. Both populations learn throughout the run,
# with learning rates 5e-6 for excitation and 3.3325e-5 for inhibition.
# The lower panel shows the postsynaptic rate in 60-second bins. Red dashed
# lines mark the inhibitory target bounds (10 and 30 Hz); the navy dotted line
# marks the fixed excitatory target (10 Hz).

execution_seconds_combined = @elapsed begin
  result_combined = simulate_combined_competition()
end
combined_weight_figure = plot(
    result_combined.weight_times_e./3600,result_combined.weights_e;
    color=:navy,linewidth=3,label=false,legend=false,
    xlabel="time (hours)",ylabel="weight magnitude",
    title="Both excitatory and inhibitory",
    ylims=(0.0,1.05max(weight_max_e_combined,weight_max_i_combined)),
    size=(1000,550),left_margin=6Plots.mm,bottom_margin=4Plots.mm)
plot!(combined_weight_figure,
    result_combined.weight_times_i./3600,result_combined.weights_i;
    color=:red,linewidth=3,label=false)
combined_rate_figure = plot(
    result_combined.rate_times./3600,result_combined.rates_post;
    color=:black,linewidth=1.5,label="postsynaptic rate",
    xlabel="time (hours)",ylabel="rate (Hz)",ylims=(0.0,Inf),legend=false)
hline!(combined_rate_figure,[r_min_i_combined,r_max_i_combined];
    color=:red,linestyle=:dash,linewidth=2,label="I target bounds (10–30 Hz)")
hline!(combined_rate_figure,[r_target_e_combined];
    color=:navy,linestyle=:dot,linewidth=2,label="E target (10 Hz)")
combined_competition_figure = plot(combined_weight_figure,combined_rate_figure;
    layout=grid(2,1;heights=[2/3,1/3]),link=:x,xlims=(0.0,simulation_duration_combined/3600),
    size=(1000,825),left_margin=6Plots.mm,bottom_margin=4Plots.mm);

## #src
for (times,weights,final_weights,weight_max) in ( #src
    (result_combined.weight_times_e,result_combined.weights_e,result_combined.final_weights_e,weight_max_e_combined), #src
    (result_combined.weight_times_i,result_combined.weights_i,result_combined.final_weights_i,weight_max_i_combined)) #src
  @assert size(weights) == (length(times),size(final_weights,2)) #src
  @assert times[1] == 0.0 #src
  @assert all(diff(times) .> 0.0) #src
  @assert all(isfinite,weights) #src
  @assert all(0.0 .<= weights .<= weight_max) #src
  @assert all(weights[1,:] .== weight_start_combined) #src
  @assert all(isfinite,final_weights) #src
  @assert all(0.0 .<= final_weights .<= weight_max) #src
end #src
@assert all(isfinite,result_combined.rates_post) #src
@assert length(result_combined.rate_times) == length(result_combined.rates_post) #src
display(combined_competition_figure) #src
## #src
combined_competition_figure

# Simulation execution time, including first-use compilation and excluding plotting:

println("Execution time: ",round(execution_seconds_combined/60;digits=3)," minutes.") #hide
