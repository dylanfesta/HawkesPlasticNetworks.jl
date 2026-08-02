#=
This file is a sub-section of HawkesPlasticNetworks.jl.
It deals with analytical solutions to the Hawkes process equations.
And other analytic convenience functions.
It considers only simple cases for the time being
=#

# @inline function interaction_kernel(t::Real,pop::PopulationExpKernelExcitatory)
#   τ = pop.trace.τ
#    return interaction_kernel(t,KernelExp(τ))
# end
# @inline function interaction_kernel_upper(t::Real,psker::PopulationStateExpKernel)
#   τ = psker.traces[1].τ
#    return interaction_kernel_upper(t,KernelExp(τ))
# end
# @inline function interaction_kernel_fourier(ω::Real,psker::PopulationStateExpKernel)
#   τ = psker.traces[1].τ
#    return interaction_kernel_fourier(ω,KernelExp(τ))
# end


"""
    expected_rate_EI(
        w_ee::Real,w_ie::Real,
        w_ei::Real,w_ii::Real,
        h_e::Real,h_i::Real)

Computes the expected firing rates for a two population EI network
Weight arguments are in absolute value.
Assumes this linear dynamics.

τ dr_e/dt = -r_e + (w_ee r_e - w_ei r_i + h_e)
τ dr_i/dt = -r_i + (w_ie r_e - w_ii r_i + h_i)

So the solution is (Id-Wmat)^-1 * hvec.

Returns: r_e,r_i

"""
function expected_rate_EI(
    w_ee::Real,w_ie::Real,w_ei::Real,w_ii::Real,
    h_e::Real,h_i::Real)
  wmat = [w_ee -abs(w_ei); w_ie -abs(w_ii)]
  hvec = [h_e; h_i]
  r_e, r_i = (I-wmat)\hvec
  return r_e,r_i
end



# Before computing motif coefficients, extract the interaction-kernel time
# constant and sign from its source population.

function get_tau_and_sign(pop::PopulationExpKernelExcitatory)
  return pop.trace.τ,+1.0
end

function get_tau_and_sign(pop::PopulationExpKernelInhibitory)
  return pop.trace.τ,-1.0
end

"""
    get_motif_coef_M10(stdp, pop_ij)

Compute the `M10` motif coefficient for the interaction `i <- j`, whose
kernel time constant and sign are supplied by `pop_ij`.
"""
function get_motif_coef_M10(
    stdp::PlasticitySymmetricSTDP,
    pop_ij::AbstractPopulation)
  τplus = stdp.τ_plus
  τminus = stdp.γ*stdp.τ_plus
  τker,sign = get_tau_and_sign(pop_ij)
  scale_plus = (stdp.B+1.0)/4.0
  scale_minus = (stdp.B-1.0)/4.0
  return sign*(scale_plus/(τker+τplus)+scale_minus/(τker+τminus))
end

"""
    get_motif_coef_M01(stdp, pop_ji)

Compute the `M01` motif coefficient for the reciprocal interaction `j <- i`,
whose kernel time constant and sign are supplied by `pop_ji`.
"""
function get_motif_coef_M01(
    stdp::PlasticitySymmetricSTDP,
    pop_ji::AbstractPopulation)
  return get_motif_coef_M10(stdp,pop_ji)
end

function get_motif_coef_M10(
    stdp::PlasticityAsymmetricSTDP,
    pop_ij::AbstractPopulation)
  τplus = stdp.τ_plus
  τker,sign = get_tau_and_sign(pop_ij)
  scale_plus = (stdp.B+1.0)/2.0
  return sign*scale_plus/(τker+τplus)
end

function get_motif_coef_M01(
    stdp::PlasticityAsymmetricSTDP,
    pop_ji::AbstractPopulation)
  τminus = stdp.γ*stdp.τ_plus
  τker,sign = get_tau_and_sign(pop_ji)
  scale_minus = (stdp.B-1.0)/2.0
  return sign*scale_minus/(τker+τminus)
end



"""
    analytic_wei_with_motifs_1D(
        αpre::Real,αpost::Real,bias::Real,
        M10::Real,M01::Real,wie::Real,he::Real,rinh::Real)

Analytic solution for a two neruon motif with one excitatory and one inhibitory neuron.
Where the inhibitory neuron is kept at a fixed rate by ad-hoc varying input.

This is the toy-model proposed in Festa,Cusseddu,Gjorgjieva 2026

"""
function analytic_wei_two_neuron_motif_fixed_rhin(
  αpre::Real,αpost::Real,bias::Real,
    M10::Real,M01::Real,wie::Real,he::Real,rinh::Real)
  somefactor = αpost+bias*rinh+M01*wie
  _up = αpre*rinh + he*somefactor
  _down =  rinh*(somefactor+M10)
  return max(0.0,_up/_down)
end

function analytic_wei_two_neuron_motif_fixed_rhin(
  plasticity::Union{PlasticitySymmetricSTDP,PlasticityAsymmetricSTDP},
  pop_ij::AbstractPopulation,
  pop_ji::AbstractPopulation,
  wie::Real,he::Real,rinh::Real)
  M10 = get_motif_coef_M10(plasticity,pop_ij)
  M01 = get_motif_coef_M01(plasticity,pop_ji)
  return analytic_wei_two_neuron_motif_fixed_rhin(
    plasticity.αpre,plasticity.αpost,plasticity.B,
    M10,M01,wie,he,rinh)
end


#=

########
# Symmetric
########

function M_10(stdp::STDPSymmetric{R},kerij::KernelExp{R,NT}) where {R,NT}
  A,τ,θ,γ = get_parameters(stdp)
  τplus = τ
  τminus = γ*τ
  τij = kerij.τ
  Gj = (τij+τminus)/(τij+τplus)
  return ker_sign(NT,R)*A*(Gj+θ) / (τij+τminus) 
end
function M_01(stdp::STDPSymmetric{R},kerji::KernelExp{R,NT}) where {R,NT}
  A,τ,θ,γ = get_parameters(stdp)
  τplus = τ
  τminus = γ*τ
  τji = kerji.τ
  Gi = (τji+τminus)/(τji+τplus)
  return ker_sign(NT,R)*A*(Gi+θ) / (τji+τminus) 
end


########
# Asymmetric
########

function M_10(stdp::STDPAsymmetric{R},kerij::KernelExp{R,NT}) where {R,NT}
  A,τ,_,_ = get_parameters(stdp)
  τplus = τ
  τij = kerij.τ
  return ker_sign(NT,R)*A/(τij+τplus) 
end
function M_01(stdp::STDPAsymmetric{R},kerji::KernelExp{R,NT}) where {R,NT}
  A,τ,θ,γ = get_parameters(stdp)
  τminus = γ*τ
  τji = kerji.τ
  return ker_sign(NT,R)*θ*A/(τji+τminus) 
end



=#
