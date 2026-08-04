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
    analytic_wei_two_neuron_motif_fixed_rhin(
        αpre::Real,αpost::Real,bias::Real,
        M10::Real,M01::Real,wie::Real,he::Real,rinh::Real)

Analytic solution for a two-neuron motif with one excitatory and one inhibitory
neuron, where the inhibitory neuron is kept at a fixed rate by varying its
external input.

`M10` and `M01` are signed motif coefficients. In particular, `M10` is
negative for an inhibitory presynaptic population. The learned I→E weight is a
nonnegative magnitude, so the signed first-order drift contains
`M10*w_ei*rinh`.

This is the toy-model proposed in Festa,Cusseddu,Gjorgjieva 2026

"""
function analytic_wei_two_neuron_motif_fixed_rhin(
  αpre::Real,αpost::Real,bias::Real,
    M10::Real,M01::Real,wie::Real,he::Real,rinh::Real)
  rate_and_reciprocal = αpost+bias*rinh+M01*wie
  numerator = αpre*rinh + he*rate_and_reciprocal
  denominator = rinh*(rate_and_reciprocal-M10)
  return max(0.0,numerator/denominator)
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


"""
    analytic_alphaprepost(target1, target2;
        rinh, hexc, B, M10, M01)

Compute the rate-dependent constants `αpre` and `αpost` for two desired
operating points of the fixed-`rinh` two-neuron E/I motif.

Each target is a named tuple `(wexc=..., rexc=...)`, where `wexc` is the fixed
E→I weight and `rexc` is the desired excitatory rate. The corresponding
learned I→E weight is `(hexc-rexc)/rinh`; consequently, `rexc == hexc`
specifies a zero learned weight.

With the package's signed `M10` convention, each target satisfies

```math
r_{\\mathrm{inh}}\\alpha_{\\mathrm{pre}} +
r_{\\mathrm{exc}}\\alpha_{\\mathrm{post}} = -\\left[
B r_{\\mathrm{inh}}r_{\\mathrm{exc}} +
M_{01}w_{\\mathrm{exc}}r_{\\mathrm{exc}} +
M_{10}(h_{\\mathrm{exc}}-r_{\\mathrm{exc}})\\right].
```

The two target rates must differ, and both solutions must be stable fixed
points of the analytic weight dynamics. Returns
`(αpre=value, αpost=value)`.
"""
function analytic_alphaprepost(
    target1::NamedTuple{(:wexc,:rexc),<:Tuple{Real,Real}},
    target2::NamedTuple{(:wexc,:rexc),<:Tuple{Real,Real}};
    rinh::Real,hexc::Real,B::Real,M10::Real,M01::Real)
  rinh_f,hexc_f,B_f,M10_f,M01_f,wexc1,rexc1,wexc2,rexc2 = promote(
    float(rinh),float(hexc),float(B),float(M10),float(M01),
    float(target1.wexc),float(target1.rexc),
    float(target2.wexc),float(target2.rexc))
  values = (rinh_f,hexc_f,B_f,M10_f,M01_f,wexc1,rexc1,wexc2,rexc2)
  if !all(isfinite,values)
    throw(ArgumentError("analytic parameters and targets must be finite"))
  end
  if !(rinh_f > 0)
    throw(ArgumentError("rinh must be positive"))
  end
  if !(hexc_f > 0)
    throw(ArgumentError("hexc must be positive"))
  end
  if !(-1 <= B_f <= 1)
    throw(ArgumentError("B must be between -1 and 1"))
  end
  if !(wexc1 >= 0)
    throw(ArgumentError("target wexc values must be nonnegative"))
  end
  if !(wexc2 >= 0)
    throw(ArgumentError("target wexc values must be nonnegative"))
  end
  if !(0 <= rexc1 <= hexc_f)
    throw(ArgumentError("target rexc values must lie between zero and hexc"))
  end
  if !(0 <= rexc2 <= hexc_f)
    throw(ArgumentError("target rexc values must lie between zero and hexc"))
  end
  if rexc1 == rexc2
    throw(ArgumentError("target rexc values must differ"))
  end

  constant1 = B_f*rinh_f*rexc1 + M01_f*wexc1*rexc1 +
    M10_f*(hexc_f-rexc1)
  constant2 = B_f*rinh_f*rexc2 + M01_f*wexc2*rexc2 +
    M10_f*(hexc_f-rexc2)
  αpost = (constant1-constant2)/(rexc2-rexc1)
  αpre = (-constant1-rexc1*αpost)/rinh_f

  stability1 = αpost+B_f*rinh_f+M01_f*wexc1-M10_f
  stability2 = αpost+B_f*rinh_f+M01_f*wexc2-M10_f
  if !(stability1 > 0)
    throw(ArgumentError("both targets must be stable fixed points"))
  end
  if !(stability2 > 0)
    throw(ArgumentError("both targets must be stable fixed points"))
  end
  return (αpre=αpre,αpost=αpost)
end


"""
    analytic_alphaprepost(plasticity, post, pre, target1, target2;
        rinh, hexc)

Compute `αpre` and `αpost` using `plasticity` as a template for `B` and
the STDP kernel parameters. Population arguments follow `post <- pre`: `post`
must be excitatory and `pre` inhibitory. The implied learned weights must lie
within the template's weight bounds.
"""
function analytic_alphaprepost(
    plasticity::Union{PlasticitySymmetricSTDP,PlasticityAsymmetricSTDP},
    post::PopulationExpKernelExcitatory,
    pre::PopulationExpKernelInhibitory,
    target1::NamedTuple{(:wexc,:rexc),<:Tuple{Real,Real}},
    target2::NamedTuple{(:wexc,:rexc),<:Tuple{Real,Real}};
    rinh::Real,hexc::Real)
  M10 = get_motif_coef_M10(plasticity,pre)
  M01 = get_motif_coef_M01(plasticity,post)
  alphas = analytic_alphaprepost(
    target1,target2;rinh=rinh,hexc=hexc,B=plasticity.B,M10=M10,M01=M01)

  rinh_f,hexc_f = promote(float(rinh),float(hexc))
  for target in (target1,target2)
    winh = (hexc_f-float(target.rexc))/rinh_f
    if !(plasticity.weight_min <= winh <= plasticity.weight_max)
      throw(ArgumentError(
        "target implies an I→E weight outside the plasticity bounds"))
    end
  end
  return alphas
end


"""
    analytic_alphaprepost_rule(
        plasticity, post, pre, target1, target2;
        rinh, hexc, weights=nothing)

Create a fresh plasticity rule of the same type as `plasticity`, replacing its
rate-dependent constants with [`analytic_alphaprepost`](@ref). Learning rate,
kernel parameters, and weight bounds are preserved, while all traces are reset.

When `weights` is supplied, its zeros define the structural-zero mask. When it
is omitted, the returned rule assumes all-to-all plasticity with dimensions
determined by `post <- pre`. The supplied matrix is used only to construct the
mask and is not retained by the rule.
"""
function analytic_alphaprepost_rule(
    plasticity::PlasticitySymmetricSTDP,
    post::PopulationExpKernelExcitatory,
    pre::PopulationExpKernelInhibitory,
    target1::NamedTuple{(:wexc,:rexc),<:Tuple{Real,Real}},
    target2::NamedTuple{(:wexc,:rexc),<:Tuple{Real,Real}};
    rinh::Real,hexc::Real,
    weights::Union{Nothing,Matrix{Float64}}=nothing)
  alphas = analytic_alphaprepost(
    plasticity,post,pre,target1,target2;rinh=rinh,hexc=hexc)
  mask_weights = weights
  if isnothing(mask_weights)
    mask_weights = ones(Float64,nneurons(post),nneurons(pre))
  end
  if size(mask_weights) != (nneurons(post),nneurons(pre))
    throw(DimensionMismatch("weights must have size (n_post, n_pre)"))
  end
  return PlasticitySymmetricSTDP(
    plasticity.η,plasticity.B,alphas.αpre,alphas.αpost,
    plasticity.τ_plus,plasticity.γ,mask_weights;
    weight_min=plasticity.weight_min,weight_max=plasticity.weight_max)
end

function analytic_alphaprepost_rule(
    plasticity::PlasticityAsymmetricSTDP,
    post::PopulationExpKernelExcitatory,
    pre::PopulationExpKernelInhibitory,
    target1::NamedTuple{(:wexc,:rexc),<:Tuple{Real,Real}},
    target2::NamedTuple{(:wexc,:rexc),<:Tuple{Real,Real}};
    rinh::Real,hexc::Real,
    weights::Union{Nothing,Matrix{Float64}}=nothing)
  alphas = analytic_alphaprepost(
    plasticity,post,pre,target1,target2;rinh=rinh,hexc=hexc)
  mask_weights = weights
  if isnothing(mask_weights)
    mask_weights = ones(Float64,nneurons(post),nneurons(pre))
  end
  if size(mask_weights) != (nneurons(post),nneurons(pre))
    throw(DimensionMismatch("weights must have size (n_post, n_pre)"))
  end
  return PlasticityAsymmetricSTDP(
    plasticity.η,plasticity.B,alphas.αpre,alphas.αpost,
    plasticity.τ_plus,plasticity.γ,mask_weights;
    weight_min=plasticity.weight_min,weight_max=plasticity.weight_max)
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
