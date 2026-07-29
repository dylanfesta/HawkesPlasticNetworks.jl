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
