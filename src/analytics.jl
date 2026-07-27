#= 
This file is a sub-section of HawkesPlasticNetworks.jl.
It deals with analytical solutions to the Hawkes process equations.
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
