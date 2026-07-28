using HawkesPlasticNetworks
using Test

mutable struct TestPlasticityRule <: HawkesPlasticNetworks.AbstractPlasticityRule
    calls::Vector{Tuple{Symbol,Symbol,Float64,Symbol,Int}}
end

function HawkesPlasticNetworks.apply_plasticity!(
        rule::TestPlasticityRule,
        connection::HawkesPlasticNetworks.ConnectionWithWeights,
        t_fire::Real,
        population_fire_label::Symbol,
        neuron_fire_idx::Integer)
    push!(rule.calls,(
        connection.post_pop_label,
        connection.pre_pop_label,
        Float64(t_fire),
        population_fire_label,
        Int(neuron_fire_idx)))
    return nothing
end

@testset "HawkesPlasticNetworks.jl" begin
    @testset "TracePlasticity construction" begin
        trace = HawkesPlasticNetworks.Trace(20.0, 3)

        @test trace.val == zeros(3)
        @test trace.τ == 20.0
        @test trace.t_last == 0.0
    end

    @testset "Population convenience construction" begin
        pop_exc = HawkesPlasticNetworks.PopulationExpKernelExcitatory(
            2,0.5;label="exc")
        pop_inh = HawkesPlasticNetworks.PopulationExpKernelInhibitory(
            3,0.25;label="inh")
        pop_inh_from_trace = HawkesPlasticNetworks.PopulationExpKernelInhibitory(
            1,HawkesPlasticNetworks.Trace(0.75,1);label="inh_trace")

        @test pop_exc.label == :exc
        @test pop_exc.trace.τ == 0.5
        @test pop_exc.trace.val == zeros(2)
        @test pop_inh isa HawkesPlasticNetworks.PopulationExpKernelInhibitory
        @test pop_inh.label == :inh
        @test pop_inh.trace.τ == 0.25
        @test pop_inh.trace.val == zeros(3)
        @test pop_inh_from_trace isa
            HawkesPlasticNetworks.PopulationExpKernelInhibitory
        @test pop_inh_from_trace.trace.τ == 0.75
    end

    @testset "RecorderPopulationRate construction" begin
        rec = HawkesPlasticNetworks.RecorderPopulationRate(
            :exc, 2, 25.0; Tstart=5.0, Δt=10.0)

        @test rec.times == [10.0, 20.0]
        @test rec.rates == [0.0, 0.0]
        @test_throws ArgumentError HawkesPlasticNetworks.RecorderPopulationRate(
            :exc, 0, 10.0)
        @test_throws ArgumentError HawkesPlasticNetworks.RecorderPopulationRate(
            :exc, 2, 10.0; Δt=0.0)
        @test_throws ArgumentError HawkesPlasticNetworks.RecorderPopulationRate(
            :exc, 2, -1.0)
    end

    @testset "RecorderPopulationRate recording and content" begin
        rec = HawkesPlasticNetworks.RecorderPopulationRate(
            :exc, 2, 25.0; Tstart=5.0, Δt=10.0)

        for t_fire in (5.0, 14.999, 15.0)
            HawkesPlasticNetworks.record_stuff!(rec,t_fire,1,:exc,1)
        end
        HawkesPlasticNetworks.record_stuff!(rec,10.0,1,:inh,1)
        HawkesPlasticNetworks.record_stuff!(rec,25.0,1,:exc,1)

        @test rec.rates == [0.1, 0.05]

        content = HawkesPlasticNetworks.get_content(rec)
        @test content.population_label == :exc
        @test content.times == [10.0, 20.0]
        @test content.rates == [0.1, 0.05]

        content.rates[1] = -1.0
        @test rec.rates[1] == 0.1

        @test HawkesPlasticNetworks.reset!(rec) === nothing
        @test rec.rates == [0.0, 0.0]
    end

    @testset "RecorderPopulationTrain starts at the first slot" begin
        rec = HawkesPlasticNetworks.RecorderPopulationTrain(:exc,2)
        HawkesPlasticNetworks.record_stuff!(rec,1.0,1,:exc,2)

        @test HawkesPlasticNetworks.get_content(rec).times == [1.0]
        @test HawkesPlasticNetworks.get_content(rec).neurons == [2]
    end

    @testset "Connected population rates and network reset" begin
        pop = HawkesPlasticNetworks.PopulationExpKernelExcitatory(
            1,1.0;label="unit")
        connection = HawkesPlasticNetworks.ConnectionWithWeights(
            pop,fill(0.5,1,1),pop)
        connected = HawkesPlasticNetworks.ConnectedPopulationExpKernel(
            pop,[0.6],(connection,pop))
        rec = HawkesPlasticNetworks.RecorderPopulationRate(
            pop,10.0;Δt=2.0)
        network = HawkesPlasticNetworks.RecurrentNetworkExpKernel(
            (connected,),(rec,))
        rates = zeros(1)

        HawkesPlasticNetworks.compute_rates!(rates,0.0,connected)
        @test rates == [0.6]
        pop.trace.val[1] = 2.0
        HawkesPlasticNetworks.compute_rates_upper!(rates,0.0,connected)
        @test rates == [1.6]

        HawkesPlasticNetworks.set_initial_rates!(connected,0.7)
        @test pop.trace.val == [0.7]
        rec.rates[1] = 1.0
        @test HawkesPlasticNetworks.reset!(network) === nothing
        @test pop.trace.val == [0.0]
        @test rec.rates == zeros(5)
    end

    @testset "Inhibitory populations subtract positive connection weights" begin
        pop_post = HawkesPlasticNetworks.PopulationExpKernelInhibitory(
            1,2.0;label="post")
        pop_exc = HawkesPlasticNetworks.PopulationExpKernelExcitatory(
            1,2.0;label="exc")
        pop_inh = HawkesPlasticNetworks.PopulationExpKernelInhibitory(
            1,2.0;label="inh")
        connection_exc = HawkesPlasticNetworks.ConnectionWithWeights(
            pop_post,fill(0.6,1,1),pop_exc)
        connection_inh = HawkesPlasticNetworks.ConnectionWithWeights(
            pop_post,fill(0.4,1,1),pop_inh)
        connected = HawkesPlasticNetworks.ConnectedPopulationExpKernel(
            pop_post,[1.0],
            (connection_exc,pop_exc),(connection_inh,pop_inh))
        rates = zeros(1)

        HawkesPlasticNetworks.set_initial_rates!(pop_exc,2.0)
        HawkesPlasticNetworks.set_initial_rates!(pop_inh,3.0)
        @test HawkesPlasticNetworks.compute_rates!(
            rates,0.0,connected) === nothing
        @test rates ≈ [1.0]
    end

    @testset "Recorder iteration" begin
        rec_exc = HawkesPlasticNetworks.RecorderPopulationRate(
            :exc, 2, 10.0; Δt=10.0)
        rec_inh = HawkesPlasticNetworks.RecorderPopulationRate(
            :inh, 4, 10.0; Δt=10.0)

        @test HawkesPlasticNetworks.recorders_iterator!(
            5.0,1,:exc,2,(rec_exc,rec_inh)) === nothing
        @test rec_exc.rates == [0.05]
        @test rec_inh.rates == [0.0]
        @test HawkesPlasticNetworks.recorders_iterator!(
            5.0,1,:exc,2,()) === nothing
    end

    @testset "Tuple-recursive base cases" begin
        @test HawkesPlasticNetworks.compute_next_spike_population_iterator(
            1.0,()) == (Inf,0,:nope,-1)
        @test HawkesPlasticNetworks.burn_spike_iterator!(
            1.0,1,1,()) === nothing
        @test HawkesPlasticNetworks.burn_spike_iterator_legacy!(
            1.0,1,1,()) === nothing

        pop_a = HawkesPlasticNetworks.PopulationExpKernelExcitatory(
            2,HawkesPlasticNetworks.Trace(20.0,2);label="a")
        pop_b = HawkesPlasticNetworks.PopulationExpKernelExcitatory(
            2,HawkesPlasticNetworks.Trace(20.0,2);label="b")
        connected_a = HawkesPlasticNetworks.ConnectedPopulationExpKernel(
            pop_a,(),(),zeros(2))
        connected_b = HawkesPlasticNetworks.ConnectedPopulationExpKernel(
            pop_b,(),(),zeros(2))

        @test HawkesPlasticNetworks.burn_spike_iterator!(
            2.0,2,1,(connected_a,connected_b)) === nothing
        @test pop_a.trace.t_last == 0.0
        @test pop_b.trace.t_last == 2.0
        @test isapprox(pop_b.trace.val[1],inv(20.0))
    end

    @testset "Plasticity rule folding" begin
        calls = Tuple{Symbol,Symbol,Float64,Symbol,Int}[]
        rule_a = TestPlasticityRule(calls)
        rule_b = TestPlasticityRule(calls)
        trace = HawkesPlasticNetworks.Trace(20.0,2)
        post = HawkesPlasticNetworks.PopulationExpKernelExcitatory(
            2,trace;label="post")
        pre = HawkesPlasticNetworks.PopulationExpKernelExcitatory(
            2,HawkesPlasticNetworks.Trace(20.0,2);label="pre")
        plastic_connection = HawkesPlasticNetworks.ConnectionWithWeights(
            zeros(2,2),:post,:pre,(rule_a,rule_b),true)
        fixed_connection = HawkesPlasticNetworks.ConnectionWithWeights(
            zeros(2,2),:post,:fixed,(rule_a,),false)
        connected_post = HawkesPlasticNetworks.ConnectedPopulationExpKernel(
            post,zeros(2),(plastic_connection,pre),(fixed_connection,pre))

        @test HawkesPlasticNetworks.apply_plasticity!(
            3.5,:pre,2,(connected_post,)) === nothing
        @test calls == [
            (:post,:pre,3.5,:pre,2),
            (:post,:pre,3.5,:pre,2),
        ]
        @test HawkesPlasticNetworks.apply_plasticity!(
            3.5,:pre,2,()) === nothing
    end
end
