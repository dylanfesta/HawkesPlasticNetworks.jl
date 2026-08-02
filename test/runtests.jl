using HawkesPlasticNetworks
using Test

mutable struct TestPlasticityRule <: HawkesPlasticNetworks.AbstractPlasticityRule
    calls::Vector{Tuple{Symbol,Symbol,Float64,Symbol,Int}}
end

include("plasticity_asymmetric_stdp.jl")
include("plasticity_heterosynaptic_rules.jl")
include("plasticity_homeostatic_scaling.jl")
include("plasticity_symmetric_stdp.jl")
include("plasticity_vogels_sprekeler.jl")
include("weight_matrix_utilities.jl")

function HawkesPlasticNetworks.apply_plasticity!(
        rule::TestPlasticityRule,
        connection::HawkesPlasticNetworks.AbstractConnectionWithWeights,
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

    @testset "Dynamics construction validation" begin
        @test_throws ArgumentError HawkesPlasticNetworks.Trace(0.0,1)
        @test_throws ArgumentError HawkesPlasticNetworks.Trace(Inf,1)
        @test_throws ArgumentError HawkesPlasticNetworks.Trace(1.0,-1)
        @test_throws ArgumentError HawkesPlasticNetworks.PopulationExpKernelExcitatory(
            0,1.0)
        @test_throws DimensionMismatch HawkesPlasticNetworks.PopulationExpKernelExcitatory(
            2,HawkesPlasticNetworks.Trace(1.0,1))

        invalid_trace = HawkesPlasticNetworks.Trace(1.0,1)
        invalid_trace.val[1] = -1.0
        @test_throws ArgumentError HawkesPlasticNetworks.PopulationExpKernelInhibitory(
            1,invalid_trace)

        post = HawkesPlasticNetworks.PopulationExpKernelExcitatory(
            1,1.0;label="post")
        pre = HawkesPlasticNetworks.PopulationExpKernelExcitatory(
            1,1.0;label="pre")
        zero_connection = HawkesPlasticNetworks.ConnectionWithWeights(
            post,zeros(1,1),pre)
        @test zero_connection.weights == zeros(1,1)
        for invalid_weight in (-1.0,Inf,NaN)
            @test_throws ArgumentError HawkesPlasticNetworks.ConnectionWithWeights(
                post,fill(invalid_weight,1,1),pre)
        end
        @test_throws DimensionMismatch HawkesPlasticNetworks.ConnectionWithWeights(
            post,zeros(2,1),pre)

        @test_throws DimensionMismatch HawkesPlasticNetworks.ConnectedPopulationExpKernel(
            post,Float64[],(zero_connection,pre))
        @test_throws ArgumentError HawkesPlasticNetworks.ConnectedPopulationExpKernel(
            post,[Inf],(zero_connection,pre))

        wrong_label_connection = HawkesPlasticNetworks.ConnectionWithWeights(
            zeros(1,1),:wrong,:pre,(),false)
        @test_throws ArgumentError HawkesPlasticNetworks.ConnectedPopulationExpKernel(
            post,[1.0],(wrong_label_connection,pre))

        wrong_size_connection = HawkesPlasticNetworks.ConnectionWithWeights(
            zeros(2,1),:post,:pre,(),false)
        @test_throws DimensionMismatch HawkesPlasticNetworks.ConnectedPopulationExpKernel(
            post,[1.0],(wrong_size_connection,pre))

        @test_throws ArgumentError HawkesPlasticNetworks.set_initial_rates!(
            post,-1.0)
        @test_throws ArgumentError HawkesPlasticNetworks.set_initial_rates!(
            post,[NaN])
        @test_throws DimensionMismatch HawkesPlasticNetworks.set_initial_rates!(
            post,[1.0,2.0])
    end

    @testset "RecorderPopulationRate construction" begin
        rec = HawkesPlasticNetworks.RecorderPopulationRate(
            :exc, 2, 25.0; Tstart=5.0, Δt=10.0)

        @test rec.times == [10.0, 20.0, 30.0]
        @test rec.rates == [0.0, 0.0, 0.0]
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

        @test rec.rates == [0.1, 0.05, 0.05]
        @test rec.t_last[] == 25.0

        content = HawkesPlasticNetworks.get_content(rec)
        @test content.population_label == :exc
        @test content.times == [10.0, 20.0]
        @test content.rates == [0.1, 0.05]

        content.rates[1] = -1.0
        @test rec.rates[1] == 0.1

        @test HawkesPlasticNetworks.reset!(rec) === nothing
        @test rec.rates == [0.0, 0.0, 0.0]
        @test rec.t_last[] == -Inf
        @test isempty(HawkesPlasticNetworks.get_content(rec).times)
    end

    @testset "RecorderPopulationRate omits the current partial bin" begin
        rec = HawkesPlasticNetworks.RecorderPopulationRate(
            :exc, 1, 30.0; Δt=10.0)

        HawkesPlasticNetworks.record_stuff!(rec,9.0,1,:inh,1)
        @test isempty(HawkesPlasticNetworks.get_content(rec).times)

        HawkesPlasticNetworks.record_stuff!(rec,10.0,1,:inh,1)
        content = HawkesPlasticNetworks.get_content(rec)
        @test content.times == [5.0]
        @test content.rates == [0.0]

        HawkesPlasticNetworks.record_stuff!(rec,21.0,1,:exc,1)
        content = HawkesPlasticNetworks.get_content(rec)
        @test content.times == [5.0,15.0]
        @test content.rates == [0.0,0.0]
    end

    @testset "RecorderPopulationTrain starts at the first slot" begin
        rec = HawkesPlasticNetworks.RecorderPopulationTrain(:exc,2)
        HawkesPlasticNetworks.record_stuff!(rec,1.0,1,:exc,2)

        @test HawkesPlasticNetworks.get_content(rec).times == [1.0]
        @test HawkesPlasticNetworks.get_content(rec).neurons == [2]
    end

    @testset "WeightMatrixRecorder records independent snapshots" begin
        weights = [1.0 2.0 3.0; 4.0 5.0 6.0]
        rec = HawkesPlasticNetworks.WeightMatrixRecorder(
            weights,2.0,6.0;Tstart=1.0)

        @test rec.weights isa Array{Float64,3}
        @test size(rec.weights) == (3,2,3)
        @test HawkesPlasticNetworks.record_stuff!(
            rec,0.5,1,:exc,1) === nothing
        HawkesPlasticNetworks.record_stuff!(rec,1.0,1,:exc,1)
        weights[1,2] = 20.0
        HawkesPlasticNetworks.record_stuff!(rec,2.5,1,:exc,1)
        HawkesPlasticNetworks.record_stuff!(rec,3.0,1,:exc,1)
        weights[2,3] = 60.0
        HawkesPlasticNetworks.record_stuff!(rec,5.0,1,:exc,1)
        HawkesPlasticNetworks.record_stuff!(rec,7.0,1,:exc,1)

        content = HawkesPlasticNetworks.get_content(rec)
        @test content.times == [1.0,3.0,5.0]
        @test size(content.weights) == (3,2,3)
        @test content.weights[1,:,:] == [1.0 2.0 3.0; 4.0 5.0 6.0]
        @test content.weights[2,:,:] == [1.0 20.0 3.0; 4.0 5.0 6.0]
        @test content.weights[3,:,:] == [1.0 20.0 3.0; 4.0 5.0 60.0]

        weights[1,1] = -1.0
        content.weights[1,1,1] = -2.0
        @test rec.weights[1,1,1] == 1.0

        @test HawkesPlasticNetworks.reset!(rec) === nothing
        @test rec.k_write[] == 0
        @test all(isnan,rec.times)
        @test all(isnan,rec.weights)
        @test rec.do_every.t_last[] == -Inf
    end

    @testset "WeightMatrixRecorder validates its recording window" begin
        weights = zeros(0,2)

        rec = HawkesPlasticNetworks.WeightMatrixRecorder(weights,1.0,2.0)
        @test size(rec.weights) == (3,0,2)
        @test_throws ArgumentError HawkesPlasticNetworks.WeightMatrixRecorder(
            weights,0.0,2.0)
        @test_throws ArgumentError HawkesPlasticNetworks.WeightMatrixRecorder(
            weights,1.0,-1.0)
        @test_throws ArgumentError HawkesPlasticNetworks.WeightMatrixRecorder(
            weights,1.0,Inf)
    end

    @testset "DoEveryDt gates arbitrary callbacks" begin
        calls = Tuple[]
        rec = HawkesPlasticNetworks.DoEveryDt(
            (args...) -> push!(calls,args),2.0;Tstart=1.0,Tend=5.0)

        HawkesPlasticNetworks.record_stuff!(rec,0.0,:early)
        HawkesPlasticNetworks.record_stuff!(rec,1.0,:first)
        HawkesPlasticNetworks.record_stuff!(rec,2.0,:too_soon)
        HawkesPlasticNetworks.record_stuff!(rec,3.0,:second)
        HawkesPlasticNetworks.record_stuff!(rec,6.0,:late)
        @test calls == [(1.0,:first),(3.0,:second)]

        @test HawkesPlasticNetworks.reset!(rec) === nothing
        HawkesPlasticNetworks.record_stuff!(rec,2.0,:after_reset)
        @test calls[end] == (2.0,:after_reset)
        @test_throws ArgumentError HawkesPlasticNetworks.DoEveryDt(identity,0.0)
        @test_throws ArgumentError HawkesPlasticNetworks.DoEveryDt(
            identity,1.0;Tstart=2.0,Tend=1.0)
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
        @test rec.rates == zeros(6)
        @test rec.t_last[] == -Inf
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

    @testset "Excitation-only thinning bound remains valid as inhibition decays" begin
        post = HawkesPlasticNetworks.PopulationExpKernelExcitatory(
            1,1.0;label="post_bound")
        pre_exc = HawkesPlasticNetworks.PopulationExpKernelExcitatory(
            1,1.0;label="exc_bound")
        pre_inh = HawkesPlasticNetworks.PopulationExpKernelInhibitory(
            1,0.25;label="inh_bound")
        connection_exc = HawkesPlasticNetworks.ConnectionWithWeights(
            post,fill(0.6,1,1),pre_exc)
        connection_inh = HawkesPlasticNetworks.ConnectionWithWeights(
            post,fill(0.4,1,1),pre_inh)
        connected = HawkesPlasticNetworks.ConnectedPopulationExpKernel(
            post,[0.5],
            (connection_exc,pre_exc),(connection_inh,pre_inh))
        HawkesPlasticNetworks.set_initial_rates!(pre_exc,2.0)
        HawkesPlasticNetworks.set_initial_rates!(pre_inh,3.0)

        rates = zeros(1)
        upper = zeros(1)
        HawkesPlasticNetworks.compute_rates!(rates,0.0,connected)
        initial_rate = only(rates)
        @test initial_rate ≈ 0.5
        HawkesPlasticNetworks.compute_rates_upper!(upper,0.0,connected)
        @test only(upper) ≈ 1.7

        HawkesPlasticNetworks.compute_rates!(rates,0.25,connected)
        @test only(rates) > initial_rate

        for bound_time in range(0.0,1.0;length=5)
            HawkesPlasticNetworks.compute_rates_upper!(
                upper,bound_time,connected)
            bound = only(upper)
            for rate_time in range(bound_time,2.0;length=21)
                HawkesPlasticNetworks.compute_rates!(
                    rates,rate_time,connected)
                @test only(rates) <= bound + 10eps(bound)
            end
        end

        HawkesPlasticNetworks.compute_rates!(rates,0.0,connected)
        HawkesPlasticNetworks.compute_rates_upper!(upper,0.0,connected)
        real_allocations = @allocated HawkesPlasticNetworks.compute_rates!(
            rates,0.0,connected)
        upper_allocations = @allocated HawkesPlasticNetworks.compute_rates_upper!(
            upper,0.0,connected)
        @test real_allocations == 0
        @test upper_allocations == 0

        inhibited = HawkesPlasticNetworks.ConnectedPopulationExpKernel(
            post,[-0.5],(connection_inh,pre_inh))
        HawkesPlasticNetworks.compute_rates!(rates,0.0,inhibited)
        @test rates == [0.0]
        HawkesPlasticNetworks.compute_rates_upper!(upper,0.0,inhibited)
        @test upper == [eps(Float64)]
    end

    @testset "Non-interacting weighted connections skip dynamics" begin
        pop_post = HawkesPlasticNetworks.PopulationExpKernelExcitatory(
            1,2.0;label="post")
        pop_exc = HawkesPlasticNetworks.PopulationExpKernelExcitatory(
            1,2.0;label="exc")
        pop_inh = HawkesPlasticNetworks.PopulationExpKernelInhibitory(
            1,2.0;label="inh")
        rule = TestPlasticityRule(
            Tuple{Symbol,Symbol,Float64,Symbol,Int}[])
        connection_exc = HawkesPlasticNetworks.ConnectionNonInteracting(
            pop_post,fill(10.0,1,1),pop_exc;plasticity_rules=(rule,))
        connection_inh = HawkesPlasticNetworks.ConnectionNonInteracting(
            fill(20.0,1,1),:post,:inh,(),false)
        connected = HawkesPlasticNetworks.ConnectedPopulationExpKernel(
            pop_post,[0.7],
            (connection_exc,pop_exc),(connection_inh,pop_inh))
        rates = zeros(1)

        @test connection_exc isa
            HawkesPlasticNetworks.AbstractConnectionWithWeights
        @test connection_exc.weights == fill(10.0,1,1)
        @test connection_exc.post_pop_label == :post
        @test connection_exc.pre_pop_label == :exc
        @test connection_exc.is_plastic[]
        @test !connection_inh.is_plastic[]

        HawkesPlasticNetworks.set_initial_rates!(pop_exc,2.0)
        HawkesPlasticNetworks.set_initial_rates!(pop_inh,3.0)
        @test HawkesPlasticNetworks.compute_rates!(
            rates,0.0,connected) === nothing
        @test rates == [0.7]
    end

    @testset "Non-interacting connections remain plastic" begin
        weights = fill(1.0,1,1)
        plast = HawkesPlasticNetworks.PlasticityAsymmetricSTDP(
            0.2,0.0,0.1,0.3,2.0,2.0,weights;
            weight_min=0.0,weight_max=10.0)
        connection = HawkesPlasticNetworks.ConnectionNonInteracting(
            weights,:post,:pre,(plast,),true)
        post = HawkesPlasticNetworks.PopulationExpKernelExcitatory(
            1,2.0;label="post")
        pre = HawkesPlasticNetworks.PopulationExpKernelExcitatory(
            1,2.0;label="pre")
        connected = HawkesPlasticNetworks.ConnectedPopulationExpKernel(
            post,[0.0],(connection,pre))

        @test HawkesPlasticNetworks.apply_plasticity!(
            1.0,:pre,1,(connected,)) === nothing
        @test isapprox(weights[1,1],1.02)
        @test HawkesPlasticNetworks.plasticity_off!(connection) === nothing
        @test !connection.is_plastic[]
        weights_before = copy(weights)
        @test HawkesPlasticNetworks.apply_plasticity!(
            2.0,:pre,1,(connected,)) === nothing
        @test weights == weights_before
        @test HawkesPlasticNetworks.plasticity_on!(connection) === nothing
        @test connection.is_plastic[]
    end

    @testset "Recorder iteration" begin
        rec_exc = HawkesPlasticNetworks.RecorderPopulationRate(
            :exc, 2, 10.0; Δt=10.0)
        rec_inh = HawkesPlasticNetworks.RecorderPopulationRate(
            :inh, 4, 10.0; Δt=10.0)

        @test HawkesPlasticNetworks.recorders_iterator!(
            5.0,1,:exc,2,(rec_exc,rec_inh)) === nothing
        @test rec_exc.rates == [0.05,0.0]
        @test rec_inh.rates == [0.0,0.0]
        @test rec_exc.t_last[] == 5.0
        @test rec_inh.t_last[] == 5.0
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
        fixed_pre = HawkesPlasticNetworks.PopulationExpKernelExcitatory(
            2,HawkesPlasticNetworks.Trace(20.0,2);label="fixed")
        plastic_connection = HawkesPlasticNetworks.ConnectionWithWeights(
            zeros(2,2),:post,:pre,(rule_a,rule_b),true)
        fixed_connection = HawkesPlasticNetworks.ConnectionWithWeights(
            zeros(2,2),:post,:fixed,(rule_a,),false)
        connected_post = HawkesPlasticNetworks.ConnectedPopulationExpKernel(
            post,zeros(2),
            (plastic_connection,pre),(fixed_connection,fixed_pre))

        @test !ismutabletype(typeof(plastic_connection))
        @test plastic_connection.is_plastic[]
        @test !fixed_connection.is_plastic[]
        @test HawkesPlasticNetworks.apply_plasticity!(
            3.5,:pre,2,(connected_post,)) === nothing
        @test calls == [
            (:post,:pre,3.5,:pre,2),
            (:post,:pre,3.5,:pre,2),
        ]

        @test HawkesPlasticNetworks.plasticity_off!(
            plastic_connection) === nothing
        @test !plastic_connection.is_plastic[]
        @test HawkesPlasticNetworks.plasticity_on!(
            fixed_connection) === nothing
        @test fixed_connection.is_plastic[]
        empty!(calls)
        @test HawkesPlasticNetworks.apply_plasticity!(
            4.5,:pre,1,(connected_post,)) === nothing
        @test calls == [(:post,:fixed,4.5,:pre,1)]

        @test HawkesPlasticNetworks.apply_plasticity!(
            3.5,:pre,2,()) === nothing
    end
end
