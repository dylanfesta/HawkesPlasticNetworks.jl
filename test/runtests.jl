using HawkesPlasticNetworks
using Test

@testset "HawkesPlasticNetworks.jl" begin
    @testset "TracePlasticity construction" begin
        trace = HawkesPlasticNetworks.Trace(20.0, 3)

        @test trace.val == zeros(3)
        @test trace.τ == 20.0
        @test trace.t_last == 0.0
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
end
