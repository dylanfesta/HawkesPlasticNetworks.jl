using Random

const HPN_TRACES = HawkesPlasticNetworks

@testset "CalciumTrace construction and validation" begin
    trace = HPN_TRACES.CalciumTrace(5.0,3;r_min=5,r_max=20.0)
    @test trace isa HPN_TRACES.AbstractTrace
    @test trace.trace isa HPN_TRACES.Trace
    @test length(trace) == 3
    @test trace.trace.val == zeros(3)
    @test trace.trace.τ == 5.0
    @test trace.trace.t_last == 0.0
    @test trace.r_min == 5.0
    @test trace.r_max == 20.0
    @test trace[1] == 5.0

    for τ in (0.0,-1.0,Inf,NaN)
        @test_throws ArgumentError HPN_TRACES.CalciumTrace(τ,1;r_min=0,r_max=20)
    end
    @test_throws ArgumentError HPN_TRACES.CalciumTrace(5.0,-1;r_min=0,r_max=20)
    for (lower,upper) in ((-1,20),(20,5),(Inf,Inf),(0,Inf),(NaN,20),(0,NaN))
        @test_throws ArgumentError HPN_TRACES.CalciumTrace(
            5.0,1;r_min=lower,r_max=upper)
    end

    constant_trace = HPN_TRACES.CalciumTrace(5.0,1;r_min=0,r_max=0)
    HPN_TRACES.add_firing_event_now!(constant_trace,1)
    @test constant_trace[1] == 0.0

    empty_trace = HPN_TRACES.CalciumTrace(5.0,0;r_min=5,r_max=20)
    @test length(empty_trace) == 0
    @test HPN_TRACES.propagate!(1.0,empty_trace) === nothing
    @test HPN_TRACES.reset!(empty_trace) === nothing
    @test isempty(empty_trace.trace.val)
    @test empty_trace.trace.t_last == 0.0
    @test_throws BoundsError empty_trace[1]
    @test_throws BoundsError HPN_TRACES.trace_value(empty_trace,1)
end

@testset "Trace rate readout and indexing" begin
    trace = HPN_TRACES.Trace(5.0,3)
    trace.val .= (2.0,10.0,30.0)
    @test length(trace) == 3
    for i in 1:3
        @test HPN_TRACES.trace_value(trace,i) == trace.val[i]
        @test trace[i] == HPN_TRACES.trace_value(trace,i)
    end
    for i in (0,4)
        @test_throws BoundsError HPN_TRACES.trace_value(trace,i)
        @test_throws BoundsError trace[i]
    end
end

@testset "CalciumTrace readout preserves the rate estimate" begin
    trace = HPN_TRACES.CalciumTrace(5.0,5;r_min=5,r_max=20)
    rates = [2.0,5.0,10.0,20.0,30.0]
    expected = [5.0,5.0,10.0,20.0,20.0]
    trace.trace.val .= rates
    for i in eachindex(rates)
        @test HPN_TRACES.trace_value(trace,i) == expected[i]
        @test trace[i] == HPN_TRACES.trace_value(trace,i)
    end
    @test trace.trace.val == rates
    @test trace.trace.t_last == 0.0
    for i in (0,6)
        @test_throws BoundsError HPN_TRACES.trace_value(trace,i)
        @test_throws BoundsError trace[i]
    end

    # Decay the full 30 Hz estimate, even after reading a saturated 20 Hz value.
    @test HPN_TRACES.propagate!(5.0,trace) === nothing
    @test isapprox(trace[5],30exp(-1);rtol=1e-12,atol=0.0)
    @test isapprox(trace.trace.val[1],2exp(-1);rtol=1e-12,atol=0.0)
    @test trace[1] == 5.0
end

@testset "CalciumTrace spike updates, decay, and reset" begin
    trace = HPN_TRACES.CalciumTrace(5.0,3;r_min=0,r_max=20)
    @test HPN_TRACES.propagate!(2.0,trace) === nothing
    @test HPN_TRACES.add_firing_event_now!(trace,2) === nothing
    @test HPN_TRACES.add_firing_event_now!(trace,2) === nothing
    @test isapprox(trace[2],2/5;rtol=1e-12,atol=0.0)
    @test trace[1] == trace[3] == 0.0
    @test isapprox(HPN_TRACES.trace_decay(7.0,trace),exp(-1);rtol=1e-12,atol=0.0)
    @test trace.trace.t_last == 2.0
    @test HPN_TRACES.propagate!(7.0,trace) === nothing
    @test trace.trace.t_last == 7.0
    @test isapprox(trace[2],2/5*exp(-1);rtol=1e-12,atol=0.0)
    @test HPN_TRACES.add_firing_event_now!(trace,3) === nothing
    @test isapprox(trace[3],1/5;rtol=1e-12,atol=0.0)
    @test trace[1] == 0.0
    @test_throws BoundsError HPN_TRACES.add_firing_event_now!(trace,0)
    @test_throws BoundsError HPN_TRACES.add_firing_event_now!(trace,4)
    @test HPN_TRACES.reset!(trace) === nothing
    @test trace.trace.val == zeros(3)
    @test trace.trace.t_last == 0.0
    @test trace.r_min == 0
    @test trace.r_max == 20
    @test HPN_TRACES.add_firing_event_now!(trace,1) === nothing
    @test isapprox(trace[1],1/5;rtol=1e-12,atol=0.0)
end

@testset "CalciumTrace Poisson rates after 30 seconds of warmup" begin
    rng = MersenneTwister(20261020)
    τ = 5.0
    r_min,r_max = 5.0,20.0
    warmup = 30.0
    measurement_duration = 60.0
    sample_interval = 0.5
    realizations = 128

    for rate in (2.0,10.0,30.0)
        @testset "$rate Hz" begin
            raw_sum = 0.0
            calcium_sum = 0.0
            samples = 0
            all_bounded = true
            for realization in 1:realizations
                trace = HPN_TRACES.CalciumTrace(τ,1;r_min=r_min,r_max=r_max)
                next_spike = randexp(rng)/rate
                for t in sample_interval:sample_interval:(warmup+measurement_duration)
                    while next_spike <= t
                        HPN_TRACES.propagate!(next_spike,trace)
                        HPN_TRACES.add_firing_event_now!(trace,1)
                        next_spike += randexp(rng)/rate
                    end
                    HPN_TRACES.propagate!(t,trace)
                    if t > warmup
                        raw_sum += trace.trace[1]
                        calcium_sum += trace[1]
                        samples += 1
                        if !(r_min <= trace[1] <= r_max)
                            all_bounded = false
                        end
                    end
                end
            end
            @test samples == realizations*Int(measurement_duration/sample_interval)
            @test all_bounded
            @test isapprox(raw_sum/samples,rate;rtol=0.05,atol=0.0)
            @test isapprox(calcium_sum/samples,clamp(rate,r_min,r_max);rtol=0.0,atol=0.25)
        end
    end
end
