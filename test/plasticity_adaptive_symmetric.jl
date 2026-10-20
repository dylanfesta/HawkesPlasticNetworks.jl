using Random

@testset "PlasticityAdaptiveSymmetric" begin
    @testset "Construction and validation" begin
        w = ones(2,3)
        p = HPN.PlasticityAdaptiveSymmetric(0.1,-0.5,0.02,2,250,w;r_min=9,r_max=11)
        @test p.αpre == p.αpost == 0.0
        @test length(p.trace_pre_plus) == length(p.trace_pre_minus) == 3
        @test length(p.trace_post_plus) == length(p.trace_post_minus) ==
            length(p.trace_post_calcium) == 2
        @test isapprox(p.trace_pre_plus.τ,0.02;atol=1e-14,rtol=0)
        @test isapprox(p.trace_post_minus.τ,0.04;atol=1e-14,rtol=0)
        @test isapprox(p.trace_post_calcium.trace.τ,5;atol=1e-14,rtol=0)
        @test p.trace_post_calcium.r_min == 9
        @test p.trace_post_calcium.r_max == 11
        for B in (0.0,-0.0,NaN,Inf,-Inf,1.1,-1.1)
            @test_throws ArgumentError HPN.PlasticityAdaptiveSymmetric(
                0.1,B,0.02,2,250,w;r_min=9,r_max=11)
        end
        for (τ,γ,γc) in ((0,2,250),(-1,2,250),(Inf,2,250),(NaN,2,250),
                (0.02,0,250),(0.02,-1,250),(0.02,Inf,250),(0.02,NaN,250),
                (0.02,2,0),(0.02,2,-1),(0.02,2,1),(0.02,2,2),
                (0.02,0.5,1),(0.02,2,Inf),(0.02,2,NaN),
                (1e308,2,3),(1e308,1,2))
            @test_throws ArgumentError HPN.PlasticityAdaptiveSymmetric(
                0.1,0.5,τ,γ,γc,w;r_min=9,r_max=11)
        end
        for (lo,hi) in ((-1,11),(11,9),(NaN,11),(9,Inf))
            @test_throws ArgumentError HPN.PlasticityAdaptiveSymmetric(
                0.1,0.5,0.02,2,250,w;r_min=lo,r_max=hi)
        end
        for (lo,hi) in ((-1,1),(2,1),(0,NaN))
            @test_throws ArgumentError HPN.PlasticityAdaptiveSymmetric(
                0.1,0.5,0.02,2,250,w;r_min=9,r_max=11,
                weight_min=lo,weight_max=hi)
        end
        @test_throws UndefKeywordError HPN.PlasticityAdaptiveSymmetric(
            0.1,0.5,0.02,2,250,w)
    end

    @testset "Pre/post formulas and calcium lifecycle" begin
        w = ones(2,3)
        p = HPN.PlasticityAdaptiveSymmetric(0.01,0.5,2,2,5,w;
            r_min=9,r_max=11,αpre=0.3,αpost=-0.2)
        c = HPN.ConnectionNonInteracting(w,:post,:pre,(p,),true)
        p.trace_post_plus.val .= [2,4]
        p.trace_post_minus.val .= [6,8]
        p.trace_post_calcium.trace.val .= [20,5]
        expected = 1 .+ 0.01 .* (0.3 .+ 0.375*exp(-1/2).*[2,4] .-
            0.125*exp(-1/4).*[6,8] .- 0.5.*[11,9])
        @test HPN.apply_plasticity!(p,c,1.0,:pre,2) === nothing
        @test isapprox(w[:,2],expected;atol=1e-14,rtol=1e-14)
        @test w[:,1] == ones(2)
        @test w[:,3] == ones(2)
        @test p.trace_post_calcium.trace.val == [20,5]
        @test p.trace_post_calcium.trace.t_last == 0
        @test isapprox(p.trace_pre_plus.val[2],0.5;atol=1e-14,rtol=0)
        p.trace_pre_plus.val .= [3,5,7]
        p.trace_pre_minus.val .= [4,6,8]
        before = copy(w)
        expected_post = before[1,:] .+ 0.01 .* (-0.2 .+
            0.375*exp(-1/2).*[3,5,7] .- 0.125*exp(-1/4).*[4,6,8])
        @test HPN.apply_plasticity!(p,c,2.0,:post,1) === nothing
        @test isapprox(w[1,:],expected_post;atol=1e-14,rtol=1e-14)
        @test w[2,:] == before[2,:]
        @test isapprox(p.trace_post_calcium.trace.val,
            [20*exp(-2/10)+0.1,5*exp(-2/10)];atol=1e-14,rtol=1e-14)
        @test p.trace_post_calcium.trace.t_last == 2
        @test isapprox(p.trace_post_plus.val[1],2*exp(-2/2)+0.5;
            atol=1e-14,rtol=1e-14)
        @test isapprox(p.trace_post_minus.val[1],6*exp(-2/4)+0.25;
            atol=1e-14,rtol=1e-14)
    end

    @testset "Presynaptic kernel decays before clamping and bounds once" begin
        w = [0.0 0.5; 0.5 0.5; 0.5 0.5]
        p = HPN.PlasticityAdaptiveSymmetric(0.01,1,1,2,10,w;
            r_min=9,r_max=11,weight_max=1)
        p.trace_post_plus.val .= 20
        p.trace_post_calcium.trace.val .= [20,20,0]
        t = 10log(2)
        expected = clamp.(0.5 .+ 0.01.*(10exp(-t) .- [10,10,9]),0,1)
        @test HPN._apply_adaptive_symmetric_pre!(w,p,t,1) === nothing
        @test w[1,1] == 0
        @test isapprox(w[2:3,1],expected[2:3];atol=1e-14,rtol=1e-14)
        @test w[:,2] == fill(0.5,3)
        # Fast and calcium terms must cancel before a single clamp, even when
        # the fast contribution alone would hit the upper bound.
        p.trace_post_plus.val .= 20
        p.trace_post_calcium.trace.val .= 10
        w[2,1] = 0.5
        HPN._apply_adaptive_symmetric_pre!(w,p,0.0,1)
        @test isapprox(w[2,1],0.5;atol=1e-14,rtol=0)
        p.trace_post_plus.val .= 2000
        HPN._apply_adaptive_symmetric_pre!(w,p,0.0,1)
        @test w[2,1] == 1
        p.trace_post_plus.val .= 0
        for _ in 1:11
            HPN._apply_adaptive_symmetric_pre!(w,p,0.0,1)
        end
        @test w[2,1] == 0
    end

    @testset "Calcium signs, integrated areas, and steady-rate cancellation" begin
        for B in (-1.0,-0.5,0.5,1.0)
            w = fill(10.0,1,1)
            p = HPN.PlasticityAdaptiveSymmetric(0.01,B,0.02,2,250,w;
                r_min=9,r_max=11)
            c = HPN.ConnectionWithWeights(w,:post,:pre,(p,),true)
            p.trace_post_calcium.trace.val .= 10
            HPN.apply_plasticity!(p,c,0.0,:pre,1)
            @test isapprox(w[1],10-0.1B;atol=1e-14,rtol=1e-14)
            @test sign(w[1]-10) == -sign(B)
            # Numerically integrate the actual impulse response, including the
            # normalization 1/τ. Bounds are inactive for this area calculation.
            τp,τm,τc = 0.02,0.04,5.0
            dt = 0.0001
            times = 0.0:dt:150.0
            fast = 0.0
            slow = 0.0
            for t in times
                endpoint = 1.0
                if t == first(times) || t == last(times)
                    endpoint = 0.5
                end
                fast += endpoint*2*((B+1)/4/τp*exp(-t/τp) +
                    (B-1)/4/τm*exp(-t/τm))*dt
                slow += endpoint*(-B/τc)*exp(-t/τc)*dt
            end
            @test isapprox(fast,B;atol=3e-6,rtol=0)
            @test isapprox(slow,-B;atol=1e-9,rtol=0)
            @test isapprox(fast+slow,0;atol=3e-6,rtol=0)
            # Equal normalized rate estimates cancel across the two event
            # directions when each direction occurs at its population's rate.
            HPN.reset!(p)
            p.trace_post_plus.val .= 10
            p.trace_post_minus.val .= 10
            p.trace_post_calcium.trace.val .= 10
            p.trace_pre_plus.val .= 20
            p.trace_pre_minus.val .= 20
            initial = w[1]
            HPN.apply_plasticity!(p,c,0.0,:pre,1)
            pre_delta = w[1]-initial
            HPN.reset!(p)
            p.trace_pre_plus.val .= 20
            p.trace_pre_minus.val .= 20
            initial = w[1]
            HPN.apply_plasticity!(p,c,0.0,:post,1)
            post_delta = w[1]-initial
            @test isapprox(20pre_delta+10post_delta,0;atol=1e-12,rtol=0)
        end
    end

    @testset "Self-connections read before writing; reset" begin
        w = ones(2,2)
        p = HPN.PlasticityAdaptiveSymmetric(0.01,0.5,2,2,5,w;r_min=0,r_max=11)
        c = HPN.ConnectionWithWeights(w,:self,:self,(p,),true)
        p.trace_post_plus.val .= [6,7]
        p.trace_post_minus.val .= [8,9]
        p.trace_pre_plus.val .= [2,3]
        p.trace_pre_minus.val .= [4,5]
        p.trace_post_calcium.trace.val .= [10,11]
        HPN.apply_plasticity!(p,c,0.0,:self,1)
        @test isapprox(w[1,1],1+0.01*(0.375*8-0.125*12-5);
            atol=1e-14,rtol=1e-14)
        @test isapprox(w[2,1],1+0.01*(0.375*7-0.125*9-5.5);
            atol=1e-14,rtol=1e-14)
        @test isapprox(w[1,2],1+0.01*(0.375*3-0.125*5);
            atol=1e-14,rtol=1e-14)
        @test w[2,2] == 1
        @test isapprox(p.trace_post_calcium.trace.val,[10.1,11];atol=1e-14,rtol=0)
        @test p.trace_pre_plus.val == [2.5,3]
        @test p.trace_pre_minus.val == [4.25,5]
        @test p.trace_post_plus.val == [6.5,7]
        @test p.trace_post_minus.val == [8.25,9]
        HPN.apply_plasticity!(p,c,1.0,:self,2)
        before = copy(w)
        @test HPN.reset!(p) === nothing
        for trace in (p.trace_pre_plus,p.trace_pre_minus,p.trace_post_plus,
                p.trace_post_minus,p.trace_post_calcium.trace)
            @test all(iszero,trace.val)
            @test trace.t_last == 0
        end
        @test w == before
    end

    @testset "Masks, dimensions, unrelated events, and empty populations" begin
        w = [0.0 0.5; 0.5 0.5]
        p = HPN.PlasticityAdaptiveSymmetric(1,1,1,2,10,w;
            r_min=0,r_max=11,weight_min=0.2,weight_max=1)
        c = HPN.ConnectionWithWeights(w,:post,:pre,(p,),true)
        p.trace_post_plus.val .= 10
        HPN.apply_plasticity!(p,c,0.0,:pre,1)
        @test w[1,1] == 0
        @test w[2,1] == 1
        p.trace_pre_plus.val .= 10
        HPN.apply_plasticity!(p,c,0.0,:post,1)
        @test w[1,1] == 0
        @test w[1,2] == 1
        w[1,1] = 0.5
        w[2,1] = 0
        @test HPN.refresh_mask!(p,w) === nothing
        @test p.zero_weight_mask == Bool[0 0; 1 0]
        HPN.apply_plasticity!(p,c,0.0,:pre,1)
        @test w[1,1] == 1
        @test w[2,1] == 0
        p.trace_post_plus.val .= 0
        p.trace_post_calcium.trace.val .= 11
        HPN.apply_plasticity!(p,c,0.0,:pre,2)
        @test all(w[:,2] .== 0.2)
        before = copy(w)
        @test HPN.apply_plasticity!(p,c,5.0,:other,1) === nothing
        @test w == before
        @test p.trace_post_calcium.trace.t_last == 0
        @test p.trace_pre_plus.t_last == 0
        bad = HPN.ConnectionWithWeights(ones(3,2),:post,:pre,(p,),true)
        @test_throws DimensionMismatch HPN.apply_plasticity!(p,bad,1.0,:pre,1)
        @test_throws DimensionMismatch HPN.refresh_mask!(p,ones(3,2))
        for (npost,npre,label) in ((0,2,:pre),(2,0,:post))
            empty_w = ones(npost,npre)
            empty_p = HPN.PlasticityAdaptiveSymmetric(0.1,-1,1,2,10,empty_w;
                r_min=9,r_max=11)
            empty_c = HPN.ConnectionWithWeights(empty_w,:post,:pre,(empty_p,),true)
            @test HPN.apply_plasticity!(empty_p,empty_c,0.0,label,1) === nothing
            @test isempty(empty_w)
        end
    end

    for (name,pre_type,B,drive,seed) in (
            ("Excitatory",HPN.PopulationExpKernelExcitatory,-1.0,5.0,42),
            ("Inhibitory",HPN.PopulationExpKernelInhibitory,1.0,15.0,43))
        @testset "$name single-input rate stability" begin
            Random.seed!(seed)
            pre = pre_type(1,1.0;label="pre")
            post = HPN.PopulationExpKernelExcitatory(1,1.0;label="post")
            w = fill(0.25,1,1)
            p = HPN.PlasticityAdaptiveSymmetric(1e-6,B,0.02,2,250,w;
                r_min=9,r_max=11)
            connection = HPN.ConnectionWithWeights(post,w,pre;plasticity_rules=(p,))
            connected_post = HPN.ConnectedPopulationExpKernel(post,[drive],(connection,pre))
            connected_pre = HPN.ConnectedPopulationExpKernel(pre,[20.0])
            network = HPN.RecurrentNetworkExpKernel((connected_post,connected_pre),())
            HPN.set_initial_rates!(pre,20.0)
            HPN.set_initial_rates!(post,10.0)
            p.trace_pre_plus.val .= 20
            p.trace_pre_minus.val .= 20
            p.trace_post_plus.val .= 10
            p.trace_post_minus.val .= 10
            p.trace_post_calcium.trace.val .= 10
            counts = zeros(Int,10)
            measurement_weights = Float64[]
            t = 0.0
            while true
                # Split dynamics_step! to reject the first event past 1200 s
                # before it can change state or contribute a spike count.
                tfire,popfire,label,neuron = HPN.compute_next_spike_population_iterator(
                    t,network.populations)
                if tfire >= 1200.0
                    break
                end
                HPN.burn_spike_iterator!(tfire,popfire,neuron,network.populations)
                HPN.apply_plasticity!(tfire,label,neuron,network.populations)
                if tfire >= 600.0
                    push!(measurement_weights,w[1])
                    if label == :post
                        bin = floor(Int,(tfire-600.0)/60.0)+1
                        counts[bin] += 1
                    end
                end
                t = tfire
            end
            rates = counts ./ 60.0
            @info "$name adaptive symmetric rate stability" rates=rates weight_range=extrema(measurement_weights) final_weight=w[1]
            @test all(9 .<= rates .<= 11)
            @test maximum(rates)-minimum(rates) > 0
            @test maximum(measurement_weights)-minimum(measurement_weights) > 1e-8
            @test all(isfinite,measurement_weights)
            @test all(measurement_weights .>= 0)
        end
    end
end
