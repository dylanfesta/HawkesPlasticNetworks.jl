using Random

@testset "PlasticityAdaptiveAsymmetric" begin
    @testset "Construction and validation" begin
        w = ones(2,3)
        p = HPN.PlasticityAdaptiveAsymmetric(0.1,-0.5,0.02,2,250,w;r_min=9,r_max=11)
        @test p.αpre == p.αpost == 0.0
        @test !p.inverted
        @test_throws ArgumentError HPN.PlasticityAdaptiveAsymmetric(
            -0.1,-0.5,0.02,2,250,w;r_min=9,r_max=11)
        @test length(p.trace_pre_plus) == 3
        @test length(p.trace_post_minus) ==
            length(p.trace_post_calcium) == 2
        @test isapprox(p.trace_pre_plus.τ,0.02;atol=1e-14,rtol=0)
        @test isapprox(p.trace_post_minus.τ,0.04;atol=1e-14,rtol=0)
        @test isapprox(p.trace_post_calcium.trace.τ,5;atol=1e-14,rtol=0)
        @test p.trace_post_calcium.r_min == 9
        @test p.trace_post_calcium.r_max == 11
        for B in (0.0,-0.0,NaN,Inf,-Inf,1.1,-1.1)
            @test_throws ArgumentError HPN.PlasticityAdaptiveAsymmetric(
                0.1,B,0.02,2,250,w;r_min=9,r_max=11)
        end
        for (τ,γ,γc) in ((0,2,250),(-1,2,250),(Inf,2,250),(NaN,2,250),
                (0.02,0,250),(0.02,-1,250),(0.02,Inf,250),(0.02,NaN,250),
                (0.02,2,0),(0.02,2,-1),(0.02,2,1),(0.02,2,2),
                (0.02,0.5,1),(0.02,2,Inf),(0.02,2,NaN),
                (1e308,2,3),(1e308,1,2))
            @test_throws ArgumentError HPN.PlasticityAdaptiveAsymmetric(
                0.1,0.5,τ,γ,γc,w;r_min=9,r_max=11)
        end
        for (lo,hi) in ((-1,11),(11,9),(NaN,11),(9,Inf))
            @test_throws ArgumentError HPN.PlasticityAdaptiveAsymmetric(
                0.1,0.5,0.02,2,250,w;r_min=lo,r_max=hi)
        end
        for (lo,hi) in ((-1,1),(2,1),(0,NaN))
            @test_throws ArgumentError HPN.PlasticityAdaptiveAsymmetric(
                0.1,0.5,0.02,2,250,w;r_min=9,r_max=11,
                weight_min=lo,weight_max=hi)
        end
        @test_throws UndefKeywordError HPN.PlasticityAdaptiveAsymmetric(
            0.1,0.5,0.02,2,250,w)
    end

    @testset "Normal and inverted event formulas and trace lifecycle" begin
        for inverted in (false,true), B in (-0.6,0.6)
            w = ones(2,3)
            p = HPN.PlasticityAdaptiveAsymmetric(0.01,B,2,2,5,w;
                r_min=9,r_max=11,αpre=0.3,αpost=-0.2,inverted=inverted)
            c = HPN.ConnectionNonInteracting(w,:post,:pre,(p,),true)
            if inverted
                area_post,area_pre = (B+1)/2,(B-1)/2
            else
                area_post,area_pre = (B-1)/2,(B+1)/2
            end
            p.trace_post_minus.val .= [6,8]
            p.trace_post_calcium.trace.val .= [20,5]
            expected_pre = 1 .+ 0.01 .* (0.3 .+ area_post*exp(-1/4).*[6,8] .- B.*[11,9])
            @test HPN.apply_plasticity!(p,c,1.0,:pre,2) === nothing
            @test isapprox(w[:,2],expected_pre;atol=1e-14,rtol=1e-14)
            @test w[:,1] == w[:,3] == ones(2)
            @test p.trace_post_calcium.trace.val == [20,5]
            @test p.trace_post_calcium.trace.t_last == 0
            @test isapprox(p.trace_pre_plus.val,[0,0.5,0];atol=1e-14,rtol=0)
            @test p.trace_pre_plus.t_last == 1
            p.trace_pre_plus.val .= [3,5,7]
            before = copy(w)
            expected_post = w[1,:] .+ 0.01 .* (-0.2 .+ area_pre*exp(-1/2).*[3,5,7])
            @test HPN.apply_plasticity!(p,c,2.0,:post,1) === nothing
            @test isapprox(w[1,:],expected_post;atol=1e-14,rtol=1e-14)
            @test w[2,:] == before[2,:]
            @test isapprox(p.trace_post_minus.val,[6*exp(-2/4)+0.25,8*exp(-2/4)];
                atol=1e-14,rtol=1e-14)
            @test isapprox(p.trace_post_calcium.trace.val,
                [20*exp(-2/10)+0.1,5*exp(-2/10)];atol=1e-14,rtol=1e-14)
            @test p.trace_post_calcium.trace.t_last == 2
            before = copy(w)
            @test HPN.reset!(p) === nothing
            for trace in (p.trace_pre_plus,p.trace_post_minus,p.trace_post_calcium.trace)
                @test all(iszero,trace.val)
                @test trace.t_last == 0
            end
            @test w == before
        end
    end

    @testset "Kernel calcium signs, decay before clamping, and combined bounds" begin
        for inverted in (false,true), B in (-1.0,-0.5,0.5,1.0)
            w = fill(10.0,3,1)
            p = HPN.PlasticityAdaptiveAsymmetric(0.01,B,1,2,10,w;
                r_min=9,r_max=11,inverted=inverted)
            p.trace_post_calcium.trace.val .= [20,24,0]
            t = 10log(2)
            @test HPN._apply_adaptive_asymmetric_pre!(w,p,t,1) === nothing
            @test isapprox(w[:,1],10 .- 0.01B.*[10,11,9];atol=1e-14,rtol=1e-14)
            @test all(sign.(w .- 10) .== -sign(B))
        end
        w = fill(0.5,1,1)
        p = HPN.PlasticityAdaptiveAsymmetric(0.1,1,1,2,10,w;
            r_min=0,r_max=11,weight_min=0.2,weight_max=1,inverted=true)
        p.trace_post_minus.val .= 10
        p.trace_post_calcium.trace.val .= 10
        HPN._apply_adaptive_asymmetric_pre!(w,p,0.0,1)
        @test isapprox(w[1],0.5;atol=1e-14,rtol=0)
        p.trace_post_minus.val .= 30
        HPN._apply_adaptive_asymmetric_pre!(w,p,0.0,1)
        @test w[1] == 1
        p.trace_post_minus.val .= 0
        HPN._apply_adaptive_asymmetric_pre!(w,p,0.0,1)
        @test w[1] == 0.2
    end

    @testset "Pair-window area and rate cancellation in both orientations" begin
        for inverted in (false,true), B in (-1.0,-0.5,0.5,1.0)
            w = fill(10.0,1,1)
            p = HPN.PlasticityAdaptiveAsymmetric(0.01,B,0.02,2,250,w;
                r_min=0,r_max=100,inverted=inverted)
            c = HPN.ConnectionWithWeights(w,:post,:pre,(p,),true)
            # Integrate the actual isolated-pair responses, using η=0.01.
            # The calcium response is present only for post-before-pre pairs.
            # Analytic exponential integrals avoid quadrature of the long tail.
            HPN.apply_plasticity!(p,c,0.0,:post,1)
            initial = w[1]
            lag = 0.03
            HPN.apply_plasticity!(p,c,lag,:pre,1)
            left_response = (w[1]-initial)/p.η
            calcium_response = -B/p.trace_post_calcium.trace.τ*exp(-lag/p.trace_post_calcium.trace.τ)
            fast_left_area = (left_response-calcium_response)*p.trace_post_minus.τ*
                exp(lag/p.trace_post_minus.τ)
            calcium_area = calcium_response*p.trace_post_calcium.trace.τ*
                exp(lag/p.trace_post_calcium.trace.τ)
            HPN.reset!(p)
            HPN.apply_plasticity!(p,c,0.0,:pre,1)
            initial = w[1]
            HPN.apply_plasticity!(p,c,lag,:post,1)
            fast_right_area = (w[1]-initial)/p.η*p.trace_pre_plus.τ*
                exp(lag/p.trace_pre_plus.τ)
            @test isapprox(fast_left_area+fast_right_area,B;atol=1e-12,rtol=0)
            @test isapprox(calcium_area,-B;atol=1e-14,rtol=0)
            @test isapprox(fast_left_area+fast_right_area+calcium_area,0;atol=1e-12,rtol=0)
            HPN.reset!(p)
            p.trace_post_minus.val .= 10
            p.trace_post_calcium.trace.val .= 10
            initial = w[1]
            HPN.apply_plasticity!(p,c,0.0,:pre,1)
            pre_delta = w[1]-initial
            HPN.reset!(p)
            p.trace_pre_plus.val .= 20
            initial = w[1]
            HPN.apply_plasticity!(p,c,0.0,:post,1)
            post_delta = w[1]-initial
            @test isapprox(20pre_delta+10post_delta,0;atol=1e-12,rtol=0)
        end
    end

    @testset "Self-connection event ordering" begin
        for inverted in (false,true)
            w = ones(2,2)
            p = HPN.PlasticityAdaptiveAsymmetric(0.01,0.5,2,2,5,w;
                r_min=0,r_max=11,inverted=inverted)
            c = HPN.ConnectionWithWeights(w,:self,:self,(p,),true)
            p.trace_pre_plus.val .= [2,3]
            p.trace_post_minus.val .= [4,5]
            p.trace_post_calcium.trace.val .= [10,11]
            if inverted
                a_post,a_pre = 0.75,-0.25
            else
                a_post,a_pre = -0.25,0.75
            end
            HPN.apply_plasticity!(p,c,0.0,:self,1)
            expected = [1+0.01*(a_post*4-5+a_pre*2) 1+0.01*a_pre*3;
                1+0.01*(a_post*5-5.5) 1]
            @test isapprox(w,expected;atol=1e-14,rtol=1e-14)
            @test p.trace_pre_plus.val == [2.5,3]
            @test p.trace_post_minus.val == [4.25,5]
            @test isapprox(p.trace_post_calcium.trace.val,[10.1,11];atol=1e-14,rtol=0)
        end
    end

    @testset "Masks, dimensions, unrelated events, and empty populations" begin
        w = [0.0 0.5; 0.5 0.5]
        p = HPN.PlasticityAdaptiveAsymmetric(1,1,1,2,10,w;
            r_min=0,r_max=11,weight_max=1,inverted=true)
        c = HPN.ConnectionWithWeights(w,:post,:pre,(p,),true)
        p.trace_post_minus.val .= 10
        HPN.apply_plasticity!(p,c,0.0,:pre,1)
        @test w[1,1] == 0
        @test w[2,1] == 1
        w[1,1] = 0.5
        w[2,1] = 0
        @test HPN.refresh_mask!(p,w) === nothing
        @test p.zero_weight_mask == Bool[0 0; 1 0]
        HPN.apply_plasticity!(p,c,0.0,:pre,1)
        @test w[1,1] == 1
        @test w[2,1] == 0
        before = copy(w)
        @test HPN.apply_plasticity!(p,c,5.0,:other,1) === nothing
        @test w == before
        @test p.trace_pre_plus.t_last == p.trace_post_minus.t_last ==
            p.trace_post_calcium.trace.t_last == 0
        bad = HPN.ConnectionWithWeights(ones(3,2),:post,:pre,(p,),true)
        @test_throws DimensionMismatch HPN.apply_plasticity!(p,bad,1.0,:pre,1)
        @test_throws DimensionMismatch HPN.refresh_mask!(p,ones(3,2))
        for (npost,npre,label) in ((0,2,:pre),(2,0,:post))
            empty_w = ones(npost,npre)
            empty_p = HPN.PlasticityAdaptiveAsymmetric(0.1,-1,1,2,10,empty_w;
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
            p = HPN.PlasticityAdaptiveAsymmetric(1e-6,B,0.02,2,250,w;
                r_min=9,r_max=11)
            connection = HPN.ConnectionWithWeights(post,w,pre;plasticity_rules=(p,))
            connected_post = HPN.ConnectedPopulationExpKernel(post,[drive],(connection,pre))
            connected_pre = HPN.ConnectedPopulationExpKernel(pre,[20.0])
            network = HPN.RecurrentNetworkExpKernel((connected_post,connected_pre),())
            HPN.set_initial_rates!(pre,20.0)
            HPN.set_initial_rates!(post,10.0)
            p.trace_pre_plus.val .= 20
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
            @info "$name adaptive asymmetric rate stability" rates=rates weight_range=extrema(measurement_weights) final_weight=w[1]
            @test all(9 .<= rates .<= 11)
            @test maximum(rates)-minimum(rates) > 0
            @test maximum(measurement_weights)-minimum(measurement_weights) > 1e-8
            @test all(isfinite,measurement_weights)
            @test all(measurement_weights .>= 0)
        end
    end
end
