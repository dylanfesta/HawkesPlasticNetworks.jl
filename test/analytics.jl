const HPN_ANALYTICS = HawkesPlasticNetworks

@testset "Analytic two-neuron motif parameters" begin
    target_zero = (wexc=0,rexc=40.0)
    target_rate = (wexc=0.5f0,rexc=25)
    parameters = (rinh=80,hexc=40.0,B=0,M10=-2.0f0,M01=3.0)

    @testset "Two-target solution" begin
        alphas = HPN_ANALYTICS.analytic_alphaprepost(
            target_zero,target_rate;parameters...)

        @test propertynames(alphas) == (:αpre,:αpost)
        @test isapprox(alphas.αpre,-0.25)
        @test isapprox(alphas.αpost,0.5)

        for target in (target_zero,target_rate)
            constant = parameters.B*parameters.rinh*target.rexc +
                parameters.M01*target.wexc*target.rexc +
                parameters.M10*(parameters.hexc-target.rexc)
            @test isapprox(
                parameters.rinh*alphas.αpre + target.rexc*alphas.αpost,
                -constant)

            expected_winh =
                (parameters.hexc-target.rexc)/parameters.rinh
            actual_winh =
                HPN_ANALYTICS.analytic_wei_two_neuron_motif_fixed_rhin(
                    alphas.αpre,alphas.αpost,parameters.B,
                    parameters.M10,parameters.M01,target.wexc,
                    parameters.hexc,parameters.rinh)
            @test isapprox(actual_winh,expected_winh)

            rate_and_reciprocal = alphas.αpost +
                parameters.B*parameters.rinh +
                parameters.M01*target.wexc
            drift = alphas.αpre*parameters.rinh +
                target.rexc*rate_and_reciprocal +
                parameters.M10*expected_winh*parameters.rinh
            @test isapprox(drift,0.0;atol=1e-12)
        end

        reversed = HPN_ANALYTICS.analytic_alphaprepost(
            target_rate,target_zero;parameters...)
        @test isapprox(reversed.αpre,alphas.αpre)
        @test isapprox(reversed.αpost,alphas.αpost)
    end

    @testset "Signed motif coefficients" begin
        post_exc = HPN_ANALYTICS.PopulationExpKernelExcitatory(1,0.03)
        pre_exc = HPN_ANALYTICS.PopulationExpKernelExcitatory(1,0.03)
        pre_inh = HPN_ANALYTICS.PopulationExpKernelInhibitory(1,0.03)
        symmetric = HPN_ANALYTICS.PlasticitySymmetricSTDP(
            0.1,0.0,0.0,0.0,0.05,10.0,ones(1,1))
        asymmetric = HPN_ANALYTICS.PlasticityAsymmetricSTDP(
            0.1,0.0,0.0,0.0,0.05,1.0,ones(1,1))

        for rule in (symmetric,asymmetric)
            m10_exc = HPN_ANALYTICS.get_motif_coef_M10(rule,pre_exc)
            m10_inh = HPN_ANALYTICS.get_motif_coef_M10(rule,pre_inh)
            m01_exc = HPN_ANALYTICS.get_motif_coef_M01(rule,post_exc)
            m01_inh = HPN_ANALYTICS.get_motif_coef_M01(rule,pre_inh)
            @test m10_exc > 0
            @test isapprox(m10_inh,-m10_exc)
            @test isapprox(m01_inh,-m01_exc)
        end
    end

    @testset "Target validation" begin
        @test_throws ArgumentError HPN_ANALYTICS.analytic_alphaprepost(
            target_zero,target_rate;
            merge(parameters,(M01=Inf,))...)
        @test_throws ArgumentError HPN_ANALYTICS.analytic_alphaprepost(
            target_zero,target_rate;
            merge(parameters,(rinh=0,))...)
        @test_throws ArgumentError HPN_ANALYTICS.analytic_alphaprepost(
            target_zero,target_rate;
            merge(parameters,(hexc=0,))...)
        @test_throws ArgumentError HPN_ANALYTICS.analytic_alphaprepost(
            target_zero,target_rate;
            merge(parameters,(B=1.1,))...)
        @test_throws ArgumentError HPN_ANALYTICS.analytic_alphaprepost(
            (wexc=-0.1,rexc=40.0),target_rate;parameters...)
        @test_throws ArgumentError HPN_ANALYTICS.analytic_alphaprepost(
            target_zero,(wexc=0.5,rexc=41.0);parameters...)
        @test_throws ArgumentError HPN_ANALYTICS.analytic_alphaprepost(
            target_zero,(wexc=0.5,rexc=40.0);parameters...)
        @test_throws ArgumentError HPN_ANALYTICS.analytic_alphaprepost(
            (wexc=0.0,rexc=25.0),(wexc=0.5,rexc=40.0);
            rinh=80.0,hexc=40.0,B=0.0,M10=-2.0,M01=3.0)
    end

    @testset "Plasticity template overload" begin
        post = HPN_ANALYTICS.PopulationExpKernelExcitatory(
            2,0.02;label="post")
        pre = HPN_ANALYTICS.PopulationExpKernelInhibitory(
            3,0.03;label="pre")
        template = HPN_ANALYTICS.PlasticitySymmetricSTDP(
            0.1,0.0,1.0,2.0,0.05,10.0,ones(2,3))

        actual = HPN_ANALYTICS.analytic_alphaprepost(
            template,post,pre,target_zero,target_rate;
            rinh=80.0,hexc=40.0)
        expected = HPN_ANALYTICS.analytic_alphaprepost(
            target_zero,target_rate;
            rinh=80.0,hexc=40.0,B=template.B,
            M10=HPN_ANALYTICS.get_motif_coef_M10(template,pre),
            M01=HPN_ANALYTICS.get_motif_coef_M01(template,post))
        @test isapprox(actual.αpre,expected.αpre)
        @test isapprox(actual.αpost,expected.αpost)

        bounded = HPN_ANALYTICS.PlasticitySymmetricSTDP(
            0.1,0.0,1.0,2.0,0.05,10.0,ones(2,3);
            weight_min=0.01)
        @test_throws ArgumentError HPN_ANALYTICS.analytic_alphaprepost(
            bounded,post,pre,target_zero,target_rate;
            rinh=80.0,hexc=40.0)
    end

    @testset "Fresh rule construction" begin
        post = HPN_ANALYTICS.PopulationExpKernelExcitatory(
            2,0.02;label="post")
        pre = HPN_ANALYTICS.PopulationExpKernelInhibitory(
            3,0.03;label="pre")
        mask_weights = [0.0 1.0 2.0; 3.0 0.0 4.0]
        symmetric = HPN_ANALYTICS.PlasticitySymmetricSTDP(
            0.2,0.0,1.0,2.0,0.05,10.0,ones(2,3);
            weight_max=2.0)
        symmetric.trace_pre_plus.val .= 1.0

        symmetric_new = HPN_ANALYTICS.analytic_alphaprepost_rule(
            symmetric,post,pre,target_zero,target_rate;
            rinh=80.0,hexc=40.0,weights=mask_weights)
        symmetric_alphas = HPN_ANALYTICS.analytic_alphaprepost(
            symmetric,post,pre,target_zero,target_rate;
            rinh=80.0,hexc=40.0)
        @test symmetric_new isa HPN_ANALYTICS.PlasticitySymmetricSTDP
        @test symmetric_new.η == symmetric.η
        @test symmetric_new.B == symmetric.B
        @test symmetric_new.τ_plus == symmetric.τ_plus
        @test symmetric_new.γ == symmetric.γ
        @test symmetric_new.weight_min == symmetric.weight_min
        @test symmetric_new.weight_max == symmetric.weight_max
        @test isapprox(symmetric_new.αpre,symmetric_alphas.αpre)
        @test isapprox(symmetric_new.αpost,symmetric_alphas.αpost)
        @test symmetric_new.zero_weight_mask == iszero.(mask_weights)
        @test all(iszero,symmetric_new.trace_pre_plus.val)
        @test all(iszero,symmetric_new.trace_pre_minus.val)
        @test all(iszero,symmetric_new.trace_post_plus.val)
        @test all(iszero,symmetric_new.trace_post_minus.val)
        @test_throws DimensionMismatch HPN_ANALYTICS.analytic_alphaprepost_rule(
            symmetric,post,pre,target_zero,target_rate;
            rinh=80.0,hexc=40.0,weights=ones(1,1))

        asymmetric_targets = (
            (wexc=0.0,rexc=25.0),(wexc=0.5,rexc=40.0))
        asymmetric = HPN_ANALYTICS.PlasticityAsymmetricSTDP(
            0.3,0.0,1.0,2.0,0.05,1.0,ones(2,3);
            weight_max=3.0)
        asymmetric_new = HPN_ANALYTICS.analytic_alphaprepost_rule(
            asymmetric,post,pre,asymmetric_targets...;
            rinh=80.0,hexc=40.0)
        asymmetric_alphas = HPN_ANALYTICS.analytic_alphaprepost(
            asymmetric,post,pre,asymmetric_targets...;
            rinh=80.0,hexc=40.0)
        @test asymmetric_new isa HPN_ANALYTICS.PlasticityAsymmetricSTDP
        @test asymmetric_new.η == asymmetric.η
        @test asymmetric_new.B == asymmetric.B
        @test asymmetric_new.τ_plus == asymmetric.τ_plus
        @test asymmetric_new.γ == asymmetric.γ
        @test asymmetric_new.weight_max == asymmetric.weight_max
        @test isapprox(asymmetric_new.αpre,asymmetric_alphas.αpre)
        @test isapprox(asymmetric_new.αpost,asymmetric_alphas.αpost)
        @test size(asymmetric_new.zero_weight_mask) == (2,3)
        @test !any(asymmetric_new.zero_weight_mask)
        @test all(iszero,asymmetric_new.trace_pre_plus.val)
        @test all(iszero,asymmetric_new.trace_post_minus.val)

    end
end
