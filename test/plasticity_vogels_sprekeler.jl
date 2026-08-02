@testset "PlasticityVogelsSprekeler" begin
    @testset "Weight bounds are nonnegative" begin
        @test_throws ArgumentError HPN.PlasticityVogelsSprekeler(
            0.2,0.5,2.0,ones(1,1);weight_min=-1.0,weight_max=1.0)
    end

    @testset "Target term is applied on presynaptic spikes" begin
        η = 0.2
        r_target = 0.5
        τ = 2.0

        weights_pre = ones(1,1)
        plast_pre = HPN.PlasticityVogelsSprekeler(
            η,r_target,τ,weights_pre)
        connection_pre = HPN.ConnectionWithWeights(
            weights_pre,:post,:pre,(plast_pre,),true)

        @test HPN.apply_plasticity!(
            plast_pre,connection_pre,0.0,:pre,1) === nothing
        @test isapprox(weights_pre[1,1],1.0-η*r_target)

        weights_post = ones(1,1)
        plast_post = HPN.PlasticityVogelsSprekeler(
            η,r_target,τ,weights_post)
        connection_post = HPN.ConnectionWithWeights(
            weights_post,:post,:pre,(plast_post,),true)

        @test HPN.apply_plasticity!(
            plast_post,connection_post,0.0,:post,1) === nothing
        @test weights_post[1,1] == 1.0
    end

    @testset "Pre and post traces each supply half the pair term" begin
        η = 0.2
        r_target = 0.5
        τ = 2.0
        weights = ones(1,1)
        plast = HPN.PlasticityVogelsSprekeler(
            η,r_target,τ,weights)
        connection = HPN.ConnectionWithWeights(
            weights,:post,:pre,(plast,),true)

        plast.trace_post_plus.val[1] = 4.0
        plast.trace_pre_plus.val[1] = 6.0

        HPN.apply_plasticity!(plast,connection,0.0,:pre,1)
        HPN.apply_plasticity!(plast,connection,0.0,:post,1)

        expected = 1.0 +
            η*(-r_target+0.5*4.0) +
            η*0.5*(6.0+inv(τ))
        @test isapprox(weights[1,1],expected)
    end
end
