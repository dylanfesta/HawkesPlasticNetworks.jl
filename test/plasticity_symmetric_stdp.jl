@testset "PlasticitySymmetricSTDP" begin
    @testset "Construction and validation" begin
        weights = ones(2,3)
        plast = HPN.PlasticitySymmetricSTDP(
            0.1,0.2,0.3,0.4,2.0,1.5,weights)

        @test length(plast.trace_pre_plus) == 3
        @test length(plast.trace_pre_minus) == 3
        @test length(plast.trace_post_plus) == 2
        @test length(plast.trace_post_minus) == 2
        @test plast.trace_pre_plus.τ == 2.0
        @test plast.trace_pre_minus.τ == 3.0
        @test plast.trace_post_plus.τ == 2.0
        @test plast.trace_post_minus.τ == 3.0

        @test_throws ArgumentError HPN.PlasticitySymmetricSTDP(
            0.1,1.1,0.0,0.0,2.0,1.5,weights)
        @test_throws ArgumentError HPN.PlasticitySymmetricSTDP(
            0.1,0.0,0.0,0.0,0.0,1.5,weights)
        @test_throws ArgumentError HPN.PlasticitySymmetricSTDP(
            0.1,0.0,0.0,0.0,2.0,0.0,weights)
        @test_throws ArgumentError HPN.PlasticitySymmetricSTDP(
            0.1,0.0,0.0,0.0,2.0,1.5,weights;
            weight_min=2.0,weight_max=1.0)
    end

    @testset "Pre and post updates combine both traces" begin
        weights = ones(2,3)
        plast = HPN.PlasticitySymmetricSTDP(
            0.2,0.5,0.1,-0.1,2.0,2.0,weights)
        connection = HPN.ConnectionWithWeights(
            weights,:post,:pre,(plast,),true)

        plast.trace_post_plus.val .= [2.0,4.0]
        plast.trace_post_minus.val .= [6.0,8.0]
        plast.trace_post_plus.t_last = 1.0
        plast.trace_post_minus.t_last = 1.0
        decay_plus = exp(-1.0/2.0)
        decay_minus = exp(-1.0/4.0)
        expected_pre = 1.0 + 0.2*0.1 +
            0.2*0.75*decay_plus*4.0 +
            0.2*(-0.25)*decay_minus*8.0

        @test HPN.apply_plasticity!(
            plast,connection,2.0,:pre,2) === nothing
        @test isapprox(weights[2,2],expected_pre)
        @test weights[1,1] == 1.0
        @test plast.trace_pre_plus.val[2] == 0.5
        @test plast.trace_pre_minus.val[2] == 0.25

        plast.trace_pre_plus.val .= [3.0,5.0,7.0]
        plast.trace_pre_minus.val .= [4.0,6.0,8.0]
        plast.trace_pre_plus.t_last = 2.0
        plast.trace_pre_minus.t_last = 2.0
        expected_post = 1.0 + 0.2*(-0.1) +
            0.2*0.75*decay_plus*7.0 +
            0.2*(-0.25)*decay_minus*8.0

        @test HPN.apply_plasticity!(
            plast,connection,3.0,:post,1) === nothing
        @test isapprox(weights[1,3],expected_post)
        @test plast.trace_post_plus.val[1] ==
            0.5 + 2.0*exp(-2.0/2.0)
        @test plast.trace_post_minus.val[1] ==
            0.25 + 6.0*exp(-2.0/4.0)
    end

    @testset "Self connection reads all traces before writing" begin
        weights = fill(1.0,2,2)
        plast = HPN.PlasticitySymmetricSTDP(
            0.1,0.0,0.0,0.0,2.0,2.0,weights)
        connection = HPN.ConnectionWithWeights(
            weights,:self,:self,(plast,),true)
        plast.trace_pre_plus.val .= [2.0,3.0]
        plast.trace_pre_minus.val .= [4.0,5.0]
        plast.trace_post_plus.val .= [6.0,7.0]
        plast.trace_post_minus.val .= [8.0,9.0]

        HPN.apply_plasticity!(plast,connection,0.0,:self,1)

        @test isapprox(weights[1,1],1.0 + 0.05*(6.0-8.0+2.0-4.0))
        @test plast.trace_pre_plus.val == [2.5,3.0]
        @test plast.trace_pre_minus.val == [4.25,5.0]
        @test plast.trace_post_plus.val == [6.5,7.0]
        @test plast.trace_post_minus.val == [8.25,9.0]
    end

    @testset "Structural zeros, bounds, reset, and dimensions" begin
        weights = [0.0 0.9; 0.5 0.5]
        plast = HPN.PlasticitySymmetricSTDP(
            1.0,1.0,0.0,0.0,1.0,2.0,weights;
            weight_max=1.0)
        connection = HPN.ConnectionWithWeights(
            weights,:post,:pre,(plast,),true)
        plast.trace_pre_plus.val .= 1.0

        HPN.apply_plasticity!(plast,connection,0.0,:post,1)
        @test weights[1,1] == 0.0
        @test weights[1,2] == 1.0

        HPN.reset!(plast)
        @test plast.trace_pre_plus.val == zeros(2)
        @test plast.trace_pre_minus.val == zeros(2)
        @test plast.trace_post_plus.val == zeros(2)
        @test plast.trace_post_minus.val == zeros(2)

        bad_connection = HPN.ConnectionWithWeights(
            ones(1,1),:post,:pre,(plast,),true)
        @test_throws DimensionMismatch HPN.apply_plasticity!(
            plast,bad_connection,1.0,:pre,1)
    end
end
