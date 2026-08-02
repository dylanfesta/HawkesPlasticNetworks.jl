const HPN_HET = HawkesPlasticNetworks

function heterosynaptic_allocated!(plast,connection,time)
    return @allocated HPN_HET.apply_plasticity!(
        plast,connection,time,:unrelated,1)
end

function heterosynaptic_refresh_allocated!(plast,weights)
    return @allocated HPN_HET.refresh_mask!(plast,weights)
end

@testset "PlasticityHeterosynapticNormalization" begin
    @testset "Construction and validation" begin
        weights = [0.0 1.0 2.0; 3.0 0.0 4.0]
        plast = HPN_HET.PlasticityHeterosynapticNormalization(
            5.0,0.5,2.0,weights;
            target=HPN_HET.HeterosynapticIncoming(),
            method=HPN_HET.HeterosynapticSubtractive(),
            weight_min=0.0,weight_max=3.0)

        @test plast.weight_sum_target == 5.0
        @test plast.tolerance == 0.5
        @test plast.Δt == 2.0
        @test plast.weight_min == 0.0
        @test plast.weight_max == 3.0
        @test plast.t_last == 0.0
        @test plast.zero_weight_mask == Bool[1 0 0; 0 1 0]
        @test plast.active_counts == [2,2]
        @test length(plast.group_workspace) == 2

        outgoing = HPN_HET.PlasticityHeterosynapticNormalization(
            5.0,0.5,2.0,weights;
            target=HPN_HET.HeterosynapticOutgoing(),
            method=HPN_HET.HeterosynapticDivisive())
        @test outgoing.active_counts == [1,1,2]
        @test length(outgoing.group_workspace) == 3
        @test outgoing.weight_min == 0.0
        @test outgoing.weight_max == Inf

        both = HPN_HET.PlasticityHeterosynapticNormalization(
            5.0,0.5,2.0,weights;
            target=HPN_HET.HeterosynapticBoth(),
            method=HPN_HET.HeterosynapticDivisive())
        @test length(both.group_workspace) == 3

        for bad_limit in (0.0,-1.0,Inf,NaN)
            @test_throws ArgumentError HPN_HET.PlasticityHeterosynapticNormalization(
                bad_limit,0.5,1.0,weights;
                target=HPN_HET.HeterosynapticIncoming(),
                method=HPN_HET.HeterosynapticSubtractive())
        end
        for bad_tolerance in (0.0,-1.0,Inf,NaN)
            @test_throws ArgumentError HPN_HET.PlasticityHeterosynapticNormalization(
                1.0,bad_tolerance,1.0,weights;
                target=HPN_HET.HeterosynapticIncoming(),
                method=HPN_HET.HeterosynapticSubtractive())
        end
        for bad_dt in (0.0,-1.0,Inf,NaN)
            @test_throws ArgumentError HPN_HET.PlasticityHeterosynapticNormalization(
                1.0,0.5,bad_dt,weights;
                target=HPN_HET.HeterosynapticIncoming(),
                method=HPN_HET.HeterosynapticSubtractive())
        end
        @test_throws ArgumentError HPN_HET.PlasticityHeterosynapticNormalization(
            1.0,0.5,1.0,weights;
            target=HPN_HET.HeterosynapticIncoming(),
            method=HPN_HET.HeterosynapticSubtractive(),
            weight_min=2.0,weight_max=1.0)
        @test_throws ArgumentError HPN_HET.PlasticityHeterosynapticNormalization(
            1.0,0.5,1.0,weights;
            target=HPN_HET.HeterosynapticIncoming(),
            method=HPN_HET.HeterosynapticSubtractive(),
            weight_min=-1.0,weight_max=1.0)

        empty_rows = HPN_HET.PlasticityHeterosynapticNormalization(
            1.0,0.5,1.0,zeros(0,3);
            target=HPN_HET.HeterosynapticIncoming(),
            method=HPN_HET.HeterosynapticSubtractive())
        empty_columns = HPN_HET.PlasticityHeterosynapticNormalization(
            1.0,0.5,1.0,zeros(2,0);
            target=HPN_HET.HeterosynapticOutgoing(),
            method=HPN_HET.HeterosynapticDivisive())
        @test isempty(empty_rows.group_workspace)
        @test isempty(empty_columns.group_workspace)
    end

    @testset "Incoming subtractive normalization and bounds" begin
        weights = [0.0 4.0 4.0; 1.0 2.0 2.0; 2.0 3.0 1.0]
        plast = HPN_HET.PlasticityHeterosynapticNormalization(
            5.0,0.5,1.0,weights;
            target=HPN_HET.HeterosynapticIncoming(),
            method=HPN_HET.HeterosynapticSubtractive(),
            weight_min=0.0,weight_max=10.0)
        connection = HPN_HET.ConnectionWithWeights(
            weights,:post,:pre,(plast,),true)

        @test HPN_HET.apply_plasticity!(
            plast,connection,1.0,:unrelated,1) === nothing
        @test weights[1,:] ≈ [0.0,2.5,2.5]
        @test weights[2,:] == [1.0,2.0,2.0]
        @test weights[3,:] ≈ [5/3,8/3,2/3]
        @test sum(weights[1,:]) ≈ 5.0
        @test sum(weights[3,:]) ≈ 5.0

        bounded_weights = [0.1 10.0]
        bounded = HPN_HET.PlasticityHeterosynapticNormalization(
            1.0,0.5,1.0,bounded_weights;
            target=HPN_HET.HeterosynapticIncoming(),
            method=HPN_HET.HeterosynapticSubtractive(),
            weight_min=0.0,weight_max=4.0)
        bounded_connection = HPN_HET.ConnectionWithWeights(
            bounded_weights,:post,:pre,(bounded,),true)
        HPN_HET.apply_plasticity!(
            bounded,bounded_connection,1.0,:unrelated,1)
        @test bounded_weights == [0.0 4.0]
        @test sum(bounded_weights) != 1.0
    end

    @testset "Incoming divisive normalization" begin
        weights = [0.0 2.0 6.0; 1.0 2.0 2.0]
        plast = HPN_HET.PlasticityHeterosynapticNormalization(
            4.0,0.5,1.0,weights;
            target=HPN_HET.HeterosynapticIncoming(),
            method=HPN_HET.HeterosynapticDivisive())
        connection = HPN_HET.ConnectionWithWeights(
            weights,:post,:pre,(plast,),true)

        HPN_HET.apply_plasticity!(plast,connection,1.0,:pre,2)
        @test weights[1,:] ≈ [0.0,1.0,3.0]
        @test weights[2,:] ≈ [0.8,1.6,1.6]
    end

    @testset "Tolerance controls triggering, not normalization target" begin
        for method in (
                HPN_HET.HeterosynapticSubtractive(),
                HPN_HET.HeterosynapticDivisive())
            weights = reshape([2.75,2.75],1,2)
            plast = HPN_HET.PlasticityHeterosynapticNormalization(
                5.0,0.5,1.0,weights;
                target=HPN_HET.HeterosynapticIncoming(),method=method)
            connection = HPN_HET.ConnectionWithWeights(
                weights,:post,:pre,(plast,),true)

            HPN_HET.apply_plasticity!(plast,connection,1.0,:pre,1)
            @test sum(weights) == 5.5

            weights .= 2.8
            HPN_HET.apply_plasticity!(plast,connection,2.0,:pre,1)
            @test isapprox(sum(weights),5.0;atol=1e-12)
        end
    end

    @testset "Outgoing subtractive normalization" begin
        weights = [0.0 1.0 4.0; 8.0 2.0 2.0]
        plast = HPN_HET.PlasticityHeterosynapticNormalization(
            5.0,0.5,1.0,weights;
            target=HPN_HET.HeterosynapticOutgoing(),
            method=HPN_HET.HeterosynapticSubtractive())
        connection = HPN_HET.ConnectionWithWeights(
            weights,:post,:pre,(plast,),true)

        HPN_HET.apply_plasticity!(plast,connection,1.0,:post,1)
        @test weights[:,1] == [0.0,5.0]
        @test weights[:,2] == [1.0,2.0]
        @test weights[:,3] ≈ [3.5,1.5]
    end

    @testset "Outgoing divisive normalization and upper bound" begin
        weights = [0.0 1.0 8.0; 8.0 2.0 4.0]
        plast = HPN_HET.PlasticityHeterosynapticNormalization(
            4.0,0.5,1.0,weights;
            target=HPN_HET.HeterosynapticOutgoing(),
            method=HPN_HET.HeterosynapticDivisive(),
            weight_min=0.0,weight_max=2.0)
        connection = HPN_HET.ConnectionWithWeights(
            weights,:post,:pre,(plast,),true)

        HPN_HET.apply_plasticity!(plast,connection,1.0,:post,1)
        @test weights[:,1] == [0.0,2.0]
        @test weights[:,2] == [1.0,2.0]
        @test weights[:,3] == [2.0,4/3]
        @test sum(weights[:,1]) != 4.0
    end

    @testset "Both normalizes incoming then outgoing" begin
        initial_weights = [0.0 4.0 3.0; 6.0 2.0 5.0]
        for method in (
                HPN_HET.HeterosynapticSubtractive(),
                HPN_HET.HeterosynapticDivisive())
            expected_weights = copy(initial_weights)
            incoming = HPN_HET.PlasticityHeterosynapticNormalization(
                5.0,0.25,1.0,expected_weights;
                target=HPN_HET.HeterosynapticIncoming(),method=method)
            outgoing = HPN_HET.PlasticityHeterosynapticNormalization(
                5.0,0.25,1.0,expected_weights;
                target=HPN_HET.HeterosynapticOutgoing(),method=method)
            expected_connection = HPN_HET.ConnectionWithWeights(
                expected_weights,:post,:pre,(incoming,outgoing),true)

            weights = copy(initial_weights)
            both = HPN_HET.PlasticityHeterosynapticNormalization(
                5.0,0.25,1.0,weights;
                target=HPN_HET.HeterosynapticBoth(),method=method)
            connection = HPN_HET.ConnectionWithWeights(
                weights,:post,:pre,(both,),true)

            HPN_HET.apply_plasticity!(
                incoming,expected_connection,1.0,:other,1)
            HPN_HET.apply_plasticity!(
                outgoing,expected_connection,1.0,:other,1)
            @test HPN_HET.apply_plasticity!(
                both,connection,1.0,:other,1) === nothing
            @test isapprox(weights,expected_weights;atol=1e-12)
            @test weights[1,1] == 0.0
        end
    end

    @testset "Timing, reset, mask refresh, and dimensions" begin
        weights = [0.0 4.0; 4.0 4.0]
        plast = HPN_HET.PlasticityHeterosynapticNormalization(
            4.0,0.5,2.0,weights;
            target=HPN_HET.HeterosynapticIncoming(),
            method=HPN_HET.HeterosynapticDivisive())
        connection = HPN_HET.ConnectionWithWeights(
            weights,:post,:pre,(plast,),true)

        before = copy(weights)
        HPN_HET.apply_plasticity!(plast,connection,1.99,:other,1)
        @test weights == before
        @test plast.t_last == 0.0
        HPN_HET.apply_plasticity!(plast,connection,2.0,:other,1)
        @test weights[2,:] == [2.0,2.0]
        @test plast.t_last == 2.0

        weights[1,1] = 8.0
        HPN_HET.apply_plasticity!(plast,connection,3.0,:other,1)
        @test weights[1,1] == 8.0
        HPN_HET.apply_plasticity!(plast,connection,10.0,:other,1)
        @test weights[1,1] == 8.0
        @test plast.t_last == 10.0

        @test HPN_HET.refresh_mask!(plast,weights) === nothing
        @test plast.zero_weight_mask == falses(2,2)
        HPN_HET.apply_plasticity!(plast,connection,12.0,:other,1)
        @test weights[1,:] ≈ [8/3,4/3]

        @test HPN_HET.reset!(plast) === nothing
        @test plast.t_last == 0.0
        @test all(iszero,plast.group_workspace)
        @test !any(plast.groups_to_normalize)

        wrong_weights = zeros(3,3)
        wrong_connection = HPN_HET.ConnectionWithWeights(
            wrong_weights,:post,:pre,(plast,),true)
        @test_throws DimensionMismatch HPN_HET.apply_plasticity!(
            plast,wrong_connection,2.0,:other,1)
        @test_throws DimensionMismatch HPN_HET.refresh_mask!(
            plast,wrong_weights)
    end

    @testset "Hot paths allocate no memory" begin
        for target in (
                HPN_HET.HeterosynapticIncoming(),
                HPN_HET.HeterosynapticOutgoing(),
                HPN_HET.HeterosynapticBoth())
            for method in (
                    HPN_HET.HeterosynapticSubtractive(),
                    HPN_HET.HeterosynapticDivisive())
                weights = fill(1.0,8,8)
                weights[diagind(weights)] .= 0.0
                plast = HPN_HET.PlasticityHeterosynapticNormalization(
                    4.0,0.5,1.0,weights;target=target,method=method)
                connection = HPN_HET.ConnectionWithWeights(
                    weights,:post,:pre,(plast,),true)

                heterosynaptic_allocated!(plast,connection,0.5)
                heterosynaptic_allocated!(plast,connection,1.0)
                heterosynaptic_refresh_allocated!(plast,weights)
                HPN_HET.reset!(plast)
                @test heterosynaptic_allocated!(plast,connection,0.5) == 0
                @inbounds for idx in eachindex(weights,plast.zero_weight_mask)
                    if plast.zero_weight_mask[idx]
                        weights[idx] = 0.0
                    else
                        weights[idx] = 1.0
                    end
                end
                HPN_HET.reset!(plast)
                @test heterosynaptic_allocated!(plast,connection,1.0) == 0
                HPN_HET.reset!(plast)
                @test heterosynaptic_allocated!(plast,connection,1.0) == 0
                @test heterosynaptic_refresh_allocated!(plast,weights) == 0
            end
        end
    end
end
