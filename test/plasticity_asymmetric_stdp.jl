const HPN = HawkesPlasticNetworks
using LinearAlgebra

function eager_reference_apply!(
    plast::HPN.PlasticityAsymmetricSTDP,
    connection::HPN.ConnectionWithWeights,
    t_fire::Float64,
    pop_fire_label::Symbol,
    neuron_fire_idx::Int)
    post_fired = pop_fire_label == connection.post_pop_label
    pre_fired = pop_fire_label == connection.pre_pop_label
    (post_fired || pre_fired) || return nothing

    if pre_fired
        HPN.propagate!(t_fire,plast.trace_post_minus)
    end
    if post_fired
        HPN.propagate!(t_fire,plast.trace_pre_plus)
    end

    weights = connection.weights
    if pre_fired
        trace_scale = plast.η*((plast.B-1.0)/2.0)
        bias = plast.η*plast.αpre
        @inbounds for post_idx in axes(weights,1)
            if !plast.zero_weight_mask[post_idx,neuron_fire_idx]
                weight = weights[post_idx,neuron_fire_idx]
                Δweight =
                    bias + trace_scale*plast.trace_post_minus.val[post_idx]
                weights[post_idx,neuron_fire_idx] = HPN.hardbounds(
                    weight+Δweight,plast.weight_min,plast.weight_max)
            end
        end
    end
    if post_fired
        trace_scale = plast.η*((plast.B+1.0)/2.0)
        bias = plast.η*plast.αpost
        @inbounds for pre_idx in axes(weights,2)
            if !plast.zero_weight_mask[neuron_fire_idx,pre_idx]
                weight = weights[neuron_fire_idx,pre_idx]
                Δweight =
                    bias + trace_scale*plast.trace_pre_plus.val[pre_idx]
                weights[neuron_fire_idx,pre_idx] = HPN.hardbounds(
                    weight+Δweight,plast.weight_min,plast.weight_max)
            end
        end
    end

    if post_fired
        HPN.propagate!(t_fire,plast.trace_post_minus)
        HPN.add_firing_event_now!(plast.trace_post_minus,neuron_fire_idx)
    end
    if pre_fired
        HPN.propagate!(t_fire,plast.trace_pre_plus)
        HPN.add_firing_event_now!(plast.trace_pre_plus,neuron_fire_idx)
    end
    return nothing
end

function apply_allocated!(
    plast::HPN.PlasticityAsymmetricSTDP,
    connection::HPN.ConnectionWithWeights,
    label::Symbol,
    neuron::Int)
    return @allocated HPN.apply_plasticity!(
        plast,connection,2.0,label,neuron)
end

function refresh_allocated!(
    plast::HPN.PlasticityAsymmetricSTDP,weights::Matrix{Float64})
    return @allocated HPN.refresh_mask!(plast,weights)
end

@testset "PlasticityAsymmetricSTDP" begin
    @testset "Construction and validation" begin
        weights = [0.0 1.0 2.0; 3.0 0.0 4.0]
        plast = HPN.PlasticityAsymmetricSTDP(
            0.1,0.0,-0.2,0.3,2.0,1.5,weights;
            weight_min=0.0,weight_max=5.0)

        @test length(plast.trace_pre_plus) == 3
        @test length(plast.trace_post_minus) == 2
        @test plast.trace_pre_plus.τ == 2.0
        @test plast.trace_post_minus.τ == 3.0
        @test plast.zero_weight_mask == Bool[1 0 0; 0 1 0]
        @test_throws ArgumentError HPN.PlasticityAsymmetricSTDP(
            0.1,1.1,0.0,0.0,1.0,1.0,weights)
        @test_throws ArgumentError HPN.PlasticityAsymmetricSTDP(
            0.1,0.0,0.0,0.0,0.0,1.0,weights)
        @test_throws ArgumentError HPN.PlasticityAsymmetricSTDP(
            0.1,0.0,0.0,0.0,1.0,0.0,weights)
        @test_throws ArgumentError HPN.PlasticityAsymmetricSTDP(
            0.1,0.0,0.0,0.0,1.0,1.0,weights;
            weight_min=2.0,weight_max=1.0)

        empty_plast = HPN.PlasticityAsymmetricSTDP(
            0.1,0.0,0.0,0.0,1.0,1.0,zeros(0,3))
        @test length(empty_plast.trace_pre_plus) == 3
        @test length(empty_plast.trace_post_minus) == 0
        @test size(empty_plast.zero_weight_mask) == (0,3)
    end

    @testset "Post and pre spike updates" begin
        weights = fill(1.0,2,3)
        plast = HPN.PlasticityAsymmetricSTDP(
            0.2,0.0,0.1,0.3,2.0,2.0,weights;
            weight_min=-10.0,weight_max=10.0)
        connection = HPN.ConnectionWithWeights(
            weights,:post,:pre,(plast,),true)

        plast.trace_pre_plus.val .= [2.0,4.0,6.0]
        plast.trace_pre_plus.t_last = 1.0
        before = copy(weights)
        decay_pre = exp(-(3.0-1.0)/2.0)
        expected_row =
            before[2,:] .+ 0.2 .* (0.3 .+ 0.5*decay_pre .*
            plast.trace_pre_plus.val)

        @test HPN.apply_plasticity!(
            plast,connection,3.0,:post,2) === nothing
        @test weights[2,:] ≈ expected_row
        @test weights[1,:] == before[1,:]
        @test plast.trace_post_minus.t_last == 3.0
        @test plast.trace_post_minus.val == [0.0,0.25]
        @test plast.trace_pre_plus.t_last == 1.0

        plast.trace_post_minus.val .= [3.0,5.0]
        plast.trace_post_minus.t_last = 2.0
        before = copy(weights)
        decay_post = exp(-(4.0-2.0)/4.0)
        expected_column =
            before[:,1] .+ 0.2 .* (0.1 .- 0.5*decay_post .*
            plast.trace_post_minus.val)

        @test HPN.apply_plasticity!(
            plast,connection,4.0,:pre,1) === nothing
        @test weights[:,1] ≈ expected_column
        @test weights[:,2:3] == before[:,2:3]
        @test plast.trace_pre_plus.t_last == 4.0
        @test isapprox(
            plast.trace_pre_plus.val[1],
            2.0*exp(-(4.0-1.0)/2.0)+0.5)
        @test plast.trace_post_minus.t_last == 2.0

        unchanged_weights = copy(weights)
        unchanged_pre = copy(plast.trace_pre_plus.val)
        unchanged_post = copy(plast.trace_post_minus.val)
        @test HPN.apply_plasticity!(
            plast,connection,5.0,:other,1) === nothing
        @test weights == unchanged_weights
        @test plast.trace_pre_plus.val == unchanged_pre
        @test plast.trace_post_minus.val == unchanged_post
    end

    @testset "Self connection reads before writing traces" begin
        weights = fill(1.0,3,3)
        plast = HPN.PlasticityAsymmetricSTDP(
            0.1,0.2,0.4,-0.3,2.0,1.5,weights;
            weight_min=-10.0,weight_max=10.0)
        connection = HPN.ConnectionWithWeights(
            weights,:self,:self,(plast,),true)
        plast.trace_pre_plus.val .= [1.0,2.0,3.0]
        plast.trace_post_minus.val .= [4.0,5.0,6.0]
        before = copy(weights)
        decay_pre = exp(-1.0)
        decay_post = exp(-2.0/3.0)
        pre_delta =
            0.1*(0.4 + ((0.2-1.0)/2.0)*decay_post*5.0)
        post_delta =
            0.1*(-0.3 + ((0.2+1.0)/2.0)*decay_pre*2.0)

        @test HPN.apply_plasticity!(
            plast,connection,2.0,:self,2) === nothing
        @test weights[2,2] ≈ before[2,2] + pre_delta + post_delta
        @test weights[1,2] ≈ before[1,2] +
            0.1*(0.4 - 0.4*decay_post*4.0)
        @test weights[2,3] ≈ before[2,3] +
            0.1*(-0.3 + 0.6*decay_pre*3.0)
        @test plast.trace_pre_plus.val ≈
            [decay_pre,2decay_pre+0.5,3decay_pre]
        @test plast.trace_post_minus.val ≈
            [4decay_post,5decay_post+inv(3.0),6decay_post]

        zero_diagonal = fill(1.0,3,3)
        zero_diagonal[diagind(zero_diagonal)] .= 0.0
        masked = HPN.PlasticityAsymmetricSTDP(
            1.0,1.0,1.0,1.0,1.0,1.0,zero_diagonal)
        masked_connection = HPN.ConnectionWithWeights(
            zero_diagonal,:self,:self,(masked,),true)
        HPN.apply_plasticity!(
            masked,masked_connection,1.0,:self,2)
        @test zero_diagonal[2,2] == 0.0
    end

    @testset "Structural mask and refresh" begin
        weights = [0.0 1.0; 2.0 3.0]
        plast = HPN.PlasticityAsymmetricSTDP(
            1.0,1.0,0.0,1.0,1.0,1.0,weights;
            weight_min=-10.0,weight_max=10.0)
        connection = HPN.ConnectionWithWeights(
            weights,:post,:pre,(plast,),true)

        weights[1,1] = 5.0
        HPN.apply_plasticity!(plast,connection,1.0,:post,1)
        @test weights[1,1] == 5.0

        @test HPN.refresh_mask!(plast,weights) === nothing
        HPN.apply_plasticity!(plast,connection,2.0,:post,1)
        @test weights[1,1] > 5.0

        weights[1,2] = 0.0
        HPN.apply_plasticity!(plast,connection,3.0,:post,1)
        @test weights[1,2] > 0.0
        weights[1,2] = 0.0
        HPN.refresh_mask!(plast,weights)
        HPN.apply_plasticity!(plast,connection,4.0,:post,1)
        @test weights[1,2] == 0.0
        @test_throws DimensionMismatch HPN.refresh_mask!(
            plast,zeros(3,3))
        wrong_connection = HPN.ConnectionWithWeights(
            zeros(3,3),:post,:pre,(plast,),true)
        @test_throws DimensionMismatch HPN.apply_plasticity!(
            plast,wrong_connection,5.0,:post,1)
    end

    @testset "Bounds and reset" begin
        weights = fill(0.95,1,1)
        plast = HPN.PlasticityAsymmetricSTDP(
            1.0,1.0,0.0,1.0,1.0,1.0,weights;
            weight_min=0.2,weight_max=1.0)
        connection = HPN.ConnectionWithWeights(
            weights,:post,:pre,(plast,),true)
        HPN.apply_plasticity!(plast,connection,1.0,:post,1)
        @test weights[1,1] == 1.0
        mask_before = copy(plast.zero_weight_mask)
        @test HPN.reset!(plast) === nothing
        @test plast.trace_pre_plus.val == [0.0]
        @test plast.trace_post_minus.val == [0.0]
        @test plast.zero_weight_mask == mask_before
    end

    @testset "Lazy and eager implementations agree" begin
        weights_lazy = [0.0 0.8 1.2; 0.4 0.7 0.0; 1.0 0.5 0.9]
        weights_eager = copy(weights_lazy)
        lazy = HPN.PlasticityAsymmetricSTDP(
            0.03,-0.2,0.1,-0.05,1.7,2.2,weights_lazy;
            weight_min=0.0,weight_max=2.0)
        eager = deepcopy(lazy)
        connection_lazy = HPN.ConnectionWithWeights(
            weights_lazy,:same,:same,(lazy,),true)
        connection_eager = HPN.ConnectionWithWeights(
            weights_eager,:same,:same,(eager,),true)

        for (time,neuron) in ((0.3,1),(0.8,3),(1.1,2),(2.4,1))
            HPN.apply_plasticity!(
                lazy,connection_lazy,time,:same,neuron)
            eager_reference_apply!(
                eager,connection_eager,time,:same,neuron)
        end

        @test weights_lazy ≈ weights_eager
        @test lazy.trace_pre_plus.val ≈ eager.trace_pre_plus.val
        @test lazy.trace_post_minus.val ≈ eager.trace_post_minus.val
        @test lazy.trace_pre_plus.t_last == eager.trace_pre_plus.t_last
        @test lazy.trace_post_minus.t_last == eager.trace_post_minus.t_last
    end

    @testset "Hot paths allocate no memory" begin
        weights = fill(0.5,8,8)
        weights[diagind(weights)] .= 0.0
        plast = HPN.PlasticityAsymmetricSTDP(
            0.01,0.0,0.0,0.0,2.0,1.5,weights)
        connection = HPN.ConnectionWithWeights(
            weights,:self,:self,(plast,),true)
        separate_connection = HPN.ConnectionWithWeights(
            weights,:post,:pre,(plast,),true)

        HPN.apply_plasticity!(plast,connection,1.0,:self,3)
        apply_allocated!(plast,connection,:self,3)
        apply_allocated!(plast,separate_connection,:pre,3)
        apply_allocated!(plast,separate_connection,:post,3)
        refresh_allocated!(plast,weights)
        @test apply_allocated!(plast,connection,:self,3) == 0
        @test apply_allocated!(plast,separate_connection,:pre,3) == 0
        @test apply_allocated!(plast,separate_connection,:post,3) == 0
        @test refresh_allocated!(plast,weights) == 0
    end
end
