const HPN_STDHS = HawkesPlasticNetworks

@testset "PlasticityHomeostaticScaling construction" begin
    weights = [0.0 0.2; 0.3 0.4]
    plast = HPN_STDHS.PlasticityHomeostaticScaling(
        1e-6,5.0,1,20.0,weights;
        weight_min=0.0,weight_max=1.0)

    @test plast.η == 1e-6
    @test plast.α == 5.0
    @test plast.s == 1.0
    @test plast.τ == 20.0
    @test plast.weight_min == 0.0
    @test plast.weight_max == 1.0
    @test plast.trace_post.val == zeros(2)
    @test plast.zero_weight_mask == iszero.(weights)
end

@testset "PlasticityHomeostaticScaling signed update" begin
    weights = [0.5 1.0; 0.0 2.0]
    mask = iszero.(weights)

    @test HPN_STDHS._apply_stdhs_single_trace!(
        weights,mask,8.0,1,0.1,-5.0,1.0,0.0,10.0) === nothing
    @test weights ≈ [0.65 1.3; 0.0 2.0]

    @test HPN_STDHS._apply_stdhs_single_trace!(
        weights,mask,8.0,1,0.1,5.0,-1.0,0.0,10.0) === nothing
    @test weights ≈ [0.455 0.91; 0.0 2.0]
end
