using Random

@testset "Fixed-input sparse weight matrix" begin
    weights = HawkesPlasticNetworks.generate_sparse_matrix_fixed_input(
        4,4,2.5,1.0)

    @test size(weights) == (4,4)
    @test all(isapprox.(vec(sum(weights,dims=2)),2.5))
    @test all(weights .== 2.5 / 4)

    weights_without_self =
        HawkesPlasticNetworks.generate_sparse_matrix_fixed_input(
            4,4,-3.0,1.0;self_connections=false)
    @test all(iszero,diag(weights_without_self))
    @test all(isapprox.(vec(sum(weights_without_self,dims=2)),-3.0))
    @test all(vec(sum(weights_without_self .!= 0.0,dims=1)) .> 0)

    @test_throws ArgumentError HawkesPlasticNetworks.generate_sparse_matrix_fixed_input(
        2,2,1.0,0.0)
    @test_throws ArgumentError HawkesPlasticNetworks.generate_sparse_matrix_fixed_input(
        2,2,1.0,1.1)
    @test_throws ArgumentError HawkesPlasticNetworks.generate_sparse_matrix_fixed_input(
        0,2,1.0,0.5)
    @test_throws ArgumentError HawkesPlasticNetworks.generate_sparse_matrix_fixed_input(
        2,0,1.0,0.5)

    # With one possible connection removed, both the row and column are empty.
    @test_throws ArgumentError HawkesPlasticNetworks.generate_sparse_matrix_fixed_input(
        1,1,1.0,1.0;self_connections=false)

    first_sample = HawkesPlasticNetworks.generate_sparse_matrix_fixed_input(
        10,10,1.0,0.8;rng=MersenneTwister(12))
    second_sample = HawkesPlasticNetworks.generate_sparse_matrix_fixed_input(
        10,10,1.0,0.8;rng=MersenneTwister(12))
    @test first_sample == second_sample
end

@testset "Uniform two-population weight matrix" begin
    ne = 4
    ni = 3
    weights = HawkesPlasticNetworks.generate_two_population_uniform_weight_matrix(
        ne,ni,2.0,3.0,4.0,5.0)
    excitatory = 1:ne
    inhibitory = (ne + 1):(ne + ni)

    @test size(weights) == (ne + ni,ne + ni)
    @test all(iszero,diag(weights))
    @test all(isapprox.(vec(sum(weights[excitatory,excitatory],dims=2)),2.0))
    @test all(isapprox.(vec(sum(weights[inhibitory,excitatory],dims=2)),3.0))
    @test all(isapprox.(vec(sum(weights[excitatory,inhibitory],dims=2)),-4.0))
    @test all(isapprox.(vec(sum(weights[inhibitory,inhibitory],dims=2)),-5.0))
    @test all(weights[inhibitory,excitatory] .> 0.0)
    @test all(weights[excitatory,inhibitory] .< 0.0)

    sparse_weights = HawkesPlasticNetworks.generate_two_population_uniform_weight_matrix(
        10,10,2.0,3.0,4.0,5.0;
        p_ee=0.8,p_ie=0.8,p_ei=0.8,p_ii=0.8,
        rng=MersenneTwister(42))
    @test count(iszero,sparse_weights) > 20
    @test all(isapprox.(vec(sum(sparse_weights[1:10,1:10],dims=2)),2.0))
    @test all(isapprox.(vec(sum(sparse_weights[11:20,1:10],dims=2)),3.0))
    @test all(isapprox.(vec(sum(sparse_weights[1:10,11:20],dims=2)),-4.0))
    @test all(isapprox.(vec(sum(sparse_weights[11:20,11:20],dims=2)),-5.0))

    @test_throws ArgumentError HawkesPlasticNetworks.generate_two_population_uniform_weight_matrix(
        1,2,1.0,1.0,1.0,1.0)
    @test_throws ArgumentError HawkesPlasticNetworks.generate_two_population_uniform_weight_matrix(
        2,1,1.0,1.0,1.0,1.0)
    @test_throws ArgumentError HawkesPlasticNetworks.generate_two_population_uniform_weight_matrix(
        2,2,1.0,1.0,1.0,1.0;p_ei=0.0)
end
