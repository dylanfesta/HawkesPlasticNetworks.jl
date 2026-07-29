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
end
