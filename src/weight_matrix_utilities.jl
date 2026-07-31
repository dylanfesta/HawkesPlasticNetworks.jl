#= 
This is a sub-section of HawkesPlasticNetworks.jl

Contains utility functions to generate weight matrices,
with minimal dependency on the rest of the package

Note that I use the usual convention post <- pre when ordering
arguments and for matrix indices.

=#


"""
   generate_sparse_matrix_fixed_input(n_post::Integer,n_pre::Integer,
   w::Real,p::Real=0.1; self_connections::Bool=true,
   rng::AbstractRNG=Random.default_rng())

Generate a sparse matrix, with density `p` so that each row
sums to `w`. An `ArgumentError` is thrown if the sampled connectivity
contains a row or column with no connections.

Arguments:
- n_post::Integer: number of postsynaptic neurons
- n_pre::Integer: number of presynaptic neurons
- w::Real: weight of each connection
- p::Real: probability of each connection
- self_connections::Bool: whether to include self-connections
- rng::AbstractRNG: random-number generator used to sample connectivity

Returns:
- w::Matrix{Float64}: weight matrix

"""
function generate_sparse_matrix_fixed_input(n_post::Integer,n_pre::Integer,
  w::Real,p::Real=0.1; self_connections::Bool=true,
  rng::AbstractRNG=Random.default_rng())

  if min(n_post,n_pre) < 1
    throw(ArgumentError("population sizes must be positive"))
  end
  if !(0.0 < p <= 1.0)
    throw(ArgumentError("p must satisfy 0 < p <= 1"))
  end

  # fill with ones
  weights = Float64.(rand(rng,n_post,n_pre) .< p)
  if !self_connections
    weights[diagind(weights)] .= 0.0
  end

  # count
  row_sums = sum(weights,dims=2)
  column_sums = sum(weights,dims=1)
  if any(iszero,vcat(vec(row_sums),vec(column_sums)))
    throw(ArgumentError(
      "sampled connectivity contains an empty row or column; increase p"))
  end
  # replace ones with w divided by number of inputs
  # (this broadcasts by row)
  weights .*= w ./ row_sums
  return weights
end

"""
    generate_two_population_uniform_weight_matrix(ne::Integer, ni::Integer,
        wee::Real, wie::Real, wei::Real, wii::Real;
        p_ee::Real=1.0, p_ie::Real=1.0,
        p_ei::Real=1.0, p_ii::Real=1.0,
        rng::AbstractRNG=Random.default_rng())

Generate the weight matrix for uniformly weighted excitatory and inhibitory
populations. Matrix indices and block names follow the `post <- pre`
convention. Thus, for example, `wie` and `p_ie` describe the `I <- E`
block. Each row of a block sums to the corresponding requested weight.

The inhibitory blocks are made negative using `-abs`. Recurrent blocks have
no self-connections. Each sparsity parameter is the probability that a
connection is present and must satisfy `0 < p <= 1`.
"""
function generate_two_population_uniform_weight_matrix(ne::Integer,ni::Integer,
  wee::Real,wie::Real,wei::Real,wii::Real;
  p_ee::Real=1.0,p_ie::Real=1.0,p_ei::Real=1.0,p_ii::Real=1.0,
  rng::AbstractRNG=Random.default_rng())

  if ne < 2
    throw(ArgumentError("ne must be at least 2"))
  end
  if ni < 2
    throw(ArgumentError("ni must be at least 2"))
  end

  n = ne + ni
  excitatory = 1:ne
  inhibitory = (ne + 1):n
  weights = Matrix{Float64}(undef,n,n)

  weights[excitatory,excitatory] .= generate_sparse_matrix_fixed_input(
    ne,ne,wee,p_ee;self_connections=false,rng=rng)
  weights[inhibitory,excitatory] .= generate_sparse_matrix_fixed_input(
    ni,ne,wie,p_ie;rng=rng)
  weights[excitatory,inhibitory] .= generate_sparse_matrix_fixed_input(
    ne,ni,-abs(wei),p_ei;rng=rng)
  weights[inhibitory,inhibitory] .= generate_sparse_matrix_fixed_input(
    ni,ni,-abs(wii),p_ii;self_connections=false,rng=rng)

  return weights
end

"""
scale_rows_to_sum!(mat::AbstractMatrix{Float64},row_sum::Real)

Scales each row of `mat` so that the row sums to `row_sum`.
(this corresponds to scaling all incoming weights)
Throws an error if any row is zero.
"""
function scale_rows_to_sum!(mat::AbstractMatrix{Float64},row_sum::Real)
  row_sums = sum(mat,dims=2)
  if any(iszero,vec(row_sums))
    throw(ArgumentError("matrix contains an empty row"))
  end
  mat .*= row_sum ./ row_sums
  return nothing
end

  
                            
