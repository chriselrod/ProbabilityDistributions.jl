function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    precompile(Tuple{Type{ProbabilityDistributions.NormalCholeskyConfiguration{Float64}}})
    precompile(Tuple{typeof(ProbabilityDistributions.gamma_quote),Int64,Type{Float64},Tuple{Bool,Bool,Bool},Tuple{Bool,Bool,Bool},Bool,Tuple{Bool,Bool,Bool}})
    precompile(Tuple{typeof(ProbabilityDistributions.multivariate_normal_SMLT_quote),ProbabilityDistributions.NormalCholeskyConfiguration{Float64}})
    precompile(Tuple{typeof(ProbabilityDistributions.univariate_normal_quote),Int64,Type{Float64},Bool,Union{Nothing, Bool},Union{Nothing, Bool},Tuple{Bool,Bool,Bool},Tuple{Bool,Bool,Bool},Bool,Bool})
    precompile(Tuple{typeof(ProbabilityDistributions.∂multivariate_normal_SMLT_quote),ProbabilityDistributions.NormalCholeskyConfiguration{Float64}})
end
