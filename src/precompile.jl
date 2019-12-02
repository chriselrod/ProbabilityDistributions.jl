function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    # precompile(Tuple{ProbabilityDistributions.var"##s76#286",Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any})
    precompile(Tuple{typeof(ProbabilityDistributions.∂multivariate_normal_SMLT_quote),ProbabilityDistributions.NormalCholeskyConfiguration{Float64}})
    # precompile(Tuple{ProbabilityDistributions.var"##s76#196",Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any})
    precompile(Tuple{typeof(ProbabilityDistributions.multivariate_normal_SMLT_quote),ProbabilityDistributions.NormalCholeskyConfiguration{Float64}})
    precompile(Tuple{typeof(ProbabilityDistributions.track_mu_store),Int64,Int64,Type,Int64,Bool,Int64,Int64,Int64,Bool,Bool,Bool,Bool,Bool,Int64})
    precompile(Tuple{typeof(ProbabilityDistributions.loadδfnmadd_quote),Int64,Int64,Int64,DataType,Symbol,Symbol,Int64,Int64,Symbol,Symbol,Symbol,Symbol,Bool,Bool,Int64,Int64,Int64,Bool})
    precompile(Tuple{typeof(ProbabilityDistributions.loadδfnmadd_quote),Int64,Int64,Symbol,DataType,Int64,Int64,Int64,Int64,Symbol,Symbol,Symbol,Symbol,Bool,Bool,Int64,Int64,Int64,Bool})
    precompile(Tuple{typeof(ProbabilityDistributions.loadδfnmadd_quote),Int64,Int64,Symbol,DataType,Symbol,Symbol,Int64,Int64,Symbol,Symbol,Symbol,Symbol,Bool,Bool,Int64,Int64,Int64,Bool})
    precompile(Tuple{typeof(ProbabilityDistributions.loadδfnmadd_quote),Int64,Int64,Int64,DataType,Int64,Int64,Int64,Int64,Symbol,Symbol,Symbol,Symbol,Bool,Bool,Int64,Int64,Int64,Bool})
    # precompile(Tuple{ProbabilityDistributions.var"##s25#21",Any,Any,Any,Any,Any,Any,Any,Any,Any,Any})
    precompile(Tuple{typeof(ProbabilityDistributions.track_mu_store),Symbol,Int64,Type,Int64,Bool,Int64,Int64,Int64,Bool,Bool,Bool,Bool,Bool,Int64})
    # precompile(Tuple{ProbabilityDistributions.var"##s76#254",Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any})
end
