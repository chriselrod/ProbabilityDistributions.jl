module ProbabilityDistributions

using   SIMDPirates, SLEEFPirates, SpecialFunctions, DistributionParameters,
        PaddedMatrices, StructuredMatrices, ScatteredArrays, LinearAlgebra,
        VectorizationBase, LoopVectorization, StackPointers, Parameters

using PaddedMatrices:
    DynamicPtrMatrix, AbstractMutableFixedSizeArray, simplify_expr,
    AbstractFixedSizeVector, AbstractFixedSizeMatrix, AbstractFixedSizeArray, AbstractPaddedMatrix,
    AbstractMutableFixedSizeVector, AbstractMutableFixedSizeMatrix
    
using ScatteredArrays: AbstractScatteredArray
using StructuredMatrices: AbstractAutoregressiveMatrix, AbstractLowerTriangularMatrix
using DistributionParameters: AbstractCorrCholesky, AbstractCovarCholesky, AbstractFixedSizeCovarianceMatrix, invdiag, logdiag
using VectorizationBase: pick_vector_width, pick_vector_width_shift
using ReverseDiffExpressionsBase: isinitialized, uninitialized
using MacroTools: postwalk

function return_expression(return_expr)
    length(return_expr.args) == 1 ? return_expr.args[1] : return_expr
end
function return_expression(return_expr, sp::Bool, spexpr = :sp)
    expr = length(return_expr.args) == 1 ? return_expr.args[1] : return_expr
    sp ? Expr(:tuple, spexpr, expr) : expr
end

include("distribution_functions.jl")
include("normal/normal.jl")
include("normal/univariate_normal.jl")
include("normal/multivariate_normal.jl")
# include("normal/matrix_normal.jl")

const STACK_POINTER_SUPPORTED_METHODS = Set{Symbol}()
@def_stackpointer_fallback ∂lsgg ∂Gamma ∂LKJ ∂Normal ∂Beta ∂Bernoulli_logit ∂Binomial_logit #∂EₘₐₓNMA
@def_stackpointer_noalloc Normal ∂Normal!

function __init__()
    @add_stackpointer_method ∂lsgg ∂Gamma ∂LKJ ∂Normal ∂Beta ∂Bernoulli_logit ∂Binomial_logit ∂EₘₐₓNMA
    @add_stackpointer_noalloc Normal ∂Normal!

    if VERSION > v"1.3.0-rc1"
        Threads.@spawn precompile(multivariate_normal_SMLT_quote, (NormalCholeskyConfiguration{Float64},))
        Threads.@spawn precompile(∂multivariate_normal_SMLT_quote, (NormalCholeskyConfiguration{Float64},))
        Threads.@spawn precompile(univariate_normal_quote, (Int,Float64,Bool,Bool,Bool,NTuple{3,Bool},NTuple{3,Bool},Bool,Bool))
    else
        precompile(multivariate_normal_SMLT_quote, (NormalCholeskyConfiguration{Float64},))
        precompile(∂multivariate_normal_SMLT_quote, (NormalCholeskyConfiguration{Float64},))
        precompile(univariate_normal_quote, (Int,Float64,Bool,Bool,Bool,NTuple{3,Bool},NTuple{3,Bool},Bool,Bool))
    end
end

end # module
