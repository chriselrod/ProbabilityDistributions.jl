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

abstract type AbstractProbabilityDistribution{track} end
struct Updated∇ end

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

# _precompile_()




end # module
