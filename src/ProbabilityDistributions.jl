module ProbabilityDistributions

using   SIMDPirates, SLEEFPirates, SpecialFunctions,
        PaddedMatrices, StructuredMatrices, ScatteredArrays, StaticArrays, LinearAlgebra,
        VectorizationBase, LoopVectorization

using PaddedMatrices: AbstractFixedSizePaddedVector, AbstractFixedSizePaddedMatrix
using ScatteredArrays: AbstractScatteredArray
using StructuredMatrices: AbstractAutoregressiveMatrix
using DistributionParameters: LKJ_Correlation_Cholesky

function return_expression(return_expr)
    length(return_expr.args) == 1 ? return_expr.args[1] : return_expr
end

include("distribution_functions.jl")
include("normal_distribution.jl")

end # module
