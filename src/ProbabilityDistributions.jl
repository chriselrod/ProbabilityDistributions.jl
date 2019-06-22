module ProbabilityDistributions

using   SIMDPirates, SLEEFPirates, SpecialFunctions, DistributionParameters,
        PaddedMatrices, StructuredMatrices, ScatteredArrays, StaticArrays, LinearAlgebra,
        VectorizationBase, LoopVectorization

using PaddedMatrices: StackPointer, @support_stack_pointer,
    AbstractFixedSizePaddedVector, AbstractFixedSizePaddedMatrix, AbstractPaddedMatrix
using ScatteredArrays: AbstractScatteredArray
using StructuredMatrices: AbstractAutoregressiveMatrix
using DistributionParameters: LKJCorrCholesky

function return_expression(return_expr)
    length(return_expr.args) == 1 ? return_expr.args[1] : return_expr
end

include("distribution_functions.jl")
include("normal_distribution.jl")

end # module
