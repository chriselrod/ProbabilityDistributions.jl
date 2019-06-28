module ProbabilityDistributions

using   SIMDPirates, SLEEFPirates, SpecialFunctions, DistributionParameters,
        PaddedMatrices, StructuredMatrices, ScatteredArrays, StaticArrays, LinearAlgebra,
        VectorizationBase, LoopVectorization

using PaddedMatrices: StackPointer, DynamicPtrMatrix,
    AbstractFixedSizePaddedVector, AbstractFixedSizePaddedMatrix, AbstractPaddedMatrix
using ScatteredArrays: AbstractScatteredArray
using StructuredMatrices: AbstractAutoregressiveMatrix
using DistributionParameters: AbstractLKJCorrCholesky
using SIMDPirates: extract_data, vbroadcast, vadd, vsub, vfnmadd, vfmsub, vfnmsub, vsum, vload, vstore!
using VectorizationBase: pick_vector_width, pick_vector_width_shift

function return_expression(return_expr)
    length(return_expr.args) == 1 ? return_expr.args[1] : return_expr
end

include("distribution_functions.jl")
include("normal_distribution.jl")

const STACK_POINTER_SUPPORTED_METHODS = Set{Symbol}()
PaddedMatrices.@support_stack_pointer ∂lsgg
PaddedMatrices.@support_stack_pointer ∂LKJ
PaddedMatrices.@support_stack_pointer Normal
PaddedMatrices.@support_stack_pointer ∂Normal

function __init__()
    for m ∈ (:∂lsgg, :∂LKJ, :Normal, :∂Normal)
        push!(PaddedMatrices.STACK_POINTER_SUPPORTED_METHODS, m)
    end
end

end # module
