module ProbabilityDistributions

using   SIMDPirates, SLEEFPirates, SpecialFunctions, DistributionParameters,
        PaddedMatrices, StructuredMatrices, ScatteredArrays, LinearAlgebra,
        VectorizationBase, LoopVectorization, StackPointers

using PaddedMatrices: StackPointer, DynamicPtrMatrix,
    AbstractFixedSizePaddedVector, AbstractFixedSizePaddedMatrix, AbstractFixedSizePaddedArray, AbstractPaddedMatrix,
    AbstractMutableFixedSizePaddedVector, AbstractMutableFixedSizePaddedMatrix,
    AbstractMutableFixedSizePaddedArray, simplify_expr
using ScatteredArrays: AbstractScatteredArray
using StructuredMatrices: AbstractAutoregressiveMatrix, AbstractLowerTriangularMatrix
using DistributionParameters: AbstractLKJCorrCholesky, AbstractFixedSizeCovarianceMatrix
using SIMDPirates: extract_data, vbroadcast, vadd, vsub, vfnmadd, vfmsub, vfnmsub, vsum, vload, vstore!
using VectorizationBase: pick_vector_width, pick_vector_width_shift
using LoopVectorization: @vvectorize

function return_expression(return_expr)
    length(return_expr.args) == 1 ? return_expr.args[1] : return_expr
end
function return_expression(return_expr, sp::Bool, spexpr = :sp)
    expr = length(return_expr.args) == 1 ? return_expr.args[1] : return_expr
    sp ? Expr(:tuple, spexpr, expr) : expr
end


include("distribution_functions.jl")
include("normal/univariate_normal.jl")
include("normal/multivariate_normal.jl")
include("normal/matrix_normal.jl")

# const STACK_POINTER_SUPPORTED_METHODS = Set{Symbol}()
@def_stackpointer_fallback ∂lsgg ∂Gamma ∂LKJ Normal ∂Normal Normal_fmadd ∂Normal_fmadd

function __init__()
    @add_stackpointer_method ∂lsgg ∂Gamma ∂LKJ Normal ∂Normal Normal_fmadd ∂Normal_fmadd
end

end # module
