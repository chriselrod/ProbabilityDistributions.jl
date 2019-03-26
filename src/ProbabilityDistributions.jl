module ProbabilityDistributions

using   SIMDPirates, SLEEFPirates,
        PaddedMatrices, StructuredMatrices, ScatteredArrays, StaticArrays,
        VectorizationBase, LoopVectorization

using PaddedMatrices: AbstractFixedSizePaddedMatrix
using ScatteredArrays: AbstractScatteredArray
using StructuredMatrices: AbstractAutoregressiveMatrix
using DistributionParameters: LKJ_Correlation_Cholesky

include("distribution_functions.jl")
include("normal_distribution.jl")

end # module
