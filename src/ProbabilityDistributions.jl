module ProbabilityDistributions

using PaddedMatrices, SIMDPirates, SLEEFPirates, LoopVectorization

include("distribution_functions.jl")
include("normal_distribution.jl")

end # module
