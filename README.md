# ProbabilityDistributions

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://chriselrod.github.io/ProbabilityDistributions.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://chriselrod.github.io/ProbabilityDistributions.jl/dev)
[![CI](https://github.com/chriselrod/ProbabilityDistributions.jl/workflows/CI/badge.svg)](https://github.com/chriselrod/ProbabilityDistributions.jl/actions?query=workflow%3ACI)
[![CI (Julia nightly)](https://github.com/chriselrod/ProbabilityDistributions.jl/workflows/CI%20(Julia%20nightly)/badge.svg)](https://github.com/chriselrod/ProbabilityDistributions.jl/actions?query=workflow%3A%22CI+%28Julia+nightly%29%22)
[![Codecov](https://codecov.io/gh/chriselrod/ProbabilityDistributions.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/chriselrod/ProbabilityDistributions.jl)

API is:
Distributions are structs, with unnormalized `logdensity` and `∂logdensity!` methods.
```julia
logdensity(Noram{(true,false,true)}(), y, μ, σ)
∂logdensity!((∂y, ∂μ, ∂σ) Noram{(true,false,true)}(), y, μ, σ)
```
`logdensity` returns the `target` density in an unreduced form.
`∂logdensity!` returns a tuple `(target,(∂y, ∂μ, ∂σ))`

The `true`/`false` specifies whether the corresponding parameter is to be treated as a constant or as an unknown for the purpose of normalization.
`∂logdensity!` will calculate gradients for each unknown. If the corresponding entry, e.g. `∂y` is `nothing`, it will return the calculated gradient.
Otherwise, it will return `Zero()`.



