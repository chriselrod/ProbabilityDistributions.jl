# ProbabilityDistributions

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://chriselrod.github.io/ProbabilityDistributions.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://chriselrod.github.io/ProbabilityDistributions.jl/dev)
[![Build Status](https://travis-ci.com/chriselrod/ProbabilityDistributions.jl.svg?branch=master)](https://travis-ci.com/chriselrod/ProbabilityDistributions.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/chriselrod/ProbabilityDistributions.jl?svg=true)](https://ci.appveyor.com/project/chriselrod/ProbabilityDistributions-jl)
[![Codecov](https://codecov.io/gh/chriselrod/ProbabilityDistributions.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/chriselrod/ProbabilityDistributions.jl)
[![Coveralls](https://coveralls.io/repos/github/chriselrod/ProbabilityDistributions.jl/badge.svg?branch=master)](https://coveralls.io/github/chriselrod/ProbabilityDistributions.jl?branch=master)


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



