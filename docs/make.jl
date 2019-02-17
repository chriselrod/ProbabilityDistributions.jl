using Documenter, ProbabilityDistributions

makedocs(;
    modules=[ProbabilityDistributions],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/chriselrod/ProbabilityDistributions.jl/blob/{commit}{path}#L{line}",
    sitename="ProbabilityDistributions.jl",
    authors="Chris Elrod",
    assets=[],
)

deploydocs(;
    repo="github.com/chriselrod/ProbabilityDistributions.jl",
)
