using Documenter, LagrangianStochasticConvection

DocMeta.setdocmeta!(LagrangianStochasticConvection, :DocTestSetup, :(using LagrangianStochasticConvection); recursive = true)

makedocs(
    sitename = "LagrangianStochasticConvection.jl",
    modules = [LagrangianStochasticConvection],
    repo=Remotes.GitLab("gitlab.obspm.fr", "jphilidet", "LagrangianStochasticConvection"),
    pages = [
        "index.md",
        "tutorials.md",
        "kernel.md",
        "particles.md",
        "publicAPI.md",
        "otherreferences.md"
    ]
)

# deploydocs(
#     repo="gitlab.obspm.fr/jphilidet/lagrangian_stochastic_convection"
# )