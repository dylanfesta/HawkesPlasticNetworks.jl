using Documenter
using HawkesPlasticNetworks
using Literate

DocMeta.setdocmeta!(HawkesPlasticNetworks, :DocTestSetup, :(using HawkesPlasticNetworks); recursive=true)

docs_dir = @__DIR__
generated_dir = joinpath(docs_dir,"src","generated")
Literate.markdown(
    joinpath(docs_dir,"literate","01_single_unit.jl"),
    generated_dir;
    flavor=Literate.DocumenterFlavor(),
)
Literate.markdown(
    joinpath(docs_dir,"literate","02_ei_network.jl"),
    generated_dir;
    flavor=Literate.DocumenterFlavor(),
)

makedocs(;
    modules=[HawkesPlasticNetworks],
    authors="Dylan Festa <dylan.festa@gmail.com>",
    repo="https://github.com/dylanfesta/HawkesPlasticNetworks.jl/blob/{commit}{path}#{line}",
    sitename="HawkesPlasticNetworks.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://dylanfesta.github.io/HawkesPlasticNetworks.jl",
        repolink="https://github.com/dylanfesta/HawkesPlasticNetworks.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Examples" => [
            "Single self-interacting unit" => "generated/01_single_unit.md",
            "One excitatory and one inhibitory unit" =>
                "generated/02_ei_network.md",
        ],
    ],
)

deploydocs(;
    repo="github.com/dylanfesta/HawkesPlasticNetworks.jl",
    devbranch="main",
)
