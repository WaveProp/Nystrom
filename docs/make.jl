using Nystrom
using Documenter

DocMeta.setdocmeta!(Nystrom, :DocTestSetup, :(using Nystrom); recursive=true)

makedocs(;
    modules=[Nystrom],
    authors="Luiz M. Faria <maltezfaria@gmail.com> and contributors",
    repo="https://github.com/WaveProp/Nystrom.jl/blob/{commit}{path}#{line}",
    sitename="Nystrom.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://WaveProp.github.io/Nystrom.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/WaveProp/Nystrom.jl",
    devbranch="main",
)
