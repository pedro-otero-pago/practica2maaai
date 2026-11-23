# =============================================================
#  rfe.jl — Filtro wrapper RFE con regresión logística
#  En cada ronda se ajusta una RL y se elimina la fracción
#  `descarte` de variables con menor importancia, hasta quedar k.
# =============================================================

using MLJ
import MLJModelInterface, MLJBase

RegLog = @load LogisticClassifier pkg=MLJLinearModels verbosity=0

mutable struct RFEFilter <: MLJModelInterface.Supervised
    k::Int # número de variables conservadas al final
    descarte::Float64 # fracción eliminada en cada ronda
end
RFEFilter(; k::Int = 32, descarte::Float64 = 0.5) = RFEFilter(k, descarte)

# Importancia de cada variable: suma del valor absoluto de su coef. de la RL sobre todas las clases
function importanciasRL(Xmat, nombres, y)
    mach = machine(RegLog(), MLJBase.table(Xmat, names = nombres), y)
    fit!(mach, verbosity = 0)
    coefs = fitted_params(mach).coefs # nombre -> coeficiente/s
    imp = Dict(Symbol(first(p)) => sum(abs.(last(p))) for p in coefs)
    return [imp[n] for n in nombres]
end

function MLJModelInterface.fit(model::RFEFilter, verbosity::Int, X, y)
    Xmat = Float64.(MLJBase.matrix(X))
    nombres = collect(MLJBase.schema(X).names)
    activas = collect(1:size(Xmat,2))
    k = min(model.k, length(activas))
    rondas = [length(activas)]

    while length(activas) > k
        imp = importanciasRL(Xmat[:, activas], nombres[activas], y)
        quitar = max(1, floor(Int, model.descarte * length(activas)))
        quedan = max(k, length(activas) - quitar)
        mejores = sortperm(imp, rev = true)[1:quedan]
        activas = activas[sort(mejores)] # conservamos el orden original
        push!(rondas, length(activas))
    end

    fitresult = (idxs = activas, selected_names = nombres[activas])
    report = (selected_features = nombres[activas], rondas = rondas)
    return fitresult, nothing, report
end

function MLJModelInterface.transform(model::RFEFilter, fitresult, X)
    Xmat = MLJBase.matrix(X)
    return MLJBase.table(Xmat[:, fitresult.idxs], names = fitresult.selected_names)
end

MLJModelInterface.input_scitype(::Type{<:RFEFilter}) = Table(Continuous)
MLJModelInterface.target_scitype(::Type{<:RFEFilter}) = AbstractVector{<:Finite}
MLJModelInterface.output_scitype(::Type{<:RFEFilter}) = Table(Continuous)