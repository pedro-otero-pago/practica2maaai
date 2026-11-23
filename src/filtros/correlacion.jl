# =============================================================
#  correlacion.jl — Filtros por correlación (Pearson, Spearman, Kendall)
#  La clase tiene 6 valores sin orden: cada variable se puntúa con el
#  máximo |ρ| frente al indicador de cada clase (uno contra resto).
# =============================================================
using Statistics, StatsBase, MLJ
import MLJModelInterface, MLJBase


# Puntuación de una variable frente a la clase (6 clases), uno contra resto:
# para cada clase c se correlaciona x con el indicador (y .== c)
# y se devuelve el máximo |ρ|.
function puntuacionCorrelacion(x::AbstractVector{<:Real}, y::AbstractVector, metodo::Symbol)
    xf = Float64.(x)
    mejor = 0.0
    for c in unique(y)
        indicador = Float64.(y .== c)
        ρ = if metodo == :pearson
            cor(xf, indicador)
        elseif metodo == :spearman
            corspearman(xf, indicador)
        elseif metodo == :kendall
            corkendall(xf, indicador)
        else
            error("Método desconocido: $metodo")
        end
        isnan(ρ) && continue   # variable constante en este fold
        mejor = max(mejor, abs(ρ))
    end
    return mejor
end


mutable struct CorrelacionFilter <: MLJModelInterface.Supervised
    k::Int           # número de variables que se conservan
    metodo::Symbol   # :pearson, :spearman o :kendall
end
CorrelacionFilter(; k::Int = 32, metodo::Symbol = :pearson) = CorrelacionFilter(k, metodo)

function MLJModelInterface.fit(model::CorrelacionFilter, verbosity::Int, X, y)
    Xmat    = Float64.(MLJBase.matrix(X))
    nombres = collect(MLJBase.schema(X).names)
    puntos  = [puntuacionCorrelacion(Xmat[:, j], y, model.metodo) for j in 1:size(Xmat, 2)]
    k       = min(model.k, length(puntos))
    idxs    = sortperm(puntos, rev = true)[1:k]      # las k de mayor |ρ|
    fitresult = (idxs = idxs, selected_names = nombres[idxs])
    report    = (correlaciones = puntos, selected_features = nombres[idxs])
    return fitresult, nothing, report
end

function MLJModelInterface.transform(model::CorrelacionFilter, fitresult, X)
    Xmat = MLJBase.matrix(X)
    return MLJBase.table(Xmat[:, fitresult.idxs], names = fitresult.selected_names)
end

MLJModelInterface.input_scitype(::Type{<:CorrelacionFilter})  = Table(Continuous)
MLJModelInterface.target_scitype(::Type{<:CorrelacionFilter}) = AbstractVector{<:Finite}
MLJModelInterface.output_scitype(::Type{<:CorrelacionFilter}) = Table(Continuous)