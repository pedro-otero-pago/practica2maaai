# =============================================================
#  anova.jl — Filtro ANOVA (SelectKBest por estadístico F)
#  Patrón ANOVAFilter del Tutorial 02, corregido: en HypothesisTests,
#  SStᵢ y SSeᵢ son vectores (un valor por grupo), así que
#  F = (ΣSStᵢ / DFt) / (ΣSSeᵢ / DFe). La versión del tutorial daba F = 0
#  para todas las variables sin error (lo ocultaba el try/catch).
# =============================================================
using HypothesisTests, MLJ
import MLJModelInterface, MLJBase

mutable struct ANOVAFilter <: MLJModelInterface.Supervised
    k::Int          # número de variables que se conservan
end
ANOVAFilter(; k::Int = 32) = ANOVAFilter(k)

"""Estadístico F del ANOVA de un factor de la variable x agrupada por la clase y."""
function estadisticoF(x::AbstractVector, y::AbstractVector)
    grupos = [x[y .== c] for c in unique(y)]     # un vector de valores por clase
    t = OneWayANOVATest(grupos...)
    F = (sum(t.SStᵢ) / t.DFt) / (sum(t.SSeᵢ) / t.DFe)   # fórmula modificada
    return isfinite(F) ? F : 0.0                        # variable constante -> F no definido -> 0
end

function MLJModelInterface.fit(model::ANOVAFilter, verbosity::Int, X, y)
    Xmat    = Float64.(MLJBase.matrix(X))
    nombres = collect(MLJBase.schema(X).names)         
    fstats  = [estadisticoF(Xmat[:, j], y) for j in 1:size(Xmat, 2)]
    k       = min(model.k, length(fstats))
    idxs    = sortperm(fstats, rev = true)[1:k]       # las k de F más alto
    fitresult = (idxs = idxs, selected_names = nombres[idxs])
    report    = (fstats = fstats, selected_features = nombres[idxs])
    return fitresult, nothing, report
end

function MLJModelInterface.transform(model::ANOVAFilter, fitresult, X)
    Xmat = MLJBase.matrix(X)
    return MLJBase.table(Xmat[:, fitresult.idxs], names = fitresult.selected_names)
end

MLJModelInterface.input_scitype(::Type{<:ANOVAFilter})  = Table(Continuous)
MLJModelInterface.target_scitype(::Type{<:ANOVAFilter}) = AbstractVector{<:Finite}
MLJModelInterface.output_scitype(::Type{<:ANOVAFilter}) = Table(Continuous)