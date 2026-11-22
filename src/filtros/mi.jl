# =============================================================
#  mi.jl — Filtro por Información Mutua (SelectKBest por I(X;Y))
#  Cada variable continua se discretiza en B intervalos de igual
#  frecuencia (cuantiles), robustos a los valores extremos de DR.
# =============================================================
using Statistics, MLJ
import MLJModelInterface, MLJBase

"""Información mutua (en nats) entre la variable x, discretizada en B cuantiles, y la clase y."""
function informacionMutua(x::AbstractVector, y::AbstractVector; B::Int = 10)
    N = length(x)

    # 1. Puntos de corte: cuantiles 1/B, 2/B, …, (B-1)/B
    cortes = quantile(x, (1:B-1) ./ B)

    # 2. Intervalo de cada valor (1..B): nº de cortes ≤ valor, + 1
    bins = searchsortedlast.(Ref(cortes), x) .+ 1

    # 3. Tabla de conteos B × K
    clases = unique(y)
    n = [count((bins .== b) .& (y .== c)) for b in 1:B, c in clases]

    # 4. Probabilidades conjunta y marginales
    p  = n ./ N
    pb = sum(p, dims = 2)          # p(b): suma por filas
    pc = sum(p, dims = 1)        # p(c): suma por columnas

    # 5. I(X;Y), solo celdas con p > 0
    return sum(p[i, j] * log(p[i, j] / (pb[i] * pc[j]))
               for i in 1:B, j in 1:length(clases) if p[i, j] > 0)
end

mutable struct MIFilter <: MLJModelInterface.Supervised
    k::Int          # número de variables que se conservan
end
MIFilter(; k::Int = 32) = MIFilter(k)

function MLJModelInterface.fit(model::MIFilter, verbosity::Int, X, y)
    Xmat    = Float64.(MLJBase.matrix(X))
    nombres = collect(MLJBase.schema(X).names)
    mis     = [informacionMutua(Xmat[:, j], y) for j in 1:size(Xmat, 2)]
    k       = min(model.k, length(mis))
    idxs    = sortperm(mis, rev = true)[1:k]       # las k de mayor información mutua
    fitresult = (idxs = idxs, selected_names = nombres[idxs])
    report    = (mi = mis, selected_features = nombres[idxs])
    return fitresult, nothing, report
end

function MLJModelInterface.transform(model::MIFilter, fitresult, X)
    Xmat = MLJBase.matrix(X)
    return MLJBase.table(Xmat[:, fitresult.idxs], names = fitresult.selected_names)
end

MLJModelInterface.input_scitype(::Type{<:MIFilter})  = Table(Continuous)
MLJModelInterface.target_scitype(::Type{<:MIFilter}) = AbstractVector{<:Finite}
MLJModelInterface.output_scitype(::Type{<:MIFilter}) = Table(Continuous)