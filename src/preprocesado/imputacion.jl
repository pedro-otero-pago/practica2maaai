# ============================================================
#  Práctica 2 — MAAAI · imputacion.jl
#  Imputación KNN como nodo MLJ: se ajusta SOLO con entrenamiento
#  y después se aplica a validación y test (sin data leakage).
# ============================================================

using MLJ
import MLJModelInterface as MMI
using Statistics

mutable struct ImputadorKNN <: MMI.Unsupervised
    k::Int        # número de vecinos
end
ImputadorKNN(; k = 5) = ImputadorKNN(k)

function MMI.fit(m::ImputadorKNN, verbosity, X)
    M = MMI.matrix(X)                                  # filas × variables (puede tener missing)
    completas = [all(!ismissing, fila) for fila in eachrow(M)]
    ref = Float64.(M[completas, :])                    # posibles vecinos: filas completas de train
    rango = vec(maximum(ref, dims = 1) .- minimum(ref, dims = 1))
    rango[rango .== 0] .= 1.0                          # evita dividir entre 0
    nombres = MMI.schema(X).names                      # nombres de las columnas, para devolverlas igual
    fitresult = (ref = ref, rango = rango, nombres = nombres)
    return fitresult, nothing, (vecinosPosibles = size(ref, 1),)
end

function MMI.transform(m::ImputadorKNN, fr, Xnew)
    M = Matrix{Union{Missing, Float64}}(MMI.matrix(Xnew))
    for i in 1:size(M, 1)
        faltan = findall(ismissing, M[i, :])
        if isempty(faltan)
            continue                                   # fila completa: no se toca
        end
        obs = setdiff(1:size(M, 2), faltan)            # variables que sí tiene la fila
        x = Float64.(M[i, obs])
        # Distancia euclídea (escalada por el rango) a cada fila de referencia
        d = [sum(((fr.ref[r, obs] .- x) ./ fr.rango[obs]) .^ 2) for r in 1:size(fr.ref, 1)]
        vecinos = sortperm(d)[1:m.k]                   # los k más cercanos
        for j in faltan
            M[i, j] = mean(fr.ref[vecinos, j])
        end
    end
    return MMI.table(Float64.(M); names = fr.nombres)
end

MMI.input_scitype(::Type{<:ImputadorKNN})  = Table(Union{Continuous, Missing})
MMI.output_scitype(::Type{<:ImputadorKNN}) = Table(Continuous)