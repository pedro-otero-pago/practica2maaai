# ============================================================
#  Práctica 2 — MAAAI · minmax.jl
#  MLJ de normalización Min-Max: se ajusta con entrenamiento
#  Se aplica a validación y test (sin data leakage).
#  Va en el pipeline después del imputador KNN (imputacion.jl)
# ============================================================

using MLJ
import MLJModelInterface as MMI

# Transformador no supervisado. Los hiperparámetros son el rango de salida.
# Fórmula -> x' = lower + (x - min) / (max - min) * (upper - lower)
mutable struct NormalizadorMinMax <: MMI.Unsupervised
    lower::Float64
    upper::Float64
end
NormalizadorMinMax(; lower = 0.0, upper = 1.0) = NormalizadorMinMax(Float64(lower), Float64(upper))

# fit: aprende los valores mínimo y máximo de cada columna con los datos que recibe
# En la CrossValidation, estos datos se obtienen de los datos de entrenamiento de cada fold
function MMI.fit(m::NormalizadorMinMax, verbosity, X)
    M = Float64.(MMI.matrix(X))     # matriz de filas por variables (sin datos faltantes)
    mins = vec(minimum(M, dims=1))
    maxs = vec(maximum(M, dims=1))
    nombres = MMI.schema(X).names   # devolver el nombre de las columnas
    fitresult = (mins = mins, maxs = maxs, nombres = nombres)
    return fitresult, nothing, (minimos = mins, maximos = maxs)
end

# transform: aplica lo aprendido en fit a cualquier dato (entrenamiento, validación o test).
# En validación y test los valores pueden salir fuera de [lower, upper]: es normal
# (deriva temporal), no se recortan.

# Si una columna es constante (max == min) se divide entre 1 para evitar dividir entre 0.
function MMI.transform(m::NormalizadorMinMax, fr, Xnueva)
    M = Float64.(MMI.matrix(Xnueva))
    rango = fr.maxs .- fr.mins
    rango[rango .== 0] .= 1.0
    Z = m.lower .+ (M .- permutedims(fr.mins)) ./ permutedims(rango) .* (m.upper - m.lower)
    return MMI.table(Z; names = fr.nombres)
end

MMI.input_scitype(::Type{<:NormalizadorMinMax}) = Table(Continuous)
MMI.output_scitype(::Type{<:NormalizadorMinMax}) = Table(Continuous)