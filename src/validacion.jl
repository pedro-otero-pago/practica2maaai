# ============================================================
#  Práctica 2 — MAAAI · validacion.jl
#  Validación cruzada 5-fold por batches (batch-wise)
# ============================================================

using DataFrames

# Cada fold de validación son dos batches consecutivos completos
# Ningún batch está en validación y entrenamiento a la vez
# Los folds respetan el orden temporal de los batches (no se mezclan)
const FOLDS_VALIDACION = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]


# Recibimos el batch de cada fila del conjunto de train (limpio.batch[idxTrain])
# Devolvemos un vector de pares (idxTrain, idxVal) para cada fold de validación cruzada
function foldsPorBatch(batches::AbstractVector{<:Integer}, grupos = FOLDS_VALIDACION)
    pares = Tuple{Vector{Int}, Vector{Int}}[]
    for g in grupos
        val = findall(b -> b in g, batches)         # filas del batch en el grupo de validación
        train = findall(b -> !(b in g), batches)    # filas del batch fuera del grupo de validación
        push!(pares, (train, val))
    end
    return pares
end


# Comprobaciones de los folds para main.jl
# 1. Ningún batch aparece a la vez en train y validación
# 2. Cada fila está exactamente una vez en el conjunto de validación
function comprobarFolds(folds, batches::AbstractVector{<:Integer}, classes::AbstractVector{<:Integer})
    n = length(batches)
    for (k, (tr, va)) in enumerate(folds)
        @assert isempty(intersect(batches[tr], batches[va])) # Hay batches en ambos conj. a la vez
        @assert isempty(intersect(tr, va)) # Hay filas en ambos conj. a la vez
    end
    @assert sort(reduce(vcat, [f[2] for f in folds])) == collect(1:n) # Cada fila debe estar en el conj. de validación 1 vez

    todas = sort(unique(classes))
    resumen = DataFrame(fold = Int[], batchesValidacion = Vector{Int}[], filasTrain = Int[], 
                        filasValidacion = Int[], clasesAusentesEnVal = Vector{Int}[])
    for (k, (tr, va)) in enumerate(folds)
        push!(resumen, (k, sort(unique(batches[va])), length(tr), length(va), setdiff(todas, unique(classes[va]))))
    end
    return resumen
end