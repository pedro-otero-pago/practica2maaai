# ============================================================
#  Práctica 2 — MAAAI · validacion.jl
#  Validación cruzada temporal de 5 folds por batches (ventana creciente)
# ============================================================

using DataFrames

# Validación cruzada TEMPORAL con ventana creciente (indicación del profesor):
# cada fold valida un bloque de batches y entrena SOLO con los batches anteriores.
# El batch 1 no se valida nunca: no hay datos anteriores con los que entrenar.
const FOLDS_VALIDACION = [[2], [3, 4], [5, 6], [7, 8], [9, 10]]


# Recibimos el batch de cada fila del conjunto de train (limpio.batch[idxTrain])
# Devolvemos un vector de pares (idxTrain, idxVal) para cada fold de validación cruzada
function foldsPorBatch(batches::AbstractVector{<:Integer}, grupos = FOLDS_VALIDACION)
    pares = Tuple{Vector{Int}, Vector{Int}}[]
    for g in grupos
        val = findall(b -> b in g, batches)         # filas del batch en el grupo de validación
        train = findall(b -> b < minimum(g), batches)          # filas de los batches ANTERIORES
        push!(pares, (train, val))
    end
    return pares
end


# Comprobaciones de los folds para main.jl
# 1. Todo el entrenamiento de cada fold es anterior a su validación
# 2. Ninguna fila está a la vez en entrenamiento y validación
# 3. Se validan una sola vez todas las filas salvo las del batch 1
function comprobarFolds(folds, batches::AbstractVector{<:Integer}, classes::AbstractVector{<:Integer})
    for (k, (tr, va)) in enumerate(folds)
        @assert maximum(batches[tr]) < minimum(batches[va])   # todo el train es anterior a la validación
        @assert isempty(intersect(tr, va))                      # ninguna fila en los dos
    end
    # Cada fila está como mucho una vez en validación, y se validan todas salvo las del batch 1
    @assert sort(reduce(vcat, [f[2] for f in folds])) == findall(b -> b > 1, batches)

    todas = sort(unique(classes))
    resumen = DataFrame(fold = Int[], batchesTrain = Vector{Int}[], batchesValidacion = Vector{Int}[],
                    filasTrain = Int[], filasValidacion = Int[], clasesAusentesEnVal = Vector{Int}[])
    for (k, (tr, va)) in enumerate(folds)
        push!(resumen, (k, sort(unique(batches[tr])), sort(unique(batches[va])), length(tr), length(va), setdiff(todas, unique(classes[va]))))
    end
    return resumen
end