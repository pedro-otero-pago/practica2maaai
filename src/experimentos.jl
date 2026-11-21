# =============================================================
#  experimentos.jl — Validación cruzada por batches de una red
#  sobre una rejilla de hiperparámetros (TunedModel + Grid)
# =============================================================
using MLJ, DataFrames

# balanced_accuracy primero: TunedModel elige la mejor configuración con la 1.ª medida.
# Sin F1 macro: en el fold 2 (sin tolueno en validación) cuenta la clase 6 como 0.
const MEDIDAS = [balanced_accuracy, accuracy]

"""Valor de un hiperparámetro dada su ruta, p. ej. :(clasificador.K) → modelo.clasificador.K"""
valorEn(modelo, ruta) = foldl(getproperty, Symbol.(split(string(ruta), ".")); init = modelo)

"""
Evalúa `red` con cada combinación de `rangos` (uno o varios `range`) usando los folds
por batches. Devuelve una fila por configuración: valores probados, media y valores por fold.
"""
function evaluarRejilla(red, rangos, X, y, folds)
    rangos = rangos isa AbstractVector ? rangos : [rangos]
    tm = TunedModel(model      = red,
                    tuning     = Grid(shuffle = false),     # sin barajar: orden reproducible
                    resampling = folds,
                    range      = rangos,
                    measures   = MEDIDAS)
    mach = machine(tm, X, y)
    fit!(mach, verbosity = 0)
    h = report(mach).history

    return DataFrame(
        configuracion = [join(["$(r.field) = $(valorEn(x.model, r.field))" for r in rangos], ", ") for x in h],
        balanced_acc  = [x.measurement[1] for x in h],
        accuracy      = [x.measurement[2] for x in h],
        balanced_fold = [x.per_fold[1] for x in h],
        accuracy_fold = [x.per_fold[2] for x in h])
end