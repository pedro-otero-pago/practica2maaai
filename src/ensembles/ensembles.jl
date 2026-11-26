# =============================================================
#  ensembles.jl — Bagging (KNN, k = 5), AdaBoost (stumps) y EvoTree
#  sin reducción y con PCA (95 % de la varianza), validación temporal
# =============================================================
using MLJ, DataFrames, Statistics
import MLJMultivariateStatsInterface as MVS

AdaBoostStump = @load AdaBoostStumpClassifier pkg=DecisionTree verbosity=0
EvoTree       = @load EvoTreeClassifier pkg=EvoTrees verbosity=0

const N_BAGGING       = [10, 50]     # enunciado
const N_ADABOOST      = 5            # enunciado
const N_EVOTREE       = [50, 100]    # enunciado
const ETA_EVOTREE     = 0.2          # enunciado

# Las dos variantes de datos que permite el enunciado
reduccionesEnsembles() = [("Sin reducción", nothing),
                          ("PCA 95 %", MVS.PCA(variance_ratio = 0.95))]

"""Bagging de KNN (K = 5). rng = 104: muestreo bootstrap reproducible."""
nuevoBagging() = EnsembleModel(model = KNN(K = 5), n = 10, rng = 104)

"""AdaBoost con stumps (árboles de profundidad 1). rng = 104 para reproducibilidad."""
nuevoAdaBoost() = AdaBoostStump(n_iter = N_ADABOOST, rng = 104)

"""EvoTree (gradient boosting). Sin submuestreo de filas ni columnas: es determinista."""
nuevoEvoTree() = EvoTree(nrounds = 50, eta = ETA_EVOTREE)

"""
Valida Bagging, AdaBoost y EvoTree, sin reducción y con PCA al 95 %,
con los folds temporales. Devuelve una fila por configuración.
"""
function experimentosEnsembles(X, y, folds)
    # (nombre, constructor, hiperparámetro de la rejilla, valores)
    modelos = [("Bagging (KNN, K = 5)", nuevoBagging,  :(clasificador.n),       N_BAGGING),
               ("AdaBoost (stumps)",    nuevoAdaBoost, :(clasificador.n_iter),  [N_ADABOOST]),
               ("EvoTree",              nuevoEvoTree,  :(clasificador.nrounds), N_EVOTREE)]
    tablas = DataFrame[]

    for (nombreRed, reduccion) in reduccionesEnsembles()
        for (nombre, nuevo, ruta, valores) in modelos
            red    = RedClasificacion(reduccion = reduccion, clasificador = nuevo())
            rangos = [range(red, ruta, values = valores)]
            res    = evaluarRejilla(red, rangos, X, y, folds)
            insertcols!(res, 1, :modelo => nombre, :reduccion => nombreRed)
            push!(tablas, res)
        end
    end

    return vcat(tablas...)
end


"""
Entrena `red` con el entrenamiento de un fold y devuelve, para cada clase de su
validación: filas reales, veces que se predice y fracción acertada (recall).
"""
function aciertoPorClase(red, X, y, fold)
    tr, va = fold
    maq = machine(red, X, y)
    fit!(maq, rows = tr, verbosity = 0)
    pred = predict_mode(maq, rows = va)
    yv   = y[va]
    cls  = levels(yv)
    return DataFrame(clase     = cls,
                     reales    = [sum(yv .== c) for c in cls],
                     predichas = [sum(pred .== c) for c in cls],
                     acierto   = [round(mean(pred[yv .== c] .== c), digits = 2) for c in cls])
end