# =============================================================
#  finales.jl — las cinco mejores combinaciones de Stacking y boosting,
#  entrenados con todo el entrenamiento y evaluados una vez en el test
# =============================================================

using MLJ, DataFrames
import MLJMultivariateStatsInterface as MVS

XGB = @load XGBoostClassifier pkg = XGBoost verbosity = 0
LGBM = @load LGBMClassifier pkg = LightGBM verbosity = 0
CAT = @load CatBoostClassifier pkg = CatBoost verbosity = 0

function mejorKNN()
    return RedClasificacion(reduccion = MVS.LDA(regcoef = 1e-3), clasificador = KNN(K = 10))
end

function mejorSVM()
    return RedClasificacion(filtro = RFEFilter(k = 16), clasificador = SVM(cost = 1.0))
end

function mejorMLP()
    filtro = CorrelacionFilter(metodo = :spearman)
    filtro.k = 64
    mlp = nuevoMLP()
    mlp.builder = CapasMLP([100, 50])
    return RedClasificacion(filtro = filtro, clasificador = mlp)
end

function mejorBagging()
    bag = nuevoBagging()
    bag.n = 50
    return RedClasificacion(clasificador = bag)
end

function mejorAdaBoost()
    return RedClasificacion(reduccion = MVS.PCA(variance_ratio = 0.95), clasificador = nuevoAdaBoost())
end 

cincoMejores() = [("KNN (LDA, K = 10)", mejorKNN()),
                    ("SVM (RFE k = 16, C = 1)", mejorSVM()),
                    ("MLP (Spearman k = 64, [100, 50])", mejorMLP()),
                    ("Bagging (n = 10)", mejorBagging()),
                    ("AdaBoost (PCA 95 %)", mejorAdaBoost())]

# Stacking: base = SVM + KNN + MLP; metaclasificador = MLP.
# CV(nfolds = 5) sin barajar: las predicciones de los base salen de bloques contiguos.
nuevoStacking() = Stack(metalearner = nuevoMLP(),
                        resampling  = CV(nfolds = 5),
                        svm = mejorSVM(), knn = mejorKNN(), mlp = mejorMLP())

# XGBoost, LightGBM y CatBoost con sus valores por defecto en la misma red
# imputador y Min-Max con ajuste solo en train
boostings() = [("XGBoost", RedClasificacion(clasificador = XGB())),
                ("LightGBM", RedClasificacion(clasificador = LGBM())),
                ("CatBoost", RedClasificacion(clasificador = CAT()))]

modelosCompuestos() = vcat([("Stacking (SVM + KNN + MLP → MLP)", nuevoStacking())], boostings())

"""
Entreno de cada modelo con todo el train y medición del del balanced_accuracy y accuracy en test.
"""
function evaluacionFinal(modelos, Xtrain, ytrain, Xtest, ytest)
    res = DataFrame(modelo = String[], balanced_acc = Float64[], accuracy = Float64[], segundos = Float64[])
    for (nombre, modelo) in modelos
        t = @elapsed begin
            maq = machine(modelo, Xtrain, ytrain)
            fit!(maq, verbosity = 0)
            pred = predict_mode(maq, Xtest)
            push!(res, (nombre, balanced_accuracy(pred, ytest), accuracy(pred, ytest), 0.0))
        end
        res.segundos[end] = round(t, digits = 1)
    end
    return res
end