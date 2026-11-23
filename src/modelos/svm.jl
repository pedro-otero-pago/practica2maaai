# =============================================================
#  svm.jl — Experimentos con SVM (C = 0.1, 0.5, 1.0)
#  sin filtro y con filtro RFE (k del filtro = 16, 32, 64)
# =============================================================

using MLJ, DataFrames

SVM = @load ProbabilisticSVC pkg=LIBSVM verbosity=0

const VALORES_C_SVM = [0.1, 0.5, 1.0] # Parámetros determinados del enunciado

function experimentosSVM(X, y, folds)
    filtros = [("Sin filtro", nothing), ("RFE", RFEFilter())]
    tablas = DataFrame[]

    for (nombre, filtro) in filtros
        red = RedClasificacion(filtro = filtro, clasificador = SVM())
        rangos = Any[range(red, :(clasificador.cost), values = VALORES_C_SVM)]
        if filtro !== nothing
            pushfirst!(rangos, range(red, :(filtro.k), values = VALORES_K_FILTRO))
        end
        res = evaluarRejilla(red, rangos, X, y, folds)
        insertcols!(res, 1, :modelo => "SVM", :filtro => nombre)
        push!(tablas, res)
    end

    return vcat(tablas...)
end