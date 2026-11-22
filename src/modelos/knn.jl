# =============================================================
#  knn.jl — Experimentos con KNN (k = 1, 10, 20) sin filtro,
#  con filtro ANOVA y con filtro MI (k del filtro = 16, 32, 64)
# =============================================================
using MLJ, DataFrames

KNN = @load KNNClassifier pkg=NearestNeighborModels verbosity=0

const VALORES_K_KNN    = [1, 10, 20]     # enunciado
const VALORES_K_FILTRO = [16, 32, 64]    # 1/8, 1/4 y 1/2 de las 128 variables (acordado)

"""
Valida KNN con cada filtro (sin filtro, ANOVA, MI) sobre la rejilla de K
y de k del filtro. Devuelve una tabla con una fila por configuración.
"""
function experimentosKNN(X, y, folds)
    filtros = [("Sin filtro", nothing), ("ANOVA", ANOVAFilter()), ("MI", MIFilter())]
    tablas  = DataFrame[]

    for (nombre, filtro) in filtros
        red    = RedClasificacion(filtro = filtro, clasificador = KNN())
        rangos = [range(red, :(clasificador.K), values = VALORES_K_KNN)]
        if filtro !== nothing
            pushfirst!(rangos, range(red, :(filtro.k), values = VALORES_K_FILTRO))
        end
        res = evaluarRejilla(red, rangos, X, y, folds)
        insertcols!(res, 1, :modelo => "KNN", :filtro => nombre)
        push!(tablas, res)
    end

    return vcat(tablas...)
end