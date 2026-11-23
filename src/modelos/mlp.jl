# =============================================================
#  mlp.jl — Experimentos con MLP (arquitecturas [50], [100] y [100, 50])
#  sin filtro y con filtros de Pearson, Spearman y Kendall (k = 16, 32, 64)
# =============================================================
using MLJ, DataFrames
import MLJFlux, Flux

ClasificadorMLP = @load NeuralNetworkClassifier pkg=MLJFlux verbosity=0

const ARQUITECTURAS_MLP = [(50,), (100,), (100, 50)]   # enunciado
const EPOCAS_MLP = 50    # el enunciado no lo fija (MLJFlux trae 10 por defecto)
const LOTE_MLP   = 64    # MLJFlux trae batch_size = 1 por defecto: demasiado lento

"""MLP base; la arquitectura se cambia en la rejilla. rng = 104 para reproducibilidad."""
nuevoMLP() = ClasificadorMLP(builder    = MLJFlux.MLP(hidden = (50,), σ = Flux.relu),
                             epochs     = EPOCAS_MLP,
                             batch_size = LOTE_MLP,
                             rng        = 104)

"""
Valida el MLP con cada filtro (sin filtro, Pearson, Spearman, Kendall) sobre la
rejilla de arquitecturas y de k del filtro. Devuelve una fila por configuración.
"""
function experimentosMLP(X, y, folds)
    filtros = [("Sin filtro", nothing),
               ("Pearson",  CorrelacionFilter(metodo = :pearson)),
               ("Spearman", CorrelacionFilter(metodo = :spearman)),
               ("Kendall",  CorrelacionFilter(metodo = :kendall))]
    builders = [MLJFlux.MLP(hidden = h, σ = Flux.relu) for h in ARQUITECTURAS_MLP]
    tablas   = DataFrame[]

    for (nombre, filtro) in filtros
        red    = RedClasificacion(filtro = filtro, clasificador = nuevoMLP())
        rangos = Any[range(red, :(clasificador.builder), values = builders)]
        if filtro !== nothing
            pushfirst!(rangos, range(red, :(filtro.k), values = VALORES_K_FILTRO))
        end
        res = evaluarRejilla(red, rangos, X, y, folds)
        insertcols!(res, 1, :modelo => "MLP", :filtro => nombre)
        push!(tablas, res)
    end

    return vcat(tablas...)
end