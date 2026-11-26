# =============================================================
#  mlp.jl — Experimentos con MLP (arquitecturas [50], [100] y [100, 50])
#  sin filtro y con filtros de Pearson, Spearman y Kendall (k = 16, 32, 64)
# =============================================================
using MLJ, DataFrames
import MLJFlux, Flux

ClasificadorMLP = @load NeuralNetworkClassifier pkg=MLJFlux verbosity=0

const ARQUITECTURAS_MLP = [[50], [100], [100, 50]]   # enunciado
const EPOCAS_MLP = 50    # el enunciado no lo fija (MLJFlux trae 10 por defecto)
const LOTE_MLP   = 64    # MLJFlux trae batch_size = 1 por defecto: demasiado lento


# Builder propio: MLJFlux.MLP fija el número de capas en el tipo (MLP{1}, MLP{2}),
# lo que impide poner [50] y [100, 50] en la misma rejilla. Con un Vector{Int}
# todas las arquitecturas son del mismo tipo.
mutable struct CapasMLP <: MLJFlux.Builder
    hidden::Vector{Int}
end

function MLJFlux.build(b::CapasMLP, rng, n_in, n_out)
    init   = Flux.glorot_uniform(rng)
    capas  = Any[]
    entrada = n_in
    for h in b.hidden
        push!(capas, Flux.Dense(entrada => h, Flux.relu; init = init))
        entrada = h
    end
    push!(capas, Flux.Dense(entrada => n_out; init = init))   # salida: una neurona por clase
    return Flux.Chain(capas...)
end


"""MLP base; la arquitectura se cambia en la rejilla. rng = 104 para reproducibilidad."""
nuevoMLP() = ClasificadorMLP(builder    = CapasMLP([50]),
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

    tablas   = DataFrame[]

    for (nombre, filtro) in filtros
        red    = RedClasificacion(filtro = filtro, clasificador = nuevoMLP())
        rangos = Any[range(red, :(clasificador.builder.hidden), values = ARQUITECTURAS_MLP)]
        if filtro !== nothing
            pushfirst!(rangos, range(red, :(filtro.k), values = VALORES_K_FILTRO))
        end
        res = evaluarRejilla(red, rangos, X, y, folds)
        insertcols!(res, 1, :modelo => "MLP", :filtro => nombre)
        push!(tablas, res)
    end

    return vcat(tablas...)
end