# =============================================================
#  reduccion.jl — PCA, LDA e ICA dentro de la red (clasificador KNN, K = 10)
#  Los modelos se toman con el módulo explícito (MVS): el @load daba el mismo
#  tipo (PCA) para los tres.
# =============================================================
using MLJ, DataFrames
import MLJMultivariateStatsInterface as MVS
import MLJModelInterface as MMI
using Random

"""
ICA con reintentos: FastICA parte de una matriz aleatoria y a veces no converge.
Si falla, se vuelve a ajustar con otra semilla (fija, para que sea reproducible).
Al terminar, se restaura el estado del generador para no afectar las ejecuciones a posteriori
"""
# Ambas las 2 funciones de reintentos se han constrido con el fin de evitar fallos y errores en la ejecución del main
mutable struct ICAReintentos <: MMI.Unsupervised
    outdim::Int
    maxiter::Int
    tol::Float64
    intentos::Int
    semilla::Int
end
ICAReintentos(; outdim = 10, maxiter = 1000, tol = 1e-4, intentos = 10, semilla = 104) =
    ICAReintentos(outdim, maxiter, tol, intentos, semilla)

function MMI.fit(m::ICAReintentos, verbosity, X)
    estado = copy(Random.default_rng()) # estado del generador antes de tocarlo
    try
        for intento in 1:m.intentos
            Random.seed!(m.semilla + intento)
            ica = MVS.ICA(outdim = m.outdim, maxiter = m.maxiter, tol = m.tol)
            try
                fr, _, _ = MMI.fit(ica, 0, X)
                return (modelo = ica, fitresult = fr), nothing, (intentoUsado = intento,)
            catch e
                occursin("converge", sprint(showerror, e)) || rethrow()   # solo se reintenta si no converge
            end
        end
        error("ICA no converge tras $(m.intentos) intentos (outdim = $(m.outdim))")
    finally
        copy!(Random.default_rng(), estado) # se restaura siempre
    end
end

MMI.transform(m::ICAReintentos, fr, X) = MMI.transform(fr.modelo, fr.fitresult, X)

MMI.input_scitype(::Type{<:ICAReintentos})  = Table(Continuous)
MMI.output_scitype(::Type{<:ICAReintentos}) = Table(Continuous)

"""
Se evalua cada valor de ruta por separado. En caso de que la red no converja (ICA):
la fila se registra como tal en vez de parar la ejecución ("no converge")
"""
function evaluarRejillaSegura(red, ruta, valores, X, y, folds)
    filas = DataFrame[]
    for v in valores
        try
            res = evaluarRejilla(red, [range(red, ruta, values = [v])], X, y, folds)
            res.estado = fill("ok", nrow(res))
            push!(filas, res)
        catch e
            occursin("converge", sprint(showerror, e)) || rethrow() # Culaquier otro error para
            push!(filas, DataFrame(configuracion = "$ruta = $v",
                                    balanced_acc = missing, accuracy = missing,
                                    balanced_fold = missing, accuracy_fold = missing,
                                    estado = "no converge"))
        end
    end
    return vcat(filas...)
end

function experimentosReduccion(X, y, folds)
    casos = [("PCA", MVS.PCA(), :(reduccion.variance_ratio), [0.95, 0.99]),
            ("LDA", MVS.LDA(), :(reduccion.regcoef), [1e-6, 1e-3, 1e-1]),
            ("ICA", ICAReintentos(), :(reduccion.outdim), [5, 10, 20, 40, 60, 100])]
    tablas = DataFrame[]

    for (nombre, reduccion, ruta, valores) in casos
        red = RedClasificacion(reduccion = reduccion, clasificador = KNN(K = 10))
        res = evaluarRejillaSegura(red, ruta, valores, X, y, folds)
        insertcols!(res, 1, :modelo => "KNN (K = 10)", :reduccion => nombre)
        push!(tablas, res)
    end

    return vcat(tablas...)
end