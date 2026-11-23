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
end

MMI.transform(m::ICAReintentos, fr, X) = MMI.transform(fr.modelo, fr.fitresult, X)

MMI.input_scitype(::Type{<:ICAReintentos})  = Table(Continuous)
MMI.output_scitype(::Type{<:ICAReintentos}) = Table(Continuous)

function experimentosReduccion(X, y, folds)
    casos = [("PCA", MVS.PCA(), :(reduccion.variance_ratio), [0.95, 0.99]),
            ("LDA", MVS.LDA(), :(reduccion.regcoef), [1e-6, 1e-3, 1e-1]),
            ("ICA", ICAReintentos(), :(reduccion.outdim), [5, 10, 20, 40])]
    tablas = DataFrame[]

    for (nombre, reduccion, ruta, valores) in casos
        red = RedClasificacion(reduccion = reduccion, clasificador = KNN(K = 10))
        rangos = [range(red, ruta, values = valores)]
        res = evaluarRejilla(red, rangos, X, y, folds)
        insertcols!(res, 1, :modelo => "KNN (K = 10)", :reduccion => nombre)
        push!(tablas, res)
    end

    return vcat(tablas...)
end