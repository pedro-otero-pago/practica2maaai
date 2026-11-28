# =============================================================
#  votacion.jl — Random Forest y Hard Voting (B3)
# =============================================================
using MLJ

RandomForest = @load RandomForestClassifier pkg=DecisionTree verbosity=0

const N_ARBOLES_RF     = 500     
const PROFUNDIDAD_RF   = 10     

"""
Random Forest dentro de la red, sin filtro ni reducción: usa las 128 variables tras
imputar y normalizar (igual que XGBoost, LightGBM y CatBoost), lo que permite analizar
después la importancia de cada variable. rng = 104 para que sea reproducible.
"""
nuevoRandomForest() = RedClasificacion(clasificador = RandomForest(n_trees = N_ARBOLES_RF,
                                                                   max_depth = PROFUNDIDAD_RF,
                                                                   rng = 104))

using Random, Statistics
import MLJModelInterface

"""
Hard Voting con un único modelo base: reparte el entrenamiento al azar en `particiones`
partes (semilla fija) y entrena una copia del modelo en cada una. Predice por mayoría.
"""
mutable struct VotacionSVM <: MLJModelInterface.Probabilistic
    modelo              # modelo base (la mejor SVM: la red completa)
    particiones::Int    # 3 (enunciado)
    semilla::Int        # 104 (enunciado)
end
VotacionSVM(; modelo, particiones = 3, semilla = 104) = VotacionSVM(modelo, particiones, semilla)

function MLJModelInterface.fit(m::VotacionSVM, verbosity::Int, X, y)
    n = length(y)
    # Orden aleatorio reproducible con un generador PROPIO (no toca el global)
    orden  = randperm(MersenneTwister(m.semilla), n)
    # Reparto en `particiones` trozos: posiciones 1, 4, 7… / 2, 5, 8… / 3, 6, 9…
    trozos = [orden[i:m.particiones:n] for i in 1:m.particiones]

    maquinas = [begin
                    maq = machine(deepcopy(m.modelo), selectrows(X, t), y[t])
                    fit!(maq, verbosity = 0)
                    maq
                end for t in trozos]

    fitresult = (maquinas = maquinas, clases = classes(y[1]))
    report    = (tamanos = length.(trozos),)
    return fitresult, nothing, report
end

"""
Gas ganador en una medición: el más votado por las SVM. Si hay empate (con 3 votantes
y 6 gases solo ocurre cuando las tres votan distinto), gana el gas con mayor
probabilidad media entre los empatados.
"""
function decidirVoto(votos, probs, clases)
    conteo    = [count(==(c), votos) for c in clases]       # votos que recibe cada gas
    empatados = clases[conteo .== maximum(conteo)]              # gases con el máximo de votos
    length(empatados) == 1 && return empatados[1]           # sin empate: el más votado
    medias = [mean(pdf(p, c) for p in probs) for c in empatados]  # probabilidad media de cada empatado
    return empatados[argmax(medias)]                           # el de mayor probabilidad media
end

function MLJModelInterface.predict_mode(m::VotacionSVM, fr, Xnew)
    probs = [MLJ.predict(maq, Xnew) for maq in fr.maquinas]   # distribución de cada SVM
    votos = [mode.(p) for p in probs]                         # el gas que vota cada SVM
    pred  = similar(votos[1])
    for i in eachindex(pred)
        pred[i] = decidirVoto([v[i] for v in votos], [p[i] for p in probs], fr.clases)
    end
    return pred
end

# Hard Voting: la "probabilidad" es 1 para el gas elegido y 0 para el resto (como en el Tutorial 04)
function MLJModelInterface.predict(m::VotacionSVM, fr, Xnew)
    yhat = MLJModelInterface.predict_mode(m, fr, Xnew)
    return [UnivariateFinite(fr.clases, [c == yhat[i] ? 1.0 : 0.0 for c in fr.clases])
            for i in eachindex(yhat)]
end

MLJModelInterface.input_scitype(::Type{<:VotacionSVM})  = Table(Union{Missing, Continuous})
MLJModelInterface.target_scitype(::Type{<:VotacionSVM}) = AbstractVector{<:Finite}