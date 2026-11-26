# =============================================================
#  red.jl — Learning Network (patrón ANOVA_Pipeline, Tutorial 02)
#  imputador -> minmax -> [filtro] -> [reducción] -> clasificador
# =============================================================
using MLJ
import MLJBase
import MLJModelInterface
import MLJBase: source, machine, node


"""
Red de clasificación: imputador -> Min–Max -> [filtro] -> [reducción] -> clasificador
"""
mutable struct RedClasificacion <: MLJBase.ProbabilisticNetworkComposite
    imputador
    normalizador
    filtro
    reduccion
    clasificador
end

# Constructor por nombre. Imputador y Min–Max llevan valores por defecto;
# filtro y reducción son opcionales (nothing); el clasificador es obligatorio.
RedClasificacion(; imputador = ImputadorKNN(), normalizador = NormalizadorMinMax(),
                   filtro = nothing, reduccion = nothing, clasificador) =
    RedClasificacion(imputador, normalizador, filtro, reduccion, clasificador)

# Declaramos la entrada y salida de los modelos base para Stack
MLJModelInterface.input_scitype(::Type{<:RedClasificacion})  = Table(Union{Missing, Continuous})
MLJModelInterface.target_scitype(::Type{<:RedClasificacion}) = AbstractVector{<:Finite}

function MLJBase.prefit(model::RedClasificacion, verbosity, X, y)
    Xs = source(X)
    ys = source(y)

    # Etapa 1: imputación (no supervisada)
    m_imp = machine(:imputador, Xs)
    Z1    = MLJ.transform(m_imp, Xs)

    # Etapa 2: Min–Max (no supervisada)
    m_norm = machine(:normalizador, Z1)
    Z2     = MLJ.transform(m_norm, Z1)

    # Etapa 3: filtro (supervisado) u omitido
    if model.filtro === nothing
        Z3 = Z2
    else
        m_filtro = machine(:filtro, Z2, ys)
        Z3       = MLJ.transform(m_filtro, Z2)
    end

    # Etapa 4: reducción. Puede ser supervisada (LDA) o no (PCA, ICA)
    if model.reduccion === nothing
        Z4 = Z3
    elseif model.reduccion isa MLJModelInterface.Supervised
        m_red = machine(:reduccion, Z3, ys)
        Z4    = MLJ.transform(m_red, Z3)
    else
        m_red = machine(:reduccion, Z3)
        Z4    = MLJ.transform(m_red, Z3)
    end

    # Etapa 5: clasificador (supervisado)
    m_clf = machine(:clasificador, Z4, ys)
    yhat  = MLJ.predict(m_clf, Z4)

    return (predict = yhat,)
end