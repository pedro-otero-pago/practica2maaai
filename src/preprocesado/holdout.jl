# ============================================================
#  Práctica 2 — MAAAI · holdout.jl
#  Partición holdout temporal: el 10 % más reciente es el test
# ============================================================

function holdoutTemporal(datos::DataFrame, porcentajeTest::Real = 0.1)
    N = nrow(datos)
    nTest = round(Int, porcentajeTest * N)
    ordenTemporal = sortperm(datos.orden)          # se utiliza sorperm para asegurarse de que en caso 
                                                   # de que se eliminen filas, siga funcionando en el orden adecuado
    idxTest = last(ordenTemporal, nTest)           # las nTest más recientes
    idxTrain = first(ordenTemporal, N - nTest)     # el resto

    return (idxTrain, idxTest)
end